from __future__ import annotations

from datetime import UTC, datetime

import pytest

from monitoring.explanation_drift import (
    ExplanationDriftPolicy,
    ExplanationProfileIdentity,
)
from monitoring.models import AlertState
from monitoring.store import MonitoringStore

IDENTITY = ExplanationProfileIdentity("credit-risk", "2026.10", "tree-shap-v3")
NOW = datetime(2026, 10, 22, tzinfo=UTC)
BASELINE = {"income": -0.7, "debt": 0.8, "age": -0.1, "history": 0.2}
STABLE = {"income": -0.68, "debt": 0.82, "age": -0.09, "history": 0.19}
SHIFTED = {"income": 0.7, "debt": -0.8, "age": 0.9, "history": -0.6}
POLICY = ExplanationDriftPolicy(
    top_k=2,
    min_top_k_overlap=0.5,
    min_cosine_similarity=0.8,
    min_sign_agreement=0.75,
    max_normalized_l1_shift=0.5,
)


def test_explanation_baseline_is_content_addressed_and_versioned(store: MonitoringStore) -> None:
    first = store.register_explanation_baseline(IDENTITY, BASELINE, created_at=NOW)
    replay = store.register_explanation_baseline(
        IDENTITY,
        dict(reversed(list(BASELINE.items()))),
        created_at=NOW,
    )
    replacement = store.register_explanation_baseline(
        IDENTITY,
        {**BASELINE, "debt": 0.9},
        created_at=NOW,
    )

    assert replay.baseline_id == first.baseline_id
    assert replay.fingerprint == first.fingerprint
    assert replacement.baseline_id != first.baseline_id
    assert store.active_explanation_baseline(IDENTITY) == replacement


def test_explanation_baseline_survives_store_restart(tmp_path) -> None:
    database = tmp_path / "monitoring.db"
    baseline = MonitoringStore(database).register_explanation_baseline(IDENTITY, BASELINE)

    restored = MonitoringStore(database).active_explanation_baseline(IDENTITY)

    assert restored is not None
    assert restored.baseline_id == baseline.baseline_id
    assert restored.as_mapping() == BASELINE


def test_explanation_breaches_open_durable_alert_and_persist_policy(
    store: MonitoringStore,
) -> None:
    baseline = store.register_explanation_baseline(IDENTITY, BASELINE)

    first_report, pending = store.evaluate_explanation_window(
        IDENTITY,
        SHIFTED,
        window_id="window-1",
        policy=POLICY,
        evaluated_at=NOW,
    )
    _, opened = store.evaluate_explanation_window(
        IDENTITY,
        SHIFTED,
        window_id="window-2",
        policy=POLICY,
        evaluated_at=NOW,
    )
    history = store.recent_explanation_runs(IDENTITY)

    assert not first_report.passed
    assert pending is not None and pending["state"] == AlertState.PENDING
    assert opened is not None and opened["state"] == AlertState.OPEN
    assert opened["feature_name"] == "explanation/tree-shap-v3"
    assert history[0]["baseline_id"] == baseline.baseline_id
    assert history[0]["policy"]["top_k"] == 2
    assert len(history) == 2


def test_duplicate_explanation_window_rolls_back_before_alert_transition(
    store: MonitoringStore,
) -> None:
    store.register_explanation_baseline(IDENTITY, BASELINE)
    store.evaluate_explanation_window(
        IDENTITY,
        SHIFTED,
        window_id="window-1",
        policy=POLICY,
    )

    with pytest.raises(ValueError, match="already been evaluated"):
        store.evaluate_explanation_window(
            IDENTITY,
            SHIFTED,
            window_id="window-1",
            policy=POLICY,
        )

    _, opened = store.evaluate_explanation_window(
        IDENTITY,
        SHIFTED,
        window_id="window-2",
        policy=POLICY,
    )
    assert opened is not None
    assert opened["state"] == AlertState.OPEN
    assert opened["consecutive_breaches"] == 2
    assert len(store.recent_explanation_runs(IDENTITY)) == 2


def test_stable_explanation_windows_resolve_existing_alert(store: MonitoringStore) -> None:
    store.register_explanation_baseline(IDENTITY, BASELINE)
    _, opened = store.evaluate_explanation_window(
        IDENTITY,
        SHIFTED,
        window_id="breach",
        policy=POLICY,
        open_after=1,
    )
    _, still_open = store.evaluate_explanation_window(
        IDENTITY,
        STABLE,
        window_id="stable-1",
        policy=POLICY,
    )
    stable_report, resolved = store.evaluate_explanation_window(
        IDENTITY,
        STABLE,
        window_id="stable-2",
        policy=POLICY,
    )

    assert opened is not None and opened["state"] == AlertState.OPEN
    assert still_open is not None and still_open["state"] == AlertState.OPEN
    assert stable_report.passed
    assert resolved is not None and resolved["state"] == AlertState.RESOLVED


def test_explanation_evaluation_requires_registered_baseline(store: MonitoringStore) -> None:
    with pytest.raises(ValueError, match="baseline not registered"):
        store.evaluate_explanation_window(IDENTITY, STABLE, window_id="window-1")
