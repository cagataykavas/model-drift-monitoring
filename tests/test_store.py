from __future__ import annotations

from datetime import UTC, datetime

import pytest

from monitoring.models import AlertState, BaselineIdentity, FeatureKind, Severity
from monitoring.store import MonitoringStore

IDENTITY = BaselineIdentity("fraud", "2026.09", "amount", FeatureKind.NUMERIC)
NOW = datetime(2026, 9, 9, tzinfo=UTC)


def test_baseline_registration_is_versioned_and_idempotent(store: MonitoringStore) -> None:
    first = store.register_baseline(IDENTITY, list(range(20)), created_at=NOW)
    replay = store.register_baseline(IDENTITY, list(range(20)), created_at=NOW)
    replacement = store.register_baseline(IDENTITY, list(range(1, 21)), created_at=NOW)

    assert replay.baseline_id == first.baseline_id
    assert replacement.baseline_id != first.baseline_id
    assert store.active_baseline(IDENTITY).baseline_id == replacement.baseline_id


def test_feature_kind_mismatch_is_rejected(store: MonitoringStore) -> None:
    store.register_baseline(IDENTITY, list(range(20)))
    wrong = BaselineIdentity("fraud", "2026.09", "amount", FeatureKind.CATEGORICAL)
    with pytest.raises(ValueError, match="kind"):
        store.active_baseline(wrong)


def test_alert_requires_consecutive_breaches_and_resolves_after_stability(
    store: MonitoringStore,
) -> None:
    first = store.observe_alert(IDENTITY, Severity.ALERT, open_after=2)
    assert first["state"] == AlertState.PENDING
    opened = store.observe_alert(IDENTITY, Severity.ALERT, open_after=2)
    assert opened["state"] == AlertState.OPEN
    assert opened["occurrences"] == 1

    acknowledged = store.acknowledge_alert(opened["alert_key"])
    assert acknowledged["state"] == AlertState.ACKNOWLEDGED
    still_active = store.observe_alert(IDENTITY, Severity.STABLE, resolve_after=2)
    assert still_active["state"] == AlertState.ACKNOWLEDGED
    resolved = store.observe_alert(IDENTITY, Severity.STABLE, resolve_after=2)
    assert resolved["state"] == AlertState.RESOLVED


def test_only_open_alert_can_be_acknowledged(store: MonitoringStore) -> None:
    pending = store.observe_alert(IDENTITY, Severity.WARNING, open_after=2)
    with pytest.raises(ValueError, match="open"):
        store.acknowledge_alert(pending["alert_key"])


def test_delayed_labels_join_predictions_by_entity(store: MonitoringStore) -> None:
    store.ingest_predictions(
        "fraud",
        "1",
        "window-1",
        [
            ("a", 0.1, NOW),
            ("b", 0.9, NOW),
            ("unlabeled", 0.8, NOW),
        ],
    )
    store.ingest_labels([("a", 0, NOW), ("b", 1, NOW)])
    report = store.evaluate_performance("fraud", "1", "window-1")
    assert report.joined_count == 2
    assert report.accuracy == 1.0


def test_performance_requires_at_least_one_joined_label(store: MonitoringStore) -> None:
    store.ingest_predictions("fraud", "1", "window", [("a", 0.2, NOW)])
    with pytest.raises(ValueError, match="non-zero"):
        store.evaluate_performance("fraud", "1", "window")
