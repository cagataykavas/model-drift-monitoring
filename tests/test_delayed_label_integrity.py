from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from monitoring.api import create_app
from monitoring.store import IngestionConflict, MonitoringStore

PREDICTED_AT = datetime(2026, 10, 1, tzinfo=UTC)


def test_prediction_replay_is_idempotent_and_reports_actual_inserts(
    store: MonitoringStore,
) -> None:
    values = [("account-1", 0.2, PREDICTED_AT)]

    assert store.ingest_predictions("credit", "v1", "oct", values) == 1
    assert store.ingest_predictions("credit", "v1", "oct", values) == 0


@pytest.mark.parametrize(
    "replacement",
    [
        ("account-1", 0.8, PREDICTED_AT),
        ("account-1", 0.2, PREDICTED_AT + timedelta(seconds=1)),
    ],
)
def test_conflicting_prediction_replay_is_rejected(
    store: MonitoringStore,
    replacement: tuple[str, float, datetime],
) -> None:
    store.ingest_predictions("credit", "v1", "oct", [("account-1", 0.2, PREDICTED_AT)])

    with pytest.raises(IngestionConflict, match="prediction idempotency"):
        store.ingest_predictions("credit", "v1", "oct", [replacement])


def test_prediction_batch_rolls_back_before_a_conflict(store: MonitoringStore) -> None:
    original = ("account-1", 0.2, PREDICTED_AT)
    store.ingest_predictions("credit", "v1", "oct", [original])

    with pytest.raises(IngestionConflict):
        store.ingest_predictions(
            "credit",
            "v1",
            "oct",
            [("account-2", 0.3, PREDICTED_AT), ("account-1", 0.9, PREDICTED_AT)],
        )

    assert store.ingest_predictions("credit", "v1", "oct", [("account-2", 0.3, PREDICTED_AT)]) == 1


def test_label_replay_is_idempotent_but_revision_requires_a_versioned_path(
    store: MonitoringStore,
) -> None:
    observed_at = PREDICTED_AT + timedelta(days=30)
    label = [("account-1", 1, observed_at)]

    assert store.ingest_labels(label) == 1
    assert store.ingest_labels(label) == 0
    with pytest.raises(IngestionConflict, match="label idempotency"):
        store.ingest_labels([("account-1", 0, observed_at)])
    with pytest.raises(IngestionConflict, match="label idempotency"):
        store.ingest_labels([("account-1", 1, observed_at + timedelta(seconds=1))])


@pytest.mark.parametrize("target", ["prediction", "label", "as_of"])
def test_naive_timestamps_fail_closed(store: MonitoringStore, target: str) -> None:
    naive = PREDICTED_AT.replace(tzinfo=None)

    with pytest.raises(ValueError, match="timezone-aware"):
        if target == "prediction":
            store.ingest_predictions("credit", "v1", "oct", [("a", 0.2, naive)])
        elif target == "label":
            store.ingest_labels([("a", 1, naive)])
        else:
            store.evaluate_performance("credit", "v1", "oct", as_of=naive)


def test_as_of_snapshot_excludes_labels_observed_later(store: MonitoringStore) -> None:
    store.ingest_predictions(
        "credit",
        "v1",
        "oct",
        [("a", 0.9, PREDICTED_AT), ("b", 0.1, PREDICTED_AT)],
    )
    store.ingest_labels(
        [
            ("a", 1, PREDICTED_AT + timedelta(days=1)),
            ("b", 0, PREDICTED_AT + timedelta(days=3)),
        ]
    )
    as_of = PREDICTED_AT + timedelta(days=2)

    report = store.evaluate_performance("credit", "v1", "oct", as_of=as_of)

    assert report.prediction_count == 2
    assert report.joined_count == 1
    assert report.label_coverage == 0.5
    assert report.evaluated_at == as_of
    assert (
        store.evaluate_performance(
            "credit", "v1", "oct", as_of=PREDICTED_AT + timedelta(days=4)
        ).joined_count
        == 2
    )


def test_label_observed_before_prediction_is_rejected_as_leakage(
    store: MonitoringStore,
) -> None:
    store.ingest_predictions("credit", "v1", "oct", [("a", 0.9, PREDICTED_AT + timedelta(days=1))])
    store.ingest_labels([("a", 1, PREDICTED_AT)])

    with pytest.raises(ValueError, match="predates its prediction"):
        store.evaluate_performance("credit", "v1", "oct", as_of=PREDICTED_AT + timedelta(days=2))


def test_prediction_after_as_of_is_rejected(store: MonitoringStore) -> None:
    store.ingest_predictions("credit", "v1", "oct", [("a", 0.9, PREDICTED_AT + timedelta(days=1))])

    with pytest.raises(ValueError, match="later than evaluation as_of"):
        store.evaluate_performance("credit", "v1", "oct", as_of=PREDICTED_AT)


def test_minimum_label_coverage_gate_rejects_partial_evidence(
    store: MonitoringStore,
) -> None:
    store.ingest_predictions(
        "credit",
        "v1",
        "oct",
        [("a", 0.9, PREDICTED_AT), ("b", 0.1, PREDICTED_AT)],
    )
    store.ingest_labels([("a", 1, PREDICTED_AT + timedelta(days=1))])

    with pytest.raises(ValueError, match=r"1/2 \(0.500000 < 0.750000\)"):
        store.evaluate_performance(
            "credit",
            "v1",
            "oct",
            as_of=PREDICTED_AT + timedelta(days=2),
            minimum_label_coverage=0.75,
        )


@pytest.mark.parametrize("coverage", [-0.1, 1.1, float("nan")])
def test_invalid_label_coverage_policy_is_rejected(store: MonitoringStore, coverage: float) -> None:
    with pytest.raises(ValueError, match="minimum_label_coverage"):
        store.evaluate_performance("credit", "v1", "oct", minimum_label_coverage=coverage)


def test_api_exposes_replay_conflicts_and_as_of_coverage(store: MonitoringStore) -> None:
    api = TestClient(create_app(store))
    predicted = PREDICTED_AT.isoformat()
    predictions = {
        "model_id": "credit",
        "model_version": "v1",
        "window_id": "oct",
        "predictions": [
            {"entity_id": "a", "score": 0.9, "predicted_at": predicted},
            {"entity_id": "b", "score": 0.1, "predicted_at": predicted},
        ],
    }
    assert api.post("/v1/predictions", json=predictions).json() == {"accepted": 2}
    assert api.post("/v1/predictions", json=predictions).json() == {"accepted": 0}

    conflicting = {**predictions, "predictions": [{**predictions["predictions"][0], "score": 0.1}]}
    assert api.post("/v1/predictions", json=conflicting).status_code == 409

    labels = {
        "labels": [
            {
                "entity_id": "a",
                "label": 1,
                "observed_at": (PREDICTED_AT + timedelta(days=1)).isoformat(),
            },
            {
                "entity_id": "b",
                "label": 0,
                "observed_at": (PREDICTED_AT + timedelta(days=3)).isoformat(),
            },
        ]
    }
    assert api.post("/v1/labels", json=labels).json() == {"accepted": 2}

    path = "/v1/models/credit/versions/v1/performance/oct/evaluate"
    as_of = PREDICTED_AT + timedelta(days=2)
    response = api.post(path, params={"as_of": as_of.isoformat()})
    assert response.status_code == 200
    assert response.json()["prediction_count"] == 2
    assert response.json()["joined_count"] == 1
    assert response.json()["label_coverage"] == 0.5
    assert (
        api.post(
            path,
            params={"as_of": as_of.isoformat(), "minimum_label_coverage": 0.75},
        ).status_code
        == 422
    )


def test_api_rejects_naive_ingestion_timestamp(store: MonitoringStore) -> None:
    api = TestClient(create_app(store))
    response = api.post(
        "/v1/predictions",
        json={
            "model_id": "credit",
            "model_version": "v1",
            "window_id": "oct",
            "predictions": [
                {"entity_id": "a", "score": 0.9, "predicted_at": "2026-10-01T00:00:00"}
            ],
        },
    )

    assert response.status_code == 422
    assert "timezone-aware" in response.json()["detail"]
