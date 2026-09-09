from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from fastapi.testclient import TestClient

from monitoring.api import create_app
from monitoring.store import MonitoringStore


def client(store: MonitoringStore) -> TestClient:
    return TestClient(create_app(store))


def baseline_payload(name: str, values, kind: str = "numeric") -> dict:
    return {
        "model_id": "fraud",
        "model_version": "1",
        "feature_name": name,
        "kind": kind,
        "values": values,
    }


def test_window_api_applies_fdr_persists_history_and_opens_alert(store: MonitoringStore) -> None:
    api = client(store)
    rng = np.random.default_rng(3)
    reference = rng.normal(size=500).tolist()
    assert api.post("/v1/baselines", json=baseline_payload("amount", reference)).status_code == 200
    assert (
        api.post(
            "/v1/baselines",
            json=baseline_payload("country", ["TR"] * 400 + ["DE"] * 100, "categorical"),
        ).status_code
        == 200
    )
    request = {
        "window_id": "2026-09-09T10:00Z",
        "alert_after_consecutive_windows": 1,
        "features": [
            baseline_payload("amount", rng.normal(1, 1, 300).tolist()),
            baseline_payload("country", ["US"] * 300, "categorical"),
        ],
    }
    response = api.post("/v1/windows/evaluate", json=request)
    assert response.status_code == 200
    reports = response.json()["reports"]
    assert {report["severity"] for report in reports} == {"alert"}
    assert all(report["alert"]["state"] == "open" for report in reports)
    numeric_ks = next(
        metric for metric in reports[0]["metrics"] if metric["name"] == "ks_statistic"
    )
    assert numeric_ks["adjusted_pvalue"] is not None
    history = api.get(
        "/v1/models/fraud/versions/1/features/amount/history", params={"kind": "numeric"}
    )
    assert history.status_code == 200
    assert history.json()[0]["window_id"] == "2026-09-09T10:00Z"


def test_duplicate_window_is_a_conflict(store: MonitoringStore) -> None:
    api = client(store)
    values = list(range(20))
    api.post("/v1/baselines", json=baseline_payload("amount", values))
    request = {"window_id": "same", "features": [baseline_payload("amount", values)]}
    assert api.post("/v1/windows/evaluate", json=request).status_code == 200
    assert api.post("/v1/windows/evaluate", json=request).status_code == 409


def test_delayed_label_api_produces_performance_report(store: MonitoringStore) -> None:
    api = client(store)
    now = datetime(2026, 9, 9, tzinfo=UTC).isoformat()
    predictions = {
        "model_id": "fraud",
        "model_version": "1",
        "window_id": "w1",
        "predictions": [
            {"entity_id": "a", "score": 0.1, "predicted_at": now},
            {"entity_id": "b", "score": 0.9, "predicted_at": now},
        ],
    }
    labels = {
        "labels": [
            {"entity_id": "a", "label": 0, "observed_at": now},
            {"entity_id": "b", "label": 1, "observed_at": now},
        ]
    }
    assert api.post("/v1/predictions", json=predictions).status_code == 200
    assert api.post("/v1/labels", json=labels).status_code == 200
    report = api.post("/v1/models/fraud/versions/1/performance/w1/evaluate")
    assert report.status_code == 200
    assert report.json()["roc_auc"] == 1.0


def test_health_metrics_and_missing_baseline(store: MonitoringStore) -> None:
    api = client(store)
    assert api.get("/health").json() == {"status": "ok"}
    assert "drift_open_alerts" in api.get("/metrics").text
    response = api.post(
        "/v1/windows/evaluate",
        json={"window_id": "w", "features": [baseline_payload("missing", list(range(20)))]},
    )
    assert response.status_code == 404
