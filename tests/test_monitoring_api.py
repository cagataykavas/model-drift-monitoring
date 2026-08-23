from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

import monitoring.api as api_module
from monitoring.store import MonitoringStore


def client_for(tmp_path: Path) -> TestClient:
    api_module.store = MonitoringStore(tmp_path / "drift.db")
    return TestClient(api_module.app)


def test_stable_and_shifted_batches_are_recorded(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    rng = np.random.default_rng(4)
    baseline = rng.normal(0, 1, 500).tolist()
    stable = rng.normal(0.03, 1.02, 300).tolist()
    shifted = rng.normal(1.0, 1.15, 300).tolist()

    registered = client.put(
        "/models/fraud-v1/baselines/amount_zscore",
        json={"values": baseline},
    )
    assert registered.status_code == 200

    first = client.post(
        "/models/fraud-v1/evaluate/amount_zscore",
        json={"values": stable, "psi_warning": 0.1, "psi_alert": 0.25},
    )
    assert first.status_code == 200
    assert first.json()["severity"] in {"stable", "warning"}

    second = client.post(
        "/models/fraud-v1/evaluate/amount_zscore",
        json={"values": shifted, "psi_warning": 0.1, "psi_alert": 0.25},
    )
    assert second.status_code == 200
    assert second.json()["severity"] == "alert"

    history = client.get("/models/fraud-v1/history/amount_zscore")
    assert history.status_code == 200
    assert len(history.json()) == 2
    assert history.json()[0]["run_id"] > history.json()[1]["run_id"]


def test_missing_baseline_returns_404(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.post(
        "/models/unknown/evaluate/feature",
        json={"values": [float(i) for i in range(20)]},
    )
    assert response.status_code == 404
