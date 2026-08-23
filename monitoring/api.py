from __future__ import annotations

from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from drift import drift_report, prediction_drift
from monitoring.store import MonitoringStore


class BaselineRequest(BaseModel):
    values: list[float] = Field(min_length=20)


class EvaluateRequest(BaseModel):
    values: list[float] = Field(min_length=20)
    psi_warning: float = Field(default=0.10, ge=0.0)
    psi_alert: float = Field(default=0.25, ge=0.0)
    prediction_scores: bool = False


DATABASE_PATH = Path(os.getenv("DRIFT_DATABASE_PATH", "drift.db"))
store = MonitoringStore(DATABASE_PATH)
app = FastAPI(
    title="Model Drift Monitoring",
    version="0.2.0",
    description=(
        "Persistent feature/prediction drift monitoring with PSI, KS tests, "
        "severity thresholds and historical run tracking."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.put("/models/{model_id}/baselines/{feature_name}")
def set_baseline(model_id: str, feature_name: str, request: BaselineRequest) -> dict:
    store.set_baseline(model_id, feature_name, request.values)
    return {
        "model_id": model_id,
        "feature_name": feature_name,
        "observations": len(request.values),
        "status": "registered",
    }


@app.post("/models/{model_id}/evaluate/{feature_name}")
def evaluate(model_id: str, feature_name: str, request: EvaluateRequest) -> dict:
    reference = store.get_baseline(model_id, feature_name)
    if reference is None:
        raise HTTPException(status_code=404, detail="baseline not registered")
    if request.psi_alert < request.psi_warning:
        raise HTTPException(status_code=422, detail="psi_alert must be >= psi_warning")

    if request.prediction_scores:
        report = prediction_drift(reference, request.values)
        # prediction_drift uses default thresholds; normalize severity for requested policy.
        psi = float(report["psi"])
        report["severity"] = (
            "alert"
            if psi >= request.psi_alert
            else "warning"
            if psi >= request.psi_warning
            else "stable"
        )
    else:
        report = drift_report(
            reference,
            request.values,
            psi_warning=request.psi_warning,
            psi_alert=request.psi_alert,
        )

    report.update(
        {
            "model_id": model_id,
            "feature_name": feature_name,
            "reference_observations": len(reference),
            "current_observations": len(request.values),
        }
    )
    run_id = store.record_run(model_id, feature_name, report)
    report["run_id"] = run_id
    return report


@app.get("/models/{model_id}/history/{feature_name}")
def history(
    model_id: str,
    feature_name: str,
    limit: int = Query(default=20, ge=1, le=200),
) -> list[dict]:
    return store.recent_runs(model_id, feature_name, limit=limit)
