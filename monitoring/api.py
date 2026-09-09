from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel, Field

from monitoring.engine import apply_false_discovery_control, evaluate_feature
from monitoring.models import AlertState, BaselineIdentity, FeatureKind
from monitoring.store import MonitoringStore


class BaselineRequest(BaseModel):
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)
    kind: FeatureKind
    values: list[float | str | None] = Field(min_length=20)


class FeatureWindowRequest(BaseModel):
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)
    kind: FeatureKind
    values: list[float | str | None] = Field(min_length=20)


class EvaluateWindowRequest(BaseModel):
    window_id: str = Field(min_length=1, max_length=200)
    features: list[FeatureWindowRequest] = Field(min_length=1, max_length=200)
    alert_after_consecutive_windows: int = Field(default=2, ge=1, le=20)
    resolve_after_stable_windows: int = Field(default=2, ge=1, le=20)


class PredictionItem(BaseModel):
    entity_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    predicted_at: datetime


class PredictionBatch(BaseModel):
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    predictions: list[PredictionItem] = Field(min_length=1, max_length=10000)


class LabelItem(BaseModel):
    entity_id: str = Field(min_length=1)
    label: int = Field(ge=0, le=1)
    observed_at: datetime


class LabelBatch(BaseModel):
    labels: list[LabelItem] = Field(min_length=1, max_length=10000)


def identity_for(value: BaselineRequest | FeatureWindowRequest) -> BaselineIdentity:
    return BaselineIdentity(value.model_id, value.model_version, value.feature_name, value.kind)


def create_app(store: MonitoringStore | None = None) -> FastAPI:
    monitoring_store = store or MonitoringStore(Path(os.getenv("DRIFT_DATABASE_PATH", "drift.db")))
    registry = CollectorRegistry()
    evaluations = Counter(
        "drift_feature_evaluations_total",
        "Evaluated feature windows",
        ["severity"],
        registry=registry,
    )
    open_alerts = Gauge("drift_open_alerts", "Open or acknowledged drift alerts", registry=registry)
    application = FastAPI(
        title="Model Drift Control Plane",
        version="0.3.0",
        description=(
            "Versioned numeric/categorical/prediction drift, false-discovery control, "
            "delayed-label performance and alert lifecycle management."
        ),
    )
    application.state.store = monitoring_store

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/v1/baselines")
    def register_baseline(request: BaselineRequest) -> dict[str, Any]:
        try:
            baseline = monitoring_store.register_baseline(identity_for(request), request.values)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "baseline_id": baseline.baseline_id,
            "fingerprint": baseline.fingerprint,
            "active": baseline.active,
            "observations": len(baseline.values),
            "identity": {
                "model_id": baseline.identity.model_id,
                "model_version": baseline.identity.model_version,
                "feature_name": baseline.identity.feature_name,
                "kind": baseline.identity.kind,
            },
        }

    @application.post("/v1/windows/evaluate")
    def evaluate_window(request: EvaluateWindowRequest) -> dict[str, Any]:
        pending = []
        identities: set[tuple[str, str, str]] = set()
        for feature in request.features:
            identity = identity_for(feature)
            identity_key = (identity.model_id, identity.model_version, identity.feature_name)
            if identity_key in identities:
                raise HTTPException(status_code=422, detail="feature appears twice in one window")
            identities.add(identity_key)
            baseline = monitoring_store.active_baseline(identity)
            if baseline is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"baseline not registered for {identity.feature_name}",
                )
            try:
                pending.append(
                    evaluate_feature(baseline, feature.values, window_id=request.window_id)
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        reports = apply_false_discovery_control(pending)
        rendered = []
        for report in reports:
            try:
                run_id = monitoring_store.record_feature_report(report)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            alert = monitoring_store.observe_alert(
                report.identity,
                report.severity,
                open_after=request.alert_after_consecutive_windows,
                resolve_after=request.resolve_after_stable_windows,
            )
            evaluations.labels(severity=report.severity).inc()
            rendered.append({**report.to_dict(), "run_id": run_id, "alert": alert})
        return {"window_id": request.window_id, "reports": rendered}

    @application.get(
        "/v1/models/{model_id}/versions/{model_version}/features/{feature_name}/history"
    )
    def feature_history(
        model_id: str,
        model_version: str,
        feature_name: str,
        kind: Annotated[FeatureKind, Query()],
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
    ) -> list[dict[str, Any]]:
        return monitoring_store.recent_feature_runs(
            BaselineIdentity(model_id, model_version, feature_name, kind), limit=limit
        )

    @application.post("/v1/predictions")
    def ingest_predictions(request: PredictionBatch) -> dict[str, int]:
        count = monitoring_store.ingest_predictions(
            request.model_id,
            request.model_version,
            request.window_id,
            [(item.entity_id, item.score, item.predicted_at) for item in request.predictions],
        )
        return {"accepted": count}

    @application.post("/v1/labels")
    def ingest_labels(request: LabelBatch) -> dict[str, int]:
        count = monitoring_store.ingest_labels(
            [(item.entity_id, item.label, item.observed_at) for item in request.labels]
        )
        return {"accepted": count}

    @application.post(
        "/v1/models/{model_id}/versions/{model_version}/performance/{window_id}/evaluate"
    )
    def evaluate_performance(model_id: str, model_version: str, window_id: str) -> dict[str, Any]:
        try:
            report = monitoring_store.evaluate_performance(model_id, model_version, window_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return report.to_dict()

    @application.get("/v1/alerts")
    def alerts(
        state: Annotated[AlertState | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> list[dict]:
        return monitoring_store.list_alerts(state=state, limit=limit)

    @application.post("/v1/alerts/{alert_key}/acknowledge")
    def acknowledge(alert_key: str) -> dict:
        try:
            return monitoring_store.acknowledge_alert(alert_key)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/metrics")
    def metrics() -> Response:
        active = monitoring_store.list_alerts(limit=500)
        open_alerts.set(
            sum(item["state"] in {AlertState.OPEN, AlertState.ACKNOWLEDGED} for item in active)
        )
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
