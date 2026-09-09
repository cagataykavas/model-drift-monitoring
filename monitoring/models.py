from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class FeatureKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    PREDICTION = "prediction"


class Severity(StrEnum):
    STABLE = "stable"
    WARNING = "warning"
    ALERT = "alert"


class AlertState(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class BaselineIdentity:
    model_id: str
    model_version: str
    feature_name: str
    kind: FeatureKind

    def __post_init__(self) -> None:
        for name, value in (
            ("model_id", self.model_id),
            ("model_version", self.model_version),
            ("feature_name", self.feature_name),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")


@dataclass(frozen=True, slots=True)
class Baseline:
    baseline_id: int
    identity: BaselineIdentity
    values: tuple[float | str | None, ...]
    fingerprint: str
    created_at: datetime
    active: bool


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float
    threshold_warning: float | None = None
    threshold_alert: float | None = None
    pvalue: float | None = None
    adjusted_pvalue: float | None = None

    def severity(self) -> Severity:
        if self.threshold_alert is not None and self.value >= self.threshold_alert:
            return Severity.ALERT
        if self.threshold_warning is not None and self.value >= self.threshold_warning:
            return Severity.WARNING
        return Severity.STABLE


@dataclass(frozen=True, slots=True)
class FeatureReport:
    identity: BaselineIdentity
    baseline_id: int
    window_id: str
    reference_count: int
    current_count: int
    missing_rate_reference: float
    missing_rate_current: float
    metrics: tuple[MetricResult, ...]
    severity: Severity
    evaluated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["identity"]["kind"] = self.identity.kind.value
        value["severity"] = self.severity.value
        value["evaluated_at"] = self.evaluated_at.astimezone(UTC).isoformat()
        return value


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    model_id: str
    model_version: str
    window_id: str
    joined_count: int
    positive_rate: float
    mean_score: float
    brier_score: float
    log_loss: float
    roc_auc: float | None
    accuracy: float
    precision: float
    recall: float
    expected_calibration_error: float
    evaluated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evaluated_at"] = self.evaluated_at.astimezone(UTC).isoformat()
        return value


def now_utc() -> datetime:
    return datetime.now(UTC)
