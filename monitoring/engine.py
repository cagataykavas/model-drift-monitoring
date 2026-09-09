from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from monitoring.metrics import (
    benjamini_hochberg,
    categorical_drift,
    missing_rate,
    numeric_drift,
    report_severity,
)
from monitoring.models import Baseline, FeatureKind, FeatureReport, MetricResult, now_utc


def evaluate_feature(
    baseline: Baseline,
    current_values: Sequence[float | str | None],
    *,
    window_id: str,
    evaluated_at: datetime | None = None,
) -> FeatureReport:
    if len(current_values) < 20:
        raise ValueError("monitoring window requires at least 20 observations")
    if baseline.identity.kind in {FeatureKind.NUMERIC, FeatureKind.PREDICTION}:
        try:
            reference_numeric = [
                None if value is None else float(value) for value in baseline.values
            ]
            current_numeric = [None if value is None else float(value) for value in current_values]
        except (TypeError, ValueError) as exc:
            raise ValueError("numeric feature contains a non-numeric value") from exc
        metrics = numeric_drift(reference_numeric, current_numeric)
    else:
        reference_categories = [None if value is None else str(value) for value in baseline.values]
        current_categories = [None if value is None else str(value) for value in current_values]
        metrics = categorical_drift(reference_categories, current_categories)
    return FeatureReport(
        identity=baseline.identity,
        baseline_id=baseline.baseline_id,
        window_id=window_id,
        reference_count=len(baseline.values),
        current_count=len(current_values),
        missing_rate_reference=missing_rate(baseline.values),
        missing_rate_current=missing_rate(current_values),
        metrics=metrics,
        severity=report_severity(metrics),
        evaluated_at=evaluated_at or now_utc(),
    )


def apply_false_discovery_control(
    reports: Sequence[FeatureReport],
) -> tuple[FeatureReport, ...]:
    locations: list[tuple[int, int]] = []
    pvalues: list[float] = []
    for report_index, report in enumerate(reports):
        for metric_index, metric in enumerate(report.metrics):
            if metric.pvalue is not None:
                locations.append((report_index, metric_index))
                pvalues.append(metric.pvalue)
    adjusted = benjamini_hochberg(pvalues)
    mutable_metrics = [list(report.metrics) for report in reports]
    for (report_index, metric_index), adjusted_value in zip(locations, adjusted, strict=True):
        metric: MetricResult = mutable_metrics[report_index][metric_index]
        mutable_metrics[report_index][metric_index] = replace(
            metric, adjusted_pvalue=adjusted_value
        )
    return tuple(
        replace(
            report,
            metrics=tuple(metrics),
            severity=report_severity(metrics),
        )
        for report, metrics in zip(reports, mutable_metrics, strict=True)
    )
