from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from monitoring.models import MetricResult, Severity

T = TypeVar("T")
EPSILON = 1e-6


def canonical_fingerprint(values: Sequence[float | str | None]) -> str:
    normalized = [None if value is None else value for value in values]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def missing_rate(values: Sequence[object | None]) -> float:
    if not values:
        return 0.0
    missing = sum(
        value is None or (isinstance(value, float) and math.isnan(value)) for value in values
    )
    return missing / len(values)


def finite_numeric(values: Sequence[float | None]) -> np.ndarray:
    result = np.asarray([value for value in values if value is not None], dtype=float)
    if result.size == 0:
        raise ValueError("numeric window has no non-missing observations")
    if not np.isfinite(result).all():
        raise ValueError("numeric observations must be finite or null")
    return result


def probability_vector(counts: np.ndarray) -> np.ndarray:
    probabilities = counts.astype(float) / max(float(counts.sum()), 1.0)
    clipped = np.clip(probabilities, EPSILON, None)
    return clipped / clipped.sum()


def numeric_drift(
    reference_values: Sequence[float | None],
    current_values: Sequence[float | None],
    *,
    bins: int = 10,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
    effect_warning: float = 0.10,
    effect_alert: float = 0.25,
) -> tuple[MetricResult, ...]:
    if bins < 2:
        raise ValueError("bins must be at least two")
    reference = finite_numeric(reference_values)
    current = finite_numeric(current_values)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        center = float(reference[0])
        edges = np.asarray([center - 0.5, center + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    reference_distribution = probability_vector(np.histogram(reference, bins=edges)[0])
    current_distribution = probability_vector(np.histogram(current, bins=edges)[0])
    psi = float(
        np.sum(
            (current_distribution - reference_distribution)
            * np.log(current_distribution / reference_distribution)
        )
    )
    ks = ks_2samp(reference, current, method="auto")
    scale = max(float(np.std(reference)), EPSILON)
    normalized_wasserstein = float(wasserstein_distance(reference, current) / scale)
    missing_delta = abs(missing_rate(reference_values) - missing_rate(current_values))
    return (
        MetricResult("psi", psi, psi_warning, psi_alert),
        MetricResult("ks_statistic", float(ks.statistic), pvalue=float(ks.pvalue)),
        MetricResult(
            "normalized_wasserstein",
            normalized_wasserstein,
            effect_warning,
            effect_alert,
        ),
        MetricResult("missing_rate_delta", missing_delta, 0.05, 0.15),
    )


def categorical_drift(
    reference_values: Sequence[str | None],
    current_values: Sequence[str | None],
    *,
    js_warning: float = 0.10,
    js_alert: float = 0.25,
) -> tuple[MetricResult, ...]:
    if not reference_values or not current_values:
        raise ValueError("categorical windows cannot be empty")
    reference = Counter(
        "__MISSING__" if value is None else str(value) for value in reference_values
    )
    current = Counter("__MISSING__" if value is None else str(value) for value in current_values)
    categories = sorted(reference.keys() | current.keys())
    ref = probability_vector(np.asarray([reference[key] for key in categories]))
    cur = probability_vector(np.asarray([current[key] for key in categories]))
    js = float(jensenshannon(ref, cur, base=2.0) ** 2)
    unseen_count = sum(count for key, count in current.items() if key not in reference)
    unseen_rate = unseen_count / len(current_values)
    missing_delta = abs(missing_rate(reference_values) - missing_rate(current_values))
    return (
        MetricResult("jensen_shannon", js, js_warning, js_alert),
        MetricResult("unseen_category_rate", unseen_rate, 0.01, 0.10),
        MetricResult("missing_rate_delta", missing_delta, 0.05, 0.15),
    )


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    if any(not 0 <= value <= 1 for value in pvalues):
        raise ValueError("p-values must be in [0, 1]")
    count = len(pvalues)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, pvalues[original_index] * count / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted


def report_severity(metrics: Sequence[MetricResult], *, adjusted_alpha: float = 0.05) -> Severity:
    severities: list[Severity] = []
    for metric in metrics:
        severity = metric.severity()
        if metric.pvalue is not None:
            pvalue = metric.adjusted_pvalue if metric.adjusted_pvalue is not None else metric.pvalue
            if pvalue > adjusted_alpha:
                severity = Severity.STABLE
        severities.append(severity)
    if Severity.ALERT in severities:
        return Severity.ALERT
    if Severity.WARNING in severities:
        return Severity.WARNING
    return Severity.STABLE
