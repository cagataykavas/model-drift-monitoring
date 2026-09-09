from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from scipy.stats import rankdata

from monitoring.models import PerformanceReport, now_utc


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket_ids = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
    value = 0.0
    for bucket in range(bins):
        mask = bucket_ids == bucket
        if mask.any():
            value += float(mask.mean()) * abs(float(scores[mask].mean() - labels[mask].mean()))
    return value


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def performance_report(
    *,
    model_id: str,
    model_version: str,
    window_id: str,
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
    evaluated_at: datetime | None = None,
) -> PerformanceReport:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have equal non-zero length")
    probability = np.asarray(scores, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("scores must be finite probabilities")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("labels must be binary")
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    predicted = (probability >= threshold).astype(int)
    true_positive = int(np.sum((predicted == 1) & (truth == 1)))
    false_positive = int(np.sum((predicted == 1) & (truth == 0)))
    false_negative = int(np.sum((predicted == 0) & (truth == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return PerformanceReport(
        model_id=model_id,
        model_version=model_version,
        window_id=window_id,
        joined_count=len(truth),
        positive_rate=float(truth.mean()),
        mean_score=float(probability.mean()),
        brier_score=float(np.mean((probability - truth) ** 2)),
        log_loss=float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))),
        roc_auc=roc_auc(probability, truth),
        accuracy=float(np.mean(predicted == truth)),
        precision=precision,
        recall=recall,
        expected_calibration_error=expected_calibration_error(probability, truth),
        evaluated_at=evaluated_at or now_utc(),
    )
