from __future__ import annotations

import numpy as np
import pytest

from monitoring.engine import apply_false_discovery_control, evaluate_feature
from monitoring.metrics import (
    benjamini_hochberg,
    categorical_drift,
    missing_rate,
    numeric_drift,
)
from monitoring.models import Baseline, BaselineIdentity, FeatureKind, Severity, now_utc


def baseline(values: list[float | str | None], kind: FeatureKind) -> Baseline:
    return Baseline(
        baseline_id=1,
        identity=BaselineIdentity("fraud", "1", "feature", kind),
        values=tuple(values),
        fingerprint="hash",
        created_at=now_utc(),
        active=True,
    )


def metric(results, name: str):
    return next(value for value in results if value.name == name)


def test_numeric_shift_has_large_psi_and_effect_size() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 2000).tolist()
    shifted = rng.normal(1.0, 1.1, 1000).tolist()
    results = numeric_drift(reference, shifted)
    assert metric(results, "psi").value > 0.25
    assert metric(results, "normalized_wasserstein").value > 0.25
    assert metric(results, "ks_statistic").pvalue < 0.001


def test_constant_reference_remains_numerically_safe() -> None:
    results = numeric_drift([1.0] * 100, [1.0] * 100)
    assert metric(results, "psi").value == pytest.approx(0)
    assert metric(results, "normalized_wasserstein").value == pytest.approx(0)


def test_categorical_monitor_detects_unseen_values_and_missingness() -> None:
    results = categorical_drift(
        ["TR"] * 80 + ["DE"] * 20,
        ["TR"] * 40 + ["DE"] * 20 + ["US"] * 20 + [None] * 20,
    )
    assert metric(results, "unseen_category_rate").value == pytest.approx(0.4)
    assert metric(results, "missing_rate_delta").value == pytest.approx(0.2)


def test_missing_rate_treats_nan_and_null_as_missing() -> None:
    assert missing_rate([1.0, None, float("nan"), 2.0]) == 0.5


def test_benjamini_hochberg_is_monotonic_in_rank_order() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([1.1])


def test_multi_feature_evaluation_attaches_adjusted_pvalues() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(size=500).tolist()
    reports = [
        evaluate_feature(
            baseline(reference, FeatureKind.NUMERIC),
            rng.normal(shift, 1, 300).tolist(),
            window_id=f"window-{index}",
        )
        for index, shift in enumerate((0.0, 0.5, 1.0))
    ]
    controlled = apply_false_discovery_control(reports)
    ks_metrics = [metric(report.metrics, "ks_statistic") for report in controlled]
    assert all(value.adjusted_pvalue is not None for value in ks_metrics)
    assert controlled[-1].severity is Severity.ALERT


def test_numeric_feature_rejects_non_numeric_window() -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        evaluate_feature(
            baseline([1.0] * 20, FeatureKind.NUMERIC),
            ["bad"] * 20,
            window_id="window",
        )
