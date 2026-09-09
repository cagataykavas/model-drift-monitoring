from __future__ import annotations

import pytest

from monitoring.performance import performance_report


def test_perfect_ranking_has_perfect_auc_and_accuracy() -> None:
    report = performance_report(
        model_id="fraud",
        model_version="1",
        window_id="w1",
        scores=[0.01, 0.2, 0.8, 0.99],
        labels=[0, 0, 1, 1],
    )
    assert report.roc_auc == 1.0
    assert report.accuracy == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.brier_score < 0.03


def test_single_class_window_reports_auc_as_unavailable() -> None:
    report = performance_report(
        model_id="fraud",
        model_version="1",
        window_id="w1",
        scores=[0.1, 0.2, 0.3],
        labels=[0, 0, 0],
    )
    assert report.roc_auc is None


def test_invalid_probability_and_label_are_rejected() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        performance_report(model_id="m", model_version="1", window_id="w", scores=[1.2], labels=[1])
    with pytest.raises(ValueError, match="binary"):
        performance_report(model_id="m", model_version="1", window_id="w", scores=[0.2], labels=[2])
