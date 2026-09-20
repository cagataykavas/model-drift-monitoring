import pytest

from monitoring.explanation_drift import ExplanationDriftPolicy, evaluate_explanation_drift

POLICY = ExplanationDriftPolicy(
    top_k=2,
    min_top_k_overlap=0.5,
    min_cosine_similarity=0.8,
    min_sign_agreement=0.75,
    max_normalized_l1_shift=0.5,
)


def test_stable_attribution_profile_passes_with_deterministic_evidence():
    report = evaluate_explanation_drift(
        {"income": -0.5, "debt": 0.8, "age": -0.1, "history": 0.3},
        {"income": -0.48, "debt": 0.82, "age": -0.09, "history": 0.29},
        policy=POLICY,
    )

    assert report.passed
    assert report.baseline_top_features == ("debt", "income")
    assert report.to_dict()["reasons"] == []


def test_direction_and_magnitude_shift_fail_multiple_gates():
    report = evaluate_explanation_drift(
        {"income": -0.7, "debt": 0.8, "age": -0.1, "history": 0.2},
        {"income": 0.7, "debt": -0.8, "age": 0.9, "history": -0.6},
        policy=POLICY,
    )

    assert not report.passed
    assert "cosine_similarity_below_policy" in report.reasons
    assert "sign_agreement_below_policy" in report.reasons
    assert "normalized_l1_shift_above_policy" in report.reasons
    assert report.shifts[0].sign_changed


def test_top_feature_replacement_is_visible():
    report = evaluate_explanation_drift(
        {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.1},
        {"a": 0.1, "b": 0.1, "c": 0.9, "d": 0.8},
        policy=POLICY,
    )
    assert report.top_k_overlap == 0.0
    assert report.reasons[0] == "top_k_overlap_below_policy"


@pytest.mark.parametrize(
    ("baseline", "current", "message"),
    [
        ({"a": 1.0}, {"a": 1.0}, "at least two"),
        ({"a": 1.0, "b": 0.0}, {"a": 1.0, "c": 0.0}, "match exactly"),
        ({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.0}, "non-zero magnitude"),
        ({"a": float("nan"), "b": 1.0}, {"a": 1.0, "b": 1.0}, "finite"),
    ],
)
def test_invalid_evidence_fails_closed(baseline, current, message):
    with pytest.raises(ValueError, match=message):
        evaluate_explanation_drift(baseline, current, policy=POLICY)


def test_top_k_cannot_exceed_feature_count():
    with pytest.raises(ValueError, match="top_k"):
        evaluate_explanation_drift(
            {"a": 1.0, "b": 0.5},
            {"a": 1.0, "b": 0.5},
            policy=ExplanationDriftPolicy(top_k=3),
        )
