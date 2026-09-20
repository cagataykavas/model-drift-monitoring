from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class ExplanationDriftPolicy:
    top_k: int = 5
    min_top_k_overlap: float = 0.6
    min_cosine_similarity: float = 0.8
    min_sign_agreement: float = 0.8
    max_normalized_l1_shift: float = 0.5

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        for name, value in (
            ("min_top_k_overlap", self.min_top_k_overlap),
            ("min_cosine_similarity", self.min_cosine_similarity),
            ("min_sign_agreement", self.min_sign_agreement),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_normalized_l1_shift < 0:
            raise ValueError("max_normalized_l1_shift must be non-negative")


@dataclass(frozen=True)
class FeatureAttributionShift:
    feature: str
    baseline: float
    current: float
    absolute_shift: float
    sign_changed: bool


@dataclass(frozen=True)
class ExplanationDriftReport:
    passed: bool
    reasons: tuple[str, ...]
    top_k_overlap: float
    cosine_similarity: float
    sign_agreement: float
    normalized_l1_shift: float
    baseline_top_features: tuple[str, ...]
    current_top_features: tuple[str, ...]
    shifts: tuple[FeatureAttributionShift, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "top_k_overlap": self.top_k_overlap,
            "cosine_similarity": self.cosine_similarity,
            "sign_agreement": self.sign_agreement,
            "normalized_l1_shift": self.normalized_l1_shift,
            "baseline_top_features": list(self.baseline_top_features),
            "current_top_features": list(self.current_top_features),
            "shifts": [asdict(item) for item in self.shifts],
        }


def _validate(values: Mapping[str, float], name: str) -> dict[str, float]:
    if len(values) < 2 or any(not feature.strip() for feature in values):
        raise ValueError(f"{name} must contain at least two named features")
    result = {feature: float(value) for feature, value in values.items()}
    if any(not isfinite(value) for value in result.values()):
        raise ValueError(f"{name} attributions must be finite")
    return result


def evaluate_explanation_drift(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    policy: ExplanationDriftPolicy | None = None,
) -> ExplanationDriftReport:
    """Compare signed aggregate attribution profiles across monitoring windows."""
    active_policy = policy or ExplanationDriftPolicy()
    reference = _validate(baseline, "baseline")
    observed = _validate(current, "current")
    if set(reference) != set(observed):
        raise ValueError("baseline and current feature sets must match exactly")
    if active_policy.top_k > len(reference):
        raise ValueError("top_k cannot exceed the feature count")

    features = sorted(reference)
    baseline_norm = sqrt(sum(reference[name] ** 2 for name in features))
    current_norm = sqrt(sum(observed[name] ** 2 for name in features))
    if baseline_norm == 0 or current_norm == 0:
        raise ValueError("attribution profiles must have non-zero magnitude")

    cosine = sum(reference[name] * observed[name] for name in features) / (
        baseline_norm * current_norm
    )
    cosine = max(-1.0, min(1.0, cosine))
    baseline_top = tuple(
        sorted(features, key=lambda name: (-abs(reference[name]), name))[: active_policy.top_k]
    )
    current_top = tuple(
        sorted(features, key=lambda name: (-abs(observed[name]), name))[: active_policy.top_k]
    )
    overlap = len(set(baseline_top) & set(current_top)) / active_policy.top_k

    comparable = [name for name in features if reference[name] != 0 or observed[name] != 0]
    sign_matches = sum(
        (reference[name] > 0) == (observed[name] > 0)
        for name in comparable
        if reference[name] != 0 and observed[name] != 0
    )
    sign_agreement = sign_matches / len(comparable) if comparable else 1.0
    l1_shift = sum(abs(observed[name] - reference[name]) for name in features)
    normalized_l1 = l1_shift / sum(abs(reference[name]) for name in features)

    reasons: list[str] = []
    if overlap < active_policy.min_top_k_overlap:
        reasons.append("top_k_overlap_below_policy")
    if cosine < active_policy.min_cosine_similarity:
        reasons.append("cosine_similarity_below_policy")
    if sign_agreement < active_policy.min_sign_agreement:
        reasons.append("sign_agreement_below_policy")
    if normalized_l1 > active_policy.max_normalized_l1_shift:
        reasons.append("normalized_l1_shift_above_policy")

    shifts = tuple(
        FeatureAttributionShift(
            feature=name,
            baseline=reference[name],
            current=observed[name],
            absolute_shift=abs(observed[name] - reference[name]),
            sign_changed=(reference[name] > 0) != (observed[name] > 0),
        )
        for name in sorted(
            features, key=lambda item: (-abs(observed[item] - reference[item]), item)
        )
    )
    return ExplanationDriftReport(
        passed=not reasons,
        reasons=tuple(reasons),
        top_k_overlap=overlap,
        cosine_similarity=cosine,
        sign_agreement=sign_agreement,
        normalized_l1_shift=normalized_l1,
        baseline_top_features=baseline_top,
        current_top_features=current_top,
        shifts=shifts,
    )
