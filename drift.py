from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def population_stability_index(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference, current, psi_warning: float = 0.1, psi_alert: float = 0.25) -> dict:
    psi = population_stability_index(reference, current)
    ks = ks_2samp(reference, current)
    severity = "alert" if psi >= psi_alert else "warning" if psi >= psi_warning else "stable"
    return {
        "psi": psi,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "severity": severity,
    }


def prediction_drift(reference_probabilities, current_probabilities) -> dict:
    report = drift_report(reference_probabilities, current_probabilities)
    report["reference_mean_score"] = float(np.mean(reference_probabilities))
    report["current_mean_score"] = float(np.mean(current_probabilities))
    return report


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 5000)
    shifted = rng.normal(0.55, 1.15, 2500)
    print(drift_report(baseline, shifted))
