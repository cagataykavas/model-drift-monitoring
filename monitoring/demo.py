from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from monitoring.engine import apply_false_discovery_control, evaluate_feature
from monitoring.models import BaselineIdentity, FeatureKind
from monitoring.store import MonitoringStore


def run_reference_scenario(database_path: Path) -> dict:
    if database_path.exists():
        database_path.unlink()
    store = MonitoringStore(database_path)
    rng = np.random.default_rng(42)
    identities = (
        BaselineIdentity("credit-risk", "2026.09", "income_zscore", FeatureKind.NUMERIC),
        BaselineIdentity("credit-risk", "2026.09", "country", FeatureKind.CATEGORICAL),
        BaselineIdentity("credit-risk", "2026.09", "default_score", FeatureKind.PREDICTION),
    )
    baselines = (
        store.register_baseline(identities[0], rng.normal(0, 1, 1000).tolist()),
        store.register_baseline(identities[1], ["TR"] * 700 + ["DE"] * 200 + ["NL"] * 100),
        store.register_baseline(identities[2], rng.beta(2, 8, 1000).tolist()),
    )
    window_summaries = []
    for index in range(1, 3):
        reports = apply_false_discovery_control(
            (
                evaluate_feature(
                    baselines[0], rng.normal(0.9, 1.15, 400).tolist(), window_id=f"w{index}"
                ),
                evaluate_feature(baselines[1], ["TR"] * 100 + ["US"] * 300, window_id=f"w{index}"),
                evaluate_feature(baselines[2], rng.beta(4, 5, 400).tolist(), window_id=f"w{index}"),
            )
        )
        summary = []
        for report in reports:
            run_id = store.record_feature_report(report)
            alert = store.observe_alert(report.identity, report.severity)
            summary.append(
                {
                    "feature": report.identity.feature_name,
                    "severity": report.severity,
                    "run_id": run_id,
                    "alert_state": alert["state"] if alert else None,
                }
            )
        window_summaries.append({"window_id": f"w{index}", "features": summary})

    now = datetime(2026, 9, 9, tzinfo=UTC)
    scores = rng.uniform(0, 1, 200)
    labels = (scores + rng.normal(0, 0.2, 200) >= 0.5).astype(int)
    store.ingest_predictions(
        "credit-risk",
        "2026.09",
        "w2",
        [(f"entity-{index}", float(score), now) for index, score in enumerate(scores)],
    )
    store.ingest_labels(
        [(f"entity-{index}", int(label), now) for index, label in enumerate(labels)]
    )
    performance = store.evaluate_performance("credit-risk", "2026.09", "w2")
    return {
        "model_id": "credit-risk",
        "model_version": "2026.09",
        "baseline_ids": [baseline.baseline_id for baseline in baselines],
        "windows": window_summaries,
        "active_alerts": store.list_alerts(),
        "delayed_label_performance": performance.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic drift-control evidence")
    parser.add_argument("--database", type=Path, default=Path("artifacts/reference.db"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference-report.json"))
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    report = run_reference_scenario(args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
