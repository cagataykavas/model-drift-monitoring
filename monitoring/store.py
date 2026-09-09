from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monitoring.metrics import canonical_fingerprint
from monitoring.models import (
    AlertState,
    Baseline,
    BaselineIdentity,
    FeatureKind,
    FeatureReport,
    PerformanceReport,
    Severity,
    now_utc,
)
from monitoring.performance import performance_report

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS baseline_versions (
    baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_kind TEXT NOT NULL,
    values_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE(model_id, model_version, feature_name, fingerprint)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_baseline
ON baseline_versions(model_id, model_version, feature_name) WHERE active = 1;

CREATE TABLE IF NOT EXISTS monitoring_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    window_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    report_json TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    UNIQUE(model_id, model_version, feature_name, window_id),
    FOREIGN KEY (baseline_id) REFERENCES baseline_versions(baseline_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_lookup
ON monitoring_runs(model_id, model_version, feature_name, run_id DESC);

CREATE TABLE IF NOT EXISTS alerts (
    alert_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    state TEXT NOT NULL,
    severity TEXT NOT NULL,
    consecutive_breaches INTEGER NOT NULL,
    consecutive_stable INTEGER NOT NULL,
    occurrences INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    acknowledged_at TEXT,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    window_id TEXT NOT NULL,
    score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
    predicted_at TEXT NOT NULL,
    PRIMARY KEY(model_id, model_version, entity_id)
);

CREATE TABLE IF NOT EXISTS labels (
    entity_id TEXT PRIMARY KEY,
    label INTEGER NOT NULL CHECK(label IN (0, 1)),
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    window_id TEXT NOT NULL,
    report_json TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    UNIQUE(model_id, model_version, window_id)
);
"""


class MonitoringStore:
    """Durable registry for baselines, windows, alert state and delayed labels."""

    def __init__(self, database_path: str | Path = "drift.db") -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self) -> Any:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _baseline_from_row(row: sqlite3.Row) -> Baseline:
        return Baseline(
            baseline_id=int(row["baseline_id"]),
            identity=BaselineIdentity(
                model_id=row["model_id"],
                model_version=row["model_version"],
                feature_name=row["feature_name"],
                kind=FeatureKind(row["feature_kind"]),
            ),
            values=tuple(json.loads(row["values_json"])),
            fingerprint=row["fingerprint"],
            created_at=datetime.fromisoformat(row["created_at"]),
            active=bool(row["active"]),
        )

    def register_baseline(
        self,
        identity: BaselineIdentity,
        values: Sequence[float | str | None],
        *,
        created_at: datetime | None = None,
    ) -> Baseline:
        if len(values) < 20:
            raise ValueError("baseline requires at least 20 observations")
        payload = json.dumps(list(values), separators=(",", ":"), allow_nan=False)
        value_hash = canonical_fingerprint(values)
        now = created_at or now_utc()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM baseline_versions WHERE model_id = ? AND model_version = ? "
                "AND feature_name = ? AND fingerprint = ?",
                (identity.model_id, identity.model_version, identity.feature_name, value_hash),
            ).fetchone()
            if existing is not None:
                if not existing["active"]:
                    connection.execute(
                        "UPDATE baseline_versions SET active = 0 WHERE model_id = ? "
                        "AND model_version = ? AND feature_name = ?",
                        (identity.model_id, identity.model_version, identity.feature_name),
                    )
                    connection.execute(
                        "UPDATE baseline_versions SET active = 1 WHERE baseline_id = ?",
                        (existing["baseline_id"],),
                    )
                    existing = connection.execute(
                        "SELECT * FROM baseline_versions WHERE baseline_id = ?",
                        (existing["baseline_id"],),
                    ).fetchone()
                return self._baseline_from_row(existing)
            connection.execute(
                "UPDATE baseline_versions SET active = 0 WHERE model_id = ? AND model_version = ? "
                "AND feature_name = ?",
                (identity.model_id, identity.model_version, identity.feature_name),
            )
            cursor = connection.execute(
                "INSERT INTO baseline_versions(model_id, model_version, feature_name, feature_kind, "
                "values_json, fingerprint, active, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    identity.model_id,
                    identity.model_version,
                    identity.feature_name,
                    identity.kind,
                    payload,
                    value_hash,
                    now.astimezone(UTC).isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM baseline_versions WHERE baseline_id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._baseline_from_row(row)

    def active_baseline(self, identity: BaselineIdentity) -> Baseline | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM baseline_versions WHERE model_id = ? AND model_version = ? "
                "AND feature_name = ? AND active = 1",
                (identity.model_id, identity.model_version, identity.feature_name),
            ).fetchone()
        if row is None:
            return None
        baseline = self._baseline_from_row(row)
        if baseline.identity.kind is not identity.kind:
            raise ValueError("requested feature kind does not match registered baseline")
        return baseline

    def record_feature_report(self, report: FeatureReport) -> int:
        payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._transaction() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO monitoring_runs(baseline_id, model_id, model_version, feature_name, "
                    "window_id, severity, report_json, evaluated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        report.baseline_id,
                        report.identity.model_id,
                        report.identity.model_version,
                        report.identity.feature_name,
                        report.window_id,
                        report.severity,
                        payload,
                        report.evaluated_at.astimezone(UTC).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("window_id has already been evaluated for this feature") from exc
        return int(cursor.lastrowid)

    def recent_feature_runs(
        self, identity: BaselineIdentity, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, report_json FROM monitoring_runs WHERE model_id = ? "
                "AND model_version = ? AND feature_name = ? ORDER BY run_id DESC LIMIT ?",
                (
                    identity.model_id,
                    identity.model_version,
                    identity.feature_name,
                    max(1, min(limit, 200)),
                ),
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            report = json.loads(row["report_json"])
            report["run_id"] = row["run_id"]
            values.append(report)
        return values

    def ingest_predictions(
        self,
        model_id: str,
        model_version: str,
        window_id: str,
        values: Sequence[tuple[str, float, datetime]],
    ) -> int:
        with self._transaction() as connection:
            for entity_id, score, predicted_at in values:
                connection.execute(
                    "INSERT INTO predictions(model_id, model_version, entity_id, window_id, score, "
                    "predicted_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    (
                        model_id,
                        model_version,
                        entity_id,
                        window_id,
                        score,
                        predicted_at.astimezone(UTC).isoformat(),
                    ),
                )
        return len(values)

    def ingest_labels(self, values: Sequence[tuple[str, int, datetime]]) -> int:
        with self._transaction() as connection:
            for entity_id, label, observed_at in values:
                connection.execute(
                    "INSERT INTO labels(entity_id, label, observed_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(entity_id) DO UPDATE SET label = excluded.label, "
                    "observed_at = excluded.observed_at",
                    (entity_id, label, observed_at.astimezone(UTC).isoformat()),
                )
        return len(values)

    def evaluate_performance(
        self, model_id: str, model_version: str, window_id: str
    ) -> PerformanceReport:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.score, l.label FROM predictions p JOIN labels l USING(entity_id) "
                "WHERE p.model_id = ? AND p.model_version = ? AND p.window_id = ? "
                "ORDER BY p.entity_id",
                (model_id, model_version, window_id),
            ).fetchall()
        report = performance_report(
            model_id=model_id,
            model_version=model_version,
            window_id=window_id,
            scores=[row["score"] for row in rows],
            labels=[row["label"] for row in rows],
        )
        payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO performance_runs(model_id, model_version, window_id, report_json, "
                "evaluated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(model_id, model_version, window_id) "
                "DO UPDATE SET report_json = excluded.report_json, evaluated_at = excluded.evaluated_at",
                (
                    model_id,
                    model_version,
                    window_id,
                    payload,
                    report.evaluated_at.astimezone(UTC).isoformat(),
                ),
            )
        return report

    def observe_alert(
        self,
        identity: BaselineIdentity,
        severity: Severity,
        *,
        open_after: int = 2,
        resolve_after: int = 2,
        observed_at: datetime | None = None,
    ) -> dict[str, Any] | None:
        now = observed_at or now_utc()
        key = f"{identity.model_id}:{identity.model_version}:{identity.feature_name}"
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM alerts WHERE alert_key = ?", (key,)).fetchone()
            breach = severity in {Severity.WARNING, Severity.ALERT}
            if row is None and not breach:
                return None
            if row is None:
                breaches = 1
                state = AlertState.OPEN if breaches >= open_after else AlertState.PENDING
                connection.execute(
                    "INSERT INTO alerts(alert_key, model_id, model_version, feature_name, state, "
                    "severity, consecutive_breaches, consecutive_stable, occurrences, first_seen_at, "
                    "last_seen_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
                    (
                        key,
                        identity.model_id,
                        identity.model_version,
                        identity.feature_name,
                        state,
                        severity,
                        breaches,
                        int(state is AlertState.OPEN),
                        now.isoformat(),
                        now.isoformat(),
                        None,
                    ),
                )
            else:
                breaches = int(row["consecutive_breaches"]) + 1 if breach else 0
                stable = 0 if breach else int(row["consecutive_stable"]) + 1
                state = AlertState(row["state"])
                occurrences = int(row["occurrences"])
                resolved_at = row["resolved_at"]
                if (
                    breach
                    and breaches >= open_after
                    and state
                    in {
                        AlertState.PENDING,
                        AlertState.RESOLVED,
                    }
                ):
                    state = AlertState.OPEN
                    occurrences += 1
                    resolved_at = None
                elif (
                    not breach
                    and stable >= resolve_after
                    and state
                    in {
                        AlertState.OPEN,
                        AlertState.ACKNOWLEDGED,
                    }
                ):
                    state = AlertState.RESOLVED
                    resolved_at = now.isoformat()
                connection.execute(
                    "UPDATE alerts SET state = ?, severity = ?, consecutive_breaches = ?, "
                    "consecutive_stable = ?, occurrences = ?, last_seen_at = ?, resolved_at = ? "
                    "WHERE alert_key = ?",
                    (
                        state,
                        severity,
                        breaches,
                        stable,
                        occurrences,
                        now.isoformat(),
                        resolved_at,
                        key,
                    ),
                )
            result = connection.execute(
                "SELECT * FROM alerts WHERE alert_key = ?", (key,)
            ).fetchone()
        return dict(result)

    def acknowledge_alert(self, alert_key: str, *, acknowledged_at: datetime | None = None) -> dict:
        now = acknowledged_at or now_utc()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE alerts SET state = ?, acknowledged_at = ? WHERE alert_key = ? AND state = ?",
                (AlertState.ACKNOWLEDGED, now.isoformat(), alert_key, AlertState.OPEN),
            )
            if cursor.rowcount != 1:
                raise ValueError("only an open alert can be acknowledged")
            row = connection.execute(
                "SELECT * FROM alerts WHERE alert_key = ?", (alert_key,)
            ).fetchone()
        return dict(row)

    def list_alerts(self, *, state: AlertState | None = None, limit: int = 100) -> list[dict]:
        query = "SELECT * FROM alerts"
        parameters: list[Any] = []
        if state is not None:
            query += " WHERE state = ?"
            parameters.append(state)
        query += " ORDER BY last_seen_at DESC LIMIT ?"
        parameters.append(max(1, min(limit, 500)))
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]

    # Compatibility methods retained for the original v0.2 API.
    def set_baseline(self, model_id: str, feature_name: str, values: list[float]) -> None:
        self.register_baseline(
            BaselineIdentity(model_id, "default", feature_name, FeatureKind.NUMERIC), values
        )

    def get_baseline(self, model_id: str, feature_name: str) -> list[float] | None:
        baseline = self.active_baseline(
            BaselineIdentity(model_id, "default", feature_name, FeatureKind.NUMERIC)
        )
        return list(baseline.values) if baseline else None

    def record_run(self, model_id: str, feature_name: str, report: dict) -> int:
        identity = BaselineIdentity(model_id, "default", feature_name, FeatureKind.NUMERIC)
        baseline = self.active_baseline(identity)
        if baseline is None:
            raise ValueError("baseline not registered")
        feature_report = FeatureReport(
            identity=identity,
            baseline_id=baseline.baseline_id,
            window_id=str(report.get("window_id", f"legacy-{now_utc().timestamp()}")),
            reference_count=int(report.get("reference_observations", len(baseline.values))),
            current_count=int(report.get("current_observations", 0)),
            missing_rate_reference=0,
            missing_rate_current=0,
            metrics=(),
            severity=Severity(str(report["severity"])),
            evaluated_at=now_utc(),
        )
        return self.record_feature_report(feature_report)

    def recent_runs(self, model_id: str, feature_name: str, limit: int = 20) -> list[dict]:
        return self.recent_feature_runs(
            BaselineIdentity(model_id, "default", feature_name, FeatureKind.NUMERIC), limit=limit
        )
