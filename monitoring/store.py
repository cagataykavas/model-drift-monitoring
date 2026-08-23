from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import threading


SCHEMA = """
CREATE TABLE IF NOT EXISTS baselines (
    model_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    values_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(model_id, feature_name)
);

CREATE TABLE IF NOT EXISTS monitoring_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_monitoring_model_feature
    ON monitoring_runs(model_id, feature_name, id DESC);
"""


class MonitoringStore:
    def __init__(self, database_path: str | Path = "drift.db") -> None:
        self.database_path = str(database_path)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def set_baseline(self, model_id: str, feature_name: str, values: list[float]) -> None:
        payload = json.dumps([float(value) for value in values], separators=(",", ":"))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO baselines(model_id, feature_name, values_json)
                VALUES (?, ?, ?)
                ON CONFLICT(model_id, feature_name) DO UPDATE SET
                    values_json = excluded.values_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (model_id, feature_name, payload),
            )

    def get_baseline(self, model_id: str, feature_name: str) -> list[float] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT values_json FROM baselines WHERE model_id = ? AND feature_name = ?",
                (model_id, feature_name),
            ).fetchone()
        return [float(value) for value in json.loads(row["values_json"])] if row else None

    def record_run(self, model_id: str, feature_name: str, report: dict) -> int:
        payload = json.dumps(report, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO monitoring_runs(model_id, feature_name, severity, report_json)
                VALUES (?, ?, ?, ?)
                """,
                (model_id, feature_name, str(report["severity"]), payload),
            )
            return int(cursor.lastrowid)

    def recent_runs(self, model_id: str, feature_name: str, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, severity, report_json, created_at
                FROM monitoring_runs
                WHERE model_id = ? AND feature_name = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (model_id, feature_name, max(1, min(limit, 200))),
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["report_json"])
            payload.update({"run_id": row["id"], "created_at": row["created_at"]})
            result.append(payload)
        return result
