from __future__ import annotations

from pathlib import Path

import pytest

from monitoring.store import MonitoringStore


@pytest.fixture
def store(tmp_path: Path) -> MonitoringStore:
    return MonitoringStore(tmp_path / "monitoring.db")
