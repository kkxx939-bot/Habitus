"""容量快照：落盘、后写覆盖先写、占比只对有上限的读数算。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from habitus.infrastructure.lifecycle import CapacityReading, CapacitySnapshot, CapacitySnapshotStore

AT = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)


def test_publish_and_read_back(tmp_path: Path) -> None:
    store = CapacitySnapshotStore(tmp_path)
    store.publish(CapacitySnapshot("sources", AT, (CapacityReading("envelopes", 850, 1000),)))
    store.publish(CapacitySnapshot("ledger", AT, (CapacityReading("index", 12),)))
    snapshots = {item.participant: item for item in store.read_all()}
    assert snapshots["sources"].readings[0].share == pytest.approx(0.85)
    assert snapshots["ledger"].readings[0].share is None


def test_later_snapshot_replaces_earlier(tmp_path: Path) -> None:
    store = CapacitySnapshotStore(tmp_path)
    store.publish(CapacitySnapshot("sources", AT, (CapacityReading("envelopes", 1),)))
    store.publish(CapacitySnapshot("sources", AT, (CapacityReading("envelopes", 2),)))
    (snapshot,) = store.read_all()
    assert snapshot.readings[0].count == 2


def test_empty_store_reads_nothing(tmp_path: Path) -> None:
    assert CapacitySnapshotStore(tmp_path).read_all() == ()


@pytest.mark.parametrize(("name", "count", "limit"), [("", 1, None), ("x", -1, None), ("x", 1, 0), ("x", True, None)])
def test_reading_validates(name: str, count: int, limit: int | None) -> None:
    with pytest.raises(ValueError):
        CapacityReading(name, count, limit)


def test_snapshot_rejects_duplicate_names_and_naive_time() -> None:
    with pytest.raises(ValueError, match="distinct"):
        CapacitySnapshot("p", AT, (CapacityReading("a", 1), CapacityReading("a", 2)))
    with pytest.raises(ValueError, match="timezone-aware"):
        CapacitySnapshot("p", datetime(2026, 10, 19), ())
