"""容量快照：落盘、后写覆盖先写、占比只对有上限的读数算。"""

import json
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


@pytest.mark.parametrize(("name", "count", "limit"), [("", 1, None), ("x", -1, None), ("x", 1, 0)])
def test_reading_validates(name: str, count: int, limit: int | None) -> None:
    with pytest.raises(ValueError):
        CapacityReading(name, count, limit)


def test_snapshot_rejects_duplicate_names_and_naive_time() -> None:
    with pytest.raises(ValueError, match="distinct"):
        CapacitySnapshot("p", AT, (CapacityReading("a", 1), CapacityReading("a", 2)))
    with pytest.raises(ValueError, match="timezone-aware"):
        CapacitySnapshot("p", datetime(2026, 10, 19), ())


@pytest.mark.parametrize(
    ("count", "limit", "message"),
    [
        ("850", 1000, "capacity count must be a non-negative integer"),
        (True, 1000, "capacity count must be a non-negative integer"),
        (850.0, None, "capacity count must be a non-negative integer"),
        (850, "1000", "capacity limit must be a positive integer"),
        (850, False, "capacity limit must be a positive integer"),
    ],
)
def test_read_all_rejects_wrong_types_on_disk(tmp_path: Path, count: object, limit: object, message: str) -> None:
    """落盘快照里类型错的读数在读回解码时就被拒（W16：类型检查在边界，不靠值对象构造器）。"""

    store = CapacitySnapshotStore(tmp_path)
    store.publish(CapacitySnapshot("sources", AT, (CapacityReading("envelopes", 1, 2),)))
    path = tmp_path / "lifecycle" / "capacity" / "sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["readings"][0].update(count=count, limit=limit)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        store.read_all()
