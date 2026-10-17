"""机会表落盘：一条 lane 一晚一份、整张重写、回读比对、只留最近几晚、不往回写、坏文件硬拒。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.opportunities import KindDay, KindDays, OpportunityTable
from habitus.runtime.foresight_opportunities import OpportunityStore

CST = timezone(timedelta(hours=8))
NIGHT = date(2026, 7, 27)


def table(night: date, lane: str = "session", kinds: int = 1) -> OpportunityTable:
    first = datetime(night.year, night.month, night.day, 9, 0, tzinfo=CST) - timedelta(days=1)
    return OpportunityTable(
        lane=lane,
        night=night,
        relation_night=night,
        slot_minutes=15,
        window_slots=3,
        present_days=(night - timedelta(days=1),),
        relations=(),
        kinds=tuple(
            KindDays(f"s-k{index:04d}", f"类{index}", (KindDay(first.date(), "behavior://a.md", first, 1, 1),))
            for index in range(1, kinds + 1)
        ),
    )


def test_tables_are_written_per_night_read_back_and_pruned(tmp_path) -> None:
    store = OpportunityStore(tmp_path / "foresight", retained_nights=2)
    assert store.lanes() == () and store.nights("session") == () and store.latest("session") is None
    path = store.write(table(NIGHT))
    assert path == store.root / "opportunities" / "session" / f"{NIGHT.isoformat()}.json"
    assert store.read("session", NIGHT) == table(NIGHT) and store.lanes() == ("session",)
    # 同一晚重跑：整张替换
    store.write(table(NIGHT, kinds=2))
    assert store.read("session", NIGHT) == table(NIGHT, kinds=2)
    # 往后写几晚：只留最近两晚
    for offset in (1, 2, 3):
        store.write(table(NIGHT + timedelta(days=offset)))
    assert store.nights("session") == (NIGHT + timedelta(days=2), NIGHT + timedelta(days=3))
    assert store.latest_night("session") == NIGHT + timedelta(days=3)
    latest = store.latest("session")
    assert latest is not None and latest.night == NIGHT + timedelta(days=3)
    # 比盘上最晚一晚更早的不写
    with pytest.raises(ForesightError, match="not rewritten"):
        store.write(table(NIGHT))
    # 另一条 lane 各自一份
    store.write(table(NIGHT, lane="physical"))
    assert store.lanes() == ("physical", "session") and store.nights("physical") == (NIGHT,)


def test_corrupt_or_missing_tables_are_refused(tmp_path) -> None:
    store = OpportunityStore(tmp_path / "foresight")
    with pytest.raises(ForesightError, match="does not exist"):
        store.read("session", NIGHT)
    path = store.write(table(NIGHT))
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ForesightError, match="canonical"):
        store.read("session", NIGHT)
    (path.parent / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ForesightError, match="unexpected entry"):
        store.nights("session")
    with pytest.raises(ForesightError):
        store.read("../escape", NIGHT)
    with pytest.raises(ForesightError, match="positive"):
        OpportunityStore(tmp_path / "x", retained_nights=0)
