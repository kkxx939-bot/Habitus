"""机会表落盘：一条 lane 一晚一份、整张重写、回读比对、只留最近几晚、不往回写、坏文件硬拒。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.opportunities import KindDays, OpportunityTable
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.temporal import OccurrenceDay

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
        relations=(),
        kinds=tuple(
            KindDays(
                f"s-k{index:04d}",
                f"类{index}",
                (OccurrenceDay(first.date(), "behavior://a.md", first, 1, 1, "behavior://a.md", first),),
            )
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
    # 别的名字（备份、副本、笔记）不是哪一晚：跳过不报，写下一晚也不受影响（段 6 Q2）
    for stray in ("notes.txt", f"{NIGHT.isoformat()}.json.bak", f"{NIGHT.isoformat()} copy.json", "x.json"):
        (path.parent / stray).write_text("x", encoding="utf-8")
    assert store.nights("session") == (NIGHT,) and store.latest_night("session") == NIGHT
    store.write(table(NIGHT + timedelta(days=1)))
    assert store.nights("session") == (NIGHT, NIGHT + timedelta(days=1))
    with pytest.raises(ForesightError):
        store.read("../escape", NIGHT)
    with pytest.raises(ForesightError, match="positive"):
        OpportunityStore(tmp_path / "x", retained_nights=0)


def test_a_corrupt_latest_table_falls_back_to_the_night_before_and_is_counted(tmp_path) -> None:
    """段 6 Q2：最晚那一晚的表读不出来就往前退一晚（留的那几代备用就是为这个），坏的那一份计入观测；一晚都读不出来是 None。"""

    store = OpportunityStore(tmp_path / "foresight")
    store.write(table(NIGHT))
    latest = store.write(table(NIGHT + timedelta(days=1)))
    latest.write_text("{", encoding="utf-8")
    found = store.latest("session")
    assert found is not None and found.night == NIGHT
    assert store.take_unreadable() == (str(latest),) and store.take_unreadable() == ()
    (store.root / "opportunities" / "session" / f"{NIGHT.isoformat()}.json").write_text("[]", encoding="utf-8")
    assert store.latest("session") is None and len(store.take_unreadable()) == 2
