"""账本按天目录的释放（26 号清单 6.11）：结完低水位只往前推、撤回时退回；查漏报改成一份水位、旧的每天标记并进来；
先删索引再删预期；漏报只留最近一段；单元的档只留最新一条与最近一段。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from habitus.foresight.ledger import Delivery, Expectation, Omission, Settlement, UnitLevel
from habitus.foundation.lifecycle import covers
from habitus.runtime.foresight_ledger import ForesightLedgerStore

CST = timezone(timedelta(hours=8))
FIRST = date(2026, 12, 1)


def expectation(day: date, identity: str) -> Expectation:
    formed = datetime(day.year, day.month, day.day, 9, 0, tzinfo=CST)
    return Expectation(
        expectation_id=identity,
        kind_token="s-k0002",
        day=day,
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance=f"behavior://{identity}.md",
        window=(formed, formed + timedelta(minutes=30)),
        deadline=formed + timedelta(hours=3),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def seeded(tmp_path) -> ForesightLedgerStore:  # type: ignore[no-untyped-def]
    store = ForesightLedgerStore(tmp_path / "ledger")
    for offset in range(4):
        day = FIRST + timedelta(days=offset)
        store.record_expectation(expectation(day, f"e{offset}"))
    return store


def test_the_final_low_watermark_moves_only_over_a_contiguous_prefix_and_backs_off_on_reopen(tmp_path) -> None:
    store = seeded(tmp_path)
    at = datetime(2026, 12, 20, tzinfo=CST)
    store.mark_final(FIRST + timedelta(days=1), at=at)  # 第 2 天先结完：第 1 天还没结，水位不动
    assert store.final_through() is None
    store.mark_final(FIRST, at=at)
    assert store.final_through() == FIRST + timedelta(days=1)
    assert store.open_days() == (FIRST + timedelta(days=2), FIRST + timedelta(days=3))
    store.reopen(FIRST)  # 重结把第 1 天改回过期待决：水位退回
    assert store.final_through() == FIRST - timedelta(days=1)
    assert FIRST in store.open_days()
    assert store.days_with_expectations(since=FIRST + timedelta(days=2)) == (
        FIRST + timedelta(days=2),
        FIRST + timedelta(days=3),
    )


def test_scanning_is_one_watermark_marked_out_of_order(tmp_path) -> None:
    store = seeded(tmp_path)
    assert not covers(store.scan_watermark(start=FIRST), FIRST)
    store.mark_scanned(FIRST, start=FIRST)
    store.mark_scanned(FIRST + timedelta(days=2), start=FIRST)  # 乱序：第 3 天先查
    watermark = store.scan_watermark(start=FIRST)
    assert watermark.through == FIRST and FIRST + timedelta(days=2) in watermark.above
    store.mark_scanned(FIRST + timedelta(days=1), start=FIRST)
    assert store.scan_watermark(start=FIRST).through == FIRST + timedelta(days=2)


def test_a_released_day_takes_its_index_first_and_then_every_record_of_the_day(tmp_path) -> None:
    store = seeded(tmp_path)
    item = expectation(FIRST, "e0")
    store.record_delivery(FIRST, Delivery("e0", item.formed_at, 0, False, "影子", 0.0))
    store.record_settlement(Settlement("e0", "s-k0002", FIRST, item.deadline, "准时", "behavior://x.md"))
    store.mark_final(FIRST, at=item.deadline)
    assert store.day_of("e0") == FIRST
    store.discard_day(FIRST)
    assert store.day_of("e0") is None and store.expectations_on(FIRST) == ()
    assert store.deliveries_on(FIRST) == () and store.settlements_on(FIRST) == ()
    assert FIRST not in store.days_with_expectations() and store.day_of("e1") == FIRST + timedelta(days=1)
    assert store.record_expectation(item)  # 索引与预期都没了：同一身份可以重建（之后不会再有触发建它）


def test_omissions_keep_only_the_recent_days_and_unit_levels_keep_the_latest_and_a_recent_span(tmp_path) -> None:
    store = seeded(tmp_path)
    for offset in range(3):
        day = FIRST + timedelta(days=offset)
        store.record_omission(
            Omission(f"behavior://o{offset}.md", "s-k0003", day, datetime(2026, 12, 20, tzinfo=CST), False)
        )
    assert store.discard_omissions_before(FIRST + timedelta(days=2)) == 2
    assert store.omissions_on(FIRST) == () and len(store.omissions_on(FIRST + timedelta(days=2))) == 1
    now = datetime(2027, 3, 1, tzinfo=CST)
    for days_ago, level in ((150, 1), (100, 0), (90, 1), (10, 0)):
        store.record_unit_level(UnitLevel("s-k0002", now - timedelta(days=days_ago), level))
    assert store.unit_kinds() == ("s-k0002",)
    assert store.trim_unit_levels("s-k0002", before=now - timedelta(days=60)) == 2
    latest = store.unit_level_of("s-k0002")
    assert latest is not None and latest.at == now - timedelta(days=10)
    assert replace(latest, level=0) == latest
