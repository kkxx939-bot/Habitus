"""账本落盘：一条一文件、同身份重写幂等、内容不同是损坏、按日子读回、未结算的能列出来。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.ledger import Delivery, Expectation, Omission, Response, Ruling, Settlement
from habitus.runtime.foresight_ledger import ForesightLedgerStore

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 12, 10, 14, 7, tzinfo=CST)


def expectation(identity: str = "e1") -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token="s-k0002",
        day=FORMED.date(),
        formed_at=FORMED,
        source="事件",
        trigger="rel",
        trigger_instance="behavior://flight.md",
        window=(FORMED + timedelta(minutes=4), FORMED + timedelta(minutes=51)),
        deadline=FORMED + timedelta(hours=3),
        basis=(),
        opportunities=13,
        hits=10,
        observed_from=FORMED,
    )


def test_records_are_written_once_and_read_back_by_day(tmp_path) -> None:
    store = ForesightLedgerStore(tmp_path / "ledger")
    item = expectation()
    assert store.record_expectation(item) is True
    assert store.record_expectation(item) is False  # 同身份同内容：幂等
    with pytest.raises(ForesightError, match="different content"):
        store.record_expectation(replace(item, hits=9))
    assert store.expectations_on(FORMED.date()) == (item,)
    assert store.expectation(FORMED.date(), "e1") == item and store.expectation(FORMED.date(), "nope") is None
    assert store.days_with_expectations() == (FORMED.date(),)
    assert store.unsettled_expectations(FORMED.date()) == (item,)

    day = FORMED.date()
    delivery = Delivery("e1", FORMED + timedelta(minutes=1), 1, True, "说了", 1.0, "要订酒店吗")
    response = Response("e1", FORMED + timedelta(minutes=2), "负", "conv-1#3-9")
    settlement = Settlement("e1", "s-k0002", day, FORMED + timedelta(days=1), "准时", "behavior://hotel.md")
    ruling = Ruling("e1", FORMED + timedelta(days=2), "后补", "behavior://late.md", 90)
    omission = Omission("behavior://other.md", "s-k0003", day, FORMED + timedelta(days=1), False)
    assert store.record_delivery(day, delivery) and store.record_response(day, response)
    assert store.record_settlement(settlement) and store.record_ruling(day, ruling) and store.record_omission(omission)
    assert store.deliveries_on(day) == (delivery,) and store.responses_on(day) == (response,)
    assert store.settlements_on(day) == (settlement,) and store.rulings_on(day) == (ruling,)
    assert store.omissions_on(day) == (omission,)
    assert store.unsettled_expectations(day) == ()
    # 漏报的扫描标记不混进记录：查过的日子记在一份水位里（26 号清单 6.11）
    from habitus.foundation.lifecycle import covers

    assert not covers(store.scan_watermark(start=day), day)
    store.mark_scanned(day, start=day)
    assert covers(store.scan_watermark(start=day), day) and store.omissions_on(day) == (omission,)


def test_a_corrupt_record_is_skipped_and_reported_not_fatal(tmp_path) -> None:
    store = ForesightLedgerStore(tmp_path / "ledger")
    store.record_expectation(expectation())
    store.record_expectation(expectation("e2"))
    path = tmp_path / "ledger" / "expectations" / "2026" / "12" / "10" / "e1.json"
    path.write_text("{", encoding="utf-8")
    # 一个坏文件不让整层停：那一份跳过、路径记下来，取走即清
    assert [item.expectation_id for item in store.expectations_on(FORMED.date())] == ["e2"]
    assert store.take_unreadable() == (str(path),) and store.take_unreadable() == ()


def test_a_well_formed_file_of_the_wrong_shape_is_skipped_too_and_a_caller_can_keep_its_own_list(tmp_path) -> None:
    store = ForesightLedgerStore(tmp_path / "ledger")
    store.record_expectation(expectation())
    store.record_expectation(expectation("e2"))
    path = tmp_path / "ledger" / "expectations" / "2026" / "12" / "10" / "e1.json"
    path.write_text('{"record": "expectation", "schema_version": "foresight_ledger_v5"}', encoding="utf-8")
    mine: list[str] = []
    assert [item.expectation_id for item in store.expectations_on(FORMED.date(), unreadable=mine)] == ["e2"]
    # 自己的列表里有它，共享的那一份不受影响（别的读口取走共享的不会吞掉这一次的）
    assert mine == [str(path)] and store.take_unreadable() == ()


def test_the_index_is_written_before_the_expectation_so_a_crash_between_them_does_not_duplicate(
    tmp_path, monkeypatch
) -> None:
    store = ForesightLedgerStore(tmp_path / "ledger")
    item = expectation()

    def crash(*_args: object) -> bool:
        raise OSError("disk went away")

    monkeypatch.setattr(store, "_create", crash)
    with pytest.raises(OSError):
        store.record_expectation(item)
    monkeypatch.undo()
    # 索引已经占了：第二天同一身份不另建，前一天重来照写
    shift = timedelta(days=1)
    tomorrow = replace(item, day=FORMED.date() + shift, formed_at=FORMED + shift, deadline=item.deadline + shift)
    assert store.day_of("e1") == FORMED.date()
    assert store.record_expectation(tomorrow) is False
    assert store.record_expectation(item) is True
    assert store.expectations_on(FORMED.date()) == (item,)


def test_a_final_day_drops_out_of_the_open_days(tmp_path) -> None:
    store = ForesightLedgerStore(tmp_path / "ledger")
    store.record_expectation(expectation())
    day = FORMED.date()
    assert store.open_days() == (day,) and not store.is_final(day)
    assert store.mark_final(day, at=FORMED + timedelta(days=1))
    assert store.open_days() == () and store.is_final(day)
    assert ForesightLedgerStore(tmp_path / "ledger").is_final(day)  # 落了盘
    assert store.settlements_on(day) == ()  # 标记不混进记录


def test_an_identity_is_recorded_on_one_day_only(tmp_path) -> None:
    store = ForesightLedgerStore(tmp_path / "ledger")
    item = expectation()
    assert store.day_of("e1") is None and not store.has_expectation("e1")
    assert store.record_expectation(item) is True
    assert store.day_of("e1") == FORMED.date() and store.has_expectation("e1")
    # 跨天的触发第二天重算出同一个身份：账本按身份查重，不再写一份
    shift = timedelta(days=1)
    tomorrow = replace(
        item,
        day=FORMED.date() + shift,
        formed_at=FORMED + shift,
        window=(item.window[0] + shift, item.window[1] + shift),
        deadline=item.deadline + shift,
    )
    assert store.record_expectation(tomorrow) is False
    assert store.expectations_on(tomorrow.day) == () and store.day_of("e1") == FORMED.date()


def test_account_records_live_under_the_account_and_unit_records_under_the_behaviour(tmp_path) -> None:
    from habitus.foresight.ledger import AccountFact, AccountLevel, UnitFeedback, UnitSetting

    store = ForesightLedgerStore(tmp_path / "ledger")
    item = expectation()
    account = item.account
    assert store.facts_of(account) == () and store.level_of(account) is None
    first = AccountFact(account, FORMED, False, 1.0, "过期待决", "e1")
    revised = AccountFact(account, FORMED, False, 0.0, "后补", "e1")  # 同一刻改判：文件名按种类分开
    assert store.append_fact(first) and store.append_fact(revised) and not store.append_fact(first)
    assert {fact.kind for fact in store.facts_of(account)} == {"过期待决", "后补"}
    # 设置与提醒反馈按行为（裁定 61）
    assert store.setting_of("s-k0002") is None and store.feedback_of("s-k0002") == ()
    store.record_setting(UnitSetting("s-k0002", FORMED, "以后别提了"))
    store.record_setting(UnitSetting("s-k0002", FORMED + timedelta(days=1), "照常"))
    setting = store.setting_of("s-k0002")
    assert setting is not None and setting.setting == "照常"
    assert [item.setting for item in store.settings_of("s-k0002")] == ["以后别提了", "照常"]
    bad = UnitFeedback("s-k0002", FORMED, False, "e1", "负")
    good = UnitFeedback("s-k0002", FORMED + timedelta(days=1), True, "e2")
    assert store.record_feedback(good) and store.record_feedback(bad) and not store.record_feedback(bad)
    assert store.feedback_of("s-k0002") == (bad, good) and store.feedback_of("s-k0009") == ()
    store.record_level(AccountLevel(account, FORMED, 1))
    store.record_level(AccountLevel(account, FORMED + timedelta(days=2), 0))
    assert [level.level for level in store.levels_of(account)] == [1, 0] and store.level_of(account).level == 0  # type: ignore[union-attr]
    other = replace(item, trigger="weekday=2").account
    store.append_fact(AccountFact(other, FORMED, True, 1.0, "准时", "e2"))
    assert store.accounts() == tuple(sorted((account, other)))


def test_an_index_entry_without_its_expectation_file_does_not_count_as_written(tmp_path) -> None:
    """先占索引再写预期：中间断了，索引在、预期文件不在——这一条不算已有，下一拍照写（2026-10-17 四方评审）。"""

    store = ForesightLedgerStore(tmp_path / "ledger")
    item = expectation()
    store._index_create(item.expectation_id, item.day)  # 模拟断在两次写之间
    assert store.day_of(item.expectation_id) == item.day and not store.has_expectation(item.expectation_id)
    assert store.record_expectation(item) and store.has_expectation(item.expectation_id)


def test_an_omission_noticed_again_after_a_crash_is_not_a_conflict(tmp_path) -> None:
    """同一条记录的漏报只记一次：写了漏报、还没标这一天查过就断，下一晚重放时 ``noticed_at`` 不同，不当成账本损坏。"""

    store = ForesightLedgerStore(tmp_path / "ledger")
    first = Omission("behavior://e.md", "s-k0003", FORMED.date(), FORMED, False)
    assert store.record_omission(first)
    assert not store.record_omission(replace(first, noticed_at=FORMED + timedelta(days=1)))
    assert store.omissions_on(FORMED.date()) == (first,)


def test_an_oversized_record_is_refused_at_write_time(tmp_path) -> None:
    """写入就查大小：超限的不落盘（以前落了盘、读回时被当坏文件跳过，结算永远结不到它）。"""

    store = ForesightLedgerStore(tmp_path / "ledger")
    basis = tuple(f"behavior://{'x' * 200}-{n}.md" for n in range(6_000))
    huge = replace(expectation(), basis=basis, hits=len(basis), opportunities=len(basis))
    with pytest.raises(ForesightError, match="exceeds"):
        store.record_expectation(huge)
    assert store.expectations_on(FORMED.date()) == ()
    assert store.day_of(huge.expectation_id) is None  # 连索引都没占
