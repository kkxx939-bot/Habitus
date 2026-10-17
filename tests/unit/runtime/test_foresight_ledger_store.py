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
    response = Response("e1", FORMED + timedelta(minutes=2), "这次不用")
    settlement = Settlement("e1", "s-k0002", day, FORMED + timedelta(days=1), "准时", "behavior://hotel.md")
    ruling = Ruling("e1", FORMED + timedelta(days=2), "未判")
    omission = Omission("behavior://other.md", "s-k0003", day, FORMED + timedelta(days=1), False)
    assert store.record_delivery(day, delivery) and store.record_response(day, response)
    assert store.record_settlement(settlement) and store.record_ruling(day, ruling) and store.record_omission(omission)
    assert store.deliveries_on(day) == (delivery,) and store.responses_on(day) == (response,)
    assert store.settlements_on(day) == (settlement,) and store.rulings_on(day) == (ruling,)
    assert store.omissions_on(day) == (omission,)
    assert store.unsettled_expectations(day) == ()
    # 漏报的扫描标记不混进记录
    assert not store.scanned(day)
    assert store.mark_scanned(day, scanned_at=FORMED + timedelta(days=1), found=1)
    assert store.scanned(day) and store.omissions_on(day) == (omission,)


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


def test_account_records_live_under_the_account_and_are_read_back_by_account(tmp_path) -> None:
    from habitus.foresight.ledger import AccountFact, AccountLevel, AccountSetting, AccountSeverity

    store = ForesightLedgerStore(tmp_path / "ledger")
    item = expectation()
    account = item.account
    assert store.facts_of(account) == () and store.severity_of(account) is None and store.level_of(account) is None
    first = AccountFact(account, FORMED, "预测", False, 1.0, "过期待决", "e1")
    revised = AccountFact(account, FORMED, "预测", False, 0.0, "后补", "e1")  # 同一刻改判：文件名按种类分开
    assert store.append_fact(first) and store.append_fact(revised) and not store.append_fact(first)
    assert {fact.kind for fact in store.facts_of(account)} == {"过期待决", "后补"}
    old = AccountSeverity(account, FORMED, 0.5, "", FORMED + timedelta(days=30))
    new = AccountSeverity(account, FORMED + timedelta(days=31), 4.0, "补办要排队", FORMED + timedelta(days=91))
    store.record_severity(old)
    store.record_severity(new)
    assert store.severity_of(account) == new
    store.record_setting(AccountSetting(account, FORMED, "以后别提了"))
    store.record_setting(AccountSetting(account, FORMED + timedelta(days=1), "照常"))
    setting = store.setting_of(account)
    assert setting is not None and setting.setting == "照常"
    store.record_level(AccountLevel(account, FORMED, 1))
    store.record_level(AccountLevel(account, FORMED + timedelta(days=2), 0))
    assert [level.level for level in store.levels_of(account)] == [1, 0] and store.level_of(account).level == 0  # type: ignore[union-attr]
    other = replace(item, trigger="weekday=2").account
    store.append_fact(AccountFact(other, FORMED, "预测", True, 1.0, "准时", "e2"))
    assert store.accounts() == tuple(sorted((account, other)))
