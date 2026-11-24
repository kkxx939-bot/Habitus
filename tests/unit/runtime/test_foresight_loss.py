"""loss 尺子在组合根（裁定 114）：读数在算出来的那一拍随递送落盘（时窗开的依据与成员、时窗末），并进已决定单元的、出口追加的
没有；结算之后从账上现算，事后改判下一次自动反映；尺子算不出来不让结算失败；换代报告里那一节按行为的明细只留本地。"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from habitus.foresight.ledger import AccountFact, Delivery, Readings, Settlement
from habitus.foresight.reliability import reading
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_loss import measure_loss
from habitus.runtime.foresight_settlement import SettlementReport
from habitus.runtime.generation_summary import flatten, settlement_summary
from tests.unit.runtime.test_foresight_delivery import DAY0, earn, expectation, stage


def test_readings_are_stored_on_the_deliveries_of_the_tick_that_computed_them(tmp_path) -> None:
    run, ledger = stage(tmp_path)
    earn(ledger, trigger="rel")
    earn(ledger, trigger="clock")
    ledger.record_expectation(expectation("now", DAY0, trigger="rel"))
    ledger.record_expectation(
        expectation("later", DAY0 + timedelta(minutes=30), trigger="clock")
    )  # 还没到时窗、一并决定
    at = DAY0 + timedelta(minutes=16)
    asyncio.run(run.run_once(at))
    opened = {item.expectation_id: item for item in ledger.deliveries_on(DAY0.date())}
    basis, member = opened["now"].loss_readings, opened["later"].loss_readings
    assert basis is not None and member is not None
    assert basis.unit == member.unit and basis.layer == member.layer  # 同一个单元
    assert basis.account == run.standing_of(expectation("now", DAY0, trigger="rel"), at).prediction
    assert member.account == run.standing_of(expectation("later", DAY0, trigger="clock"), at).prediction
    # 同一刻后来才形成、并进已决定过的单元的：这一拍没算单元，不带读数（没量到）
    ledger.record_expectation(expectation("late", DAY0 + timedelta(minutes=5), trigger="late", window=(15, 45)))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=21)))
    (late,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "late"]
    assert late.reason == "单元已选" and late.loss_readings is None
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=41)))
    (closing,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.reason == "待取"]
    assert closing.expectation_id == "now" and closing.loss_readings is not None  # 时窗末重算的那一次


def _settled(ledger: ForesightLedgerStore, kind: str, strength: float = 1.0) -> str:
    """一条结完的预期：带读数的递送、结算、一笔事实、那天标结完。返回账身份。"""

    item = expectation("e1", DAY0)
    ledger.record_expectation(item)
    probe = reading(3.0, 4.0, alpha=0.05)
    opened = Delivery(
        "e1", DAY0 + timedelta(minutes=16), 1, False, "等时窗末", 0.0, loss_readings=Readings(probe, "同一组", probe)
    )
    ledger.record_delivery(DAY0.date(), opened)
    ledger.record_settlement(Settlement("e1", item.kind_token, DAY0.date(), item.deadline, "过期待决", None, None))
    ledger.append_fact(AccountFact(item.account, item.formed_at, False, strength, kind, "e1"))
    ledger.mark_final(DAY0.date(), at=item.deadline)
    return item.account


def test_a_later_revision_of_the_outcome_shows_in_the_next_measurement(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    account = _settled(ledger, "过期待决")
    now = DAY0 + timedelta(days=2)
    first = measure_loss(ledger, now=now)
    assert first.failed is None and first.days == 1 and first.loss is not None
    assert [(item.came, item.weight) for item in first.loss.scored] == [(False, 1.0)]
    # 重结把它改成准时（后写的覆盖先写的）：下一次算按"来了"
    ledger.append_fact(AccountFact(account, DAY0, True, 1.0, "准时", "e1", 0, written_at=now))
    second = measure_loss(ledger, now=now + timedelta(hours=1))
    assert second.loss is not None and [(item.came, item.weight) for item in second.loss.scored] == [(True, 1.0)]
    # 后补按单元历史的口径仍是没来、分量打折
    ledger.append_fact(AccountFact(account, DAY0, False, 0.25, "后补", "e1", 30, written_at=now + timedelta(hours=2)))
    third = measure_loss(ledger, now=now + timedelta(hours=3))
    assert third.loss is not None and [(item.came, item.weight) for item in third.loss.scored] == [(False, 0.25)]


def test_days_not_yet_final_are_left_out_and_a_broken_ledger_does_not_fail_the_settlement(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    ledger.record_expectation(expectation("open", DAY0))
    found = measure_loss(ledger, now=DAY0 + timedelta(days=1))
    assert found.days == 0 and found.loss is not None and found.loss.expectations == 0

    class Broken:
        def days_with_expectations(self, *, since):  # type: ignore[no-untyped-def]
            raise OSError("disk")

    broken = measure_loss(Broken(), now=DAY0)  # type: ignore[arg-type]
    assert broken.loss is None and broken.failed == "OSError"


def test_the_report_section_keeps_the_per_behaviour_detail_out_of_span_attributes(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    _settled(ledger, "过期待决")
    loss = measure_loss(ledger, now=DAY0 + timedelta(days=2))
    summary = settlement_summary(SettlementReport(1, {"过期待决": 1}, 0, 0, (), loss=loss), due=1)
    section = summary["loss"]
    assert isinstance(section, dict)
    assert section["total"] == section["by_kind"]["s-k0002"] == section["by_source"]["日子"]  # type: ignore[index]
    assert section["by_spoken"] == {"没说出口": section["total"]}
    attributes = flatten({"settlement": summary})
    assert "settlement.loss.total.unit.log_loss" in attributes
    assert not any("by_kind" in key for key in attributes)
    failed = settlement_summary(SettlementReport(0, {}, 0, 0, (), loss=measure_loss(None, now=DAY0)), due=0)  # type: ignore[arg-type]
    assert failed["loss"] == {"days": 0, "failed": "AttributeError"}


def test_bad_files_the_ruler_reads_stay_out_of_the_shared_list_and_later_days_are_skipped(tmp_path) -> None:
    """尺子读到的坏预期、坏递送记在它自己的清单里，不进结算观测取走的那一份；时钟回退时账上晚于此刻的结完日子不算。"""

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    _settled(ledger, "过期待决")
    broken = ledger.root / "deliveries" / DAY0.strftime("%Y/%m/%d") / "zz--20261001T000000000000.json"
    broken.write_text("{not json", encoding="utf-8")
    found = measure_loss(ledger, now=DAY0 + timedelta(days=2))
    assert found.loss is not None and found.loss.moments == 1
    assert ledger.take_unreadable() == ()
    earlier = measure_loss(ledger, now=DAY0 - timedelta(days=1))
    assert earlier.days == 0 and earlier.loss is not None and earlier.loss.expectations == 0
