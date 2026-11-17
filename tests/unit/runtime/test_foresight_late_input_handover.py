"""依据记了影子之后才来的输入自己再决定一次（裁定 103）：单元记影子时只看了当时在场的那几条预期；之后才形成、并进它的那条
没被看过，到了自己的时窗末按全部输入重算单元、照时窗末那一套走。记影子那一拍就在场的成员照旧不接手
（``test_foresight_candidate_basis_handover.test_a_speakable_basis_the_unit_kept_quiet_keeps_the_day_subsumed``）。

情形（数据集第一期 90 天那一遍，「看邮件 → 回邮件」的形状）：日子那条 09:10–09:40 先到时窗，它那本账没攒够，单元记影子；
他做完前因之后事件那条才形成（09:50–10:20、最晚时限 13:00），关系成立、账在提醒档。
"""

from __future__ import annotations

import asyncio

from habitus.runtime.foresight_delivery import JudgeMaterials
from tests.unit.runtime.test_foresight_candidate_basis_handover import CAUSE, at, reasons
from tests.unit.runtime.test_foresight_delivery import DAY0, earn, expectation, stage


def moment(tmp_path, *, event_hits: int):  # type: ignore[no-untyped-def]
    run, ledger = stage(tmp_path)
    earn(ledger, trigger="interval", hits=1)
    earn(ledger, trigger="rel", source="事件", hits=event_hits)
    ledger.record_expectation(expectation("day", DAY0, trigger="interval"))
    materials = JudgeMaterials(CAUSE, lambda uris: {})
    asyncio.run(run.run_once(at(16), materials))
    assert reasons(ledger, "day") == ["影子"]
    ledger.record_expectation(expectation("ev", DAY0, trigger="rel", source="事件", window=(50, 80), deadline_hours=4))
    asyncio.run(run.run_once(at(51), materials))
    assert reasons(ledger, "ev") == ["单元已选"]
    return run, ledger, materials


def test_an_input_that_arrived_after_the_shadow_decides_for_itself(tmp_path) -> None:
    run, ledger, materials = moment(tmp_path, event_hits=8)
    assert asyncio.run(run.run_once(at(66), materials)).decided == 0  # 事件那条 10:20 才到时窗末
    asyncio.run(run.run_once(at(81), materials))
    assert reasons(ledger, "ev") == ["单元已选", "待取"]
    assert reasons(ledger, "day") == ["影子"]
    assert asyncio.run(run.run_once(at(82), materials)).decided == 0  # 只接手一次


def test_a_late_input_that_is_weak_too_stays_quiet_and_is_not_asked_again(tmp_path) -> None:
    run, ledger, materials = moment(tmp_path, event_hits=1)
    asyncio.run(run.run_once(at(81), materials))
    assert reasons(ledger, "ev") == ["单元已选", "影子"]
    for minutes in (82, 100):
        assert asyncio.run(run.run_once(at(minutes), materials)).decided == 0


def test_a_late_input_already_done_is_recorded_as_done(tmp_path) -> None:
    run, ledger, _materials = moment(tmp_path, event_hits=8)
    done = JudgeMaterials(CAUSE, lambda uris: {}, happened=lambda item: item.expectation_id == "ev")
    asyncio.run(run.run_once(at(81), done))
    assert reasons(ledger, "ev") == ["单元已选", "已发生"]


def test_two_late_inputs_only_one_takes_over(tmp_path) -> None:
    run, ledger, materials = moment(tmp_path, event_hits=8)
    earn(ledger, trigger="rel2", source="事件", hits=8)
    ledger.record_expectation(
        expectation("ev2", DAY0, trigger="rel2", source="事件", window=(55, 80), deadline_hours=4)
    )
    asyncio.run(run.run_once(at(56), materials))
    assert reasons(ledger, "ev2") == ["单元已选"]
    for minutes in (81, 82, 100):
        asyncio.run(run.run_once(at(minutes), materials))
    taken = [name for name in ("ev", "ev2") if reasons(ledger, name) != ["单元已选"]]
    assert len(taken) == 1 and reasons(ledger, taken[0]) == ["单元已选", "待取"]
