"""预测错了的诊断（裁定 54；裁定 55 新版不再分原因）：按顺序过关、无关的老实记说不清；数据可不可信的几关在前；预算内只是标注；
校准表把"算出的"和"实际的"并排、对不上的标出来。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from habitus.foresight.diagnosis import (
    ABANDONED,
    BROKEN,
    POSTPONED,
    RELATION_GONE,
    UNEXPLAINED,
    UNSEEN,
    VOIDED,
    MissFacts,
    Settled,
    calibration,
    cause_of,
    diagnose,
)

BASE = MissFacts("e", "acct", "s-k0001", date(2026, 9, 1), "日子", spoken=True)


def test_the_first_gate_that_holds_names_the_cause_and_nothing_left_is_unexplained() -> None:
    assert cause_of(BASE) == UNEXPLAINED
    # 预算内只是标注：照样往下找原因（裁定 55 第 6 条）
    assert cause_of(replace(BASE, within_budget=True, relation_gone=True)) == RELATION_GONE
    assert cause_of(replace(BASE, within_budget=True)) == UNEXPLAINED
    assert cause_of(replace(BASE, unseen=True, postponed=True)) == UNSEEN  # 数据可不可信排在推迟之前
    assert cause_of(replace(BASE, voided=True, unseen=True)) == VOIDED
    assert cause_of(replace(BASE, abandoned=True, postponed=True)) == ABANDONED
    # 推迟（之后补做了）排在规律断了之前
    assert cause_of(replace(BASE, broken=True, postponed=True)) == POSTPONED
    assert cause_of(replace(BASE, broken=True)) == BROKEN
    assert cause_of(replace(BASE, notice_driven=True)).startswith("疑点")


def test_calibration_flags_a_slice_whose_claimed_rate_lies_outside_what_actually_happened() -> None:
    """周二跑步那一类：触发算出 65%，账上 22 次里只中 4 次——对不上，标出来；样本少的标不出来。"""

    rows = [Settled("日子", "v1", 0.65, index < 4) for index in range(22)]
    rows += [Settled("事件:chain", "v1", 0.7, index < 7) for index in range(10)]
    rows += [Settled("日子:interval", "v1", 0.9, False), Settled("日子:interval", "v1", 0.9, True)]
    found = {row.slice: row for row in calibration(rows, alpha=0.05)}
    assert found["日子"].off and found["日子"].expectations == 22
    assert not found["事件:chain"].off
    assert not found["日子:interval"].off  # 两条：区间宽，不下结论
    # 实现版本分开比：修了之后新版本单独一行
    fixed = calibration([*rows, *[Settled("日子", "v2", 0.65, index % 3 != 0) for index in range(30)]], alpha=0.05)
    assert [(row.slice, row.implementation, row.off) for row in fixed if row.slice == "日子"] == [
        ("日子", "v1", True),
        ("日子", "v2", False),
    ]


def test_counts_can_be_restricted_to_the_reminders_that_were_said() -> None:
    found = diagnose([BASE, replace(BASE, expectation_id="shadow", spoken=False, within_budget=True)], [], alpha=0.05)
    assert found.counts()[UNEXPLAINED] == 2 and found.within_budget() == 1
    assert found.within_budget(spoken_only=True) == 0
