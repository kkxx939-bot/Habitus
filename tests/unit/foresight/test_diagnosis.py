"""预测错了的诊断（裁定 54、55）：按顺序过关、无关的老实记说不清；对照答案找到的原因排在数据可不可信之后；预算内只是标注；
校准表把"算出的"和"实际的"并排、对不上的标出来；调整表数"今天又在、不说"之后他果然没做的几次。"""

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

BASE = MissFacts("e", "acct", "s-k0001", date(2026, 9, 1), "钟面", spoken=True)


def test_the_first_gate_that_holds_names_the_cause_and_nothing_left_is_unexplained() -> None:
    assert cause_of(BASE) == UNEXPLAINED
    # 预算内只是标注：照样往下找原因（裁定 55 第 6 条）
    assert cause_of(replace(BASE, within_budget=True, relation_gone=True)) == RELATION_GONE
    assert cause_of(replace(BASE, within_budget=True)) == UNEXPLAINED
    assert cause_of(replace(BASE, unseen=True, found="别的起因")) == UNSEEN  # 数据可不可信排在找到的原因之前
    assert cause_of(replace(BASE, voided=True, unseen=True)) == VOIDED
    assert cause_of(replace(BASE, abandoned=True, found=POSTPONED)) == ABANDONED
    # 对照答案找到的原因排在规律断了之前：断了说的是"连着错"，找到的说的是"这一次为什么"
    assert cause_of(replace(BASE, broken=True, found="别的起因")) == "别的起因"
    assert cause_of(replace(BASE, broken=True, found="还没进关联")) == "还没进关联"
    assert cause_of(replace(BASE, broken=True)) == BROKEN
    assert cause_of(replace(BASE, notice_driven=True)).startswith("疑点")


def test_calibration_flags_a_slice_whose_claimed_rate_lies_outside_what_actually_happened() -> None:
    """周二跑步那一类：触发算出 65%，账上 22 次里只中 4 次——对不上，标出来；样本少的标不出来。"""

    rows = [Settled("钟面", "v1", 0.65, index < 4) for index in range(22)]
    rows += [Settled("事件:chain", "v1", 0.7, index < 7) for index in range(10)]
    rows += [Settled("日子:interval", "v1", 0.9, False), Settled("日子:interval", "v1", 0.9, True)]
    found = {row.slice: row for row in calibration(rows, alpha=0.05)}
    assert found["钟面"].off and found["钟面"].expectations == 22
    assert not found["事件:chain"].off
    assert not found["日子:interval"].off  # 两条：区间宽，不下结论
    # 实现版本分开比：修了之后新版本单独一行
    fixed = calibration([*rows, *[Settled("钟面", "v2", 0.65, index % 3 != 0) for index in range(30)]], alpha=0.05)
    assert [(row.slice, row.implementation, row.off) for row in fixed if row.slice == "钟面"] == [
        ("钟面", "v1", True),
        ("钟面", "v2", False),
    ]


def test_counts_can_be_restricted_to_the_reminders_that_were_said() -> None:
    found = diagnose([BASE, replace(BASE, expectation_id="shadow", spoken=False, within_budget=True)], [], alpha=0.05)
    assert found.counts()[UNEXPLAINED] == 2 and found.within_budget() == 1
    assert found.within_budget(spoken_only=True) == 0


def test_the_adjustment_table_counts_how_often_holding_back_was_right() -> None:
    """因为"上次的原因今天又在"没说的几次：别的起因用了 3 次、2 次他果然没做；线索没了用了 1 次、他做了。"""

    found = diagnose(
        [], [], alpha=0.05, held=[("别的起因", True), ("别的起因", True), ("别的起因", False), ("线索没了", False)]
    )
    assert (found.adjustments["别的起因"].held, found.adjustments["别的起因"].right) == (3, 2)
    assert (found.adjustments["线索没了"].held, found.adjustments["线索没了"].right) == (1, 0)


def test_the_substitution_table_flags_what_he_does_when_the_matter_does_not_come() -> None:
    """该订菜没订的 10 次里 8 次接着缴水电，订了的 20 次里接着缴水电的只有 2 次：标出来（"B 没来时他常接着做 X"）。
    缴水电本来就常做（落空后与命中后差不多）的不标；样本少的标不出来。"""

    from habitus.foresight.diagnosis import Followed, substitutions

    rows = [Followed("订菜", "水电" if index < 8 else "看日程", missed=True) for index in range(10)]
    rows += [Followed("订菜", "水电" if index < 2 else "看日程", missed=False) for index in range(20)]
    found = {row.next_kind: row for row in substitutions(rows, alpha=0.05)}
    assert found["水电"].flagged and (found["水电"].after_misses, found["水电"].misses) == (8, 10)
    assert (found["水电"].after_hits, found["水电"].hits) == (2, 20)
    assert not found["看日程"].flagged
    same = [Followed("看日程", "水电" if index % 2 else None, missed=index < 10) for index in range(30)]
    assert not any(row.flagged for row in substitutions(same, alpha=0.05))
    few = [Followed("买药", "水电", missed=True), Followed("买药", "看日程", missed=False)]
    assert not any(row.flagged for row in substitutions(few, alpha=0.05))
    # 甲盘探针的真实一行：「买猫用品」落空 1 次接着记录跑步、命中 8 次都没有——一次不下结论（精确检验 p = 1/9）
    once = [Followed("买猫用品", "记录跑步", missed=True), *(Followed("买猫用品", "看日程", missed=False) for _ in range(8))]
    (row,) = [item for item in substitutions(once, alpha=0.05) if item.next_kind == "记录跑步"]
    assert not row.flagged and abs(row.p_value - 1 / 9) < 1e-12
