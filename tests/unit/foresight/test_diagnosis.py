"""预测错了的诊断（裁定 54；裁定 55 新版不再分原因）：按顺序过关、无关的老实记说不清；数据可不可信的几关在前；预算内只是标注；
校准表把"算出的"和"实际的"并排、对不上的标出来。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone

from habitus.foresight import Standing
from habitus.foresight.diagnosis import (
    ABANDONED,
    BROKEN,
    POSTPONED,
    RELATION_GONE,
    UNEXPLAINED,
    UNSEEN,
    VOIDED,
    BasisSeen,
    MissFacts,
    Settled,
    calibration,
    calibration_sample,
    cause_of,
    counts_as_miss,
    diagnose,
    miss_facts,
)
from habitus.foresight.ledger import VOID, AccountFact, Antecedent, Expectation
from habitus.foresight.reliability import Reliability

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


CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 9, 1, 9, 0, tzinfo=CST)


def _expectation(**overrides: object) -> Expectation:
    values: dict[str, object] = dict(
        expectation_id="run-0901",
        kind_token="s-k0003",
        day=FORMED.date(),
        formed_at=FORMED,
        source="事件",
        trigger="session|s-k0001|s-k0003|chain",
        trigger_instance="behavior://wake.md",
        window=(FORMED + timedelta(minutes=10), FORMED + timedelta(minutes=40)),
        deadline=FORMED + timedelta(hours=2),
        basis=(),
        opportunities=10,
        hits=7,
        observed_from=FORMED,
        implementation="v1",
    )
    values.update(overrides)
    return Expectation(**values)  # type: ignore[arg-type]


def _fact(kind: str, *, good: bool = False, strength: float = 1.0) -> AccountFact:
    item = _expectation()
    return AccountFact(item.account, item.formed_at, good, strength, kind, item.expectation_id)


def _standing(*, lower: float = 0.5, break_at: datetime | None = None) -> Standing:
    return Standing(level=1, threshold=0.67, prediction=Reliability(5.0, 8.0, 0.6, lower, 0.8), break_at=break_at)


@dataclass(frozen=True)
class _Row:
    kind_token: str
    status: str | None
    at: datetime


def _miss(fact: AccountFact, **overrides: object) -> MissFacts:
    values: dict[str, object] = dict(
        spoken=True,
        standing=_standing(),
        current=lambda token: (token,),
        window=(),
        today_note=None,
        basis=(),
        listed=None,
    )
    item = overrides.pop("item", _expectation())
    values.update(overrides)
    return miss_facts(item, fact, **values)  # type: ignore[arg-type]


def test_only_real_opportunities_without_a_reminder_enter_the_calibration_table() -> None:
    item = _expectation()
    sample = calibration_sample(item, _fact("准时", good=True))
    assert sample == Settled("事件:chain", "v1", 0.7, True)
    assert calibration_sample(item, _fact("过期待决")) == Settled("事件:chain", "v1", 0.7, False)
    assert calibration_sample(item, _fact(VOID, strength=0.0)) is None
    assert calibration_sample(item, _fact("过期待决", strength=0.0)) is None  # 观测空白
    assert calibration_sample(item, _fact("提醒后发生", good=True, strength=0.0)) is None
    assert calibration_sample(_expectation(opportunities=0, hits=0), _fact("准时", good=True)) is None


def test_a_miss_is_overdue_or_caught_up_or_voided_after_being_overdue() -> None:
    assert counts_as_miss(_fact("过期待决"), "过期待决")
    assert counts_as_miss(_fact("后补"), "过期待决")
    assert counts_as_miss(_fact(VOID, strength=0.0), "过期待决")
    assert not counts_as_miss(_fact(VOID, strength=0.0), "准时")
    assert not counts_as_miss(_fact("准时", good=True), "准时")


def test_the_flags_of_a_miss_are_read_from_the_fact_the_window_the_basis_and_the_standing() -> None:
    plain = _miss(_fact("过期待决"))
    assert (plain.slice, plain.voided, plain.unseen, plain.postponed, plain.abandoned) == (
        "事件:chain",
        False,
        False,
        False,
        False,
    )
    assert _miss(_fact("过期待决", strength=0.0)).unseen
    voided = _miss(_fact(VOID, strength=0.0))
    assert voided.voided and not voided.unseen
    assert _miss(_fact("后补")).postponed
    inside = FORMED + timedelta(minutes=30)
    assert _miss(_fact("过期待决"), window=[_Row("s-k0003", "abandoned", inside)]).abandoned
    assert not _miss(_fact("过期待决"), window=[_Row("s-k0003", "abandoned", FORMED)]).abandoned  # 起算那一刻不算
    assert not _miss(_fact("过期待决"), window=[_Row("s-k0003", "interrupted", inside)]).abandoned
    assert _miss(_fact("过期待决"), standing=_standing(break_at=FORMED)).broken
    assert not _miss(_fact("过期待决"), standing=_standing(break_at=FORMED + timedelta(minutes=1))).broken
    assert _miss(_fact("过期待决"), standing=_standing(lower=0.7)).within_budget
    assert _miss(_fact("过期待决"), listed=set()).relation_gone
    assert not _miss(_fact("过期待决"), listed=None).relation_gone
    assert _miss(_fact("过期待决"), current=lambda _token: ("s-k0004",)).reclassified
    absent = _expectation(antecedents=(Antecedent("session|s-k0009|s-k0003|chain", present=False, upward=True),))
    assert _miss(_fact("过期待决"), item=absent).antecedent_absent
    differs = _miss(_fact("过期待决"), today_note="调休上班", basis=[BasisSeen(None, None), BasisSeen("周一", None)])
    assert differs.day_type_differs
    assert not _miss(_fact("过期待决"), today_note="调休上班", basis=[]).day_type_differs
    assert _miss(_fact("过期待决"), basis=[BasisSeen(None, "host_agent")]).notice_driven
