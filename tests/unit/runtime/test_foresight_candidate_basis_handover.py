"""依据不能开口也算没说成（裁定 56 第 12 条、裁定 92 第 2 条②；`37-材料/readme/fix-candidate-basis`）：同一刻只有候选关系（或巧合）
撑着的事件那条先到时窗，单元只能从它里面取依据、记影子，还没到时窗的日子那条被「单元已选」吸收；它到了自己的时窗末照常接手——
按同一组输入重算单元、依据换成能开口的它。依据不能开口的单元也不写单元的档（滞回读的上一次不被强制的 0 冲掉）。
后到时窗、经"并进已决定的依据"吸收的同样接手。回归：关系成立且效应够大时成员照旧被吸收；依据不能开口但这回已经做了、
或依据能开口而单元按读数决定不说的，成员不接手。

情形（30 天回放 04-27「订工作餐 → 买猫用品」、边界场景 s4 03-15 的同一个形状）：事件那条 09:10–09:40、最晚时限 11:00；日子那条
09:50–10:20、最晚时限 13:00，两本账都在提醒档，这个行为单元上一次的档是 1。
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

from habitus.foresight.ledger import UnitLevel
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.foresight.opportunities.model import RelationOpportunities
from habitus.runtime.foresight_delivery import JudgeMaterials
from tests.unit.runtime.test_foresight_delivery import DAY0, earn, expectation, stage

KIND = "s-k0002"


def table(status: str, large_enough: bool | None) -> OpportunityTable:
    relation = RelationOpportunities(
        key="rel",
        antecedent="a",
        consequent=KIND,
        antecedent_label="订工作餐",
        consequent_label="买猫用品",
        antecedent_classes=("a",),
        consequent_classes=(KIND,),
        segment="chain",
        status=status,
        upward=True,
        items=(),
        reading=TreeReading(0.9, 0.5, large_enough),
    )
    return OpportunityTable("session", date(2026, 9, 30), date(2026, 9, 30), 15, 3, (relation,), ())


CANDIDATE = table("candidate", None)
COINCIDENCE = table("established", False)
CAUSE = table("established", True)


def at(minutes: int):  # type: ignore[no-untyped-def]
    return DAY0 + timedelta(minutes=minutes)


def reasons(ledger, identity: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        item.reason
        for item in sorted(ledger.deliveries_on(DAY0.date()), key=lambda item: item.decided_at)
        if item.expectation_id == identity
    ]


def moment(tmp_path, *, hits: int = 8, day_later: bool = False, previous: int | None = 1):  # type: ignore[no-untyped-def]
    run, ledger = stage(tmp_path)
    earn(ledger, trigger="rel", source="事件", hits=hits)
    earn(ledger, trigger="interval", hits=hits)
    if previous is not None:
        ledger.record_unit_level(UnitLevel(KIND, DAY0 - timedelta(days=1), previous))
    ledger.record_expectation(expectation("ev", DAY0, trigger="rel", source="事件"))
    day = expectation("day", DAY0, trigger="interval", window=(50, 80), deadline_hours=4)
    if not day_later:
        ledger.record_expectation(day)
    return run, ledger, day


def unit_level(ledger) -> int | None:  # type: ignore[no-untyped-def]
    level = ledger.unit_level_of(KIND)
    return level.level if level is not None else None


def test_the_day_speaks_when_a_candidate_alone_opened_the_moment(tmp_path) -> None:
    run, ledger, _day = moment(tmp_path)
    materials = JudgeMaterials(CANDIDATE, lambda uris: {})
    asyncio.run(run.run_once(at(16), materials))
    assert reasons(ledger, "ev") == ["影子"] and reasons(ledger, "day") == ["单元已选"]
    assert unit_level(ledger) == 1  # 依据不能开口：不写单元的档
    for minutes in (41, 51):
        assert asyncio.run(run.run_once(at(minutes), materials)).decided == 0  # 日子那条 10:20 才到时窗末
    asyncio.run(run.run_once(at(81), materials))
    assert reasons(ledger, "day") == ["单元已选", "待取"]
    assert asyncio.run(run.run_once(at(82), materials)).decided == 0  # 接手只一次
    assert unit_level(ledger) == 1


def test_the_day_that_joined_a_candidate_basis_later_also_speaks(tmp_path) -> None:
    """日子那条在候选那条决定之后才形成：到时窗时并进已决定的依据（「单元已选」），到了时窗末一样接手。"""

    run, ledger, day = moment(tmp_path, day_later=True)
    materials = JudgeMaterials(CANDIDATE, lambda uris: {})
    asyncio.run(run.run_once(at(16), materials))
    ledger.record_expectation(day)
    asyncio.run(run.run_once(at(51), materials))
    assert reasons(ledger, "day") == ["单元已选"]
    asyncio.run(run.run_once(at(81), materials))
    assert reasons(ledger, "day") == ["单元已选", "待取"]


def test_a_coincidental_basis_hands_over_the_same_way(tmp_path) -> None:
    run, ledger, _day = moment(tmp_path)
    materials = JudgeMaterials(COINCIDENCE, lambda uris: {})
    asyncio.run(run.run_once(at(16), materials))
    assert reasons(ledger, "ev") == ["影子"] and unit_level(ledger) == 1
    asyncio.run(run.run_once(at(81), materials))
    assert reasons(ledger, "day") == ["单元已选", "待取"]
    assert unit_level(ledger) == 1


def test_an_established_cause_keeps_the_day_subsumed(tmp_path) -> None:
    run, ledger, _day = moment(tmp_path)
    materials = JudgeMaterials(CAUSE, lambda uris: {})
    asyncio.run(run.run_once(at(16), materials))
    asyncio.run(run.run_once(at(41), materials))
    assert reasons(ledger, "ev") == ["等时窗末", "待取"]
    asyncio.run(run.run_once(at(81), materials))
    assert reasons(ledger, "day") == ["单元已选"]  # 依据还等着宿主：同一刻只说一次
    assert unit_level(ledger) == 1


def test_a_candidate_basis_that_already_happened_keeps_the_day_subsumed(tmp_path) -> None:
    """这回已经做了是单元替这一刻做的决定，不是没说成。"""

    run, ledger, _day = moment(tmp_path)
    materials = JudgeMaterials(CANDIDATE, lambda uris: {}, happened=lambda item: item.expectation_id == "ev")
    asyncio.run(run.run_once(at(16), materials))
    assert reasons(ledger, "ev") == ["已发生"]
    for minutes in (81, 200):
        asyncio.run(run.run_once(at(minutes), materials))
    assert reasons(ledger, "day") == ["单元已选"]


def test_a_speakable_basis_the_unit_kept_quiet_keeps_the_day_subsumed(tmp_path) -> None:
    """依据能开口、单元按读数决定影子：那是单元的决定，成员不接手。"""

    run, ledger, _day = moment(tmp_path, hits=1, previous=None)
    materials = JudgeMaterials(CAUSE, lambda uris: {})
    asyncio.run(run.run_once(at(16), materials))
    assert reasons(ledger, "ev") == ["影子"] and reasons(ledger, "day") == ["单元已选"]
    for minutes in (81, 200):
        asyncio.run(run.run_once(at(minutes), materials))
    assert reasons(ledger, "day") == ["单元已选"]


def test_a_basis_already_taken_by_the_host_is_as_good_as_spoken(tmp_path) -> None:
    """F8（用户 11-10「取走等同说出口」）：依据那条「待取」已被宿主取走（还没说出口）、换代后关系变回候选——依据不撤，同一刻的
    成员也不接手，出口上同一刻最多一条请求。对照：没被取走的，依据撤下（「依据失效」）、成员照现有规则接手。"""

    for taken in (True, False):
        claimed: set[str] = set()
        run, ledger = stage(tmp_path / str(taken), claimed=claimed.__contains__)
        earn(ledger, trigger="rel", source="事件", hits=8)
        earn(ledger, trigger="interval", hits=8)
        ledger.record_unit_level(UnitLevel(KIND, DAY0 - timedelta(days=1), 1))
        ledger.record_expectation(expectation("ev", DAY0, trigger="rel", source="事件"))
        ledger.record_expectation(expectation("day", DAY0, trigger="interval", window=(50, 80), deadline_hours=4))
        cause = JudgeMaterials(CAUSE, lambda uris: {})
        asyncio.run(run.run_once(at(15), cause))
        asyncio.run(run.run_once(at(41), cause))
        assert reasons(ledger, "ev")[-1] == "待取"
        if taken:
            claimed.add("ev")
        candidate = JudgeMaterials(CANDIDATE, lambda uris: {})
        for minute in (56, 81):
            asyncio.run(run.run_once(at(minute), candidate))
        if taken:
            assert reasons(ledger, "ev")[-1] == "待取" and "待取" not in reasons(ledger, "day")
        else:
            assert reasons(ledger, "ev")[-1] == "依据失效" and reasons(ledger, "day")[-1] == "待取"
        open_now = [row.request_id for row in run.outlet.published[-1]]  # type: ignore[union-attr]
        assert open_now == (["ev"] if taken else ["day"])  # 同一刻只挂一条
