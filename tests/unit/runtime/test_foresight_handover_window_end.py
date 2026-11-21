"""接手到时窗末的两处边界（裁定 103 第 1、2 条，裁定 92 第 2 条②；2026-11-17 的 90 天真跑复盘）。

Q1：成员越过自己时窗终点的第一拍正好等于它的最晚时限，这一拍时限还在，要接手（S1–S3）。
Q2：同一刻两条成员能接手、只有一条到了自己的时窗末，依据只从到了时窗末的那条里选（S6–S9，``units.basis`` 的 ``due``）。

共同的摆法：拍对齐 15 分钟槽。依据「day」09:10–09:40，09:15 那一拍记影子；成员 09:15 之后才形成，10:00 那一拍并进它。
场景号对应 ``~/Desktop/Habitus/experiments/2026-11-18-接手边界/提示词.md`` 第一部分第 3 段。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from habitus.foresight.judge import Judge, Judgement, Package
from habitus.foresight.ledger import Expectation
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.foresight.opportunities.model import RelationOpportunities
from habitus.foresight.units import Input, basis, tree_reading
from habitus.runtime.foresight_delivery import DeliveryStage, JudgeMaterials
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.runtime.test_foresight_delivery import DAY0, LIKE, Outlet, earn, expectation, stage

KIND = "s-k0002"
TICK = 900.0
#: 成员都到了时窗、并进依据的那一拍（10:00）；X 已过时窗终点（10:10）、Y 还没到（10:40）的那一拍（10:15）。
JOINED, T = 60, 75
#: Y 过了自己时窗终点（10:40）之后的第一拍（10:45）。
Y_CLOSES = 105


def at(minutes: int) -> datetime:
    return DAY0 + timedelta(minutes=minutes)


def reasons(ledger, identity: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        item.reason
        for item in sorted(ledger.deliveries_on(DAY0.date()), key=lambda item: item.decided_at)
        if item.expectation_id == identity
    ]


def decided_at(ledger, identity: str, reason: str) -> datetime:  # type: ignore[no-untyped-def]
    (found,) = [
        item.decided_at
        for item in ledger.deliveries_on(DAY0.date())
        if item.expectation_id == identity and item.reason == reason
    ]
    return found


def relation(key: str, *, status: str = "established", large_enough: bool | None = True, lift: float = 0.3):
    return RelationOpportunities(
        key=key,
        antecedent="a",
        consequent=KIND,
        antecedent_label="前因",
        consequent_label="后果",
        antecedent_classes=("a",),
        consequent_classes=(KIND,),
        segment="chain",
        status=status,
        upward=True,
        items=(),
        reading=TreeReading(0.5 + lift, 0.5, large_enough),
    )


def table(*relations: RelationOpportunities) -> OpportunityTable:
    return OpportunityTable("session", date(2026, 9, 30), date(2026, 9, 30), 15, 3, relations, ())


#: Y 是成立、效应够大、抬升更高的关系：按 ``units.basis`` 它赢。
Y_LIFTS_HIGHER = table(relation("rel-x", lift=0.2), relation("rel-y", lift=0.4))
#: 两条都不是"效应够大"：按 ``units.basis`` Y 那本账下界更高就赢。
Y_ACCOUNT_HIGHER = table(relation("rel-x", large_enough=None), relation("rel-y", large_enough=None))


def member(identity: str, trigger: str, *, window: tuple[int, int], deadline: int) -> Expectation:
    """事件那条成员：``window`` 与 ``deadline`` 都是 09:00 之后的分钟数。"""

    made = expectation(identity, DAY0, trigger=trigger, source="事件", window=window)
    return replace(made, deadline=at(deadline))


class RecordingJudge(Judge):
    """真实的 Judge（脚本化模型后端），记下每一拍交给模型的小包。"""

    def __init__(self) -> None:
        client, self.provider = recording_client([LIKE, LIKE])
        super().__init__(client)
        self.packages: list[Package] = []

    async def judge(self, package: Package) -> Judgement:
        self.packages.append(package)
        return await super().judge(package)


# ── Q1：越过时窗终点的第一拍正好等于最晚时限（时窗 09:50–10:13，最晚时限 10:15）────────────────


def closing_on_the_deadline(
    tmp_path, *, event_hits: int, materials: JudgeMaterials, **overrides
) -> tuple[DeliveryStage, object]:  # type: ignore[no-untyped-def]
    run, ledger = stage(tmp_path, tick_seconds=TICK, **overrides)
    earn(ledger, trigger="interval", hits=1)
    earn(ledger, trigger="rel", source="事件", hits=event_hits)
    ledger.record_expectation(expectation("day", DAY0, trigger="interval"))
    asyncio.run(run.run_once(at(15), materials))
    assert reasons(ledger, "day") == ["影子"]
    ledger.record_expectation(member("ev", "rel", window=(50, 73), deadline=75))
    asyncio.run(run.run_once(at(JOINED), materials))
    assert reasons(ledger, "ev") == ["单元已选"]
    return run, ledger


ONE_CAUSE = table(relation("rel"))


def test_s1_a_weak_member_decides_on_the_tick_that_equals_its_deadline(tmp_path) -> None:
    """S1：单元档是 0，最晚时限那一拍记「影子」；之后各拍不再决定（S12）。"""

    materials = JudgeMaterials(ONE_CAUSE, lambda uris: {})
    run, ledger = closing_on_the_deadline(tmp_path, event_hits=1, materials=materials)
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "ev") == ["单元已选", "影子"]
    assert decided_at(ledger, "ev", "影子") == at(T)
    for minutes in (90, 105):
        assert asyncio.run(run.run_once(at(minutes), materials)).decided == 0


def test_s2_a_strong_member_waits_for_the_host_on_its_deadline_and_is_never_claimed_next_tick(tmp_path) -> None:
    """S2：单元档是 1、模型没判不像、宿主活着：最晚时限那一拍记「待取」、小包与可取清单带着它，下一拍「宿主没取」（待办 Q7 的现状）。"""

    judge, outlet = RecordingJudge(), Outlet()
    materials = JudgeMaterials(ONE_CAUSE, lambda uris: {})
    run, ledger = closing_on_the_deadline(tmp_path, event_hits=8, materials=materials, judge=judge, outlet=outlet)
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "ev") == ["单元已选", "待取"]
    assert decided_at(ledger, "ev", "待取") == at(T)
    assert [item.expectation_id for item in judge.packages[-1].candidates] == ["ev"]
    assert [row.request_id for row in outlet.published[-1]] == ["ev"]
    asyncio.run(run.run_once(at(90), materials))
    assert reasons(ledger, "ev") == ["单元已选", "待取", "宿主没取"]


def test_s3_a_member_already_done_on_its_deadline_is_recorded_as_done(tmp_path) -> None:
    """S3：最晚时限之前已经做了：那一拍记「已发生」。"""

    quiet = JudgeMaterials(ONE_CAUSE, lambda uris: {})
    run, ledger = closing_on_the_deadline(tmp_path, event_hits=8, materials=quiet)
    done = JudgeMaterials(ONE_CAUSE, lambda uris: {}, happened=lambda item: item.expectation_id == "ev")
    asyncio.run(run.run_once(at(T), done))
    assert reasons(ledger, "ev") == ["单元已选", "已发生"]


def test_s5_a_window_that_ends_on_the_deadline_still_does_not_take_over(tmp_path) -> None:
    """S5 的同类（第一部分第 4 段）：时窗终点等于最晚时限，越过终点那一拍已晚于最晚时限，账上仍只有「单元已选」。"""

    materials = JudgeMaterials(ONE_CAUSE, lambda uris: {})
    run, ledger = stage(tmp_path, tick_seconds=TICK)
    earn(ledger, trigger="interval", hits=1)
    earn(ledger, trigger="rel", source="事件", hits=8)
    ledger.record_expectation(expectation("day", DAY0, trigger="interval"))
    asyncio.run(run.run_once(at(15), materials))
    ledger.record_expectation(member("ev", "rel", window=(50, 75), deadline=75))
    for minutes in (JOINED, T, 90, 105):
        asyncio.run(run.run_once(at(minutes), materials))
    assert reasons(ledger, "ev") == ["单元已选"]


# ── Q2：两条成员只有一条到了时窗末（X 时窗末 10:10，Y 时窗末 10:40，10:15 那一拍）──────────────


def two_late_members(
    tmp_path,
    *,
    opportunities: OpportunityTable = Y_LIFTS_HIGHER,
    x_hits: int = 8,
    y_hits: int = 8,
    x_deadline: int = 240,
    **overrides,
):  # type: ignore[no-untyped-def]
    run, ledger = stage(tmp_path, tick_seconds=TICK, **overrides)
    materials = JudgeMaterials(opportunities, lambda uris: {})
    earn(ledger, trigger="interval", hits=1)
    earn(ledger, trigger="rel-x", source="事件", hits=x_hits)
    earn(ledger, trigger="rel-y", source="事件", hits=y_hits)
    ledger.record_expectation(expectation("day", DAY0, trigger="interval"))
    asyncio.run(run.run_once(at(15), materials))
    assert reasons(ledger, "day") == ["影子"]
    x = member("x", "rel-x", window=(50, 70), deadline=x_deadline)
    y = member("y", "rel-y", window=(50, 100), deadline=240)
    ledger.record_expectation(x)
    ledger.record_expectation(y)
    asyncio.run(run.run_once(at(JOINED), materials))
    assert reasons(ledger, "x") == ["单元已选"] and reasons(ledger, "y") == ["单元已选"]
    assert_y_would_win(run, opportunities, x, y)
    return run, ledger, materials


def assert_y_would_win(run: DeliveryStage, opportunities: OpportunityTable, *items: Expectation) -> None:
    """摆法的前提：两条都交给 ``units.basis`` 时，还没到时窗末的 Y 赢。"""

    inputs = [
        Input(item, run.standing_of(item, at(T), opportunities), *tree_reading(opportunities, item)) for item in items
    ]
    assert basis(inputs, due=("x", "y")).expectation.expectation_id == "y"


@pytest.mark.parametrize(
    ("opportunities", "x_hits"),
    [(Y_LIFTS_HIGHER, 8), (Y_ACCOUNT_HIGHER, 4)],
    ids=["y-lifts-higher", "y-account-higher"],
)
def test_s6_the_member_past_its_window_end_decides_and_the_other_stays_an_input(
    tmp_path, opportunities: OpportunityTable, x_hits: int
) -> None:
    """S6：T 拍由已到时窗末的 X 落决定，没到的 Y 只当输入。"""

    run, ledger, materials = two_late_members(tmp_path, opportunities=opportunities, x_hits=x_hits)
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "x") == ["单元已选", "待取"]
    assert decided_at(ledger, "x", "待取") == at(T)
    assert reasons(ledger, "y") == ["单元已选"]


def test_s7_after_the_member_chose_not_to_speak_the_other_does_not_take_over(tmp_path) -> None:
    """S7（S6 的后续，改动后才成立）：X 自己决定了不说（影子），Y 到了自己的时窗末也不接手；X 不重复决定（S12）。"""

    run, ledger, materials = two_late_members(tmp_path, x_hits=1, y_hits=1)
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "x") == ["单元已选", "影子"]
    for minutes in (90, Y_CLOSES, 120):
        assert asyncio.run(run.run_once(at(minutes), materials)).decided == 0
    assert reasons(ledger, "y") == ["单元已选"]


def test_s8_an_unreachable_member_lets_the_other_take_over_at_its_own_window_end(tmp_path) -> None:
    """S8（S6 的后续，改动后才成立）：X 记「不可达」没说成，Y 到了自己的时窗末照常接手。"""

    outlet = Outlet(alive=False)
    run, ledger, materials = two_late_members(tmp_path, outlet=outlet)
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "x") == ["单元已选", "不可达"]
    assert reasons(ledger, "y") == ["单元已选"]
    outlet.is_alive = True
    asyncio.run(run.run_once(at(90), materials))
    assert reasons(ledger, "y") == ["单元已选"]  # Y 的时窗 10:40 才末
    asyncio.run(run.run_once(at(Y_CLOSES), materials))
    assert reasons(ledger, "y") == ["单元已选", "待取"]


def test_s8_a_member_never_claimed_lets_the_other_take_over(tmp_path) -> None:
    """S8（S6 的后续，改动后才成立）：X「待取」后过了最晚时限「宿主没取」，Y 同一拍接手。"""

    run, ledger, materials = two_late_members(tmp_path, x_deadline=90)
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "x") == ["单元已选", "待取"]
    asyncio.run(run.run_once(at(90), materials))
    assert reasons(ledger, "y") == ["单元已选"]  # 还在等宿主：别的成员不接
    asyncio.run(run.run_once(at(Y_CLOSES), materials))
    assert reasons(ledger, "x") == ["单元已选", "待取", "宿主没取"]
    assert reasons(ledger, "y") == ["单元已选", "待取"]


def test_s9_a_basis_that_cannot_speak_hands_over_to_the_member_past_its_window_end(tmp_path) -> None:
    """S9：依据是候选关系、只能记影子，记影子那一拍就拉进来的 X、Y 摆法同 S6：T 拍由 X 决定。"""

    opportunities = table(
        relation("rel-d", status="candidate", large_enough=None),
        relation("rel-x", lift=0.2),
        relation("rel-y", lift=0.4),
    )
    materials = JudgeMaterials(opportunities, lambda uris: {})
    run, ledger = stage(tmp_path, tick_seconds=TICK)
    for trigger in ("rel-d", "rel-x", "rel-y"):
        earn(ledger, trigger=trigger, source="事件", hits=8)
    x = member("x", "rel-x", window=(50, 70), deadline=240)
    y = member("y", "rel-y", window=(50, 100), deadline=240)
    ledger.record_expectation(expectation("candidate", DAY0, trigger="rel-d", source="事件"))
    ledger.record_expectation(x)
    ledger.record_expectation(y)
    asyncio.run(run.run_once(at(15), materials))
    assert reasons(ledger, "candidate") == ["影子"]
    assert reasons(ledger, "x") == ["单元已选"] and reasons(ledger, "y") == ["单元已选"]
    assert_y_would_win(run, opportunities, x, y)
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "x") == ["单元已选", "待取"]
    assert reasons(ledger, "y") == ["单元已选"]


# ── 接手的成员没说成（裁定 103 第 5 条、裁定 92 第 2 条②）────────────────────────────────


@pytest.mark.parametrize(
    "first",
    [relation("rel-x", status="candidate", large_enough=None), relation("rel-x", large_enough=False)],
    ids=["candidate", "coincidence"],
)
def test_a_member_that_cannot_speak_does_not_block_the_one_that_can(tmp_path, first: RelationOpportunities) -> None:
    """X 本来就不能开口、先到时窗末记了影子：不是它自己决定不说，能开口的 Y 到自己的时窗末照常接手。"""

    run, ledger = stage(tmp_path, tick_seconds=TICK)
    materials = JudgeMaterials(table(first, relation("rel-y", lift=0.4)), lambda uris: {})
    earn(ledger, trigger="interval", hits=1)
    earn(ledger, trigger="rel-x", source="事件", hits=8)
    earn(ledger, trigger="rel-y", source="事件", hits=8)
    ledger.record_expectation(expectation("day", DAY0, trigger="interval"))
    asyncio.run(run.run_once(at(15), materials))
    ledger.record_expectation(member("x", "rel-x", window=(50, 70), deadline=240))
    ledger.record_expectation(member("y", "rel-y", window=(50, 100), deadline=240))
    for minutes in (JOINED, T, 90, Y_CLOSES):
        asyncio.run(run.run_once(at(minutes), materials))
    assert reasons(ledger, "x") == ["单元已选", "影子"]
    assert reasons(ledger, "y") == ["单元已选", "待取"]
    assert decided_at(ledger, "y", "待取") == at(Y_CLOSES)


def test_a_member_whose_antecedent_failed_lets_the_other_take_over(tmp_path, monkeypatch) -> None:
    """X 开口前核出前因不成立：这一刻的话没说出口，Y 到自己的时窗末照常接手。"""

    run, ledger, materials = two_late_members(tmp_path)
    monkeypatch.setattr(run, "_antecedent_holds", lambda item, _materials: item.expectation_id != "x")
    for minutes in (T, 90, Y_CLOSES):
        asyncio.run(run.run_once(at(minutes), materials))
    assert reasons(ledger, "x") == ["单元已选", "前因不成立"]
    assert reasons(ledger, "y") == ["单元已选", "待取"]


def test_a_basis_whose_antecedent_failed_lets_its_member_take_over(tmp_path, monkeypatch) -> None:
    """依据自己开口前核出前因不成立，被它吸收的成员到自己的时窗末照常接手（裁定 92 第 2 条②）。"""

    run, ledger = stage(tmp_path, tick_seconds=TICK)
    materials = JudgeMaterials(table(relation("rel-b"), relation("rel-x")), lambda uris: {})
    monkeypatch.setattr(run, "_antecedent_holds", lambda item, _materials: item.expectation_id != "b")
    earn(ledger, trigger="rel-b", source="事件", hits=8)
    earn(ledger, trigger="rel-x", source="事件", hits=8)
    ledger.record_expectation(expectation("b", DAY0, trigger="rel-b", source="事件"))
    ledger.record_expectation(member("x", "rel-x", window=(50, 100), deadline=240))
    for minutes in (15, 45, 90, Y_CLOSES):
        asyncio.run(run.run_once(at(minutes), materials))
    assert reasons(ledger, "b") == ["等时窗末", "前因不成立"]
    assert reasons(ledger, "x") == ["单元已选", "待取"]


def test_a_member_whose_request_the_host_took_still_counts_as_said(tmp_path) -> None:
    """X 的请求已被宿主取走，之后换代它变回候选关系：取走等同说出口，Y 不接手（裁定 96 第 10 条）。"""

    run, ledger, materials = two_late_members(tmp_path, claimed=lambda request_id: request_id == "x")
    asyncio.run(run.run_once(at(T), materials))
    assert reasons(ledger, "x") == ["单元已选", "待取"]
    later = JudgeMaterials(
        table(relation("rel-x", status="candidate", large_enough=None), relation("rel-y", lift=0.4)), lambda uris: {}
    )
    asyncio.run(run.run_once(at(Y_CLOSES), later))
    assert reasons(ledger, "y") == ["单元已选"]
