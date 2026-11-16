"""丙二修补（用户 2026-11-16）：物理一支一整天上游一点数据都没送来（没有记录、也没有报"没在看"的空白），这一天算没在看——
和会话一支一整天没有会话同一个规矩（裁定 89 第 2 条）。那天有别的记录、只是没做这件事，照常记漏。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from habitus.foresight.ledger import AccountFact
from tests.unit.physical_downstream.site import MEDICINE, World, at

D13 = date(2026, 4, 13)
MORNING = [("吃药", 8, 0, 10)]


def _pill_facts(world: World) -> list[AccountFact]:
    ledger = world.ledger("physical")
    return [fact for account in ledger.accounts() for fact in ledger.facts_of(account) if fact.expectation_id == "pill"]


def _settle_a_day(tmp_path: Path, plan: list[list[tuple[str, int, int, int]]] | None) -> list[AccountFact]:
    """13 号吃了药；14 号有一条早上的吃药预期，14 号那天按 ``plan`` 送（None 是一点数据都不送）；15 号吃了药，16 号零点结算 14 号。"""

    world = World(tmp_path, start=at(D13, 0))
    try:
        world.physical_days(D13, [MORNING])
        day = D13 + timedelta(days=1)
        world.use("physical", MEDICINE)
        world.expect("physical", "pill", MEDICINE, window=(at(day, 8), at(day, 8, 30)), deadline=at(day, 10))
        if plan is None:
            world.night(day + timedelta(days=1))
        else:
            world.physical_days(day, plan)
        world.physical_days(day + timedelta(days=1), [MORNING])
        return _pill_facts(world)
    finally:
        world.close()


def test_a_physical_day_without_any_data_is_not_watched_and_not_a_miss(tmp_path: Path) -> None:
    facts = _settle_a_day(tmp_path, None)
    assert facts, "the physical expectation was never settled"
    assert all(fact.strength == 0.0 for fact in facts), [(fact.kind, fact.strength) for fact in facts]


def test_a_physical_day_with_other_records_but_no_pill_is_still_a_miss(tmp_path: Path) -> None:
    facts = _settle_a_day(tmp_path, [[("跑步", 18, 0, 30)]])
    assert facts, "the physical expectation was never settled"
    assert any(fact.strength == 1.0 and not fact.good for fact in facts), [(fact.kind, fact.strength) for fact in facts]
