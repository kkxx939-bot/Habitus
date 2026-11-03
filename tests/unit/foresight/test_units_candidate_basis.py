"""候选关系不当开口依据（裁定 56 第 12 条、裁定 91；由 ``37-材料/readme/repro-91`` 的复现脚本改成，断言已翻转）。

``units.basis``：没有成立且效应够大的原因时，从能当依据的输入（``speakable``）里按账下界取最高——候选关系与巧合一样只进读数、
不当依据。原来只剔巧合、不剔候选：候选那本影子账下界更高时抢走依据，``unit_level`` 见依据说不了话直接回影子。默认配置
``shadow_uses_unestablished = True``（``config/foresight.py``），候选关系照建影子预期、和日子预期同一刻进同一个单元。

情形：天天吃维生素，「离上次够久」（日子）这本账在提醒档；语义树上「工作日 → 吃维生素」还是候选（日历情境关系），它那本影子账
只在工作日建、周末偶尔忘吃不落在它身上，所以下界比日子账高。今天是工作日，两条都到了时窗、同一刻。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from habitus.foresight import Evidence, Tolerance, climb
from habitus.foresight.ledger import Expectation, account_key
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.foresight.opportunities.model import RelationOpportunities
from habitus.foresight.units import Input, PastMoment, basis, speakable, unit_level, unit_reading
from habitus.runtime.foresight_units import UnitPlanner

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 28, 9, 0, tzinfo=CST)  # 周三
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
LOSS = Tolerance(line=0.67)
KIND = "s-k0042"
REL = f"session|weekday|{KIND}|situation"
DAY_ACCOUNT = account_key(KIND, "日子", "interval")
SIT_ACCOUNT = account_key(KIND, "情境", REL)


def _expect(identity: str, source: str, trigger: str) -> Expectation:
    formed = NOW.replace(hour=0)
    return Expectation(
        expectation_id=identity,
        kind_token=KIND,
        day=formed.date(),
        formed_at=formed,
        source=source,
        trigger=trigger,
        trigger_instance=formed.isoformat(),
        window=(NOW - timedelta(minutes=30), NOW - timedelta(minutes=15)),
        deadline=formed + timedelta(days=1),
        basis=(),
        opportunities=30,
        hits=28,
        observed_from=formed,
    )


def _history() -> list[PastMoment]:
    """过去 35 天：工作日两条都在场、都来了；周末只有日子那条在场，其中 4 个周末日忘了吃。"""

    moments = []
    forgot = 0
    for back in range(35, 0, -1):
        day = NOW - timedelta(days=back)
        weekend = day.weekday() >= 5
        if weekend:
            came = not (forgot < 4 and back % 2 == 0)
            forgot += 0 if came else 1
            moments.append(PastMoment(day, frozenset({DAY_ACCOUNT}), came))
        else:
            moments.append(PastMoment(day, frozenset({DAY_ACCOUNT, SIT_ACCOUNT}), True))
    return moments


HISTORY = _history()
DAY_STANDING = climb(HISTORY, DAY_ACCOUNT, now=NOW, tolerance=LOSS, evidence=EVIDENCE)
SIT_STANDING = climb(HISTORY, SIT_ACCOUNT, now=NOW, tolerance=LOSS, evidence=EVIDENCE)
DAY_INPUT = Input(_expect("day", "日子", "interval"), DAY_STANDING)
CANDIDATE = Input(_expect("weekday", "情境", REL), SIT_STANDING, TreeReading(0.98, 0.80, None), "candidate")


def test_fixture_the_day_account_is_in_the_reminder_tier_and_the_candidate_reads_higher() -> None:
    assert DAY_STANDING.level == 1 and DAY_STANDING.prediction.lower >= LOSS.threshold
    assert SIT_STANDING.prediction.lower > DAY_STANDING.prediction.lower
    assert speakable(DAY_INPUT) and not speakable(CANDIDATE)


def test_a_candidate_relation_does_not_take_the_basis() -> None:
    reading = unit_reading((DAY_INPUT, CANDIDATE), HISTORY, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert reading.reading.lower >= LOSS.threshold  # 单元读数本身过线
    chosen = basis((DAY_INPUT, CANDIDATE), due=("day", "weekday"))
    # 候选那本账下界更高也不当依据：依据是日子，单元开口
    assert chosen.account == DAY_ACCOUNT
    assert unit_level(chosen, reading, previous=1) == 1
    # 输入顺序不影响
    assert basis((CANDIDATE, DAY_INPUT)).account == DAY_ACCOUNT
    # 只有日子那条时同样开口
    alone = unit_reading((DAY_INPUT,), HISTORY, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert unit_level(basis((DAY_INPUT,)), alone, previous=1) == 1


def test_only_unspeakable_inputs_fall_back_to_them_and_stay_in_shadow() -> None:
    """没有能当依据的输入（只剩候选）时照模块说明：在它们里面取，单独当依据记影子。"""

    reading = unit_reading((CANDIDATE,), HISTORY, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    chosen = basis((CANDIDATE,))
    assert chosen.account == SIT_ACCOUNT
    assert unit_level(chosen, reading, previous=1) == 0
    # 日子还没到时窗（不在 due 里）：依据只从到了时窗的里选，只剩候选，记影子
    assert basis((DAY_INPUT, CANDIDATE), due=("weekday",)).account == SIT_ACCOUNT


def test_a_coincidental_relation_still_does_not_take_the_basis() -> None:
    """巧合（成立但效应不够大）照旧不当依据——与候选同一条规则。"""

    coincidence = Input(CANDIDATE.expectation, SIT_STANDING, TreeReading(0.98, 0.80, False), "established")
    assert basis((DAY_INPUT, coincidence)).account == DAY_ACCOUNT


def test_through_the_runtime_planner_with_a_candidate_relation_on_the_table() -> None:
    """走递送用的 ``UnitPlanner.unit_for``：输入的关系状态从机会表读（``tree_reading``），与递送那一拍同一条路。"""

    relation = RelationOpportunities(
        key=REL,
        antecedent="weekday",
        consequent=KIND,
        antecedent_label="工作日",
        consequent_label="吃维生素",
        antecedent_classes=("weekday",),
        consequent_classes=(KIND,),
        segment="situation",
        status="candidate",
        upward=True,
        items=(),
        reading=TreeReading(0.98, 0.80, None),
        calendar_key="weekday",
        calendar_values=("工作日",),
    )
    table = OpportunityTable("session", date(2026, 10, 27), date(2026, 10, 27), 15, 3, (relation,), ())
    planner = UnitPlanner(None, evidence=EVIDENCE, prior_share=1 / 3)
    standings = {"day": DAY_STANDING, "weekday": SIT_STANDING}

    def plan(group: list[Expectation]):  # type: ignore[no-untyped-def]
        return planner.unit_for(
            group,
            due=[item.expectation_id for item in group],
            standing_of=lambda item: standings[item.expectation_id],
            previous_level=lambda kind: 1,
            feedback_of=lambda kind: None,
            setting_of=lambda kind: None,
            table=table,
            now=NOW,
        )

    both = plan([DAY_INPUT.expectation, CANDIDATE.expectation])
    # 依据是日子那条、单元开口；候选那条记「单元已选」跟着它
    assert both.basis.expectation_id == "day" and both.level == 1
    assert [item.expectation_id for item in both.members] == ["weekday"]
    only_day = plan([DAY_INPUT.expectation])
    assert only_day.basis.expectation_id == "day" and only_day.level == 1
