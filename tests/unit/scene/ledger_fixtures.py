"""账本测试共用的材料：不经映射器直接造概念命中记录、一组假设、假的机会口与覆盖口。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta

from habitus.behavior.model import BehaviorAddress
from habitus.behavior.uri import BehaviorURI
from habitus.scene.concepts import ConceptGrade, ConceptSet, GradeMeasure
from habitus.scene.hypotheses import (
    Antecedent,
    Aspect,
    Direction,
    Hypothesis,
    HypothesisOrigin,
    HypothesisSource,
    TypePrior,
)
from habitus.scene.ledger import Coverage, ObservedGap, Opportunity, OpportunityRequest, OpportunitySnapshot, WindowSpan
from habitus.scene.occurrences import ConceptHit, ConceptHits
from tests.unit.scene.concept_fixtures import ALL_CONCEPTS, LATE_RULE, concept
from tests.unit.scene.fixtures import at

NOW = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
MAPPER = "scene_concept_mapper_prompt_v2+schemaabc+emb:fake+llm:fake+concepts:0123"
CREATED = datetime(2026, 9, 26, 2, 0, tzinfo=UTC)
BASELINE = HypothesisSource(HypothesisOrigin.BASELINE)

#: 三档的「熬夜」：剂量那条判据要三个点才谈得上趋势（两点之间没有曲线）。
NIGHT_OWL_GRADES = (
    ConceptGrade("轻", GradeMeasure.START_MINUTE_OF_DAY, 120, 240, relative=True),
    ConceptGrade("中", GradeMeasure.START_MINUTE_OF_DAY, 240, 360, relative=True),
    ConceptGrade("重", GradeMeasure.START_MINUTE_OF_DAY, 360, 720, relative=True),
)
NIGHT_OWL = concept("熬夜", "入睡时刻晚于常态两小时以上", rule=LATE_RULE, grades=NIGHT_OWL_GRADES)
BOOKING = concept("约球", "和人约好了要去打球")
WAKE = concept("起床", "起床离开床")
COFFEE = concept("咖啡", "喝了一杯咖啡")
BEDTIME = concept("就寝", "上床睡觉")
CONCEPTS = ConceptSet((*ALL_CONCEPTS, NIGHT_OWL, BOOKING, WAKE, COFFEE, BEDTIME))


def uri_for(day: date, name: str, hour: int, minute: int) -> str:
    return str(BehaviorURI.from_address(BehaviorAddress.occurrence(day, name, at(day, hour, minute))))


def record(
    day: date,
    name: str,
    hour: int,
    minute: int,
    *hits: ConceptHit | str,
    situations: tuple[ConceptHit | str, ...] = (),
    kind: str | None = None,
    lasts_minutes: int = 10,
) -> ConceptHits:
    uri = uri_for(day, name, hour, minute)
    return ConceptHits(
        occurrence_uri=uri,
        kind_token=kind or name,
        last_observed_at=at(day, hour, minute) + timedelta(minutes=lasts_minutes),
        hits=tuple(item if isinstance(item, ConceptHit) else ConceptHit(item) for item in hits),
        situation_hits=tuple(item if isinstance(item, ConceptHit) else ConceptHit(item) for item in situations),
        unresolved=(),
        baseline_snapshot={},
        mapper=MAPPER,
        mapped_at=NOW,
    )


def hypothesis(
    *antecedents: Antecedent | str,
    consequent: str = "早餐",
    aspect: Aspect = Aspect.PROBABILITY,
    direction: Direction = Direction.DOWN,
    type_prior: TypePrior | None = TypePrior.INHIBITING,
    expected_at: int | None = 1,
    horizon: int = 1,
    split_by: tuple[str, ...] = (),
    released_by: tuple[str, ...] = (),
    note: str = "睡得晚起得晚，早饭常跳过",
) -> Hypothesis:
    items = tuple(item if isinstance(item, Antecedent) else Antecedent(item) for item in antecedents)
    item = Hypothesis(
        antecedents=items,
        consequent=consequent,
        aspect=aspect,
        direction=direction,
        type_prior=type_prior,
        expected_at=expected_at,
        horizon=horizon,
        split_by=split_by,
        released_by=released_by,
        note=note,
        source=BASELINE,
        created_at=CREATED,
    )
    item.validate_against(CONCEPTS)
    return item


LATE_TO_BREAKFAST = hypothesis("晚睡")
LATE_TRAVEL_TO_BREAKFAST = hypothesis(Antecedent("晚睡", "重"), "出差中")
BALL_LATE_TO_BREAKFAST = hypothesis("打球", "晚睡", note="打完球又熬夜")
EXERCISE_TO_COFFEE = hypothesis("运动", consequent="咖啡", aspect=Aspect.COUNT, direction=Direction.UP, type_prior=None, horizon=3, note="运动后多喝")
LATE_TO_WAKE = hypothesis("晚睡", consequent="起床", aspect=Aspect.TIMING, direction=Direction.UP, type_prior=None, note="睡得晚起得晚")
#: 七i 一 的补偿就寝：02:10 的锚之后**第一个**就寝机会就是当晚 23:30，所以是 ``expected_at=1``
#: （评审 A-9/C-6：原写 2 会去量后天晚上）。
LATE_TO_BEDTIME = hypothesis("晚睡", consequent="就寝", aspect=Aspect.TIMING, direction=Direction.DOWN, type_prior=None, expected_at=1, note="补偿，当晚早睡")
#: 无节律型：不数机会、没有时效；再次约球把上一次的前提作废（``released_by``）。
BOOKING_TO_BALL = hypothesis("约球", consequent="打球", direction=Direction.UP, type_prior=TypePrior.ENABLING, expected_at=None, note="约了就去")
REBOOKING_TO_BALL = hypothesis(
    "约球", consequent="打球", direction=Direction.UP, type_prior=TypePrior.ENABLING, expected_at=None, released_by=("约球",), note="约了就去，重新约就作废"
)

#: 每个后件一天的常态机会：(时, 分, 概率, 半宽分钟)。早餐一天一个；咖啡三杯三个。
DAILY_OPPORTUNITIES: Mapping[str, Sequence[tuple[int, int, float, int]]] = {
    "早餐": ((7, 45, 0.88, 90),),
    "起床": ((8, 10, 0.90, 90),),
    "就寝": ((23, 30, 0.90, 90),),
    "咖啡": ((8, 30, 0.50, 45), (14, 0, 0.40, 45), (20, 0, 0.30, 45)),
    "打球": ((19, 0, 0.20, 90),),
}


def opportunity(day: date, hour: int, minute: int, probability: float, half_minutes: int) -> Opportunity:
    centre = at(day, hour, minute)
    return Opportunity(at=centre, span=WindowSpan(centre - timedelta(minutes=half_minutes), centre + timedelta(minutes=half_minutes)), probability=probability)


def daily_snapshot(consequent: str, anchor: datetime, count: int = 16, *, generation: str = "gen-1", table: Mapping[str, Sequence[tuple[int, int, float, int]]] | None = None) -> OpportunitySnapshot:
    """从锚往后按日历日铺开机会：第一个**还没结束**（``span.end > anchor``）的机会就是"第 1 次机会"。

    锚正在其中的峰也算——晚睡·重 07:00 命中之后，当天 06:40–09:40 的起床峰就是"接下来那次机会"，人晚起正是晚在
    这个峰上。按"峰的开始晚于锚"铺会把它丢掉，当天 12:00 的起床就被算到次日峰上（时刻差 −1210 分，真值 +230；
    评审 A-2）。刀 5 的真机会口按同一条约定从树的 marginal 曲线取峰。
    """

    rows = (table or DAILY_OPPORTUNITIES)[consequent]
    found: list[Opportunity] = []
    day = anchor.date() - timedelta(days=1)
    while len(found) < count:
        for hour, minute, probability, half in rows:
            item = opportunity(day, hour, minute, probability, half)
            if item.span.end > anchor:
                found.append(item)
        day += timedelta(days=1)
    return OpportunitySnapshot(generation=generation, opportunities=tuple(found[:count]))


class TableOpportunities:
    """按后件查表铺机会的对照口；记下每次请求。查不到给 None。"""

    def __init__(self, table: Mapping[str, Sequence[tuple[int, int, float, int]]] | None = DAILY_OPPORTUNITIES, generation: str = "gen-1") -> None:
        self.table = dict(table or {})
        self.generation = generation
        self.requests: list[OpportunityRequest] = []

    def opportunities(self, request: OpportunityRequest) -> OpportunitySnapshot | None:
        self.requests.append(request)
        if request.consequent not in self.table:
            return None
        return daily_snapshot(request.consequent, request.anchor, request.count, generation=self.generation, table=self.table)


class FixedCoverage:
    """固定覆盖比例的覆盖口；与 ``dark`` 里任一段相交的机会一律没看清（覆盖 0）。记下每次问的段。"""

    def __init__(self, fraction: float = 1.0, dark: tuple[WindowSpan, ...] = ()) -> None:
        self.fraction = fraction
        self.dark = dark
        self.spans: list[WindowSpan] = []

    def coverage(self, span: WindowSpan) -> Coverage:
        self.spans.append(span)
        hit = tuple(ObservedGap(gap.start, gap.end, "未观测") for gap in self.dark if span.start < gap.end and gap.start < span.end)
        return Coverage(observed_fraction=0.0 if hit else self.fraction, gaps=hit)


__all__ = [
    "BALL_LATE_TO_BREAKFAST",
    "BEDTIME",
    "BOOKING",
    "BOOKING_TO_BALL",
    "COFFEE",
    "CONCEPTS",
    "CREATED",
    "DAILY_OPPORTUNITIES",
    "EXERCISE_TO_COFFEE",
    "LATE_TO_BEDTIME",
    "LATE_TO_BREAKFAST",
    "LATE_TO_WAKE",
    "LATE_TRAVEL_TO_BREAKFAST",
    "MAPPER",
    "NIGHT_OWL",
    "NIGHT_OWL_GRADES",
    "NOW",
    "REBOOKING_TO_BALL",
    "WAKE",
    "FixedCoverage",
    "TableOpportunities",
    "daily_snapshot",
    "hypothesis",
    "opportunity",
    "record",
    "uri_for",
]
