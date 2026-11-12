"""按条映射与一天的封口（实时化阶段 2「映射按槽」）。

**一条记录什么时候可以映射**（``mappable_from``）：它最后所见所在的槽过后再留几槽——不等那一天封口。

**一条记录的材料只由它自己决定**（``DayRecords``）：早映射、晚映射，当天映射还是封口后补映射，用的是同一个取法——

- 当天的部分取到它最后所见为止：它自己的事实、同一条 lane 当天开始时刻不晚于它最后所见的记录与空白；
- 历史的部分取它那一天的那一代的：常态由调用方按那一天给（命中历史是那一天之前已完成的日子，节律是那一代落盘的那一份），
  近几天同类记录是那一天之前 ``recent_days`` 天里同编号的记录。这条的类在最近一代之后才有、或成员才变的（``recent_given``
  答否），这份材料按"没给"，候选记未决 ``RECENT_MISSING``，下一代补判。

之后才落树的记录让当天时间线变了，按现有的续跑判据重判受影响的那一条（``current``）。

**情境不在映射时算**：按条映射写的命中记录里情境留空；一天封口时统一重算（``plan_seal``）、写回、核条数、盖日完成章
（``apply_seal``）。盖章的条件是那一天序列里的每条记录都有命中记录、口径是现在的口径、输入没变；有一条不满足就不盖章，
那一天对细分概念算看不到。情境读的是更早开始的记录的命中，命中不取决于情境，所以封口时一次读齐、与映射的先后无关。

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.scene.concepts.model import ContextScope
from habitus.scene.occurrences.mapper import (
    ConceptMapper,
    ConceptMapperError,
    MappingMaterial,
    TimelineEntry,
    recent_digest,
    timeline_digest,
)
from habitus.scene.occurrences.model import ConceptHit, ConceptHits, UnresolvedReason
from habitus.scene.occurrences.situations import SituationOutcome
from habitus.scene.occurrences.store import ConceptHitStore, DayMarker
from habitus.series import EventSeries, SeriesRecord

#: 这几种原因下，再问一遍可能有不同的答案（模型这一次没答成、或当时材料没给到）：续跑时要重问。
_RETRY_REASONS = frozenset(
    {UnresolvedReason.MODEL_FAILED, UnresolvedReason.TIMELINE_MISSING, UnresolvedReason.RECENT_MISSING}
)

SituationFor = Callable[[BehaviorDocument], SituationOutcome | Sequence[ConceptHit]]
BaselineFor = Callable[[BehaviorDocument], Mapping[str, str]]


def mappable_from(last_observed_at: datetime, *, slot_minutes: int, margin_slots: int) -> datetime:
    """这条记录最早什么时候可以映射：最后所见所在的槽（当地零点起每 ``slot_minutes`` 一槽）结束后再过 ``margin_slots`` 槽。"""

    if slot_minutes <= 0 or margin_slots < 0:
        raise ValueError("slot_minutes must be positive and margin_slots non-negative")
    midnight = last_observed_at.replace(hour=0, minute=0, second=0, microsecond=0)
    slot = (last_observed_at.hour * 60 + last_observed_at.minute) // slot_minutes
    return midnight + timedelta(minutes=(slot + 1 + margin_slots) * slot_minutes)


@dataclass(frozen=True)
class RecordMaterial:
    """一条记录的材料。``recent`` 为 None 是"没给"；``day_needed`` 是它的候选要不要当天时间线（要才进摘要与提示词）。"""

    baseline: Mapping[str, str]
    timeline: tuple[TimelineEntry, ...]
    recent: tuple[TimelineEntry, ...] | None
    day_needed: bool
    #: 候选要近几天同类记录、而这份材料按"没给"（它的类在最近一代之后才有或才变）。
    recent_withheld: bool = False

    @property
    def day_material(self) -> tuple[TimelineEntry, ...] | None:
        return self.timeline if self.day_needed else None

    def given(self) -> MappingMaterial:
        return MappingMaterial(baseline=self.baseline, timeline=self.timeline, recent=self.recent)


class DayRecords:
    """一天里各条记录的取材料口（按序列建一次；``series`` 要含这一天）。"""

    def __init__(
        self,
        series: EventSeries,
        day: date,
        mapper: ConceptMapper,
        *,
        baseline_for: BaselineFor,
        recent_given: Callable[[str], bool] | None = None,
    ) -> None:
        self.series, self.day, self.mapper = series, day, mapper
        self.baseline_for = baseline_for
        self.recent_given = recent_given
        self.records = {record.uri: record for record in series.on(day)}
        gaps = tuple(TimelineEntry.from_gap(gap) for gap in series.gaps_on(day))
        self._timelines = {
            lane: tuple(TimelineEntry.from_record(item) for item in self.records.values() if item.lane == lane) + gaps
            for lane in {item.lane for item in self.records.values()}
        }
        self._recent: dict[str, list[TimelineEntry]] | None = None

    def material(self, entry: SeriesRecord, document: BehaviorDocument) -> RecordMaterial:
        needs = self.mapper.needs(entry.kind_token) if entry.classified else frozenset()
        wants_recent = ContextScope.RECENT in needs
        withheld = wants_recent and self.recent_given is not None and not self.recent_given(entry.kind_token)
        seen_until = entry.last_observed_at
        return RecordMaterial(
            baseline=self.baseline_for(document),
            timeline=tuple(item for item in self._timelines[entry.lane] if item.started_at <= seen_until),
            recent=self._recent_of(entry.kind_token) if wants_recent and not withheld else None,
            day_needed=ContextScope.DAY in needs,
            recent_withheld=withheld,
        )

    def _recent_of(self, token: str) -> tuple[TimelineEntry, ...]:
        """这一天之前 ``recent_days`` 天序列里同编号的记录（编号是树上现在的编号）。第一次有人要时才分组。"""

        if self._recent is None:
            self._recent = {}
            since = self.day - timedelta(days=self.mapper.config.recent_days)
            for record in self.series.records:
                if since <= record.day < self.day:
                    self._recent.setdefault(record.kind_token, []).append(TimelineEntry.from_record(record))
        return tuple(self._recent.get(token, ()))


def current(existing: ConceptHits | None, mapper: ConceptMapper, entry: SeriesRecord, material: RecordMaterial) -> bool:
    """盘上这条命中记录还作数：口径是现在的口径、输入没变（``_inputs_unchanged``），映射之后这条自己的最后所见也没再变
    （按槽映射时它可能还没看完：之后又延长了，就按看完的样子重判）。"""

    return (
        existing is not None
        and existing.mapper == mapper.version_for(entry.kind_token)
        and existing.last_observed_at == entry.last_observed_at
        and _inputs_unchanged(existing, entry.kind_token, material.baseline, material.recent, material.day_material)
    )


def waiting_for_recent(
    existing: ConceptHits | None, mapper: ConceptMapper, entry: SeriesRecord, material: RecordMaterial
) -> bool:
    """这条只差"近几天同类记录"这一份没给、别的都没变：现在再映射一遍结果不会变，等下一代（盖章仍按 ``current``，不盖）。"""

    if existing is None or not material.recent_withheld or existing.recent_digest is not None:
        return False
    if UnresolvedReason.RECENT_MISSING not in existing.unresolved.values():
        return False
    settled = {
        name: reason for name, reason in existing.unresolved.items() if reason is not UnresolvedReason.RECENT_MISSING
    }
    return current(replace(existing, unresolved=settled), mapper, entry, material)


async def map_record(
    store: ConceptHitStore,
    mapper: ConceptMapper,
    document: BehaviorDocument,
    entry: SeriesRecord,
    material: RecordMaterial,
) -> ConceptHits:
    """映射一条记录、写命中记录（情境留空，封口时统一算）。不盖章；这一天原来盖过章的先撤掉——它的命中变了。"""

    record = await mapper.map(document, lane=entry.lane, classified=entry.classified, material=material.given())
    store.discard_completion(entry.day)
    store.write(record)
    return record


def day_documents(tree: BehaviorTree, series: EventSeries, day: date) -> dict[str, BehaviorDocument]:
    """序列里这一天的每条记录在行为树上的全文（按序列的先后）；序列有、树上没有就是两边对不上。"""

    by_uri = {
        str(BehaviorURI.from_address(document.address)): document
        for document in tree.read_day(BehaviorKind.OCCURRENCE, day)
    }
    uris = [record.uri for record in series.on(day)]
    missing = sorted(set(uris) - set(by_uri))
    if missing:
        raise ConceptMapperError(
            f"the series holds {len(missing)} occurrences on {day} that the tree does not: {missing[0]}"
        )
    return {uri: by_uri[uri] for uri in uris}


@dataclass(frozen=True)
class DaySeal:
    """一天封口的计划（纯读盘、纯计算）：封口后这一天的全部命中记录、其中情境变了要写回的、还不作数的条数。"""

    day: date
    #: 序列里这一天的记录数。
    expected: int
    #: 封口后的命中记录（情境已重算），按序列的先后；还不作数的那几条不在里面。
    records: tuple[ConceptHits, ...]
    rewrite: tuple[ConceptHits, ...]
    #: 还没映射、口径不是现在的、或输入变了的条数：不为零就不能盖章。
    not_current: int
    #: 盘上有、序列里没有的命中记录（叶名）：盖章前清掉。
    stale: tuple[str, ...]
    keep: frozenset[str]
    mapper: str

    @property
    def ready(self) -> bool:
        return self.not_current == 0


def plan_seal(
    documents: Mapping[str, BehaviorDocument],
    store: ConceptHitStore,
    mapper: ConceptMapper,
    records: DayRecords,
    *,
    situation_for: SituationFor,
    force: bool = False,
) -> DaySeal:
    """一天封口要做什么：核每条记录作不作数、重算情境。行为树这一天比盘上少（读不到、或缩水了）时拒绝清掉多出来的
    命中记录（多半是根指错或封口日算错），除非 ``force``。"""

    day = records.day
    on_disk = {item.address.identity_name: item for item in store.read_day(day)}
    if len(documents) < len(on_disk) and not force:
        raise ConceptMapperError(
            f"the behaviour tree shows {len(documents)} occurrences on {day} but the hit store holds {len(on_disk)}; pass force=True to drop the extra records"
        )
    keep = frozenset(document.address.identity_name for document in documents.values())
    sealed: list[ConceptHits] = []
    rewrite: list[ConceptHits] = []
    for uri, document in documents.items():
        entry = records.records[uri]
        existing = on_disk.get(document.address.identity_name)
        if existing is None or not current(existing, mapper, entry, records.material(entry, document)):
            continue
        found = situation_for(document)
        hits, checked = (found.hits, found.checked) if isinstance(found, SituationOutcome) else (tuple(found), ())
        if not _same_situations(existing, hits, checked):
            existing = replace(existing, situation_hits=tuple(hits), situations_checked=tuple(checked))
            rewrite.append(existing)
        sealed.append(existing)
    return DaySeal(
        day=day,
        expected=len(documents),
        records=tuple(sealed),
        rewrite=tuple(rewrite),
        not_current=len(documents) - len(sealed),
        stale=tuple(sorted(set(on_disk) - keep)),
        keep=keep,
        mapper=mapper.day_version(str(document.fields["kind_token"]) for document in documents.values()),
    )


def apply_seal(store: ConceptHitStore, mapper: ConceptMapper, plan: DaySeal, *, now: datetime) -> DayMarker:
    """把封口计划落盘：清掉序列里没有的命中记录、写回重算过的情境、核条数与口径、盖日完成章。"""

    if not plan.ready:
        raise ConceptMapperError(f"{plan.day} still has {plan.not_current} records that are not mapped as of now")
    store.retain_only(plan.day, plan.keep)
    for record in plan.rewrite:
        store.write(record)
    return store.complete_day(
        plan.day, records=plan.expected, completed_at=now, mapper=plan.mapper, expected=mapper.version_for
    )


def _same_situations(existing: ConceptHits, hits: Sequence[ConceptHit], checked: Sequence[str]) -> bool:
    fresh = {(hit.identity, hit.grade) for hit in hits}
    held = {(hit.identity, hit.grade) for hit in existing.situation_hits}
    return fresh == held and set(existing.situations_checked) == set(checked) | {hit.concept for hit in hits}


def _inputs_unchanged(
    existing: ConceptHits,
    token: str,
    baseline: Mapping[str, str],
    recent: Sequence[TimelineEntry] | None,
    timeline: Sequence[TimelineEntry] | None,
) -> bool:
    """续跑判据是"输入没变"，不是"没有未决"。

    - 编号变了（词表迁移改写了这一条）一律重判：候选是挂在类上的细分概念，类换了候选就换了；
    - "近几天同类记录"变了（别的日子被迁移改了编号）重判：要它的细分概念答案可能变；
    - "当天时间线"变了（同一天后来补进了记录或空白）重判：答过"没有""看不到"的，补进来的那条可能正是判据要找的；
    - 带未决的：模型这一次没答成、或当时材料没给到 → 重问；因缺常态 / 常态解不出而未决的，常态表变了才重问；
      模型答"看不到"的，材料没变就不重问（再问一遍答案不会变）。
    """

    if existing.kind_token != token:
        return False
    if existing.recent_digest != (None if recent is None else recent_digest(recent)):
        return False
    if existing.timeline_digest != (None if timeline is None else timeline_digest(timeline)):
        return False
    reasons = set(existing.unresolved.values())
    if reasons & _RETRY_REASONS:
        return False
    if reasons & {UnresolvedReason.BASELINE_MISSING, UnresolvedReason.BASELINE_UNPARSEABLE}:
        return dict(existing.baseline_snapshot) == dict(baseline)
    return True


__all__ = [
    "BaselineFor",
    "DayRecords",
    "DaySeal",
    "RecordMaterial",
    "SituationFor",
    "apply_seal",
    "current",
    "day_documents",
    "map_record",
    "mappable_from",
    "plan_seal",
    "waiting_for_recent",
]
