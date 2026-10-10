"""一条 lane 的概念时间线：关系检验读的全部事实，从事件序列与概念命中现算，不落盘。

**单位是预测树的槽**（用户 10-07：「设计成我们预测树的事件槽位吧，包括时间的设计」）。一条记录落在
"它自己的本地日期 × 当天第几槽"；为了跨天比较，槽号写成绝对槽号 ``日序号 × 每天槽数 + 当天第几槽``。

每个概念在时间线上有两种记号：

- **命中**：基础概念看记录的类编号（汇总概念看成员类，``with_ancestors``）；细分概念看映射器的命中；
- **看不到**：映射器对这条记录判过、但答"看不到"（未决），或这条记录还没映射过（它的类上挂着的细分概念全都不知道）。
  窗口里有"看不到"、又没有命中的，这一次的结果算未知，两边都不计（用户 09-27"没看到就不算"）。

**人在场**：这条 lane 当天第一条记录开始的槽到最后一条记录最后所见的槽之间，扣掉观测空白盖住的槽（两种空白都扣：
那段发生了什么不知道）。窗口只量人在场的时间，对照也只取人在场的时刻（``spans``）。人不在的时候什么都不会发生，
拿来比会让一切都显得"更常见"（第三轮评审：别的日期同钟点 36 对显著 → 同一天人在场的时刻 1 对）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from habitus.scene.concepts.model import ConceptKind, ConceptSet
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.relations.config import RelationConfig
from habitus.series import EventSeries, SeriesRecord


@dataclass(frozen=True)
class Mark:
    """一个概念在时间线上的一次命中。``slot`` 是开始的绝对槽号，``last_slot`` 是最后所见的绝对槽号（过了午夜的截在当天最后一槽）。
    ``goal`` 只给"同一件事占比"用。"""

    slot: int
    started_at: datetime
    uri: str
    goal: str | None = None
    last_slot: int | None = None

    @property
    def end(self) -> int:
        return self.slot if self.last_slot is None else max(self.slot, self.last_slot)

    @property
    def day(self) -> date:
        return self.started_at.date()


@dataclass
class LaneTimeline:
    lane: str
    slots_per_day: int
    #: 截止日开始的绝对槽号：窗口伸到这里或之后的，后面没看到。
    horizon: int
    marks: dict[str, list[Mark]] = field(default_factory=dict)
    unknown: dict[str, list[int]] = field(default_factory=dict)
    #: 日子 → 人在场的当天槽号（升序）。
    present: dict[date, tuple[int, ...]] = field(default_factory=dict)
    #: 日子 → 那天每件事（含「待定」）做完的绝对槽号与 uri，只留**它那一类的链的最后一件**（之后转移窗口内同一类没再开始）：
    #: 同一条链的对照时刻。实验组的前因是合并后的一条链、做完之后同类不会马上再来；对照也要同一个条件，
    #: 不然"同类不再来、只能是别的事"会把一切别的后果读成"更多"（合成数据：只调时长，「修改代码 → 开会 ↑」）。
    ends: dict[date, list[tuple[int, str]]] = field(default_factory=dict)
    #: uri → 记录与它的概念命中（调节条件要看前因那一条的结构字段与情境）。
    records: dict[str, SeriesRecord] = field(default_factory=dict)
    hits: Mapping[str, ConceptHits] = field(default_factory=dict)
    #: 情境概念 → 日子 → 那天成立没有（那天有记录判过它才有值；没判过的日子不在里面，算不知道）。
    situations: dict[str, dict[date, bool]] = field(default_factory=dict)

    def day_start(self, day: date) -> int:
        return day.toordinal() * self.slots_per_day

    def concepts(self) -> tuple[str, ...]:
        return tuple(sorted(self.marks))

    def days(self) -> tuple[date, ...]:
        """人在场的日子（升序）。"""

        return tuple(sorted(day for day, slots in self.present.items() if slots))


def build_timelines(
    series: EventSeries,
    concepts: ConceptSet,
    hits: Mapping[str, ConceptHits],
    config: RelationConfig,
) -> dict[str, LaneTimeline]:
    """每条 lane 一条时间线。``hits`` 是 uri → 这条记录的概念命中（映射器的产物，没映射过的不在里面）。"""

    per_day = config.slots_per_day
    lines: dict[str, LaneTimeline] = {}
    spans: dict[tuple[str, date], tuple[int, int]] = {}
    starts: dict[tuple[str, str], list[tuple[int, str]]] = {}
    pending: dict[str, list[int]] = {}
    for record in series.records:
        line = lines.setdefault(
            record.lane,
            LaneTimeline(
                lane=record.lane,
                slots_per_day=per_day,
                horizon=series.cutoff.toordinal() * per_day,
            ),
        )
        line.records[record.uri] = record
        _situations(line, record, concepts, hits.get(record.uri))
        slot = _slot(record.started_at, config.slot_minutes)
        absolute = record.day.toordinal() * per_day + slot
        last = _last_slot(record, config.slot_minutes, per_day)
        first, final = spans.get((record.lane, record.day), (slot, last))
        spans[(record.lane, record.day)] = (min(first, slot), max(final, last))
        ended = record.day.toordinal() * per_day + max(slot, last)
        line.ends.setdefault(record.day, []).append((ended, record.uri))
        starts.setdefault((record.lane, record.kind_token), []).append((absolute, record.uri))
        if not record.classified:
            # 「待定」是叫不出名的一件事：它可能就是任何一个后果，那一槽对所有行为概念都"判不了"（与预测树记删失同一个读法；
            # 裁定 21-3，第四轮评审 E13）——不能算"后果没来"
            pending.setdefault(record.lane, []).append(absolute)
            continue
        found, blind = _concepts_of(record, concepts, hits.get(record.uri))
        mark = Mark(slot=absolute, started_at=record.started_at, uri=record.uri, goal=record.goal, last_slot=ended)
        for identity in found:
            line.marks.setdefault(identity, []).append(mark)
        for identity in blind:
            line.unknown.setdefault(identity, []).append(absolute)
    covered = _gap_slots(series, config.slot_minutes, per_day)
    window = config.transition_window_slots
    followed = {
        uri
        for items in starts.values()
        for (slot, uri), (later, _next) in zip(sorted(items), sorted(items)[1:], strict=False)
        if later - slot <= window
    }
    for line in lines.values():
        line.hits = hits
        for day, finished in line.ends.items():
            line.ends[day] = sorted(item for item in finished if item[1] not in followed)
        for items in line.marks.values():
            items.sort(key=lambda item: (item.slot, item.started_at, item.uri))
        for slots in line.unknown.values():
            slots.sort()
        # 没在时间线上出现过的行为概念（同 lane）也要有一个空位：它是"从没发生过的后果"，检验里照样要问
        for identity in concepts.behaviors():
            if concepts.lane_of(identity) == line.lane:
                line.marks.setdefault(identity, [])
                if pending.get(line.lane):
                    line.unknown[identity] = sorted({*line.unknown.get(identity, []), *pending[line.lane]})
    blocked = set(covered)
    for (lane, day), (first, final) in spans.items():
        start = day.toordinal() * per_day
        lines[lane].present[day] = tuple(slot for slot in range(first, final + 1) if start + slot not in blocked)
    return lines


def _situations(line: LaneTimeline, record: SeriesRecord, concepts: ConceptSet, mapped: ConceptHits | None) -> None:
    """这条记录判过的情境概念（同 lane 的，或不属于任何 lane 的）记到它那一天：命中过一次那天就算成立。"""

    if mapped is None:
        return
    held = {hit.identity for hit in mapped.situation_hits}
    for name in mapped.situations_checked:
        if name not in concepts or concepts.lane_of(name) not in (record.lane, None):
            continue
        identity = concepts[name].identity
        days = line.situations.setdefault(identity, {})
        days[record.day] = days.get(record.day, False) or identity in held


def _concepts_of(
    record: SeriesRecord, concepts: ConceptSet, mapped: ConceptHits | None
) -> tuple[frozenset[str], frozenset[str]]:
    """这条记录命中哪些概念、哪些判不了。

    基础概念由类编号定（不经映射）；没映射过的记录，它的类上挂着的细分概念全都"看不到"。
    """

    base = concepts.base_for(record.kind_token)
    if base is None:
        return frozenset(), frozenset()
    refinements = concepts.refinements_on(record.kind_token)
    if mapped is not None and mapped.kind_token != record.kind_token:
        mapped = None  # 映射时这条还是另一个类（之后词表拆改重打了编号）：那份命中不是现在这个类的，当没映射过（E3）
    if mapped is None:
        return concepts.with_ancestors((base,)), frozenset(refinements)
    found = {base} | {hit.identity for hit in mapped.hits if hit.identity in concepts}
    blind = frozenset(identity for identity in mapped.unresolved_identities if identity in refinements)
    # 后来才加的细分概念：这条记录映射时还没有它，没判过
    blind |= frozenset(
        identity
        for identity in refinements
        if identity not in found
        and identity not in mapped.unresolved_identities
        and _added_after(concepts, identity, mapped)
    )
    return concepts.with_ancestors(found), blind


def _added_after(concepts: ConceptSet, identity: str, mapped: ConceptHits) -> bool:
    item = concepts[identity]
    return item.kind is ConceptKind.REFINEMENT and item.created_at > mapped.mapped_at


def _slot(moment: datetime, slot_minutes: int) -> int:
    return (moment.hour * 60 + moment.minute) // slot_minutes


def _last_slot(record: SeriesRecord, slot_minutes: int, per_day: int) -> int:
    """最后所见落在当天第几槽；过了午夜的截在当天最后一槽（人在场按日子算）。"""

    if record.last_observed_at.date() != record.day:
        return per_day - 1
    return _slot(record.last_observed_at, slot_minutes)


def _gap_slots(series: EventSeries, slot_minutes: int, per_day: int) -> tuple[int, ...]:
    covered: set[int] = set()
    for gap in series.gaps:
        if gap.ended_at <= gap.started_at:
            continue
        last = gap.ended_at - timedelta(microseconds=1)  # 正好停在槽边界上的，那一槽没被盖住
        start = gap.started_at.date().toordinal() * per_day + _slot(gap.started_at, slot_minutes)
        end = last.date().toordinal() * per_day + _slot(last, slot_minutes)
        covered.update(range(start, max(start, end) + 1))
    return tuple(sorted(covered))


__all__ = ["LaneTimeline", "Mark", "build_timelines"]
