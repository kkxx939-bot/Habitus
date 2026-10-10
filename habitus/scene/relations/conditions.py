"""调节条件（含组合前因）：必须编译成三种可算的模板之一，否则丢弃（语义树新方案 ``13`` ②）。

1. **结构字段**（``FIELD``）——机械判，看前因那一条：周几（工作日 / 周末）、日历备注（调休、节假日……）、地点、同在的人；
2. **时间线**（``TIMELINE``）——机械判：A 之前同一条链的窗口里出现过某一类（"讨论方案之后的修改代码"；**组合前因**
   "A 与 C 一起"就是这一种，不另起一套）；
3. **带判据句的文本概念**（``CONCEPT``）——交映射器打标签：前因那一条命中了某个情境概念（"约了人"这类）。

条件的来源有两处：机械列举的（每个够样本的前因：周末、出现过的日历备注 / 地点 / 同在的人、出现在它之前的每一类、
同 lane 的每个情境概念），以及大模型提的（第 7 步：读 A 那一条与它之前的概要、不看 B，提出"可能被什么左右"，
经 ``compile_proposal`` 编译，落不成模板的进已拒清单，不每晚换个说法反复试）。第 N 晚提的只在 N 之后的数据上检验。

**检验**：只在独立样本 ≥ 40 的前因上开（会话 lane 目前只有修改代码、调研、统计代码行数够），否则样本不够。
把这条关系的层按条件分成两组，比两组的效应：每组用同一套零假设概率算 Peto 对数优势比 ``(实测 − 期望) / 方差``，
交互 z = 两组之差 / √(1/方差₁ + 1/方差₂)。不用"每层 y − 对照率"的两样本 t：阳性很少时那个方差几乎全来自各天对照率的起伏，
会把 p 压到 0（合成数据上没埋关系也出了几十条"调节"）。

"在条件下更强"说的是两件事：条件下有效应、而且和条件外不一样——按**交并检验**取两者 p 的较大者（有效水平不变）：
条件下那一组用层内精确检验（泊松二项），交互用上面的 z。只用交互 z 不够：条件下只有两三层时，z 的正态近似失灵，
一层阳性就能把 z 推得很大（真实数据：条件下 2 层、阳性 1 层的"调节"出了几十条），精确检验在这种时候给的 p 本来就大，
最小可达 p 也是两者里较大的那个——剔除那一道就把它们标成样本不够。
前因那一条判不了条件的（没映射过、情境没判过）不进任何一组。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from habitus.scene.concepts.model import ConceptRole, ConceptSet
from habitus.scene.relations import stats
from habitus.scene.relations.measures import ChainMeasure
from habitus.scene.relations.spans import Chain, Stratum
from habitus.scene.relations.timeline import LaneTimeline


class Template(str, Enum):
    FIELD = "field"
    TIMELINE = "timeline"
    CONCEPT = "concept"


class Field(str, Enum):
    WEEKEND = "weekend"
    CALENDAR = "calendar"
    PLACE = "place"
    WITH = "with"


@dataclass(frozen=True, order=True)
class Condition:
    """``template`` 之下：``FIELD`` 用 ``field`` + ``value``（周末不要 value）；``TIMELINE`` / ``CONCEPT`` 用 ``value`` = 概念身份。"""

    template: Template
    value: str = ""
    field: Field | None = None

    @property
    def text(self) -> str:
        if self.template is Template.FIELD:
            assert self.field is not None
            return f"field:{self.field.value}" + (f"={self.value}" if self.value else "")
        return f"{self.template.value}:{self.value}"


class ConditionError(ValueError):
    """提议落不成三种模板之一。"""


def parse(text: str) -> Condition:
    """``Condition.text`` 的反面（存盘的关系身份里只有文字）。"""

    template, _, rest = text.partition(":")
    if template == Template.FIELD.value:
        name, _, value = rest.partition("=")
        return Condition(Template.FIELD, value, Field(name))
    return Condition(Template(template), rest)


def compile_proposal(proposal: Mapping[str, object], concepts: ConceptSet, *, lane: str) -> Condition:
    """把一条提议编译成模板；落不成的抛 ``ConditionError``，说清是哪一条不对（进已拒清单）。

    提议的形状：``{"template": "field", "field": "weekend" | "calendar" | "place" | "with", "value": "..."}``、
    ``{"template": "timeline", "concept": "<类名或概念名>"}``、``{"template": "concept", "concept": "<情境概念名>"}``。
    """

    try:
        template = Template(str(proposal.get("template")))
    except ValueError as exc:
        raise ConditionError(f"unknown template: {proposal.get('template')!r}") from exc
    if template is Template.FIELD:
        try:
            field = Field(str(proposal.get("field")))
        except ValueError as exc:
            raise ConditionError(f"unknown field: {proposal.get('field')!r}") from exc
        value = str(proposal.get("value") or "").strip()
        if field is Field.WEEKEND:
            return Condition(Template.FIELD, "", field)
        if not value or "=" in value or "|" in value:
            raise ConditionError(f"field {field.value} needs a plain value")
        return Condition(Template.FIELD, value, field)
    name = str(proposal.get("concept") or "").strip()
    identity = _resolve(concepts, name)
    if identity is None:
        raise ConditionError(f"no concept named {name!r}")
    item = concepts[identity]
    if concepts.lane_of(identity) not in (lane, None):
        raise ConditionError(f"concept {name!r} belongs to another lane")
    if template is Template.TIMELINE and item.role is not ConceptRole.BEHAVIOR:
        raise ConditionError(f"the timeline template needs a behaviour concept, {name!r} is not one")
    if template is Template.CONCEPT and item.role is ConceptRole.BEHAVIOR:
        raise ConditionError(f"the concept template needs a situation concept, {name!r} is a behaviour")
    return Condition(template, identity)


def mechanical(
    line: LaneTimeline,
    concepts: ConceptSet,
    antecedent: str,
    chains: Sequence[Chain],
    *,
    window: int,
    notes: Mapping[date, str],
) -> tuple[Condition, ...]:
    """机械列举的条件：在这个前因的链上真有正反两面的（全是或全不是的条件没法比）。"""

    candidates: set[Condition] = {Condition(Template.FIELD, "", Field.WEEKEND)}
    for chain in chains:
        record = line.records.get(chain.anchor.uri)
        if notes.get(chain.day):
            candidates.add(Condition(Template.FIELD, notes[chain.day], Field.CALENDAR))
        if record is not None and record.place:
            candidates.add(Condition(Template.FIELD, record.place, Field.PLACE))
        if record is not None:
            candidates.update(Condition(Template.FIELD, name, Field.WITH) for name in record.subjects)
    for identity in concepts.behaviors():
        if identity != antecedent and concepts.lane_of(identity) == line.lane and line.marks.get(identity):
            candidates.add(Condition(Template.TIMELINE, identity))
    for identity in concepts.situations():
        if concepts.lane_of(identity) in (line.lane, None) and identity in line.situations:
            candidates.add(Condition(Template.CONCEPT, identity))
    found = []
    for condition in sorted(candidates):
        judge = evaluator(line, condition, window=window, notes=notes)
        flags = {judge(chain) for chain in chains} - {None}
        if flags == {True, False}:
            found.append(condition)
    return tuple(found)


def evaluator(
    line: LaneTimeline, condition: Condition, *, window: int, notes: Mapping[date, str]
) -> Callable[[Chain], bool | None]:
    """条件 → "这次 A 满足没有"（判不了是 None）。只看前因那一条与它之前，不看 B。"""

    if condition.template is Template.FIELD:
        field = condition.field

        def by_field(chain: Chain) -> bool | None:
            record = line.records.get(chain.anchor.uri)
            if field is Field.WEEKEND:
                return chain.day.weekday() >= 5
            if field is Field.CALENDAR:
                return notes.get(chain.day) == condition.value
            if record is None:
                return None
            if field is Field.PLACE:
                return None if record.place is None else record.place == condition.value
            return condition.value in record.subjects

        return by_field
    if condition.template is Template.TIMELINE:
        marks = line.marks.get(condition.value, [])

        def by_timeline(chain: Chain) -> bool:
            low = chain.anchor.slot - window
            return any(
                low <= mark.slot <= chain.anchor.slot and mark.started_at < chain.anchor.started_at for mark in marks
            )

        return by_timeline

    def by_concept(chain: Chain) -> bool | None:
        hits = line.hits.get(chain.anchor.uri)
        if hits is None:
            return None
        names = {name for name in hits.situations_checked}
        if condition.value not in names and not any(hit.identity == condition.value for hit in hits.situation_hits):
            return None
        return any(hit.identity == condition.value for hit in hits.situation_hits)

    return by_concept


@dataclass(frozen=True)
class ModeratedMeasure:
    """带条件的量法：同一条关系的层按条件分两组，比两组的"实验组 − 对照"之差（交互）。

    效应报的是**条件下**那一组的实验组率与对照率；区间、第 3 道看的是交互（条件下比条件外多出 / 少了多少）。
    """

    inside: tuple[Stratum, ...]
    outside: tuple[Stratum, ...]
    enough: bool

    @property
    def units(self) -> int:
        return len(self.inside) + len(self.outside)

    @property
    def blocks(self) -> int:
        return len({item.block for item in (*self.inside, *self.outside)})

    def effect(self) -> stats.Effect:
        return stats.effect(self.inside)

    def tails(self, *, exact_limit: int) -> tuple[float, float]:
        z = self._z()
        if z is None or not self.inside:
            return 1.0, 1.0
        upper = 0.5 * math.erfc(z / math.sqrt(2.0))
        exact_upper, exact_lower = stats.tails(self.inside, exact_limit=exact_limit)
        return max(upper, exact_upper), max(1.0 - upper, exact_lower)

    def minimum_p(self) -> float:
        """两组的实验组各走到全阳 / 全阴时最大的 |z|。样本不够（前因不到 40、某一组没有有信息的层）就不可能显著。"""

        if not self.enough:
            return 1.0
        zs = [_interaction_z(self.inside, self.outside, forced_in, forced_out) for forced_in in (0, 1) for forced_out in (0, 1)]
        if any(z is None for z in zs):
            return 1.0
        largest = max(abs(z) for z in zs if z is not None)
        return max(min(1.0, math.erfc(largest / math.sqrt(2.0))), stats.minimum_p(self.inside))

    def interval(
        self, *, level: float, rounds: int, seed: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if len(self.inside) < 2 or len(self.outside) < 2:
            return None, None
        blocks: dict[int, list[tuple[bool, float]]] = {}
        for flag, group in ((True, self.inside), (False, self.outside)):
            for item in group:
                blocks.setdefault(item.block, []).append((flag, item.treated - item.control_rate))
        keys = sorted(blocks)
        generator = random.Random(seed)
        values: list[float] = []
        for _ in range(rounds):
            drawn = [entry for _ in keys for entry in blocks[keys[generator.randrange(len(keys))]]]
            held = [value for flag, value in drawn if flag]
            rest = [value for flag, value in drawn if not flag]
            if held and rest:
                values.append(sum(held) / len(held) - sum(rest) / len(rest))
        if len(values) < rounds * 0.9:
            return None, None
        values.sort()
        tail = (1.0 - level) / 2.0
        return (_quantile(values, tail), _quantile(values, 1.0 - tail)), None

    def heterogeneity(self) -> float | None:
        return None

    def leave_one_out(self, *, upward: bool) -> bool:
        blocks = sorted({item.block for item in (*self.inside, *self.outside)})
        if len(blocks) < 2:
            return False
        for block in blocks:
            rest = ModeratedMeasure(
                tuple(item for item in self.inside if item.block != block),
                tuple(item for item in self.outside if item.block != block),
                self.enough,
            )
            difference = rest._interaction()
            if difference is None or difference == 0.0 or (difference > 0.0) != upward:
                return False
        return True

    def subset(self, *, since: date | None = None, last_blocks: int | None = None) -> ModeratedMeasure:
        both = ChainMeasure((*self.inside, *self.outside)).subset(since=since, last_blocks=last_blocks)
        kept = set(map(id, both.strata))
        return ModeratedMeasure(
            tuple(item for item in self.inside if id(item) in kept),
            tuple(item for item in self.outside if id(item) in kept),
            self.enough,
        )

    def same_event_share(self) -> float | None:
        return ChainMeasure(self.inside).same_event_share()

    def _interaction(self) -> float | None:
        if not self.inside or not self.outside:
            return None
        return _mean_lift(self.inside) - _mean_lift(self.outside)

    def _z(self) -> float | None:
        """两组"实验组 − 对照"平均差值之差的 z（差值尺度，与区间、第 3 道同一个尺度；裁定 27 第 5 条）。"""

        if not self.enough:
            return None
        return _interaction_z(self.inside, self.outside, None, None)


def _excess(items: Sequence[Stratum], forced: int | None) -> tuple[float, float] | None:
    """一组层的平均"实验组 − 对照"与它在零假设下的方差（实验组的伯努利方差加上对照率的估计方差）。
    ``forced`` 给了就把实验组全当成它（算最小可达 p）。"""

    if not items:
        return None
    count = len(items)
    total = sum((item.treated if forced is None else forced) - item.control_rate for item in items)
    variance = sum(
        item.null_probability * (1.0 - item.null_probability) * (1.0 + 1.0 / item.controls) for item in items
    ) / (count * count)
    if variance <= 0.0:
        return None
    return total / count, variance


def _interaction_z(
    inside: Sequence[Stratum], outside: Sequence[Stratum], forced_in: int | None, forced_out: int | None
) -> float | None:
    """条件下与条件外的平均差值之差的 z。检验与"影响够大"都在差值尺度上：用优势比检验、差值判门槛，后果的基线一变
    （周末、地点常常改变基线），优势比一样的关系也会被判成"调节"（第四轮评审算法 7：基线 8% 对 35%、OR 都是 6，44–88% 判成调节）。"""

    held, rest = _excess(inside, forced_in), _excess(outside, forced_out)
    if held is None or rest is None:
        return None
    return (held[0] - rest[0]) / math.sqrt(held[1] + rest[1])


def split(measure: ChainMeasure, judge: Callable[[Chain], bool | None], *, minimum: int) -> ModeratedMeasure:
    inside: list[Stratum] = []
    outside: list[Stratum] = []
    for item in measure.strata:
        flag = judge(item.chain)
        if flag is True:
            inside.append(item)
        elif flag is False:
            outside.append(item)
    return ModeratedMeasure(tuple(inside), tuple(outside), enough=measure.units >= minimum)


def _resolve(concepts: ConceptSet, name: str) -> str | None:
    if name in concepts:
        return concepts[name].identity
    matches = [identity for identity in concepts if concepts.label_of(identity) == name]
    return matches[0] if len(matches) == 1 else None


def _mean_lift(items: Sequence[Stratum]) -> float:
    return sum(item.treated - item.control_rate for item in items) / len(items)


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


__all__ = [
    "Condition",
    "ConditionError",
    "Field",
    "ModeratedMeasure",
    "Template",
    "compile_proposal",
    "evaluator",
    "mechanical",
    "parse",
    "split",
]
