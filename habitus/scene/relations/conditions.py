"""调节条件（含组合前因）：必须编译成三种可算的模板之一，否则丢弃（语义树新方案 ``13`` ②）。

1. **结构字段**（``FIELD``）——机械判，看前因那一条：地点、同在的人；周几、日型这类日历条件是日历情境概念，走第 3 种（裁定 59）；
2. **时间线**（``TIMELINE``）——机械判：A 之前同一条链的窗口里出现过某一类（"讨论方案之后的修改代码"；**组合前因**
   "A 与 C 一起"就是这一种，不另起一套）；
3. **情境概念**（``CONCEPT``）——前因那一条命中了某个情境概念（"约了人"这类交映射器打标签；日历情境按那一天的日历机械判，
   答不出的那天不判）。

条件的来源有两处：机械列举的（每个够样本的前因：出现过的地点 / 同在的人、出现在它之前的每一类、
同 lane 的每个情境概念），以及大模型提的（第 7 步：读 A 那一条与它之前的概要、不看 B，提出"可能被什么左右"，
经 ``compile_proposal`` 编译，落不成模板的进已拒清单，不每晚换个说法反复试）。第 N 晚提的只在 N 之后的数据上检验。

**几个条件合起来**（裁定 61 第 6 条、62 第 1 条）：关系键里的条件是一个集合——各条件的文本排序后用 ``CONDITION_JOIN``
拼成一个键（``join_conditions`` / ``parse_conditions``；``+`` 是概念身份预留的分隔符，地点 / 同在的人取值里含它的不当条件）。
合起来判：任一不成立就不成立，都成立才成立，否则判不了。分组按文献（Webb 2007《Discovering Significant Patterns》附录 A 的
productive-rule 检验；裁定 63 第 1 条）：集合里每一个条件各比一次，"全部成立"对"其余成立、这一个不成立"，问的是"这一个条件在其余
条件成立时还加得上东西吗"；整条关系按最难过的那组读（每一组都过才算过）。不把几组对照混成一池——混了之后一个已成立的父条件
自己的抬升会把池子的基线拉低，什么都不解释的第二个条件也常被判成立（算法评审模拟 N=160、父抬升 0.4 时型一错误 0.55）。
只有一个条件时就是"成立 / 不成立"两组，与原来的分法相同。

**检验**：开不开由这条关系自己的最小可达 p 定（``ready_to_test``，裁定 60 第 9 条取代固定的前因 ≥ 40）：两边各自全中也能
显著才开。把这条关系的层按条件分成两组，比两组的平均差值（实验组 − 对照率；差值尺度，裁定 27 第 5 条）：交互 z = 两组之差 /
√(两组方差之和)。零假设是"两组抬升相等"，不是"都没抬升"，所以方差按两组合起来的公共抬升算（Breslow & Day 1980 §5.3：同质性
检验的期望与方差在公共效应下算；按"没有效应"算会在关系真实存在时放大 z、多认出不存在的条件——10.17 前实测 I 类错误约 0.14，
改后约 0.07，见 27 号调研）。两边都要攒到全中能显著（对称）：交互的功效由少的那一边决定（Breslow & Day 1987 §7.10）。

"在条件下更强"说的是两件事：条件下有效应、而且和条件外不一样——按**交并检验**取两者 p 的较大者（有效水平不变）：
条件下那一组用层内精确检验（泊松二项），交互用上面的 z。只用交互 z 不够：条件下只有两三层时，z 的正态近似失灵，
一层阳性就能把 z 推得很大（真实数据：条件下 2 层、阳性 1 层的"调节"出了几十条），精确检验在这种时候给的 p 本来就大，
最小可达 p 也是两者里较大的那个——剔除那一道就把它们标成样本不够。
前因那一条判不了条件的（没映射过、情境没判过）不进任何一组。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
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


# TODO(FORESIGHT-EVENT-FIRST-001)：天气等外部事实接进外部条件包（habitus/conditions），在这里注册成新字段（要有历史），
# 规格见 foresight/triggers/assemble.py。
class Field(str, Enum):
    PLACE = "place"
    WITH = "with"


@dataclass(frozen=True, order=True)
class Condition:
    """``template`` 之下：``FIELD`` 用 ``field`` + ``value``；``TIMELINE`` / ``CONCEPT`` 用 ``value`` = 概念身份。"""

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


#: 条件集合的键里各条件之间的分隔符（``concepts/model.py`` 的 ``IDENTITY_SEPARATORS`` 预留，概念身份里不会出现）。
CONDITION_JOIN = "+"


def join_conditions(conditions: Iterable[Condition]) -> str:
    """条件集合 → 关系键里的条件文本（排序、去重；空集合是空串 = 不带条件）。"""

    return CONDITION_JOIN.join(sorted({item.text for item in conditions}))


def parse_conditions(text: str) -> tuple[Condition, ...]:
    """关系键里的条件文本 → 各条件（按文本排序）；空串是空元组。"""

    if not text:
        return ()
    return tuple(parse(part) for part in text.split(CONDITION_JOIN))


def parse(text: str) -> Condition:
    """``Condition.text`` 的反面（存盘的关系身份里只有文字）。"""

    template, _, rest = text.partition(":")
    if template == Template.FIELD.value:
        name, _, value = rest.partition("=")
        return Condition(Template.FIELD, value, Field(name))
    return Condition(Template(template), rest)


def compile_proposal(proposal: Mapping[str, object], concepts: ConceptSet, *, lane: str) -> Condition:
    """把一条提议编译成模板；落不成的抛 ``ConditionError``，说清是哪一条不对（进已拒清单）。

    提议的形状：``{"template": "field", "field": "place" | "with", "value": "..."}``、
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
        if not value or "=" in value or "|" in value or CONDITION_JOIN in value:
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
) -> tuple[Condition, ...]:
    """机械列举的条件：在这个前因的链上真有正反两面的（全是或全不是的条件没法比）。"""

    candidates: set[Condition] = set()
    for chain in chains:
        record = line.records.get(chain.anchor.uri)
        if record is not None and record.place and CONDITION_JOIN not in record.place:
            candidates.add(Condition(Template.FIELD, record.place, Field.PLACE))
        if record is not None:
            candidates.update(
                Condition(Template.FIELD, name, Field.WITH) for name in record.subjects if CONDITION_JOIN not in name
            )
    for identity in concepts.behaviors():
        if identity != antecedent and concepts.lane_of(identity) == line.lane and line.marks.get(identity):
            candidates.add(Condition(Template.TIMELINE, identity))
    for identity in concepts.situations():
        if concepts.lane_of(identity) in (line.lane, None) and identity in line.situations:
            candidates.add(Condition(Template.CONCEPT, identity))
    found = []
    for condition in sorted(candidates):
        judge = evaluator(line, condition, window=window)
        flags = {judge(chain) for chain in chains} - {None}
        if flags == {True, False}:
            found.append(condition)
    return tuple(found)


def evaluator(line: LaneTimeline, condition: Condition, *, window: int) -> Callable[[Chain], bool | None]:
    """条件 → "这次 A 满足没有"（判不了是 None）。只看前因那一条与它之前，不看 B。"""

    if condition.template is Template.FIELD:
        field = condition.field

        def by_field(chain: Chain) -> bool | None:
            record = line.records.get(chain.anchor.uri)
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

    calendar = line.calendar_days(condition.value)

    def by_concept(chain: Chain) -> bool | None:
        if calendar is not None:
            return calendar.get(chain.day)  # 日历情境：那一天的日历说了算，答不出就是 None
        hits = line.hits.get(chain.anchor.uri)
        if hits is None:
            return None
        names = {name for name in hits.situations_checked}
        if condition.value not in names and not any(hit.identity == condition.value for hit in hits.situation_hits):
            return None
        return any(hit.identity == condition.value for hit in hits.situation_hits)

    return by_concept


def evaluator_all(
    line: LaneTimeline, conditions: Sequence[Condition], *, window: int
) -> Callable[[Chain], bool | None]:
    """几个条件合起来判：任一不成立 → 不成立；都成立 → 成立；其余（有判不了的）→ 判不了。空集合恒成立。"""

    judges = [evaluator(line, condition, window=window) for condition in conditions]

    def judge(chain: Chain) -> bool | None:
        flags = [item(chain) for item in judges]
        if any(flag is False for flag in flags):
            return False
        if any(flag is None for flag in flags):
            return None
        return True

    return judge


@dataclass(frozen=True)
class ModeratedMeasure:
    """带条件的量法：条件集合里**每一个条件各比一次**——"全成立"那一组对"其余成立、这一个不成立"那一组（Webb 2007
    《Discovering Significant Patterns》附录 A 的 productive-rule 检验；裁定 63 第 1 条）。整条关系按最难过的那组读：
    尾概率取各组最大（每一组都显著才显著）、最小可达 p 取各组最大（每一组都够得着才开检验）、区间取各组的外包（近端取
    最不利的——"影响够大"要每个条件都够；远端取最有利的——判"没了"要每个条件都没了）。只有一个条件时就是"成立 / 不成立"
    两组，与原来的分法相同。

    效应报的是**条件下**那一组的实验组率与对照率；区间、第 3 道看的是交互（条件下比条件外多出 / 少了多少）。
    """

    inside: tuple[Stratum, ...]
    outsides: tuple[tuple[Stratum, ...], ...]
    enough: bool

    @property
    def units(self) -> int:
        return len(self.inside) + sum(len(group) for group in self.outsides)

    @property
    def blocks(self) -> int:
        return len({item.block for item in self._all()})

    def effect(self) -> stats.Effect:
        return stats.effect(self.inside)

    def tails(self, *, exact_limit: int) -> tuple[float, float]:
        zs = self._zs()
        if zs is None or not self.inside:
            return 1.0, 1.0
        uppers = [0.5 * math.erfc(z / math.sqrt(2.0)) for z in zs]
        exact_upper, exact_lower = stats.tails(self.inside, exact_limit=exact_limit)
        return max(max(uppers), exact_upper), max(max(1.0 - upper for upper in uppers), exact_lower)

    def minimum_p(self) -> float:
        """每一组比较各算"两边的实验组各走到全阳 / 全阴时最大的 |z|"，取各组里最小的那个（最难过的那组），再与各边的精确
        最小可达 p 取大（两边对称，L10.17）。样本不够（某一边不到 ``MIN_PER_SIDE`` 次、某一组没有有信息的层）就不可能显著。"""

        if not self.enough:
            return 1.0
        largest: list[float] = []
        for outside in self.outsides:
            zs = [
                _interaction_z(self.inside, outside, forced_in, forced_out)
                for forced_in in (0, 1)
                for forced_out in (0, 1)
            ]
            if any(z is None for z in zs):
                return 1.0
            largest.append(max(abs(z) for z in zs if z is not None))
        normal = min(1.0, math.erfc(min(largest) / math.sqrt(2.0)))
        return max(normal, stats.minimum_p(self.inside), *(stats.minimum_p(outside) for outside in self.outsides))

    def interval(
        self, *, level: float, rounds: int, seed: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        found: list[tuple[float, float]] = []
        for index, outside in enumerate(self.outsides):
            spread = _bootstrap(self.inside, outside, level=level, rounds=rounds, seed=seed + index)
            if spread is None:
                return None, None
            found.append(spread)
        return (min(low for low, _ in found), max(high for _, high in found)), None

    def heterogeneity(self) -> float | None:
        return None

    def leave_one_out(self, *, upward: bool) -> bool:
        blocks = sorted({item.block for item in self._all()})
        if len(blocks) < 2:
            return False
        for block in blocks:
            inside = tuple(item for item in self.inside if item.block != block)
            for outside in self.outsides:
                rest = tuple(item for item in outside if item.block != block)
                difference = _interaction(inside, rest)
                if difference is None or difference == 0.0 or (difference > 0.0) != upward:
                    return False
        return True

    def subset(self, *, since: date | None = None, last_blocks: int | None = None) -> ModeratedMeasure:
        both = ChainMeasure(self._all()).subset(since=since, last_blocks=last_blocks)
        kept = set(map(id, both.strata))
        return ModeratedMeasure(
            tuple(item for item in self.inside if id(item) in kept),
            tuple(tuple(item for item in outside if id(item) in kept) for outside in self.outsides),
            self.enough,
        )

    def same_event_share(self) -> float | None:
        return ChainMeasure(self.inside).same_event_share()

    def _all(self) -> tuple[Stratum, ...]:
        return (*self.inside, *(item for outside in self.outsides for item in outside))

    def _zs(self) -> list[float] | None:
        """每一组比较的交互 z（差值尺度，与区间、第 3 道同一个尺度；裁定 27 第 5 条）；样本不够或哪一组量不出就没有。"""

        if not self.enough:
            return None
        zs = [_interaction_z(self.inside, outside, None, None) for outside in self.outsides]
        return None if any(z is None for z in zs) else [z for z in zs if z is not None]


def _interaction(inside: Sequence[Stratum], outside: Sequence[Stratum]) -> float | None:
    if not inside or not outside:
        return None
    return _mean_lift(inside) - _mean_lift(outside)


def _bootstrap(
    inside: Sequence[Stratum], outside: Sequence[Stratum], *, level: float, rounds: int, seed: int
) -> tuple[float, float] | None:
    """一组比较的交互区间：按块重抽样，两边各要至少 2 层。"""

    if len(inside) < 2 or len(outside) < 2:
        return None
    blocks: dict[int, list[tuple[bool, float]]] = {}
    for flag, group in ((True, inside), (False, outside)):
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
        return None
    values.sort()
    tail = (1.0 - level) / 2.0
    return _quantile(values, tail), _quantile(values, 1.0 - tail)


def _lifts(items: Sequence[Stratum], forced: int | None) -> list[float]:
    """每层的"实验组 − 对照率"。``forced`` 给了就把实验组全当成它（算最小可达 p）。"""

    return [(item.treated if forced is None else forced) - item.control_rate for item in items]


def _variance(items: Sequence[Stratum], common: float) -> float:
    """一组层的平均差值在"两组抬升相等、都等于 ``common``"下的方差：实验组的伯努利方差（来的概率 = 对照率 + 公共抬升）
    加上对照率的估计方差。"""

    def bernoulli(rate: float) -> float:
        return rate * (1.0 - rate)

    total = sum(
        bernoulli(min(1.0, max(0.0, item.control_rate + common))) + bernoulli(item.control_rate) / item.controls
        for item in items
    )
    return total / (len(items) * len(items))


def _interaction_z(
    inside: Sequence[Stratum], outside: Sequence[Stratum], forced_in: int | None, forced_out: int | None
) -> float | None:
    """条件下与条件外的平均差值之差的 z。检验与"影响够大"都在差值尺度上：用优势比检验、差值判门槛，后果的基线一变
    （周末、地点常常改变基线），优势比一样的关系也会被判成"调节"（第四轮评审算法 7：基线 8% 对 35%、OR 都是 6，44–88% 判成调节）。
    方差在公共抬升下算（见模块说明）。"""

    held, rest = _lifts(inside, forced_in), _lifts(outside, forced_out)
    if not held or not rest:
        return None
    common = (sum(held) + sum(rest)) / (len(held) + len(rest))
    variance = _variance(inside, common) + _variance(outside, common)
    if variance <= 0.0:
        return None
    return (sum(held) / len(held) - sum(rest) / len(rest)) / math.sqrt(variance)


#: 调节检验每边至少几次：区间按块重抽样要两边都有至少 2 层（``ModeratedMeasure.interval``）。
MIN_PER_SIDE = 2


def split(measure: ChainMeasure, judges: Sequence[Callable[[Chain], bool | None]]) -> ModeratedMeasure:
    """层按条件集合分组：全部成立的进条件下那一组；**恰好少一个**的按少的是哪一个各进一组（"其余成立、它不成立"，每个条件
    一组对照）；别的（有判不了的、少了两个以上）不进任何一组。只有一个条件时就是"成立 / 不成立"两组。
    条件下与每一组对照都各自够 ``MIN_PER_SIDE`` 次才算样本够（哪一组对照只有一次，那个条件的交互就量不出来）。"""

    if not judges:
        raise ValueError("a moderated measure needs at least one condition")
    inside: list[Stratum] = []
    outsides: list[list[Stratum]] = [[] for _ in judges]
    for item in measure.strata:
        flags = [judge(item.chain) for judge in judges]
        if any(flag is None for flag in flags):
            continue
        failed = [index for index, flag in enumerate(flags) if flag is False]
        if not failed:
            inside.append(item)
        elif len(failed) == 1:
            outsides[failed[0]].append(item)
    enough = len(inside) >= MIN_PER_SIDE and all(len(group) >= MIN_PER_SIDE for group in outsides)
    return ModeratedMeasure(tuple(inside), tuple(tuple(group) for group in outsides), enough=enough)


def ready_to_test(measure: ModeratedMeasure, *, alpha: float) -> bool:
    """这条带条件的关系值不值得开检验（裁定 60 第 9 条：不用固定的 40 次）：两边全分开时最小可达 p 在 ``alpha`` 以内，且每边
    至少 ``MIN_PER_SIDE`` 次。调节检验是交互的 z 与条件下精确尾概率取大（``tails``），所以用它自己的 ``minimum_p``，不借单样本
    二项的次数公式。两边要的次数都随对照率变（L10.17 起条件外与条件下对称）：α = 0.05 时对照率 0.1 要 2 次、0.3 要 4 次、0.5 要 6 次。"""

    return measure.enough and measure.minimum_p() <= alpha


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
    "CONDITION_JOIN",
    "Condition",
    "ConditionError",
    "Field",
    "ModeratedMeasure",
    "Template",
    "compile_proposal",
    "evaluator",
    "evaluator_all",
    "join_conditions",
    "mechanical",
    "parse",
    "parse_conditions",
    "MIN_PER_SIDE",
    "ready_to_test",
    "split",
]
