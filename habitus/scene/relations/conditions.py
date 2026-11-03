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
开测门交给 Tarone 的最小可达 p 是检验真实可达的下界（裁定 94 第 5 条，``ModeratedMeasure.minimum_p``）：条件下按实测层算（检验里
它是层内精确检验，阳性边际固定），条件外按假设全中 / 全不中算（只用对照数，不看实测——当天从不出现 B 的层照样有信息）；两边对照都
没有 B 时量不出的那一端跳过，不判成永远不显著；条件下（含对照）完全没有 B 的压低型不开测（检验在那里没有信息）。每边至少
``MIN_PER_SIDE`` = 3 次。前向验证用的单侧版（``minimum_toward``）同一口径。

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

from habitus.scene.concepts.model import ConceptRole, ConceptSet, members_retired
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
        if not value or not value.isprintable() or "=" in value or "|" in value or CONDITION_JOIN in value:
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
    """几个条件合起来判（``all_hold``）。"""

    return all_hold([evaluator(line, condition, window=window) for condition in conditions])


def all_hold(judges: Sequence[Callable[[Chain], bool | None]]) -> Callable[[Chain], bool | None]:
    """几个判法合起来判：任一不成立 → 不成立；都成立 → 成立；其余（有判不了的）→ 判不了。空集合恒成立。
    语义树检验与预测层白天判条件都用这一个。"""

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
        """开测门交给 Tarone 的最小可达双侧 p（裁定 60 第 9 条、L10.17、裁定 94 第 5 条）：检验本身朝两个方向各走到最干净
        （``_cleanest``：条件下按实测层、条件外按假设），取小的那个方向乘 2；再要每一组对照自己全中或全不中也能显著
        （``_side_minimum``：条件外的次数门，两边对称——交互的功效由少的那一边决定）。某一边不到 ``MIN_PER_SIDE`` 次就不可能显著。"""

        if not self._countable():
            return 1.0
        reachable = min(1.0, 2.0 * min(self._cleanest(upward=True), self._cleanest(upward=False)))
        return max(reachable, *(_side_minimum(side) for side in self.outsides))

    def minimum_toward(self, *, upward: bool) -> float:
        """只朝一侧检时的最小可达单侧 p（前向验证用它判"攒够了还到不了 α 就不判"；维持检验只存进读数）：与开测门同一口径
        （裁定 94 第 5 条），朝这一侧分得最干净（``_cleanest``），再加条件外的次数门（``_side_one_sided``）与每边至少
        ``MIN_PER_SIDE`` 次。次数按这一份（子集）自己的层数数：前向验证只看发现之后的新数据，``subset`` 留着全量的 ``enough``。"""

        if not self._countable():
            return 1.0
        return max(self._cleanest(upward=upward), *(_side_one_sided(side) for side in self.outsides))

    def _countable(self) -> bool:
        """条件下与每一组对照都至少 ``MIN_PER_SIDE`` 次（``split`` 算 ``enough`` 用的同一条）。"""

        return self.enough and _sides_enough(self.inside, self.outsides)

    def _cleanest(self, *, upward: bool) -> float:
        """朝一侧分得最干净时检验本身的单侧 p，是 ``tails`` 在同一个结果上的两半取大，不混两个方向：
        交互 z——条件下那一组全中（朝上）或全不中（朝下），每一组对照取让交互离得最远的那一端（全不中或全中；方差为 0、量不出的
        那一端跳过，不当成"永远不显著"），取各组里最难过的；条件下的精确尾概率——那几层朝这一侧全中的概率。

        **条件下按实测层**（``stats.minimum_one_sided``）：它是检验里真做的层内条件检验，固定每层阳性数（实验组 + 对照），
        自己没来、对照也一个没有的层在那个检验里怎么都没信息——所以条件下（含对照）完全没有 B 的压低型朝下到不了、不开测
        （裁定 94 第 5 条）。**条件外按假设**：z 里每组对照取假设的全中 / 全不中，次数要求在 ``_side_minimum`` /
        ``_side_one_sided``（只用对照数）；条件外不做精确检验。
        （条件下也按假设全中算时，后果几乎不出现的带条件关系都会进族、p 恒为 1：v0.11 会话 lane 05-13 那晚族从 1007 涨到 5655，
        E5 那条反被 Tarone 挤出去。）哪一组两端都量不出就到不了。"""

        sign = 1.0 if upward else -1.0
        farthest: list[float] = []
        for outside in self.outsides:
            zs = [_interaction_z(self.inside, outside, int(upward), forced_out) for forced_out in (0, 1)]
            found = [sign * z for z in zs if z is not None]
            if not found:
                return 1.0
            farthest.append(max(found))
        normal = 0.5 * math.erfc(min(farthest) / math.sqrt(2.0))
        return min(1.0, max(normal, stats.minimum_one_sided(self.inside, upward=upward)))

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
    return stats.quantile(values, tail), stats.quantile(values, 1.0 - tail)


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


#: 调节检验每边至少几次（裁定 94 第 5 条）。少的时候交互 z 的正态近似偏乐观，按同侧比（报出的单侧 p 对这个方向上真实能得到的概率）：
#: 两边对照都没有 B、条件下 n 层全中、条件外 n 层全不中，z = √(2n)；零假设（两组同一个来的概率 π）下真能得到这个结果的概率最大是
#: π^n(1 − π)^n 在 π = ½ 时的 1/4^n。n = 2：z = 2、单侧报 0.0228，真实最大 1/16 = 0.0625，偏乐观约 2.7 倍，而且真实值已过 0.05——
#: 门开了也判不出。n = 3：z ≈ 2.45、单侧报 0.0072，真实最大 1/64 ≈ 0.0156，仍偏乐观约 2.2 倍，但真实值在 0.05 以内；经族阈值
#: （Tarone 修剪后的更小的 α）之后这点乐观影响小。区间按块重抽样要两边都有至少 2 层（``ModeratedMeasure.interval``），3 也满足。
MIN_PER_SIDE = 3


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
    return ModeratedMeasure(
        tuple(inside), tuple(tuple(group) for group in outsides), enough=_sides_enough(inside, outsides)
    )


def _sides_enough(inside: Sequence[Stratum], outsides: Sequence[Sequence[Stratum]]) -> bool:
    return len(inside) >= MIN_PER_SIDE and all(len(group) >= MIN_PER_SIDE for group in outsides)


def ready_to_test(measure: ModeratedMeasure, *, alpha: float) -> bool:
    """这条带条件的关系值不值得开检验（裁定 60 第 9 条：不用固定的 40 次）：两边分得最干净时最小可达 p 在 ``alpha`` 以内，且每边
    至少 ``MIN_PER_SIDE`` 次。调节检验是交互的 z 与条件下精确尾概率取大（``tails``），所以用它自己的 ``minimum_p``，不借单样本
    二项的次数公式。两边要的次数都随对照率变（L10.17 起条件外与条件下对称）：α = 0.05、两边次数相同、每层 20 个对照时，两边对照率
    都是 0 时交互 z = √(2n)、2 次就到 0.0455，但那是正态近似偏乐观（见 ``MIN_PER_SIDE``），由每边至少 3 次挡住；都是 0.3 要各 4 次，
    都是 0.5 要各 6 次；条件外当天从不出现 B、条件下 0.1 要各 3 次。
    条件下按实测层算：自己没来、对照里也没有 B 的层不算数（见 ``ModeratedMeasure._cleanest``），所以条件下（含对照）完全没有 B 的
    压低型永远不开（裁定 94 第 5 条：检验在那里没有信息）。"""

    return measure.enough and measure.minimum_p() <= alpha


def _resolve(concepts: ConceptSet, name: str) -> str | None:
    if name in concepts:
        return concepts[name].identity
    # 名字只在在用的概念里唯一（停用的可以与在用的同名），按名字找只找在用的
    matches = [
        identity
        for identity in concepts
        if concepts.label_of(identity) == name and not members_retired(concepts, identity)
    ]
    return matches[0] if len(matches) == 1 else None


def _side_minimum(items: Sequence[Stratum]) -> float:
    """条件外的次数门（双侧，开测门用）：一组对照自己全中或全不中时最小的双侧精确 p。L10.17 的"条件外也要攒到全中能显著"。"""

    return min(1.0, 2.0 * _side_one_sided(items))


def _side_one_sided(items: Sequence[Stratum]) -> float:
    """条件外的次数门（单侧，前向验证用）：一组对照自己全中或全不中时较小的那个概率。按假设的结果算、只用对照数
    （``stats.assumed_extreme``），不看实测——条件外当天从不出现 B 的层假设全中时是有信息的。"""

    return min(stats.assumed_extreme(items, upward=True), stats.assumed_extreme(items, upward=False))


def _mean_lift(items: Sequence[Stratum]) -> float:
    return sum(item.treated - item.control_rate for item in items) / len(items)


__all__ = [
    "CONDITION_JOIN",
    "Condition",
    "ConditionError",
    "Field",
    "ModeratedMeasure",
    "Template",
    "compile_proposal",
    "evaluator",
    "all_hold",
    "evaluator_all",
    "join_conditions",
    "mechanical",
    "parse",
    "parse_conditions",
    "MIN_PER_SIDE",
    "ready_to_test",
    "split",
]
