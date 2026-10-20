"""一晚的关系检验：每条 lane 全部两两都量，数据说了算（语义树新方案 ``13`` ①③）。

```
每条 lane：
  前因 A × 后果 B × 四段跨度  →  每个都按层配好对照（spans；同一条链配两道，两道都过才算）
  → 最小可达 p 过不了线的：样本不够（不进检验族、不给区间）
  → 其余做加权 BH（FDR 10%）            第 1 道
  → 过了的：按块看稳不稳                  第 2 道
           区间下限够不够大               第 3 道
```

两条 lane 各成一个检验族（裁定 21：两条 lane 相互独立、不干扰）。不比的对（机械排除）：自己对自己（那是复现节律，
归 ``scene.recurrence``）；读同一批记录的（汇总与成员、基础与它的汇总、细分与它的源类）。

本模块只算"今晚的数据说什么"。候选要在之后的新数据上复现才成立（前向验证）、成立后按段看新数据（维持检验，从上一次判过起不重叠地攒满一段）——
这两步要跨夜的状态，由状态折叠（第 5 步）调 ``evaluate`` 在对应的子集上做。先验权重由第 7 步给，这里只收。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum

from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.calendar_situations import calendar_key
from habitus.scene.concepts.model import ConceptKind, ConceptSet, members_retired
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.relations import longspan, stats
from habitus.scene.relations.calendar import calendar_measure
from habitus.scene.relations.conditions import (
    MIN_PER_SIDE,
    Condition,
    ModeratedMeasure,
    Template,
    evaluator,
    join_conditions,
    mechanical,
    parse_conditions,
    ready_to_test,
    split,
)
from habitus.scene.relations.config import RelationConfig
from habitus.scene.relations.longspan import Episode
from habitus.scene.relations.measures import ChainMeasure, Measure, PairedMeasure
from habitus.scene.relations.spans import (
    MARKER_SEGMENTS,
    SEGMENTS,
    Chain,
    Controls,
    Pairing,
    Segment,
    Stratum,
    chains_of,
    merge_for,
)
from habitus.scene.relations.timeline import LaneTimeline, build_timelines
from habitus.series import EventSeries
from habitus.temporal.absence import Visit
from habitus.temporal.intervals import common_interval_days


@dataclass(frozen=True, order=True)
class RelationKey:
    """关系身份 = （lane, 前因, 后果, 跨度段, 条件）。条件为空 = 不带条件；带条件的是"在这些条件下 A → B 比少一个条件时
    更强 / 更弱"（调节；组合前因"A 与 C 一起"也是一种条件），条件是一个集合，写成 ``conditions.join_conditions`` 的键
    （几个 ``Condition.text`` 排序后用 ``+`` 拼起来；裁定 61 第 6 条）。"""

    lane: str
    antecedent: str
    consequent: str
    segment: Segment
    condition: str = ""

    @property
    def conditions(self) -> tuple[Condition, ...]:
        """这条关系挂的条件（没有是空元组）。"""

        return parse_conditions(self.condition)

    def parents(self) -> tuple[RelationKey, ...]:
        """少任意一个条件的那几条键（组合长在它们上面；单个条件的父是不带条件的那条；不带条件的没有父）。"""

        items = self.conditions
        return tuple(
            replace(self, condition=join_conditions(items[:index] + items[index + 1 :])) for index in range(len(items))
        )

    @property
    def text(self) -> str:
        base = f"{self.lane}|{self.antecedent}|{self.consequent}|{self.segment.value}"
        return f"{base}|{self.condition}" if self.condition else base


class Verdict(str, Enum):
    #: 最小可达 p 也过不了线：不进检验族、不给区间。
    SPARSE = "sparse"
    #: 进了检验族，没过第 1 道。
    TESTED = "tested"
    #: 过了第 1 道（今晚的数据上显著）；前向验证由状态折叠接着做。
    SIGNIFICANT = "significant"


@dataclass(frozen=True)
class Blocks:
    """一个前因的块：块长（天）= max(1 天, 合并扎堆之后相邻两次间隔的中位数)，从它第一次出现那天起算。"""

    days: float
    origin: date

    def of(self, chain: Chain) -> int:
        return math.floor((chain.day - self.origin).days / self.days)


@dataclass(frozen=True)
class RelationTest:
    key: RelationKey
    verdict: Verdict
    antecedents: int
    blocks: int
    block_days: float
    effect: stats.Effect
    p_value: float
    minimum_p: float
    upward: bool
    weight: float = 1.0
    #: 样本不够时：照现在的层结构，大约还要几次全朝一个方向的 A 才可能过线（粗估，给人看）。
    more_needed: int | None = None
    #: 第 2 道：各块效应异质性的 p、去掉任何一块方向都不变，合起来就是"时间上稳"。
    heterogeneity_p: float | None = None
    leave_one_out: bool | None = None
    stable: bool | None = None
    #: 第 3 道：差值区间、相对差区间，区间离 0 最近的一端够不够大。
    interval: tuple[float, float] | None = None
    relative_interval: tuple[float, float] | None = None
    large_enough: bool | None = None
    #: 从后果看：B 里约多少是 A 带出来的（超额比例，不另算一份证据）。
    excess_share: float | None = None
    #: 支撑例子里 A、B 目标相同的比例（只用结构字段；算法不看，高的关系对融合口径敏感）。
    same_event_share: float | None = None


@dataclass(frozen=True)
class LaneNight:
    lane: str
    cutoff: date
    tests: tuple[RelationTest, ...]
    family: int

    def significant(self) -> tuple[RelationTest, ...]:
        return tuple(test for test in self.tests if test.verdict is Verdict.SIGNIFICANT)


@dataclass(frozen=True)
class Subset:
    """在一个子集（前向验证的新数据、维持检验那一段的新数据）上的读数。"""

    antecedents: int
    blocks: int
    effect: stats.Effect
    p_value: float
    interval: tuple[float, float] | None
    relative_interval: tuple[float, float] | None
    #: 这几层就算全部最极端（全阳或全阴）能到的单侧 p（取两侧小的那侧，与同一条链入族同一口径）：大于 α 时判了必输。
    minimum_p: float


class LaneTests:
    """一条 lane 一晚的检验：时间线、对照、每个前因的链与块都只算一次。"""

    def __init__(
        self,
        line: LaneTimeline,
        concepts: ConceptSet,
        config: RelationConfig,
        cutoff: date,
        *,
        proposals: Mapping[str, tuple[tuple[Condition, date], ...]] | None = None,
        established: Iterable[RelationKey] = (),
    ) -> None:
        """``proposals``：前因 → 大模型提过的条件
        与提出的那一晚（第 7 步；只在那晚之后的数据上检验）。``established``：接的那一晚已成立的关系——第二个条件起只在
        已成立的带条件关系上加一个（裁定 62 第 1 条）。"""

        self.line = line
        self.concepts = concepts
        self.config = config
        self.cutoff = cutoff
        self.proposals = dict(proposals or {})
        self.established = frozenset(established)
        self.controls = Controls(line, config.transition_window_slots)
        # 上下界检验用的两份：后果"判不了"当 0 / 当 1（裁定 27 第 6 条）
        self._bounds = {
            0: Controls(line, config.transition_window_slots, blind_as=0),
            1: Controls(line, config.transition_window_slots, blind_as=1),
        }
        self._chains: dict[str, tuple[Chain, ...]] = {}
        self._blocks: dict[str, Blocks | None] = {}
        self._strata: dict[tuple[RelationKey, int | None, Pairing], tuple[Stratum, ...]] = {}
        self._episodes: dict[tuple[str, Segment], tuple[Episode, ...]] = {}
        self._proposed: dict[RelationKey, date] = {}

    def keys(self) -> tuple[RelationKey, ...]:
        """全部要量的关系：短跨度两两 × 四段；长跨度（行为概念的四种标志、情境概念成立的日子）；够样本的前因 × 条件。"""

        lane = self.line.lane
        # 停用的概念（类被拆改；细分 / 汇总概念成员类全停了）不再量：它不会再有新记录，量出来只会是"它没了"（裁定 27 第 9 条）
        names = [
            name
            for name in self.line.concepts()
            if self.concepts.lane_of(name) == lane and not members_retired(self.concepts, name)
        ]
        found = [
            RelationKey(lane, antecedent, consequent, segment)
            for antecedent in names
            if self.line.marks.get(antecedent)
            for consequent in names
            if comparable(self.concepts, antecedent, consequent)
            for segment in SEGMENTS
        ]
        for antecedent in names:
            for segment in MARKER_SEGMENTS:
                if self.episodes(antecedent, segment):
                    found.extend(
                        RelationKey(lane, antecedent, consequent, segment)
                        for consequent in names
                        if comparable(self.concepts, antecedent, consequent)
                    )
        for situation in sorted(self.line.situations):
            if self._derived_from_history(situation):
                continue
            if self.episodes(situation, Segment.SITUATION):
                found.extend(RelationKey(lane, situation, consequent, Segment.SITUATION) for consequent in names)
        found.extend(self._moderated(names))
        return tuple(found)

    def _derived_from_history(self, situation: str) -> bool:
        """由行为历史派生的情境（「昨日有 X」「连续 X 中」）不当前因，只当调节条件（2026-10-12 用户裁定）。

        它当前因问的问题别处已经在问：「昨日有 X 的日子 Y 多吗」= 次日段的 X → Y，「连续 X 中的日子 Y 多吗」= X 的集中期标志；
        Y 换成 X 自己就成了同义反复（真实模型盘：昨日有交材料 → 交材料 58.8% 对 0%，候选）。日历来的情境（周末、调休、
        月初）不从行为算，照旧进情境段。调节条件那条路（``conditions.mechanical`` 的概念模板）不受影响。
        """

        if situation not in self.concepts:
            return False
        rule = self.concepts[situation].situation
        return rule is not None and rule.basis.reads_history

    def _moderated(self, names: list[str]) -> list[RelationKey]:
        """带条件的关系：单个条件对每个够样本的前因都列（现状，裁定 62 第 1 条）；第二个条件起只在已成立的带条件关系上
        加一个（{C1} 成立才列 {C1, C2}），条件池同样是机械列举 + 模型提的，最多 ``max_combined_conditions`` 个。
        同一个集合从几条父关系长出来只有一条键。"""

        found: list[RelationKey] = []
        window = self.config.transition_window_slots
        pools: dict[str, dict[Condition, date | None]] = {}
        for antecedent in names:
            chains = self.chains(antecedent)
            if len(chains) < 2 * MIN_PER_SIDE:
                continue  # 两边各 2 次的最低要求；够不够开检验由每条关系自己的最小可达 p 定（``ready_to_test``）
            conditions: dict[Condition, date | None] = {
                condition: None for condition in mechanical(self.line, self.concepts, antecedent, chains, window=window)
            }
            # 模型提的条件只在提出之后的数据上检验（防"看过数据再提"）；与机械列举撞上的不算新条件，照旧在全部历史上量——
            # 不然它会把机械条件攒下的证据清掉（第四轮评审人工智能 1：第六批 21 条提议里 20 条与机械列举重复）
            conditions.update(
                {
                    condition: since
                    for condition, since in self.proposals.get(antecedent, ())
                    if condition not in conditions
                }
            )
            pools[antecedent] = conditions
            for condition, since in sorted(conditions.items(), key=lambda item: item[0].text):
                for consequent in names:
                    if not comparable(self.concepts, antecedent, consequent) or _circular(
                        self.concepts, condition, consequent
                    ):
                        continue
                    for segment in SEGMENTS:
                        key = RelationKey(self.line.lane, antecedent, consequent, segment, condition.text)
                        found.append(key)
                        if since is not None:
                            self._proposed[key] = since
        found.extend(self._grown(names, pools))
        return found

    def _grown(self, names: list[str], pools: Mapping[str, Mapping[Condition, date | None]]) -> list[RelationKey]:
        """已成立的带条件关系上再加一个条件（第二个及以上）。同一个集合从几条父关系长出来只有一条键：新加的那个条件
        是机械列举的就在全部历史上量，几条路都是模型提的才只看最早提出之后的数据。"""

        limit = self.config.thresholds.max_combined_conditions
        grown: dict[RelationKey, list[date | None]] = {}
        for parent in sorted(self.established):
            held = parent.conditions
            pool = pools.get(parent.antecedent)
            if not held or len(held) >= limit or pool is None:
                continue
            if parent.consequent not in names or parent.segment not in SEGMENTS:
                continue
            for condition, since in pool.items():
                if condition in held or _circular(self.concepts, condition, parent.consequent):
                    continue
                key = replace(parent, condition=join_conditions((*held, condition)))
                grown.setdefault(key, []).append(since)
        for key, sinces in grown.items():
            if all(since is not None for since in sinces):
                self._proposed[key] = min(since for since in sinces if since is not None)
        return sorted(grown)

    def still_defined(self, key: RelationKey) -> bool:
        """这条关系依赖的东西还在不在（冻住的行的寿命，26 号清单 5.6）：集中期 / 停了一阵随变点重算整段没了、带的条件有一个
        既不在机械列举里也不在模型提议里了（被拒、消失），就是不在了。只是前因这阵子没有新链的照样算在——它会回来。
        几个条件合起来的还要求父关系仍然成立，那一条由状态折叠按当晚的结果判（``state._drop_orphaned_combinations``）。"""

        if key.segment in (Segment.HIGH_STATE, Segment.LOW_STATE) and key.antecedent in self.line.concepts():
            if not self.episodes(key.antecedent, key.segment):
                return False
        if key.condition:
            chains = self.chains(key.antecedent)
            if not chains:
                return True
            window = self.config.transition_window_slots
            listed = {
                condition.text
                for condition in mechanical(self.line, self.concepts, key.antecedent, chains, window=window)
            }
            proposed = {condition.text for condition, _since in self.proposals.get(key.antecedent, ())}
            return all(item.text in listed or item.text in proposed for item in key.conditions)
        return True

    def episodes(self, antecedent: str, segment: Segment) -> tuple[Episode, ...]:
        cache = (antecedent, segment)
        if cache not in self._episodes:
            blocks = self.blocks(antecedent)
            self._episodes[cache] = longspan.episodes(
                self.line,
                antecedent,
                segment,
                block_days=blocks.days if blocks is not None else 1.0,
                first_seen=(self.config.thresholds.first_seen_min_days, self.config.thresholds.first_seen_min_blocks),
                alpha=self.config.thresholds.forward_alpha,
                visits=tuple(Visit(chain.day, prompted=chain.anchor.reminded) for chain in self.chains(antecedent)),
            )
        return self._episodes[cache]

    def strata(
        self,
        key: RelationKey,
        *,
        since: date | None = None,
        blind_as: int | None = None,
        pairing: Pairing = Pairing.SAME_DAY,
    ) -> tuple[Stratum, ...]:
        """短跨度关系的全部层（给了 ``since`` 就只要那天及以后的 A：前向验证只看发现之后的新数据）。条件不影响层，只影响分组。
        ``blind_as`` 给了就把后果"判不了"当成它（上下界检验）；``pairing`` 是同一条链的哪一道对照（别的段只有一种）。"""

        slot = (replace(key, condition=""), blind_as, pairing)
        if slot not in self._strata:
            controls = self.controls if blind_as is None else self._bounds[blind_as]
            blocks = self.blocks(key.antecedent)
            found: list[Stratum] = []
            for chain in merge_for(key.segment, self.chains(key.antecedent)):
                item = controls.stratum(key.segment, chain, key.consequent, antecedent=key.antecedent, pairing=pairing)
                if item is not None and blocks is not None:
                    found.append(replace(item, block=blocks.of(chain)))
            self._strata[slot] = tuple(found)
        items = self._strata[slot]
        return items if since is None else tuple(item for item in items if item.chain.day >= since)

    def _chain_measure(self, key: RelationKey, *, since: date | None = None, blind_as: int | None = None) -> Measure:
        """不带条件的短跨度关系怎么量：同一条链两道对照成对（``PairedMeasure``），别的段一道。"""

        first = ChainMeasure(self.strata(key, since=since, blind_as=blind_as))
        if key.segment is not Segment.CHAIN:
            return first
        second = ChainMeasure(self.strata(key, since=since, blind_as=blind_as, pairing=Pairing.SAME_SLOT))
        return PairedMeasure(first=first, second=second)

    def _bounded_reading(
        self, key: RelationKey, since: date | None, measure: Measure
    ) -> tuple[Measure, float, float, bool] | None:
        """细分概念当后果的第 1 道：后果"判不了"当 0、当 1 各算一遍，两遍都显著、方向一致才算过（p 取两遍里大的）。
        判不了的窗口占比超过 ``max_blind_share`` 的返回 None（不进检验族）。窗口里"有一条命中就记 1、判不了就丢"会把
        「A → 返工」读成「A → 修改代码」又量一遍（第四轮评审人工智能 3：窗口里两条修改代码时读出 97% "返工"）。"""

        exact = self.config.thresholds.exact_strata_limit
        low = self._chain_measure(key, since=since, blind_as=0)
        high = self._chain_measure(key, since=since, blind_as=1)
        if low.units and 1.0 - measure.units / low.units > self.config.thresholds.max_blind_share:
            return None
        found = []
        for bound in (low, high):
            upper, lower = bound.tails(exact_limit=exact)
            found.append((stats.two_sided(upper, lower), bound.minimum_p(), upper <= lower))
        (p_low, min_low, up_low), (p_high, min_high, up_high) = found
        if up_low != up_high:
            return measure, 1.0, max(min_low, min_high), up_low  # 两头方向都不一致：说不了
        return measure, max(p_low, p_high), max(min_low, min_high), up_low

    def refined_consequent(self, key: RelationKey) -> bool:
        """后果是细分概念的短跨度、不带条件的关系：它的"判不了"来自映射器，要做上下界检验。"""

        item = self.concepts.get(key.consequent)
        return item is not None and item.kind is ConceptKind.REFINEMENT and not key.condition and not key.segment.long

    def chains(self, antecedent: str) -> tuple[Chain, ...]:
        if antecedent not in self._chains:
            self._chains[antecedent] = chains_of(self.line, antecedent, self.config.transition_window_slots)
        return self._chains[antecedent]

    def blocks(self, antecedent: str) -> Blocks | None:
        if antecedent not in self._blocks:
            chains = self.chains(antecedent)
            if not chains:
                self._blocks[antecedent] = None
            else:
                # 块长 = max(1 天, 常见间隔)：常见间隔只在 temporal 算一处（相邻发生日隔几个日历日的中位数，裁定 59）
                common = common_interval_days([chain.day for chain in chains])
                days = max(1.0, common) if common is not None else 1.0
                self._blocks[antecedent] = Blocks(days=days, origin=chains[0].day)
        return self._blocks[antecedent]

    def night(
        self,
        weights: Mapping[RelationKey, float] | None = None,
        *,
        since: Mapping[RelationKey, date] | None = None,
        outside: frozenset[RelationKey] = frozenset(),
    ) -> LaneNight:
        """当晚的检验族。

        ``since``：这些关系只看那天及以后的数据（失效、前向没复现之后，要在新数据上重新显著才算"又回来了"——
        不然旧数据里那段强关系会让它每晚都"又过第 1 道"）。``outside``：这些关系照算读数、但不进检验族
        （候选、成立的由状态折叠各管各的，成立之后不再进全局 FDR）。
        """

        config = self.config
        weights = weights or {}
        keys = self.keys()
        since = {**self._proposed, **(since or {})}
        readings: dict[RelationKey, tuple[Measure, float, float, bool]] = {}
        blinded: set[RelationKey] = set()
        for key in keys:
            measure = self.measure(key, since=since.get(key))
            upper, lower = measure.tails(exact_limit=config.thresholds.exact_strata_limit)
            if self.refined_consequent(key):
                bounded = self._bounded_reading(key, since.get(key), measure)
                if bounded is None:
                    # 判不了的太多：留一条读数、不进检验族（该回去改触点①的区别句）
                    readings[key] = (measure, 1.0, 1.0, upper <= lower)
                    blinded.add(key)
                else:
                    readings[key] = bounded
            elif (
                key.condition
                and isinstance(measure, ModeratedMeasure)
                and not ready_to_test(measure, alpha=config.thresholds.forward_alpha)
            ):
                # 两边全分开也到不了 α：不进检验族（裁定 60 第 9 条，取代固定的前因 ≥ 40）
                readings[key] = (measure, 1.0, 1.0, upper <= lower)
            elif key.segment is Segment.CHAIN and not key.condition:
                # 同一条链只检"更多"（裁定 27 第 1 条）：一件接一件做事时，"接下来做什么"此消彼长——对照是别的事做完之后，
                # 一条强关系（讨论方案 → 修改代码）会让别的前因之后的修改代码都显得"更少"；时长差异也只会造出"更少"。
                # "做完 A 之后更不会接着做 B"在这里多半是别处关系的影子，不是一条独立的发现，对"接下来可能做什么"也没用。
                readings[key] = (measure, upper, measure.minimum_p() / 2.0, True)
            else:
                readings[key] = (measure, stats.two_sided(upper, lower), measure.minimum_p(), upper <= lower)
        texts = {key.text: key for key in readings}
        inside = [key for key in readings if key not in outside and key not in blinded]
        # 长跨度自成一族（裁定 27 第 4 条，S4）：它与短跨度问的是两类问题，同一个 q 各控各的；混在一族里，
        # 几十条长跨度要和几百条短跨度分摊误发现率，门槛被压到 q / 300 上下，永远进不了族
        family: set[str] = set()
        passed: set[str] = set()
        limits: dict[str, int] = {}
        for members in ([key for key in inside if not key.segment.long], [key for key in inside if key.segment.long]):
            group, group_passed = stats.tarone_bh(
                {key.text: readings[key][1] for key in members},
                {key.text: readings[key][2] for key in members},
                {key.text: weights.get(key, 1.0) for key in members},
                q=config.thresholds.fdr,
            )
            family |= group
            passed |= group_passed
            limits.update({key.text: max(1, len(group)) for key in members})
        tests: list[RelationTest] = []
        for text in sorted(texts):
            key = texts[text]
            measure, p_value, minimum, upward = readings[key]
            blocks = self.blocks(key.antecedent)
            base = RelationTest(
                key=key,
                verdict=Verdict.SIGNIFICANT
                if text in passed
                else Verdict.TESTED
                if text in family or (key in outside and measure.units > 0)
                else Verdict.SPARSE,
                antecedents=measure.units,
                blocks=measure.blocks,
                block_days=blocks.days if blocks is not None else 1.0,
                effect=measure.effect(),
                p_value=p_value,
                minimum_p=minimum,
                upward=upward,
                weight=weights.get(key, 1.0),
            )
            if base.verdict is Verdict.SPARSE:
                tests.append(
                    replace(base, more_needed=_more_needed(measure, config.thresholds.fdr / limits.get(text, 1)))
                )
            elif base.verdict is Verdict.SIGNIFICANT:
                tests.append(self.assess(base, measure))
            else:
                tests.append(base)
        return LaneNight(lane=self.line.lane, cutoff=self.cutoff, tests=tuple(tests), family=len(family))

    def evaluate(
        self, key: RelationKey, *, upward: bool, since: date | None = None, last_blocks: int | None = None, purpose: str
    ) -> Subset:
        """子集上的读数：``since`` 之后的新数据（前向验证、维持检验），或最近 ``last_blocks`` 个块。p 是朝 ``upward`` 的单侧。"""

        measure = self.measure(key).subset(since=since, last_blocks=last_blocks)
        upper, lower = measure.tails(exact_limit=self.config.thresholds.exact_strata_limit)
        spread, relative = measure.interval(
            level=self.config.thresholds.interval_level,
            rounds=self.config.thresholds.bootstrap_rounds,
            seed=self.seed(key, purpose),
        )
        return Subset(
            antecedents=measure.units,
            blocks=measure.blocks,
            effect=measure.effect(),
            p_value=upper if upward else lower,
            interval=spread,
            relative_interval=relative,
            minimum_p=measure.minimum_p() / 2.0,
        )

    def measure(self, key: RelationKey, *, since: date | None = None) -> Measure:
        """这个关系怎么量：长跨度按状态（``longspan``），带条件的把层按条件分两组（``conditions``），其余按层。"""

        measure: Measure
        rule = calendar_key(self.concepts, key.antecedent) if key.segment is Segment.SITUATION else None
        if rule is not None:
            # 日历情境按维度取对照（``relations.calendar``，裁定 59 第 7 条）；模型写的情境仍和前后紧挨着的平常日子比
            measure = calendar_measure(self.line, key.consequent, *rule)
        elif key.segment.long:
            measure = longspan.measure_of(self.line, key.consequent, self.episodes(key.antecedent, key.segment))
        elif key.condition:
            window = self.config.transition_window_slots
            judges = [evaluator(self.line, condition, window=window) for condition in key.conditions]
            # 带条件的只按第 1 道对照分两组比：两组的钟点偏差一样，分组之间抵掉；几个条件合起来比"全成立"对"少任意一个"
            measure = split(ChainMeasure(self.strata(key)), judges)
        else:
            measure = self._chain_measure(key)
        return measure if since is None else measure.subset(since=since)

    def seed(self, key: RelationKey, purpose: str) -> int:
        """重抽样的种子：（关系身份, 第几晚, 用途）的摘要——同一晚重跑结果逐字相同。"""

        return int(canonical_digest({"key": key.text, "night": self.cutoff.isoformat(), "purpose": purpose})[:16], 16)

    def assess(self, test: RelationTest, measure: Measure | None = None) -> RelationTest:
        """第 2、3 道与几个只报告的读数，按 ``test.upward`` 那个方向判（候选在某一晚没显著时，状态折叠也要用它）。"""

        config = self.config
        if measure is None:
            measure = self.measure(test.key)
        spread, relative = measure.interval(
            level=config.thresholds.interval_level,
            rounds=config.thresholds.bootstrap_rounds,
            seed=self.seed(test.key, "interval"),
        )
        heterogeneity = measure.heterogeneity()
        leave_one_out = measure.leave_one_out(upward=test.upward)
        effect = measure.effect()
        consequent_count = len(self.line.outcomes(test.key.consequent))
        excess = effect.antecedents * effect.difference
        return replace(
            test,
            heterogeneity_p=heterogeneity,
            leave_one_out=leave_one_out,
            stable=leave_one_out and (heterogeneity is None or heterogeneity >= config.thresholds.heterogeneity_alpha),
            interval=spread,
            relative_interval=relative,
            excess_share=excess / consequent_count if consequent_count else None,
            same_event_share=measure.same_event_share(),
            large_enough=large_enough(spread, relative, upward=test.upward, config=config),
        )


def large_enough(
    spread: tuple[float, float] | None,
    relative: tuple[float, float] | None,
    *,
    upward: bool,
    config: RelationConfig,
) -> bool:
    """第 3 道：区间离 0 最近的那一端，同时够得上绝对差与相对差。对照率常为 0、说不清相对差的，相对差那条按满足算
    （从无到有本身就是最大的相对变化）。"""

    if spread is None:
        return False
    near = spread[0] if upward else -spread[1]
    if near < config.thresholds.min_absolute_lift:
        return False
    if relative is None:
        return True
    near_relative = relative[0] if upward else -relative[1]
    return near_relative >= config.thresholds.min_relative_lift


def gone(
    spread: tuple[float, float] | None, relative: tuple[float, float] | None, *, upward: bool, config: RelationConfig
) -> bool:
    """失效：区间离 0 最远的那一端都已够不上第 3 道（有把握说它没了）。只是不显著（证据弱、区间宽）不算。"""

    if spread is None:
        return False
    far = spread[1] if upward else -spread[0]
    if far < config.thresholds.min_absolute_lift:
        return True
    if relative is None:
        return False
    far_relative = relative[1] if upward else -relative[0]
    return far_relative < config.thresholds.min_relative_lift


def comparable(concepts: ConceptSet, antecedent: str, consequent: str) -> bool:
    """这一对要不要比：不比自己；不比读同一批记录的（汇总 / 基础类有交集；细分与它的源类或含源类的汇总）。"""

    if antecedent == consequent:
        return False
    if concepts.overlaps(antecedent, consequent):
        return False
    first, second = concepts[antecedent], concepts[consequent]
    for refinement, other in ((first, second), (second, first)):
        if refinement.kind is ConceptKind.REFINEMENT and other.kind in (ConceptKind.BASE, ConceptKind.GROUP):
            if refinement.classes[0] in other.classes:
                return False
    return True


def examine_night(
    series: EventSeries,
    concepts: ConceptSet,
    hits: Mapping[str, ConceptHits],
    config: RelationConfig,
    *,
    weights: Mapping[RelationKey, float] | None = None,
) -> dict[str, LaneNight]:
    """第 ``series.cutoff`` 晚：每条 lane 一个检验族。"""

    lines = build_timelines(series, concepts, hits, config)
    return {
        lane: LaneTests(line, concepts, config, series.cutoff).night(weights) for lane, line in sorted(lines.items())
    }


def _circular(concepts: ConceptSet, condition: Condition, consequent: str) -> bool:
    """ "B 刚出现过"不是调节：条件里的那一类就是（或读同一批记录于）后果时，它说的是 B 自己扎堆，不是 A 被什么左右。"""

    if condition.template is not Template.TIMELINE:
        return False
    return condition.value == consequent or not comparable(concepts, condition.value, consequent)


def _more_needed(measure: Measure, threshold: float) -> int | None:
    """照现在的样子，还要几个全朝一个方向的单位才可能把最小可达 p 压到 ``threshold`` 以下（粗估）。"""

    if isinstance(measure, PairedMeasure):
        # 两道对照都要过：按更难过的那道估
        needed = [_more_needed(part, threshold) for part in measure.parts]
        return None if any(item is None for item in needed) else max(item for item in needed if item is not None)
    if not isinstance(measure, ChainMeasure):
        return None
    informative = [item.null_probability for item in measure.strata if 0.0 < item.null_probability < 1.0]
    if not informative:
        return None
    minimum = measure.minimum_p()
    if minimum <= threshold:
        return 0
    typical = statistics.median(min(p, 1.0 - p) for p in informative)
    if typical <= 0.0:
        return None
    return math.ceil(math.log(threshold / minimum) / math.log(max(typical, 1e-12)))


__all__ = [
    "Blocks",
    "LaneNight",
    "LaneTests",
    "RelationKey",
    "RelationTest",
    "Subset",
    "Verdict",
    "comparable",
    "examine_night",
    "gone",
    "large_enough",
]
