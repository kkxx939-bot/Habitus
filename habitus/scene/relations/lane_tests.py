"""一条 lane 一晚的检验：时间线、对照、每个前因的链与块都只算一次（``LaneTests``），量哪些交 ``candidates``、怎么量交
``evaluation``。关系的身份、读数与第 3 道的判法在 ``engine``。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import date

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.relations import longspan
from habitus.scene.relations.candidates import RelationCandidates
from habitus.scene.relations.conditions import Condition
from habitus.scene.relations.config import RelationConfig
from habitus.scene.relations.engine import Blocks, LaneNight, RelationKey, RelationTest, Subset
from habitus.scene.relations.evaluation import RelationEvaluator
from habitus.scene.relations.longspan import Episode
from habitus.scene.relations.measures import Measure
from habitus.scene.relations.spans import Chain, Controls, Pairing, Segment, Stratum, chains_of, merge_for
from habitus.scene.relations.timeline import LaneTimeline, build_timelines
from habitus.series import EventSeries
from habitus.temporal.absence import Visit
from habitus.temporal.intervals import common_interval_days


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
        #: 量哪些（枚举与冻住的行的寿命）与怎么量（读数、检验族、子集上的读数）。
        self.candidates = RelationCandidates(self)
        self.evaluator = RelationEvaluator(self, self.candidates)

    # ── 量哪些（``candidates``） ──────────────────────────────────────────────

    def keys(self) -> tuple[RelationKey, ...]:
        """全部要量的关系（``RelationCandidates.keys``）。"""

        return self.candidates.keys()

    def still_defined(self, key: RelationKey) -> bool:
        """这条关系依赖的东西还在不在（``RelationCandidates.still_defined``）。"""

        return self.candidates.still_defined(key)

    # ── 怎么量（``evaluation``） ──────────────────────────────────────────────

    def night(
        self,
        weights: Mapping[RelationKey, float] | None = None,
        *,
        since: Mapping[RelationKey, date] | None = None,
        outside: frozenset[RelationKey] = frozenset(),
    ) -> LaneNight:
        """当晚的检验族（``RelationEvaluator.night``）。"""

        return self.evaluator.night(weights, since=since, outside=outside)

    def evaluate(
        self, key: RelationKey, *, upward: bool, since: date | None = None, last_blocks: int | None = None, purpose: str
    ) -> Subset:
        """子集上的读数（``RelationEvaluator.evaluate``）。"""

        return self.evaluator.evaluate(key, upward=upward, since=since, last_blocks=last_blocks, purpose=purpose)

    def measure(self, key: RelationKey, *, since: date | None = None) -> Measure:
        """这个关系怎么量（``RelationEvaluator.measure``）。"""

        return self.evaluator.measure(key, since=since)

    def seed(self, key: RelationKey, purpose: str) -> int:
        """重抽样的种子（``RelationEvaluator.seed``）。"""

        return self.evaluator.seed(key, purpose)

    def assess(
        self, test: RelationTest, measure: Measure | None = None, *, previously_large: bool | None = None
    ) -> RelationTest:
        """第 2、3 道与几个只报告的读数（``RelationEvaluator.assess``）。"""

        return self.evaluator.assess(test, measure, previously_large=previously_large)

    def refined_consequent(self, key: RelationKey) -> bool:
        """后果是细分概念的短跨度、不带条件的关系（``RelationEvaluator.refined_consequent``）。"""

        return self.evaluator.refined_consequent(key)

    # ── 一晚只算一次的材料 ────────────────────────────────────────────────────

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


__all__ = ["LaneTests", "examine_night"]
