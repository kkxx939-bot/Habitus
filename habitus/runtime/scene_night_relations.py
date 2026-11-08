"""夜批 B4 关系：每条 lane 全部两两检验 → 接上盘上之前最近的一晚折叠 → 这一晚的关系表与迁移日志落盘（纯算法、零模型；
有模型建议时先补问调节条件与先验，都按输入摘要缓存）。

B4 读的是盘上**全部**已映射的命中（不只今天）：关系每晚全量重算，不做增量统计。模型建议（``RelationAdvice``）每次从
夜批现读——组合根与运维可以在构造后换掉它。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime

from habitus.conditions import DayTypeCalendar
from habitus.runtime.scene_advice import RelationAdvice
from habitus.scene.advice.store import AdviceStoreError
from habitus.scene.concepts.calendar_situations import value_sets_to_add
from habitus.scene.concepts.catalog import CatalogChanges, ClassCatalog
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.relations import (
    LaneTests,
    LaneTimeline,
    RelationConfig,
    RelationKey,
    Segment,
    build_timelines,
)
from habitus.scene.relations.state import LaneState, Status, Transition, fold, revert_touched
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.series import EventSeries


class RelationStage:
    def __init__(
        self,
        *,
        catalog: ClassCatalog,
        concepts: ConceptStore,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        relations: RelationStore,
        relation_config: RelationConfig,
        advice: Callable[[], RelationAdvice | None],
    ) -> None:
        self.catalog = catalog
        self.concepts = concepts
        self.hits = hits
        self.calendar = calendar
        self.relations = relations
        self.relation_config = relation_config
        self._advice = advice

    async def run(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        signals: list[str],
        *,
        vocabulary: int = 0,
        now: datetime | None = None,
    ) -> tuple[dict[str, dict[str, int]], int, dict[str, LaneTimeline]]:
        """第 ``series.cutoff`` 晚：每条 lane 检验、接上盘上之前最近的一晚折叠、落盘。返回各 lane 的计数、迁移条数与时间线。

        ``vocabulary`` 是这一晚同步到的词表版本，记在关系状态上。接的那一晚的状态记着它当时的词表版本，这之间拆分 / 合并改了
        成员的类，涉及它们的关系先退回候选（裁定 59）；按状态上记的版本算，同一晚重跑也退回得一样。折叠之后，同一件事有两个以上
        单个取值（周几、几号、日型）的日历关系处在候选或成立的，补一个取值集合的日历情境，下一晚起一起检验。"""

        hits = await asyncio.to_thread(self._all_hits)
        lines = await asyncio.to_thread(
            build_timelines, series, concepts, hits, self.relation_config, calendar=self.calendar
        )
        counts: dict[str, dict[str, int]] = {}
        moved = 0
        for lane, line in sorted(lines.items()):
            folded = await self._fold_lane(lane, line, series, concepts, signals, vocabulary=vocabulary)
            if folded is None:
                continue
            state, transitions = folded
            self._grow_value_sets(concepts, state, now or datetime.now(UTC), signals)
            counts[lane] = {status.value: state.count(status) for status in Status}
            moved += len(transitions)
            signals.extend(
                f"relation: {lane} {concepts.label_of(item.key.antecedent)} → {concepts.label_of(item.key.consequent)}"
                f"（{item.key.segment.value}{'｜' + item.key.condition if item.key.condition else ''}）"
                f"{'—' if item.before is None else item.before.value} → {item.after.value}：{item.reason}"
                for item in transitions
            )
        return counts, moved, lines

    async def _fold_lane(
        self,
        lane: str,
        line: LaneTimeline,
        series: EventSeries,
        concepts: ConceptSet,
        signals: list[str],
        *,
        vocabulary: int,
    ) -> tuple[LaneState, tuple[Transition, ...]] | None:
        """一条 lane 这一晚：退回词表变更牵到的、检验、折叠、落盘；这一晚不写（沿用上一晚的）时返回 None。"""

        previous = self._previous(lane, series.cutoff, signals)
        reverted: tuple[Transition, ...] = ()
        if previous is not None:
            touched = _touched(concepts, _changed_classes(self.catalog.changes_since(previous.vocabulary)))
            if touched:
                previous, reverted = revert_touched(previous, touched, series.cutoff)
        # 第二个条件起只长在接的那一晚已成立的带条件关系上（裁定 62 第 1 条）：退回之后再数哪些成立
        established = tuple(item.key for item in previous.of(Status.ESTABLISHED)) if previous is not None else ()
        tests = LaneTests(line, concepts, self.relation_config, series.cutoff, established=established)
        weights: dict[RelationKey, float] = {}
        advice = self._advice()
        if advice is not None:
            try:
                proposals = await advice.proposals(tests, series.cutoff, signals)
            except AdviceStoreError as exc:
                # 提议簿读不了：不带它们折叠会把接着的带条件关系当成"行没了"判失效，这条 lane 这一晚不写、沿用上一晚的
                signals.append(f"relations: {lane} 提议簿读不了（{exc}），这一晚的关系表沿用上一晚的")
                return None
            tests = LaneTests(
                line, concepts, self.relation_config, series.cutoff, proposals=proposals, established=established
            )
            weights = await advice.weights(tests, signals)
        state, transitions = await asyncio.to_thread(fold, previous, tests, weights or None)
        state = replace(state, vocabulary=vocabulary)
        transitions = (*reverted, *transitions)
        try:
            self.relations.write(state, transitions)
        except RelationStoreError as exc:
            # 这条 lane 这一晚写不出（超限、盘上已有更晚一晚）：只让它这一晚失败、明天的预测层读上一代的表，
            # 不让整晚停、别的 lane 照常（26 号清单 2.5.2）
            signals.append(f"relations: {lane} 这一晚的关系表写不出（{exc}），沿用上一晚的")
            return None
        return state, transitions

    def _all_hits(self) -> dict[str, ConceptHits]:
        return {record.occurrence_uri: record for day in self.hits.days_done() for record in self.hits.read_day(day)}

    def _grow_value_sets(self, concepts: ConceptSet, state: LaneState, now: datetime, signals: list[str]) -> None:
        held: dict[str, list[str]] = {}
        for relation in state.relations:
            key = relation.key
            if (
                key.segment is Segment.SITUATION
                and not key.condition
                and relation.upward
                and relation.status in (Status.CANDIDATE, Status.ESTABLISHED)
            ):
                held.setdefault(key.consequent, []).append(key.antecedent)
        for definition in value_sets_to_add(concepts, held, now=now):
            self.concepts.write(definition)
            signals.append(f"vocabulary: 补上日历取值集合「{definition.name}」，下一晚起一起检验（裁定 59）")

    def _previous(self, lane: str, night: date, signals: list[str]) -> LaneState | None:
        """这一晚要接的那一晚。最近那一晚的文件坏了：接在更早一个读得了的晚上之后，并报出来（第四轮评审 E10）——
        不然夜批永远写不出新的一晚，预测层也一直读着坏文件硬拒，两边互相卡死，只能人工恢复。新的一晚写出来，预测层下一拍就好了。"""

        try:
            return self.relations.previous(lane, night)
        except RelationStoreError as exc:
            for earlier in reversed(self.relations.nights(lane)):
                if earlier >= night:
                    continue
                try:
                    state = self.relations.read(lane, earlier)
                except RelationStoreError:
                    # 这一晚也读不了：接着往更早找；都读不了时下面整体报一次
                    continue
                signals.append(f"relations: {lane} 更晚的关系表读不了（{exc}），这一晚接在第 {earlier} 晚之后折叠")
                return state
            signals.append(f"relations: {lane} 之前的关系表都读不了（{exc}），这一晚从头折叠")
            return None


def _changed_classes(changes: CatalogChanges) -> frozenset[str]:
    """这一段词表变更里成员变了的类：拆出与交出的、并掉与并入的、条目被搬过的两头。"""

    found = {*changes.split_from, *changes.split_from.values(), *changes.merged_into, *changes.merged_into.values()}
    for moved in changes.moved:
        found.update((moved.source, moved.target))
    return frozenset(found)


def _touched(concepts: ConceptSet, changed: frozenset[str]) -> frozenset[str]:
    """成员变了的类牵到的行为概念：基础、细分、包含它的汇总。"""

    if not changed:
        return frozenset()
    return frozenset(identity for identity in concepts.behaviors() if set(concepts.classes_of(identity)) & changed)


__all__ = ["RelationStage"]
