"""夜批 B3 映射：今晚封口的那天 + 还没映射的、概念集变了要回填的日子，最新的先、每晚有预算（映射的 LLM 触点）。

B1 迁移重映射与 B3 共用这里的单日映射（``context`` / ``map``）。一夜的数值（``SceneNightConfig``）每次从夜批现读——
组合根与运维可以在构造后换掉它。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING

from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.infrastructure.lifecycle import RetryLedger
from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.occurrences.mapper import ConceptMapper, ConceptMapperError, DayMappingReport, map_closed_day
from habitus.scene.occurrences.situations import history_days
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.series import EventSeries

if TYPE_CHECKING:  # 只给类型检查看：夜批模块在运行时导入本模块
    from habitus.runtime.scene_night import SceneNightConfig


class MappingStage:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        timezone: tzinfo,
        subject: str | None,
        mapping_retries: RetryLedger,
        config: Callable[[], SceneNightConfig],
    ) -> None:
        self.behavior_tree = behavior_tree
        self.hits = hits
        self.calendar = calendar
        self.timezone = timezone
        self.subject = subject
        self.mapping_retries = mapping_retries
        self._config = config

    async def map_backlog(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        rhythms: Mapping[str, Rhythm],
        slack_minutes: int,
        now: datetime,
        signals: list[str],
    ) -> tuple[list[DayMappingReport], tuple[date, ...], int]:
        """要映射的日子：序列里有记录、还没以现在的概念集口径完成的（没映射过、或概念集变了要回填），最新的先；
        今晚封口的那天总在里面，其余按预算。返回（各天的报告, 映射了哪几天, 还剩几天）。"""

        recorded = sorted({record.day for record in series.records}, reverse=True)
        # 完成标记按这一天自己的口径认（那天出现的类各自的口径 + 情境；E9），条数也要与序列对得上（盖章之后又补发了记录就重做；E1）
        current = {
            day
            for day in self.hits.days_done()
            if (marker := self.hits.read_marker(day)) is not None
            and marker.records == len(series.on(day))
            and marker.mapper == mapper.day_version(record.kind_token for record in series.on(day))
        }
        pending = [day for day in recorded if day not in current]
        # 模型没答成的日子按重试账退避、次数用完就放弃，不再每晚重排、不占 14 天预算（26 号清单 2.4.6）；
        # 账的键带着这一天的映射口径，概念集一变就是新键，重新给机会。今晚封口的那天照常排进来。
        versions = {day: mapper.day_version(record.kind_token for record in series.on(day)) for day in pending}
        chosen, waiting = select_mapping_days(
            pending,
            today=series.last_day,
            budget=self._config().mapping_days_per_night,
            blocked=lambda day: self.mapping_retries.blocked(_mapping_key(day, versions[day])),
        )
        reports = []
        # 按日子先后映射（挑哪几天仍是最新的先）：读历史的派生情境（「昨天吃过早餐」「连续研发中」）要前几天先映射好
        for day in sorted(chosen):
            report = await self._map_chosen(
                day, versions[day], concepts, mapper, rhythms, slack_minutes, series, now, signals
            )
            if report is not None:
                reports.append(report)
        await self._revisit_readers(
            [report.day for report in reports], concepts, mapper, rhythms, slack_minutes, series, now, signals
        )
        backlog = len(pending) - len(chosen) - len(waiting)
        if backlog:
            signals.append(f"mapping: 还有 {backlog} 天没轮到（没映射过或概念集变了要回填），之后几晚接着补")
        if waiting:
            signals.append(f"mapping: {len(waiting)} 天模型没答成，在退避或已放弃")
        return reports, tuple(chosen), backlog

    async def _map_chosen(
        self,
        day: date,
        version: str,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        rhythms: Mapping[str, Rhythm],
        slack_minutes: int,
        series: EventSeries,
        now: datetime,
        signals: list[str],
    ) -> DayMappingReport | None:
        """映射挑中的一天、按结果记重试账；写不出的那天返回 None（不盖章，其余日子与 B4 照常）。"""

        context = self.context(day, concepts, rhythms, slack_minutes)
        if day == series.last_day:
            signals.extend(f"baseline: {key} 样本不够，引用它的候选会记未决" for key in context.baselines.missing)
            signals.extend(f"drift: {item.render()}" for item in context.baselines.drifting)
        key = _mapping_key(day, version)
        try:
            report = await self.map(day, mapper, context, series, now=now)
        except ConceptMapperError as exc:
            # 一条记录写不出（材料超了记录的上限等）：这一天不盖章、记进重试账，其余日子与 B4 照常（26 号清单 2.4.6）
            decision = self.mapping_retries.record_failure(key, exc)
            signals.append(
                f"mapping: {day} 写不出（{exc}），"
                f"{'先不再重排（概念集变了会重新给机会）' if decision.give_up else '按重试账退避后重做'}"
            )
            return None
        if report.model_failed:
            decision = self.mapping_retries.record_failure(key, MappingModelFailed(f"{report.model_failed} unresolved"))
            if decision.give_up:
                signals.append(f"mapping: {day} 模型几次都没答成，先不再重排（概念集变了会重新给机会）")
        else:
            self.mapping_retries.clear(key)
        return report

    async def _revisit_readers(
        self,
        mapped: Sequence[date],
        concepts: ConceptSet,
        mapper: ConceptMapper,
        rhythms: Mapping[str, Rhythm],
        slack_minutes: int,
        series: EventSeries,
        now: datetime,
        signals: list[str],
    ) -> None:
        """这一晚映射过的日子之后、读得到它们的那几天（派生情境往回看的天数之内），早已映射完的再过一遍：
        候选的输入没变就只就地重算情境、不问模型（续跑），这样前一天后补的命中能进后一天的「昨天…」「连续…」。"""

        depth = len(history_days(concepts, series.last_day)) - 1
        if depth <= 0 or not mapped:
            return
        done = set(self.hits.days_done())
        readers = {day + timedelta(days=step) for day in mapped for step in range(1, depth + 1)}
        for day in sorted(readers - set(mapped)):
            if day not in done or day >= series.cutoff:
                continue
            context = self.context(day, concepts, rhythms, slack_minutes)
            try:
                await self.map(day, mapper, context, series, now=now)
            except ConceptMapperError as exc:
                signals.append(f"mapping: {day} 回访重算情境时写不出（{exc}），之后照常补")

    def context(
        self, day: date, concepts: ConceptSet, rhythms: Mapping[str, Rhythm], slack_minutes: int
    ) -> SceneDayContext:
        return SceneDayContext(
            day,
            concepts=concepts,
            hits=self.hits,
            calendar=self.calendar,
            timezone=self.timezone,
            subject=self.subject,
            recent_days=self._config().recent_days,
            rhythms=rhythms,
            slack_minutes=slack_minutes,
        )

    async def map(
        self,
        day: date,
        mapper: ConceptMapper,
        context: SceneDayContext,
        series: EventSeries,
        *,
        now: datetime,
        force: bool = False,
    ) -> DayMappingReport:
        return await map_closed_day(
            self.behavior_tree,
            self.hits,
            mapper,
            day,
            now=now,
            series=series,
            situation_for=context.situation_for,
            baseline_for=context.baseline_for,
            force=force,
        )


def select_mapping_days(
    pending: Sequence[date], *, today: date | None, budget: int, blocked: Callable[[date], bool]
) -> tuple[list[date], list[date]]:
    """这一晚映射哪几天（今晚封口的那天总在里面，其余最新的先、按预算）、哪几天在重试账里等着（退避中或已放弃）。"""

    waiting = [day for day in pending if day != today and blocked(day)]
    chosen = [day for day in pending if day == today]
    chosen += [day for day in pending if day != today and day not in waiting][:budget]
    return chosen, waiting


class MappingModelFailed(RuntimeError):
    """这一天有记录模型没答成（可重试：传输、限流、输出不合格）。"""


def _mapping_key(day: date, version: str) -> str:
    return f"map:{day.isoformat()}:{version}"


__all__ = ["MappingModelFailed", "MappingStage", "select_mapping_days"]
