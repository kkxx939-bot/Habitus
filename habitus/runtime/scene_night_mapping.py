"""B3 映射的编排（实时化阶段 2「映射按槽」）：按条认欠账、按条映射，一天封口时统一重算情境、盖章。

取材料与封口的规则在领域包（``scene.occurrences.daily``）；这里只取材料、调用、落盘：

- ``plan``（纯读盘）：还没以现在的口径盖完成章的日子里，哪几条记录该映射——没有命中记录、口径不是现在的、或输入变了的。
  先后是：还没封口的日子里的新记录 → 口径或输入变了要重判的 → 历史回填（最新的先）。没有一晚的预算：模型队列一条一项地做。
- ``map_one``：映射一条（情境留空）。
- 封口在 ``runtime.scene_night_seal``（``SealStage``）：已封口的日子里每条记录都作数的，重算情境、写回、核条数、盖日完成章。
- ``rejudge_day``：词表迁移改了编号的日子，重判受影响的那几条（不盖章，章由封口盖）。

一条记录的常态取它那一天的那一代的：命中历史是那一天之前已完成的日子，节律由 ``rhythms`` 给（那一代落盘的那一份，没有就按
截止到那一天的序列现算）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, tzinfo
from typing import TYPE_CHECKING

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.tree import BehaviorTree
from habitus.conditions import DayTypeCalendar
from habitus.foundation.integrity import canonical_digest
from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.occurrences.daily import (
    DayRecords,
    current,
    day_documents,
    map_record,
    waiting_for_recent,
)
from habitus.scene.occurrences.mapper import ConceptMapper
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.series import EventSeries, SeriesRecord

if TYPE_CHECKING:  # 只给类型检查看：夜批模块在运行时导入本模块
    from habitus.runtime.scene_night import SceneNightConfig


@dataclass(frozen=True)
class PendingRecord:
    """一条该映射的记录：哪一条、它那一天的取材料口与全文、盘上有没有旧的命中记录（有就是重判）、它那一天封口了没有。"""

    entry: SeriesRecord
    document: BehaviorDocument
    records: DayRecords
    rejudged: bool
    sealed: bool

    @property
    def key(self) -> str:
        """重试账与进程内退避用的名字：这一条加它现在的口径（口径一变就是新键，重新给机会）。"""

        version = self.records.mapper.version_for(self.entry.kind_token)
        return f"map:{canonical_digest([self.entry.uri, version])[:24]}"


class MappingStage:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        timezone: tzinfo,
        subject: str | None,
        config: Callable[[], SceneNightConfig],
        rhythms: Callable[[date, ConceptSet, EventSeries], Mapping[str, Rhythm]],
        slack: Callable[[], int],
        recent_given: Callable[[], Callable[[str], bool] | None] | None = None,
    ) -> None:
        self.behavior_tree = behavior_tree
        self.hits = hits
        self.calendar = calendar
        self.timezone = timezone
        self.subject = subject
        self._config = config
        #: （日子, 概念集, 序列）→ 那一天的那一代的节律（``SceneNightlyRun.rhythms_for``）。
        self._rhythms = rhythms
        self._slack = slack
        #: 现在哪些类的"近几天同类记录"给得出（最近一代之后才有或才变的类给不出）；没接就都给。
        self._recent_given = recent_given

    # ── 认欠账、按条映射（模型队列）─────────────────────────────────────────

    def pending_days(self, series: EventSeries, mapper: ConceptMapper) -> tuple[date, ...]:
        """序列里有记录、还没以现在的口径盖完成章的日子（最新的先）：完成标记按这一天自己的口径认（那天出现的类各自的口径
        + 情境；E9），条数也要与序列对得上（盖章之后又补发了记录就重做；E1）。"""

        done = self.hits.days_done()
        pending = []
        for day in sorted({record.day for record in series.records}, reverse=True):
            marker = self.hits.read_marker(day) if day in done else None
            tokens = [record.kind_token for record in series.on(day)]
            if marker is None or marker.records != len(tokens) or marker.mapper != mapper.day_version(tokens):
                pending.append(day)
        return tuple(pending)

    def records_of(
        self, day: date, concepts: ConceptSet, mapper: ConceptMapper, series: EventSeries
    ) -> tuple[DayRecords, SceneDayContext]:
        """一天的取材料口与它的情境、常态。"""

        context = self.context(day, concepts, series)
        gate = self._recent_given() if self._recent_given is not None else None
        return DayRecords(series, day, mapper, baseline_for=context.baseline_for, recent_given=gate), context

    def plan(
        self,
        series: EventSeries,
        concepts: ConceptSet,
        mapper: ConceptMapper,
        *,
        sealed_before: date | None,
        ready: Callable[[SeriesRecord], bool],
    ) -> tuple[PendingRecord, ...]:
        """该映射的记录，按先后排好。``sealed_before`` 之前的日子已封口（None = 都没封口）；还没封口的日子里的记录要 ``ready``
        答应才算欠着（所在的槽过后留够了余量、它那一天的那一代做完了）。纯读盘。"""

        fresh: list[PendingRecord] = []
        rejudged: list[PendingRecord] = []
        backfill: list[PendingRecord] = []
        for day in self.pending_days(series, mapper):
            sealed = sealed_before is not None and day < sealed_before
            documents = day_documents(self.behavior_tree, series, day)
            records, _context = self.records_of(day, concepts, mapper, series)
            for uri, document in documents.items():
                entry = records.records[uri]
                if not sealed and not ready(entry):
                    continue
                existing = self.hits.read(document.address) if self.hits.exists(document.address) else None
                material = records.material(entry, document)
                if current(existing, mapper, entry, material) or waiting_for_recent(existing, mapper, entry, material):
                    continue
                item = PendingRecord(entry, document, records, existing is not None, sealed)
                (rejudged if item.rejudged else backfill if sealed else fresh).append(item)
        fresh.sort(key=lambda item: (item.entry.started_at, item.entry.uri))
        return (*fresh, *rejudged, *backfill)

    async def map_one(self, item: PendingRecord) -> ConceptHits:
        """映射一条（材料按它自己现取；情境留空）。写不出抛 ``ConceptMapperError``。"""

        material = item.records.material(item.entry, item.document)
        return await map_record(self.hits, item.records.mapper, item.document, item.entry, material)

    async def rejudge_day(
        self, day: date, concepts: ConceptSet, mapper: ConceptMapper, series: EventSeries
    ) -> tuple[int, int]:
        """词表迁移牵到的一天：清掉序列里已经没有的命中记录、重判不作数的那几条（章撤掉，由封口重盖）。返回（重判了几条,
        清掉了几条）。"""

        documents = day_documents(self.behavior_tree, series, day)
        keep = frozenset(document.address.identity_name for document in documents.values())
        stale = {item.address.identity_name for item in self.hits.read_day(day)} - keep
        removed = self.hits.retain_only(day, keep) if stale else ()  # 没有多出来的就不动：没变的日子章留着
        records, _context = await asyncio.to_thread(self.records_of, day, concepts, mapper, series)
        rewritten = 0
        for uri, document in documents.items():
            entry = records.records[uri]
            existing = self.hits.read(document.address) if self.hits.exists(document.address) else None
            material = records.material(entry, document)
            if current(existing, mapper, entry, material):
                continue
            rewritten += 1
            await map_record(self.hits, mapper, document, entry, material)
        return rewritten, len(removed)

    def context(self, day: date, concepts: ConceptSet, series: EventSeries) -> SceneDayContext:
        return SceneDayContext(
            day,
            concepts=concepts,
            hits=self.hits,
            calendar=self.calendar,
            timezone=self.timezone,
            subject=self.subject,
            recent_days=self._config().recent_days,
            rhythms=self._rhythms(day, concepts, series),
            slack_minutes=self._slack(),
        )


__all__ = ["MappingStage", "PendingRecord"]
