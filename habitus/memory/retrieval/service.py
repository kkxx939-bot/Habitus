"""面向 Agent 的只读记忆检索：清单 → 挑 → 找 → 判，最多再走一轮。

能查三处：四类卡片、每一轮的记录、会话摘要。整次查询（含"被用到"的记账）有一个总时间上限，到点就把已经
找到的交出去，不返回空。模型不可用时只交出档案并写明原因，不拿关键词匹配顶替"挑"和"判"。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, tzinfo
from typing import Protocol

from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.memory.conversation import ConversationAddress, ConversationSummaryMatch, ConversationSummaryReader
from habitus.memory.document import MemoryDocument
from habitus.memory.model import MemoryAddress, MemoryKind
from habitus.memory.recall import MemoryRecallLifecycle, MemoryRecallTarget, MemoryTemperature
from habitus.memory.retrieval.acknowledgement import MemoryColdProbeExpander, account_use
from habitus.memory.retrieval.assembler import MemoryContextAssembler
from habitus.memory.retrieval.catalog import CatalogCard, MemoryCatalog, MemoryCatalogReader
from habitus.memory.retrieval.context import ConversationSearchContext, ConversationSearchContextReader
from habitus.memory.retrieval.model import (
    MemoryMatchedMemory,
    MemorySearchDegradation,
    MemorySearchDegradationStage,
    MemorySearchResult,
    MemorySearchServiceConfig,
)
from habitus.memory.retrieval.records import (
    BehaviorRecord,
    BehaviorRecordSource,
    days_in_range,
    keyword_hits,
    search_records,
)
from habitus.memory.retrieval.relations import RelationExpander
from habitus.memory.retrieval.request import MemorySearchRequestNormalizer
from habitus.memory.retrieval.steps import SearchPlan, SearchSteps, SourceQuery
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.uri import MemoryURI
from habitus.model_client import BRIEFS, StructuredChatClient, model_failure

_PROFILE_URI = MemoryURI.from_address(MemoryAddress.profile())


class ConversationSummaryUseRecorder(Protocol):
    def record_use(self, references: tuple, *, used_at: datetime) -> tuple: ...


class ConversationSummaryFallbackExpander(Protocol):
    """把 Range/Archive 摘要按来源链临时补上子摘要语义（``ConversationSummaryExpander`` 满足它）。"""

    def expand(self, match: ConversationSummaryMatch, *, max_chars: int) -> ConversationSummaryMatch: ...


@dataclass
class _Found:
    """一次查询里找到的材料；编号在整次查询里不变，给模型的"判"只写编号。"""

    now: datetime
    cards: dict[str, MemoryDocument] = field(default_factory=dict)
    card_temperatures: dict[MemoryURI, MemoryTemperature | None] = field(default_factory=dict)
    card_targets: dict[MemoryURI, MemoryRecallTarget] = field(default_factory=dict)
    records: dict[str, BehaviorRecord] = field(default_factory=dict)
    full_records: set[str] = field(default_factory=set)
    summaries: dict[str, ConversationSummaryMatch] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    degradations: list[MemorySearchDegradation] = field(default_factory=list)
    cold_picked: bool = False
    rounds: int = 0
    used: tuple[str, ...] | None = None
    model_down: bool = False
    timed_out: bool = False

    def degrade(self, stage: MemorySearchDegradationStage, error: BaseException | str) -> None:
        """每个阶段只记第一次的错误类型。"""

        name = error if isinstance(error, str) else type(error).__name__
        if all(item.stage is not stage for item in self.degradations):
            self.degradations.append(MemorySearchDegradation(stage, name))

    def card_id(self, document: MemoryDocument) -> str:
        uri = MemoryURI.from_address(document.address)
        for key, value in self.cards.items():
            if MemoryURI.from_address(value.address) == uri:
                return key
        key = f"c{len(self.cards) + 1}"
        self.cards[key] = document
        return key

    def record_id(self, record: BehaviorRecord, *, full: bool) -> str:
        for key, value in self.records.items():
            if value.address == record.address:
                if full:
                    self.full_records.add(key)
                return key
        key = f"r{len(self.records) + 1}"
        self.records[key] = record
        if full:
            self.full_records.add(key)
        return key

    def summary_id(self, match: ConversationSummaryMatch) -> str:
        for key, value in self.summaries.items():
            if value.reference.identity == match.reference.identity:
                return key
        key = f"s{len(self.summaries) + 1}"
        self.summaries[key] = match
        return key


class SearchService:
    """按"清单 → 挑 → 找 → 判"交出完整、可溯源、受预算约束的记忆上下文。"""

    def __init__(
        self,
        *,
        catalog: MemoryCatalogReader,
        snapshot_reader: MemorySnapshotReader,
        client: StructuredChatClient,
        recall_lifecycle: MemoryRecallLifecycle,
        conversation_context: ConversationSearchContextReader,
        summaries: ConversationSummaryReader,
        records: BehaviorRecordSource,
        zone: tzinfo,
        config: MemorySearchServiceConfig | None = None,
        cold_probe_expander: MemoryColdProbeExpander | None = None,
        summary_use_recorder: ConversationSummaryUseRecorder | None = None,
        summary_expander: ConversationSummaryFallbackExpander | None = None,
        clock: Callable[[], datetime] | None = None,
        observer: Observer | None = None,
    ) -> None:
        resolved = config or MemorySearchServiceConfig()
        if conversation_context.config != resolved:
            raise ValueError("SearchService and Conversation context reader must share one search config")
        self.catalog = catalog
        self.snapshot_reader = snapshot_reader
        self.recall_lifecycle = recall_lifecycle
        self.conversation_context = conversation_context
        self.summaries = summaries
        self.records = records
        self.zone = zone
        self.config = resolved
        self.steps = SearchSteps(client, config=resolved)
        self.assembler = MemoryContextAssembler(zone=zone, config=resolved)
        self.relation_expander = RelationExpander(snapshot_reader=snapshot_reader, config=resolved)
        self.cold_probe_expander = cold_probe_expander
        self.summary_use_recorder = summary_use_recorder
        self.summary_expander = summary_expander
        self.clock = clock or (lambda: datetime.now(UTC))
        self.observer = observer or NullObserver()
        self.request_normalizer = MemorySearchRequestNormalizer(resolved)

    async def search(
        self,
        query: str,
        *,
        conversation: ConversationAddress | None = None,
        limit: int | None = None,
        kinds: tuple[MemoryKind, ...] = (),
    ) -> MemorySearchResult:
        request = self.request_normalizer.normalize(query, limit=limit, kinds=kinds)
        started = time.monotonic()
        deadline = started + self.config.time_limit_seconds
        now = self._timestamp()
        found = _Found(now)
        try:
            async with asyncio.timeout(max(0.0, deadline - self.config.accounting_reserve_seconds - time.monotonic())):
                await self._look(request.query, request.kinds, conversation, now, found)
        except TimeoutError:
            found.timed_out = True
            found.degrade(MemorySearchDegradationStage.TIME_LIMIT, "TimeoutError")
            found.notes.append("查资料到了时间上限，交出的是到那时已经找到的全部材料。")
        selected = (
            (await self._profile_only(found, deadline), (), ())
            if found.model_down
            else self._selected(found, found.used, request.limit)
        )
        result = await self._hand_over(request.query, request.kinds, selected, found, now, deadline)
        self.observer.record(_search_event(result, started))
        return result

    async def _look(
        self,
        query: str,
        kinds: tuple[MemoryKind, ...],
        conversation: ConversationAddress | None,
        now: datetime,
        found: _Found,
    ) -> None:
        """清单 → 挑 → 找 → 判，不够最多再走一轮；挑这一步模型不可用就记下，由调用方只交档案。"""

        context = await self._conversation(conversation, found.degrade)
        catalog = await self._catalog(kinds, now, found, found.degrade)
        background, listing = self._background(now), catalog.render()
        try:
            plan = await self.steps.pick(background=background, query=query, catalog=listing, context=context)
        except Exception as exc:
            found.degrade(MemorySearchDegradationStage.PICK, exc)
            found.model_down = True
            found.notes.append(_model_down_note(exc))
            return
        await self._gather(plan, catalog, found, found.degrade)
        found.rounds = 1
        while True:
            last = found.rounds >= self.config.max_rounds
            try:
                verdict = await self.steps.judge(
                    background=background,
                    query=query,
                    catalog=listing,
                    context=context,
                    materials=self._materials(found),
                    last=last,
                )
            except Exception as exc:
                found.degrade(MemorySearchDegradationStage.JUDGE, exc)
                found.model_down = True  # 模型不可用不分是"挑"还是"判"：只交出档案
                found.notes.append(_model_down_note(exc))
                return
            found.used = verdict.use
            if verdict.enough or last or verdict.next is None or verdict.next.empty:
                return
            await self._gather(verdict.next, catalog, found, found.degrade)
            found.rounds += 1

    async def _hand_over(
        self,
        query: str,
        kinds: tuple[MemoryKind, ...],
        selected: tuple[
            tuple[MemoryMatchedMemory, ...], tuple[BehaviorRecord, ...], tuple[ConversationSummaryMatch, ...]
        ],
        found: _Found,
        now: datetime,
        deadline: float,
    ) -> MemorySearchResult:
        """补一跳关系、在剩下的时间里记账，再拼成上下文。"""

        memories, records, summaries = selected
        memories = await self._relations(memories, deadline)
        accounting = await account_use(
            memories,
            targets=found.card_targets,
            used_at=now,
            seconds=max(0.0, deadline - time.monotonic() - 0.05),
            recall_lifecycle=self.recall_lifecycle,
            expander=self.cold_probe_expander,
        )
        if accounting.skipped:
            found.degrade(MemorySearchDegradationStage.RECALL_LIFECYCLE, accounting.error_type or "TimeoutError")
            self.observer.record(_skipped_accounting(accounting.skipped, accounting.error_type))
        await self._record_summary_use(summaries, now, deadline, found.degrade)
        assembly = self.assembler.assemble(
            accounting.memories, records=records, summaries=summaries, notes=tuple(found.notes)
        )
        return MemorySearchResult(
            query=query,
            kinds=kinds,
            memories=assembly.memories,
            records=assembly.records,
            summaries=assembly.summaries,
            context=assembly.context,
            budget_exhausted=assembly.budget_exhausted,
            rounds=found.rounds,
            timed_out=found.timed_out,
            notes=tuple(found.notes),
            degradations=tuple(found.degradations),
        )

    async def _conversation(
        self,
        address: ConversationAddress | None,
        degrade: Callable[[MemorySearchDegradationStage, BaseException | str], None],
    ) -> ConversationSearchContext | None:
        if address is None:
            return None
        try:
            return await asyncio.to_thread(self.conversation_context.read, address)
        except Exception as exc:
            degrade(MemorySearchDegradationStage.CONVERSATION_CONTEXT, exc)
            return None

    async def _catalog(
        self,
        kinds: tuple[MemoryKind, ...],
        now: datetime,
        found: _Found,
        degrade: Callable[[MemorySearchDegradationStage, BaseException | str], None],
    ) -> MemoryCatalog:
        """读各目录的概览拼清单；读不出就按没有卡片清单继续并说明。"""

        def recall_failed(error: BaseException) -> None:
            degrade(MemorySearchDegradationStage.RECALL_LIFECYCLE, error)

        try:
            catalog = await asyncio.to_thread(self.catalog.read, kinds=kinds, now=now, on_recall_failure=recall_failed)
        except Exception as exc:
            degrade(MemorySearchDegradationStage.CATALOG, exc)
            found.notes.append("记忆卡片清单这次读不出来（概览损坏），只查了每一轮的记录和会话摘要。")
            return MemoryCatalog((), ())
        found.notes.extend(catalog.notes)
        return catalog

    def _background(self, now: datetime) -> str:
        local = now.astimezone(self.zone)
        offset = local.strftime("%z")
        return (
            f"今天是 {local:%Y-%m-%d}，时间都是当地时间（UTC{offset[:3]}:{offset[3:]}）。资料有三处：\n"
            "1. 记忆卡片：从过去对话里整理出的长期结论（档案、偏好、人和项目、工具），下面给出清单，每张一行：路径 — 说明。\n"
            "2. 每一轮的记录：用户每和助手说一轮话就有一条，有时刻、名字、目标、概要、步骤；同类事情会反复出现很多次。"
            "可以按日期范围列出（只给时刻和名字），也可以按关键词搜（给整条内容；关键词是子串匹配、任一命中即算）。\n"
            "3. 会话摘要：每段对话一份，比每一轮的记录多一些前后经过和收尾时的状态；可以按日期范围和关键词找。"
        )

    async def _gather(
        self,
        plan: SearchPlan,
        catalog: MemoryCatalog,
        found: _Found,
        degrade: Callable[[MemorySearchDegradationStage, BaseException | str], None],
    ) -> None:
        cards: list[CatalogCard] = []
        for item in plan.read:
            if item in found.records:
                found.full_records.add(item)
                continue
            card = catalog.find(item)
            if card is None:
                found.notes.append(f"清单里没有 {item}，没有读。")
                continue
            if card.temperature is MemoryTemperature.COLD_2:
                if found.cold_picked:
                    continue
                found.cold_picked = True
            cards.append(card)
        if plan.card_keywords and catalog.unlisted_count:
            matched = await asyncio.to_thread(
                self.catalog.match_unlisted, catalog, plan.card_keywords, self.config.max_reads_per_round, now=found.now
            )
            for card in matched:
                if card.temperature is MemoryTemperature.COLD_2:
                    if found.cold_picked:
                        continue
                    found.cold_picked = True
                cards.append(card)
        await self._read_cards(tuple(cards), found)
        if plan.records is not None:
            try:
                await self._find_records(plan.records, found)
            except Exception as exc:
                degrade(MemorySearchDegradationStage.RECORDS, exc)
        if plan.summaries is not None:
            try:
                await self._find_summaries(plan.summaries, found)
            except Exception as exc:
                degrade(MemorySearchDegradationStage.SUMMARIES, exc)

    async def _read_cards(self, cards: tuple[CatalogCard, ...], found: _Found) -> None:
        if not cards:
            return
        unique = tuple(dict.fromkeys(cards))
        batch = await asyncio.to_thread(self.snapshot_reader.read_many, tuple(card.uri for card in unique))
        for card in unique:
            snapshot = batch.get(str(card.uri))
            if snapshot is None or not snapshot.exists or snapshot.value is None:
                found.notes.append(f"{card.path} 读的时候已经不在了。")
                continue
            document = snapshot.value
            if card.temperature is MemoryTemperature.COLD_2 and self.cold_probe_expander is not None:
                document = await asyncio.to_thread(self.cold_probe_expander.expand_for_probe, document)
            found.card_id(document)
            found.card_temperatures[card.uri] = card.temperature
            found.card_targets[card.uri] = MemoryRecallTarget(
                card.uri, snapshot.value.metadata.revision, snapshot.value.metadata.created_at
            )

    async def _find_records(self, query: SourceQuery, found: _Found) -> None:
        available = await asyncio.to_thread(self.records.days)
        days = days_in_range(available, query.date_from, query.date_to, max_days=self.config.max_record_days)
        if not days:
            return
        batch = await asyncio.to_thread(self.records.read_days, days)
        if batch.failed_days:
            found.notes.append(
                "这些日子的记录这次没读出来：" + "、".join(day.isoformat() for day in batch.failed_days) + "。"
            )
        records = tuple(sorted(batch.records, key=lambda record: (record.started_at, record.address)))
        if query.empty:  # 没给日期也没给关键词：列最近的
            for record in records[-self.config.max_record_list :]:
                found.record_id(record, full=False)
        elif query.date_from is not None or query.date_to is not None:
            for record in records[: self.config.max_record_list]:
                found.record_id(record, full=False)
        if len(records) > self.config.max_record_list and not query.keywords:
            where = "最近" if query.empty else "前"
            found.notes.append(
                f"{days[0]} 到 {days[-1]} 共 {len(records)} 条记录，只列出了{where} {self.config.max_record_list} 条。"
            )
        if query.keywords:
            hits, total = search_records(records, query.keywords, limit=self.config.max_record_hits)
            for record in hits:
                found.record_id(record, full=True)
            if total > len(hits):
                found.notes.append(f"关键词在记录里命中 {total} 条，只给了命中词最多的 {len(hits)} 条。")

    async def _find_summaries(self, query: SourceQuery, found: _Found) -> None:
        matches = await asyncio.to_thread(self.summaries.walk)
        selected: list[tuple[int, ConversationSummaryMatch]] = []
        for match in matches:
            if not _overlaps(match, query.date_from, query.date_to, self.zone):
                continue
            hits = keyword_hits(match.content, query.keywords) if query.keywords else 1
            if hits:
                selected.append((hits, match))
        selected.sort(key=lambda item: (-item[0], -item[1].started_at.timestamp(), item[1].reference.identity))
        for _hits, match in selected[: self.config.max_summary_hits]:
            if self.summary_expander is not None:
                match = await asyncio.to_thread(
                    self.summary_expander.expand, match, max_chars=self.summaries.config.max_record_chars
                )
            found.summary_id(match)

    def _materials(self, found: _Found) -> str:
        blocks: list[str] = []
        for key, document in found.cards.items():
            updated = document.metadata.updated_at.astimezone(self.zone)
            blocks.append(
                f"[{key}] 卡片 {MemoryURI.from_address(document.address)}（更新于 {updated:%Y-%m-%d %H:%M}）\n"
                f"{document.markdown_body.strip()}"
            )
        for key, record in found.records.items():
            blocks.append(f"[{key}] 记录 " + (record.text() if key in found.full_records else record.line()))
        for key, match in found.summaries.items():
            started = match.started_at.astimezone(self.zone)
            ended = match.ended_at.astimezone(self.zone)
            blocks.append(f"[{key}] 会话摘要（{started:%Y-%m-%d %H:%M} 到 {ended:%Y-%m-%d %H:%M}）\n{match.content}")
        text = "\n\n".join(blocks) or "（什么也没找到）"
        if found.notes:
            text += "\n\n说明：\n" + "\n".join(f"- {note}" for note in found.notes)
        if len(text) > self.config.max_material_chars:
            text = text[: self.config.max_material_chars - 20].rstrip() + "\n（后面的材料超出上限，没有给出）"
        return text

    def _selected(
        self,
        found: _Found,
        used: tuple[str, ...] | None,
        limit: int,
    ) -> tuple[tuple[MemoryMatchedMemory, ...], tuple[BehaviorRecord, ...], tuple[ConversationSummaryMatch, ...]]:
        """判断写了用哪些就交哪些；没走完判断（到点或判断失败）就交全部找到的。卡片按 ``limit`` 截。"""

        keys = list(used) if used is not None else [*found.cards, *found.records, *found.summaries]
        memories: list[MemoryMatchedMemory] = []
        records: list[BehaviorRecord] = []
        summaries: list[ConversationSummaryMatch] = []
        for key in dict.fromkeys(keys):
            if key in found.cards and len(memories) < limit:
                document = found.cards[key]
                uri = MemoryURI.from_address(document.address)
                memories.append(MemoryMatchedMemory(document, found.card_temperatures.get(uri)))
            elif key in found.records:
                records.append(found.records[key])
            elif key in found.summaries:
                summaries.append(found.summaries[key])
        return tuple(memories), tuple(records), tuple(summaries)

    async def _profile_only(self, found: _Found, deadline: float) -> tuple[MemoryMatchedMemory, ...]:
        """只交出档案；读档案也在总时间上限里，来不及就什么也不交。"""

        remaining = deadline - time.monotonic() - self.config.accounting_reserve_seconds / 2
        try:
            snapshot = await asyncio.wait_for(asyncio.to_thread(self.snapshot_reader.read, _PROFILE_URI), remaining)
        except Exception:
            return ()
        if not snapshot.exists or snapshot.value is None:
            return ()
        document = snapshot.value
        found.card_targets[_PROFILE_URI] = MemoryRecallTarget(
            _PROFILE_URI, document.metadata.revision, document.metadata.created_at
        )
        return (MemoryMatchedMemory(document),)

    async def _relations(
        self, memories: tuple[MemoryMatchedMemory, ...], deadline: float
    ) -> tuple[MemoryMatchedMemory, ...]:
        """一跳关系也在时间上限里补：来不及或读出错就不补。"""

        remaining = deadline - time.monotonic() - self.config.accounting_reserve_seconds / 2
        if not memories or remaining <= 0:
            return memories
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.relation_expander.expand, memories), remaining)
        except Exception:
            return memories

    async def _record_summary_use(
        self,
        summaries: tuple[ConversationSummaryMatch, ...],
        now: datetime,
        deadline: float,
        degrade: Callable[[MemorySearchDegradationStage, BaseException | str], None],
    ) -> None:
        remaining = deadline - time.monotonic() - 0.02
        if self.summary_use_recorder is None or not summaries:
            return
        if remaining <= 0:
            degrade(MemorySearchDegradationStage.SUMMARIES, "TimeoutError")
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.summary_use_recorder.record_use,
                    tuple(item.reference for item in summaries),
                    used_at=now,
                ),
                remaining,
            )
        except Exception as exc:
            degrade(MemorySearchDegradationStage.SUMMARIES, exc)

    def _timestamp(self) -> datetime:
        timestamp = self.clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("memory search clock must return a timezone-aware datetime")
        return timestamp.astimezone(UTC)


def _overlaps(match: ConversationSummaryMatch, date_from: date | None, date_to: date | None, zone: tzinfo) -> bool:
    start = match.started_at.astimezone(zone).date()
    end = match.ended_at.astimezone(zone).date()
    return (date_to is None or start <= date_to) and (date_from is None or end >= date_from)


def _skipped_accounting(skipped: int, error_type: str | None) -> ObservationEvent:
    return ObservationEvent(
        category="retrieval",
        operation="use_accounting",
        status=ObservationStatus.DEGRADED,
        duration_seconds=0.0,
        attributes={"skipped": skipped, "error_type": error_type or "TimeoutError"},
    )


def _search_event(result: MemorySearchResult, started: float) -> ObservationEvent:
    return ObservationEvent(
        category="retrieval",
        operation="search",
        status=ObservationStatus.DEGRADED if result.degradations else ObservationStatus.SUCCESS,
        duration_seconds=max(0.0, time.monotonic() - started),
        attributes={
            "rounds": result.rounds,
            "memories": len(result.memories),
            "records": len(result.records),
            "summaries": len(result.summaries),
            "timed_out": result.timed_out,
            "degradations": len(result.degradations),
            "result_count": len(result.memories) + len(result.records) + len(result.summaries),
        },
    )


def _model_down_note(error: BaseException) -> str:
    failure = model_failure(error)
    kind = None if failure is None else failure.cause_kind
    if kind is not None:
        return f"查资料用的模型这次没能用：{BRIEFS[kind]}所以只交出了档案。"
    return "查资料用的模型这次没给出能用的答复，所以只交出了档案。"


__all__ = ["ConversationSummaryFallbackExpander", "ConversationSummaryUseRecorder", "SearchService"]
