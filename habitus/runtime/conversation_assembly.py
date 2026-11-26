"""组合根里会话一侧的组装：会话原文与摘要、会话源与各消费者的回执、会话源的交付、恢复与释放。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from habitus.config import HabitusConfig
from habitus.conversation import (
    AddressWatermarks,
    BehaviorSessionOutputStore,
    ConversationConsumerDelivery,
    ConversationConsumerExecutionFence,
    ConversationConsumerOutcomeStore,
    ConversationConsumerStateInspector,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceLedger,
    ConversationSourceRecovery,
    ConversationSourceReleaser,
    ConversationSourceStore,
    PendingSources,
    SourceIndex,
)
from habitus.conversation.source.salvage import SourceSalvage
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import (
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationRetentionPlanner,
    ConversationSummaryCompactor,
    ConversationSummaryExpander,
    ConversationSummaryGenerator,
    ConversationSummaryReader,
    ConversationSummaryRetirementStore,
    ConversationSummaryService,
    ConversationSummaryStore,
    SQLiteConversationSummaryUseStore,
)
from habitus.memory.retrieval import ConversationSearchContextReader
from habitus.memory.workflow import (
    ConversationMemoryEnqueuer,
    MemoryConversationConsumer,
    MemoryConversationOutputStore,
)
from habitus.model_client import StructuredChatClient
from habitus.runtime.assembly_shared import AssemblyShared
from habitus.runtime.behavior import BehaviorBranches
from habitus.runtime.components import RuntimeConversation
from habitus.runtime.foresight import build_outlet_consumer
from habitus.runtime.foresight_outlet_consumer import OutletObservingConsumer
from habitus.runtime.lifecycle_idle import fully_sealed
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.pipeline_assembly import Downstream
from habitus.runtime.session_lane import BehaviorSessionConsumer, answered_question_tools, session_salvager


@dataclass(frozen=True)
class ConversationParts:
    """会话原文、切段与摘要那一组；检索和记忆任务都从这里取。"""

    journal: ConversationMessageJournal
    retention: ConversationRetentionPlanner
    summaries: ConversationSummaryService
    summary_store: ConversationSummaryStore
    range_summary_store: ConversationRangeSummaryStore
    summary_compactor: ConversationSummaryCompactor
    summary_use: SQLiteConversationSummaryUseStore
    summary_retirements: ConversationSummaryRetirementStore
    summary_expander: ConversationSummaryExpander
    summary_reader: ConversationSummaryReader
    search_context_reader: ConversationSearchContextReader


@dataclass(frozen=True)
class SourceStores:
    """会话源、它的账本与各消费者的回执和输出存储，加上记忆一侧的消费者。"""

    ledger: ConversationSourceLedger
    sources: ConversationSourceStore
    outcomes: ConversationConsumerOutcomeStore
    memory_outputs: MemoryConversationOutputStore
    behavior_session_outputs: BehaviorSessionOutputStore
    memory_consumer: MemoryConversationConsumer


@dataclass(frozen=True)
class SourceDelivery:
    """会话源的交付、协调、恢复与释放，以及交付名单里会话 lane 那一项。"""

    behavior_session_consumer: BehaviorSessionConsumer | OutletObservingConsumer
    inspector: ConversationConsumerStateInspector
    fence: ConversationConsumerExecutionFence
    delivery: ConversationConsumerDelivery
    coordinator: ConversationSourceCoordinator
    recovery: ConversationSourceRecovery
    releaser: ConversationSourceReleaser


def build_conversation(
    config: HabitusConfig, path_lock: PathLock, structured_chat: StructuredChatClient
) -> ConversationParts:
    conversation_config = config.conversation
    journal = ConversationMessageJournal(config.conversation_root, path_lock, config=conversation_config.journal)
    summary_store = ConversationSummaryStore(journal.layout, config=conversation_config.summary)
    summary_generator = ConversationSummaryGenerator(
        structured_chat,
        zone=config.locale.zone(),
        config=conversation_config.summary,
    )
    range_summary_store = ConversationRangeSummaryStore(journal.layout, config=conversation_config.summary)
    range_summary_generator = ConversationRangeSummaryGenerator(
        structured_chat,
        summary_config=conversation_config.summary,
        compaction_config=conversation_config.lifecycle.summary_compaction,
    )
    summary_use = SQLiteConversationSummaryUseStore(
        config.workflow_root / "conversation_summary_use.sqlite3",
        initialize=False,
    )
    summary_compactor = ConversationSummaryCompactor(
        journal,
        summary_store,
        range_summary_store,
        range_summary_generator,
        use_store=summary_use,
        config=conversation_config.lifecycle.summary_compaction,
    )
    summary_retirements = ConversationSummaryRetirementStore(config.workflow_root)
    return ConversationParts(
        journal=journal,
        retention=ConversationRetentionPlanner(conversation_config.segmentation),
        summaries=ConversationSummaryService(summary_store, summary_generator),
        summary_store=summary_store,
        range_summary_store=range_summary_store,
        summary_compactor=summary_compactor,
        summary_use=summary_use,
        summary_retirements=summary_retirements,
        summary_expander=_summary_expander(config, summary_store, range_summary_store),
        summary_reader=ConversationSummaryReader(
            journal,
            summary_compactor,
            config=conversation_config.summary_reader,
            retirement_store=summary_retirements,
        ),
        search_context_reader=ConversationSearchContextReader(
            journal,
            summary_compactor,
            config=config.memory.search_service,
            retirement_filter=summary_retirements,
        ),
    )


def _summary_expander(
    config: HabitusConfig, summary_store: ConversationSummaryStore, range_summary_store: ConversationRangeSummaryStore
) -> ConversationSummaryExpander:
    compaction = config.conversation.lifecycle.summary_compaction
    return ConversationSummaryExpander(
        summary_store,
        range_summary_store,
        max_source_reads=(
            compaction.range_to_archive.max_source_count * (1 + compaction.segment_to_range.max_source_count)
        ),
    )


def build_sources(
    config: HabitusConfig, journal: ConversationMessageJournal, enqueuer: ConversationMemoryEnqueuer
) -> SourceStores:
    root = config.conversation_root
    source_config = config.conversation.source
    ledger = ConversationSourceLedger(
        SourceIndex(
            root,
            max_addresses=source_config.max_source_addresses,
            max_markers_per_address=source_config.max_markers_per_address,
        ),
        PendingSources(root, max_entries=source_config.max_pending_sources),
        AddressWatermarks(root),
    )
    memory_outputs = MemoryConversationOutputStore(
        root,
        max_files_per_source=source_config.max_output_files_per_consumer,
        max_file_bytes=source_config.max_memory_output_bytes,
    )
    return SourceStores(
        ledger=ledger,
        sources=ConversationSourceStore(root, max_file_bytes=source_config.max_envelope_bytes, ledger=ledger),
        outcomes=ConversationConsumerOutcomeStore(root, max_file_bytes=source_config.max_outcome_bytes),
        memory_outputs=memory_outputs,
        behavior_session_outputs=BehaviorSessionOutputStore(
            root,
            max_files_per_source=source_config.max_output_files_per_consumer,
            max_file_bytes=source_config.max_behavior_session_output_bytes,
        ),
        memory_consumer=MemoryConversationConsumer(enqueuer, journal, memory_outputs),
    )


def source_settled(stores: SourceStores) -> Callable[[str], bool]:
    """这份会话源的 Behavior 到终态了没有：Outcome 只在终态才建；整份已释放（信封都删了）也算。"""

    def settled(source_id: str) -> bool:
        if stores.outcomes.read(source_id, ConversationSourceConsumer.BEHAVIOR_SESSION) is not None:
            return True
        return stores.sources.read(source_id) is None

    return settled


def build_source_delivery(
    config: HabitusConfig,
    stores: SourceStores,
    journal: ConversationMessageJournal,
    behavior: BehaviorBranches | None,
    downstream: Downstream,
    shared: AssemblyShared,
) -> SourceDelivery:
    source_config = config.conversation.source
    session_consumer = _session_consumer(stores, behavior, downstream, shared)
    inspector = ConversationConsumerStateInspector(stores.outcomes)
    fence = ConversationConsumerExecutionFence(
        shared.path_lock,
        ttl_seconds=source_config.execution_lock_ttl_seconds,
        heartbeat_interval_seconds=source_config.execution_lock_heartbeat_seconds,
        wait_seconds=source_config.execution_lock_wait_seconds,
    )
    delivery = ConversationConsumerDelivery(
        stores.sources,
        stores.outcomes,
        inspector,
        fence,
        {
            ConversationSourceConsumer.MEMORY: stores.memory_consumer,
            ConversationSourceConsumer.BEHAVIOR_SESSION: session_consumer,
        },
        observer=shared.observer,
    )
    salvage = _salvage(config, behavior, shared)
    recovery = ConversationSourceRecovery(
        stores.sources,
        delivery,
        batch_size=source_config.recovery_batch_size,
        retries=retry_ledger(
            config.conversation_root, owner="conversation-source-recovery", config=config.lifecycle.retry
        ),
        salvage=salvage,
        orphan_grace=timedelta(seconds=source_config.orphan_grace_seconds),
    )
    releaser = ConversationSourceReleaser(
        stores.sources,
        stores.outcomes,
        stores.ledger,
        delivery.consumers,
        shared.path_lock.lock_store,
        lock_ttl_seconds=source_config.execution_lock_ttl_seconds,
        fully_sealed=fully_sealed(journal),
        salvage=salvage,
    )
    return SourceDelivery(
        behavior_session_consumer=session_consumer,
        inspector=inspector,
        fence=fence,
        delivery=delivery,
        coordinator=ConversationSourceCoordinator(stores.sources, delivery),
        recovery=recovery,
        releaser=releaser,
    )


def _session_consumer(
    stores: SourceStores, behavior: BehaviorBranches | None, downstream: Downstream, shared: AssemblyShared
) -> BehaviorSessionConsumer | OutletObservingConsumer:
    # 行为侧没启用时消费者仍要注册（交付层要求名单里每一项都有实现），它把每份源落成"跳过"。
    consumer = BehaviorSessionConsumer(
        None if behavior is None else behavior.session_lane,
        stores.behavior_session_outputs,
        question_tools=answered_question_tools(shared.adapters),
    )
    if downstream.outlet is None or not downstream.foresights:
        return consumer
    # 提醒出口：同一份会话先过出口（认出 agent 的提问、写「说了」与固定选项的回话），再进会话 lane
    return build_outlet_consumer(downstream.outlet, consumer, observer=shared.observer, host_of=shared.adapters.host)


def _salvage(config: HabitusConfig, behavior: BehaviorBranches | None, shared: AssemblyShared) -> SourceSalvage:
    # 行为侧的账坏了的会话源：逐轮补救（没写过的轮落「没读懂」、进换代重试的堆，裁定 67、68）
    return SourceSalvage(
        config.conversation_root,
        salvagers={
            ConversationSourceConsumer.BEHAVIOR_SESSION: session_salvager(
                None if behavior is None else behavior.session_lane,
                question_tools=answered_question_tools(shared.adapters),
            )
        },
    )


def runtime_conversation(
    conversation: ConversationParts, stores: SourceStores, delivery: SourceDelivery
) -> RuntimeConversation:
    return RuntimeConversation(
        sources=stores.sources,
        source_outcomes=stores.outcomes,
        memory_outputs=stores.memory_outputs,
        behavior_session_outputs=stores.behavior_session_outputs,
        behavior_session_consumer=delivery.behavior_session_consumer,
        source_inspector=delivery.inspector,
        source_fence=delivery.fence,
        source_delivery=delivery.delivery,
        source_coordinator=delivery.coordinator,
        source_recovery=delivery.recovery,
        source_ledger=stores.ledger,
        source_releaser=delivery.releaser,
        journal=conversation.journal,
        retention=conversation.retention,
        summaries=conversation.summaries,
        summary_compactor=conversation.summary_compactor,
        summary_reader=conversation.summary_reader,
        summary_use=conversation.summary_use,
        summary_expander=conversation.summary_expander,
    )


__all__ = [
    "ConversationParts",
    "SourceDelivery",
    "SourceStores",
    "build_conversation",
    "build_source_delivery",
    "build_sources",
    "runtime_conversation",
    "source_settled",
]
