"""组合根里记忆任务与生命周期的组装：任务存储、入队、MemoryWorker，以及 LifecycleWorker 和它的维护参与者。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.config import HabitusConfig
from habitus.infrastructure.lifecycle import CapacitySnapshotStore, LifecycleParticipant
from habitus.memory.workflow import (
    ConversationLifecycleManager,
    ConversationMemoryEnqueuer,
    MemoryJobRunner,
    MemoryJobStore,
)
from habitus.runtime.assembly_shared import AssemblyShared
from habitus.runtime.behavior import BehaviorBranches
from habitus.runtime.components import RuntimeWorkflow
from habitus.runtime.conversation_assembly import ConversationParts, SourceDelivery, SourceStores
from habitus.runtime.lifecycle import LifecycleWorker
from habitus.runtime.lifecycle_downstream import downstream_participants
from habitus.runtime.lifecycle_idle import ConversationIdleSealParticipant
from habitus.runtime.lifecycle_memory import ConversationChainParticipant, MemoryL2Participant, conversation_lease_key
from habitus.runtime.lifecycle_sources import ConversationSourceParticipant, ConversationSourceRecoveryParticipant
from habitus.runtime.memory_assembly import MemoryParts
from habitus.runtime.pipeline_assembly import Downstream
from habitus.runtime.worker import MemoryWorker


@dataclass(frozen=True)
class WorkflowParts:
    """记忆任务那一组：任务存储、会话生命周期、入队器、Runner 与单并发的 MemoryWorker。"""

    jobs: MemoryJobStore
    conversation_lifecycle: ConversationLifecycleManager
    enqueuer: ConversationMemoryEnqueuer
    runner: MemoryJobRunner
    worker: MemoryWorker


def build_memory_workflow(
    config: HabitusConfig, memory: MemoryParts, conversation: ConversationParts, shared: AssemblyShared
) -> WorkflowParts:
    workflow_config = config.workflow
    jobs = MemoryJobStore(
        config.workflow_root, shared.path_lock, memory_root=memory.tree.root, config=workflow_config.jobs
    )
    runner = MemoryJobRunner(
        jobs,
        conversation.journal,
        memory.editor,
        memory.semantic_refresher,
        conversation.summaries,
        memory.recall_lifecycle,
        observer=shared.observer,
    )
    return WorkflowParts(
        jobs=jobs,
        conversation_lifecycle=ConversationLifecycleManager(
            conversation.summary_compactor,
            conversation.journal,
            conversation.summary_store,
            conversation.range_summary_store,
            jobs,
            conversation.summary_retirements,
            summary_config=config.conversation.lifecycle.summary_compaction,
            workflow_config=workflow_config.lifecycle,
        ),
        enqueuer=ConversationMemoryEnqueuer(conversation.journal, jobs, retention_planner=conversation.retention),
        runner=runner,
        worker=MemoryWorker(
            runner,
            workflow_config.worker,
            observer=shared.observer,
            span_controller=shared.managed_observability,
        ),
    )


def build_lifecycle(
    config: HabitusConfig,
    memory: MemoryParts,
    workflow: WorkflowParts,
    stores: SourceStores,
    delivery: SourceDelivery,
    downstream: Downstream,
    behavior: BehaviorBranches | None,
    shared: AssemblyShared,
) -> RuntimeWorkflow:
    conversation_config = config.conversation
    lifecycle_lock_store = workflow.conversation_lifecycle.journal.path_lock.lock_store
    lifecycle_worker = LifecycleWorker(
        _participants(config, memory, workflow, delivery, downstream, behavior),
        config.lifecycle.worker,
        lock_store=lifecycle_lock_store,
        lease_key=conversation_lease_key(workflow.conversation_lifecycle.journal),
        # 闲置封存每一拍都跑，其余按 lifecycle.worker 的周期（裁定 107）
        frequent=(
            ConversationIdleSealParticipant(
                workflow.enqueuer,
                delivery.releaser,
                shared.path_lock,
                conversation_config.lifecycle,
                lock_ttl_seconds=conversation_config.source.execution_lock_ttl_seconds,
                wake_memory=workflow.worker.wake,
            ),
        ),
        frequent_interval_seconds=conversation_config.lifecycle.idle_scan_interval_seconds,
        capacity=CapacitySnapshotStore(workflow.jobs.root),
        observer=shared.observer,
    )
    return RuntimeWorkflow(
        jobs=workflow.jobs,
        enqueuer=workflow.enqueuer,
        conversation_consumer=stores.memory_consumer,
        lifecycle=workflow.conversation_lifecycle,
        runner=workflow.runner,
        worker=workflow.worker,
        lifecycle_worker=lifecycle_worker,
    )


def _participants(
    config: HabitusConfig,
    memory: MemoryParts,
    workflow: WorkflowParts,
    delivery: SourceDelivery,
    downstream: Downstream,
    behavior: BehaviorBranches | None,
) -> tuple[LifecycleParticipant, ...]:
    participants: list[LifecycleParticipant] = [
        ConversationChainParticipant(
            workflow.conversation_lifecycle,
            config.conversation.lifecycle,
            lock_store=workflow.conversation_lifecycle.journal.path_lock.lock_store,
            lease_ttl_seconds=config.lifecycle.worker.lease_ttl_seconds,
        ),
        MemoryL2Participant(memory.lifecycle, memory.semantic_refresher),
        ConversationSourceParticipant(
            delivery.releaser, batch_addresses=config.conversation.source.release_batch_addresses
        ),
        ConversationSourceRecoveryParticipant(
            delivery.recovery, delivery.coordinator, wake_memory=workflow.worker.wake
        ),
    ]
    participants.extend(
        downstream_participants(config, downstream.pipelines, behavior, source_retries=delivery.recovery.retries)
    )
    return tuple(participants)


__all__ = ["WorkflowParts", "build_lifecycle", "build_memory_workflow"]
