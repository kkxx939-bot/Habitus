"""语义树与换代在组合根的组装（语义树新方案 ``13`` 第 11 步另一半；实时化方案阶段 1）。

一支一份 ``SceneNightlyRun``：词表口是行为树那一支的词表（``VocabularyCatalog``），概念集、命中记录、关系表、先验建议都住
这一支的语义树根 ``<scene_root>/<lane>``（裁定 42：``scene/<lane>/``，两支各一整套，不共用任何东西）；模型触点（先验、调节条件）走同一个 ``StructuredChatClient``；关系检验的时间尺子只认
``config.temporal``（``relation_config``）。未启用时返回 None。

换代的串法（重读放假数据 → 重试收尾 → 结算 → 读序列 → 数在用的词 → 语义树 → 机会表 → 单元历史 → 压实 → 当天的日子预期 →
完成标记）在 ``runtime.generation.Generation``，由 ``build_generation`` 装好、``build_generation_runner`` 包成换代的那一眼
（``runtime.generation_runner.GenerationRunner``：连续定稿到昨天、预测拍水位过了余量后做今天这一代），交给预测 worker
每槽先调（``runtime.foresight_worker``，同一次预测根租约之内先换代、再做预测拍）。完成标记、每代报告与
trace 都在这一支的 ``<scene_root>/<lane>/generation`` 下。换代里没有模型调用：调模型的活（出错重试、物理 lane 未封口判断的选词、
先验的补问）由模型队列做（``runtime.model_queue``，``build_model_queue``
装好、交给 ``ModelQueueWorker``）；两边共用一份 ``ModelQueueState``——换代读"队列是不是在整体退避""各类还欠着几项"，
队列读"换代是不是在做"——与一把语义树写锁（``runtime.scene_lock``）。这几样都是一支一份：两支的换代、队列、锁、租约、
完成标记各按自己的根派生，一支的失败、退避只算这一支的（``assemble_branch``）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from pathlib import Path

from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.retry import RetryStage
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.conditions import DayTypeCalendar
from habitus.config import HabitusConfig
from habitus.foundation.observability import Observer
from habitus.infrastructure.store.contracts import LockStore
from habitus.model_client import StructuredChatClient
from habitus.runtime.behavior import BehaviorRuntimeComponents, retry_stage_of
from habitus.runtime.conditions import SharedCalendar, local_calendar
from habitus.runtime.evidence import evidence_of
from habitus.runtime.foresight import ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.foresight_worker import ForesightWorker, SlotBranch
from habitus.runtime.generation import Generation
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_marks import GenerationMarks
from habitus.runtime.generation_report import GenerationReports
from habitus.runtime.generation_runner import GenerationRunner, GenerationSettings
from habitus.runtime.generation_trace import ModelCallTracer, SpanBackend, TraceFiles
from habitus.runtime.generation_vocabulary import VocabularyCounter
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.model_queue import ModelQueue, ModelQueueState, QueueJournal, QueueSettings
from habitus.runtime.model_queue_worker import ModelQueueWorker
from habitus.runtime.pipeline import ScenePipeline
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_config import relation_config
from habitus.runtime.scene_lock import SceneWriteLock, unlocked
from habitus.runtime.scene_night import SceneNightlyRun
from habitus.runtime.scene_vocabulary import VocabularyCatalog, kind_resolution
from habitus.runtime.unsealed import UnsealedFromJudgements, unreduced_gaps
from habitus.scene.advice.prior import PriorAdvisor
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.relations.store import RelationStore
from habitus.series.reader import SeriesReader
from habitus.temporal import Rulers


def scene_branch_root(config: HabitusConfig, lane: str) -> Path:
    """一支的语义树根 ``<scene_root>/<lane>``（裁定 42）：这一支的概念集、命中记录、关系表、先验缓存、完成标记、换代报告、
    trace、模型队列流水与重试账都在它下面，锁与租约的键也由它派生。"""

    return config.scene_root / lane


def build_scene_nightly(
    config: HabitusConfig,
    *,
    root: Path,
    behavior_tree: BehaviorTree,
    kind_store: BehaviorKindStore,
    structured_chat: StructuredChatClient,
    calendar: DayTypeCalendar | None = None,
) -> SceneNightlyRun | None:
    """一支的语义树（换代里语义树那一段），住 ``root``（``scene_branch_root``）；``config.scene.enabled`` 关着就是 None。
    ``calendar`` 是组合根建好、两支与预测层共用的那一份（不给就自己建）。语义树的时刻由换代交进来，这里不收时钟。"""

    if not config.scene.enabled:
        return None
    return SceneNightlyRun(
        behavior_tree=behavior_tree,
        catalog=VocabularyCatalog(kind_store),
        concepts=ConceptStore(root),
        hits=ConceptHitStore(root),
        calendar=calendar if calendar is not None else local_calendar(config),
        relations=RelationStore(
            root,
            rotate_files=config.lifecycle.transitions_rotate_files,
            fallback_generations=config.lifecycle.fallback_generations,
        ),
        relation_config=relation_config(config),
        advice=RelationAdvice(
            store=AdviceStore(root),
            prior=PriorAdvisor(structured_chat),
            zone=config.locale.zone(),
        ),
        rulers=rhythm_rulers(config),
    )


def rhythm_rulers(config: HabitusConfig) -> Rulers:
    """算时间规律（复现）的尺子只认 ``config.temporal``（裁定 59）；时段读数看最近几个发生日，取证据的窗口 M（裁定 57）。"""

    temporal = config.temporal
    return Rulers(
        slot_minutes=temporal.slot_minutes,
        merge_window_seconds=temporal.merge_window_seconds,
        pool_half_width=temporal.pool_half_width,
        shrink_slot_to_pool=temporal.shrink_slot_to_pool,
        shrink_pool_to_all_day=temporal.shrink_pool_to_all_day,
        recent=evidence_of(config).window,
    )


def build_generation(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    scene: SceneNightlyRun,
    kind_store: BehaviorKindStore,
    closed_days: Callable[[], Iterable[date]],
    foresight: ForesightRuntimeComponents | None,
    marks: GenerationMarks,
    clock: Callable[[], datetime] | None = None,
    dirty_days: DirtyDayLog | None = None,
    retry: RetryStage | None = None,
    scene_lock: SceneWriteLock | None = None,
    closed_at: Callable[[date], datetime | None] | None = None,
    unreduced_gaps: Callable[[date], int] | None = None,
    queue: ModelQueueState | None = None,
) -> Generation:
    """一支的换代的串法（``runtime.generation``；没有模型调用）：读这一支的树、脏日、词表在用状态，与这一支的队列共用一面
    （``queue``）；预测层接着时触发还要看这一支的预测拍水位过了余量（``foresight.generation_margin_slots`` 槽）。"""

    marker = None if foresight is None else foresight.watermark
    return Generation(
        behavior_tree=behavior_tree,
        series=SeriesReader(
            behavior_tree,
            () if dirty_days is None else (dirty_days,),
            name="generation",
            kinds=kind_resolution(kind_store),
        ),
        scene=scene,
        zone=config.locale.zone(),
        # 每一代开头重读放假数据（用户 10-18：节假日在换代时判断）；两支与预测层读的是同一份，换了一起换
        refresh_calendar=scene.calendar.refresh if isinstance(scene.calendar, SharedCalendar) else None,
        closed_days=closed_days,
        vocabulary=VocabularyCounter(kind_store, kind_store.config),
        marks=marks,
        clock=clock,
        foresight=(
            None
            if foresight is None
            else build_foresight_nightly(
                foresight,
                relations=RelationStore(scene.relations.root),
                relation_config=relation_config(config),
                unseen_day_types=config.foresight.unseen_day_types,
            )
        ),
        settlement=None if foresight is None else foresight.settlement,
        retry=retry,
        days=None if foresight is None else foresight.days,
        models_down=None if queue is None else queue.down,
        queue_owed=None if queue is None else queue.owed_now,
        watermark=None if marker is None else marker.read,
        margin=timedelta(minutes=config.foresight.generation_margin_slots * config.temporal.slot_minutes),
        scene_lock=scene_lock,
        generating=None if queue is None else queue.set_generating,
        closed_at=closed_at,
        unreduced_gaps=unreduced_gaps,
        lane=kind_store.lane.value,  # 是哪一支：记在 span 与报告上
    )


def assemble_branch(
    config: HabitusConfig,
    *,
    branch: BehaviorRuntimeComponents,
    calendar: DayTypeCalendar | None,
    structured_chat: StructuredChatClient,
    foresight: ForesightRuntimeComponents | None,
    marks: GenerationMarks,
    traces: TraceFiles,
    ledger_lease: Lease,
    lock_store: LockStore,
    observer: Observer,
    unsealed: UnsealedFromJudgements | None = None,
    tick: ForesightTick | None = None,
) -> ScenePipeline | None:
    """组合根的那一段，一支一份（裁定 42 两条独立管线）：这一支的语义树 + 换代 + 模型队列，与预测 worker 替这一支做的那一槽（换代的
    那一眼、预测拍 ``tick``、预测根租约）。根、锁、租约、完成标记、重试账、队列状态都是这一支的，重试的堆只编排这一支的。"""

    lane = branch.lane.value
    root = scene_branch_root(config, lane)
    scene_lock = scene_write_lock(config, root, lock_store)
    scene = build_scene_nightly(
        config, root=root, behavior_tree=branch.tree, calendar=calendar, kind_store=branch.kind_store,
        structured_chat=unlocked(structured_chat, scene_lock),
    )  # fmt: skip
    if scene is None:
        return None
    runner = branch.reduction_runner
    retry = retry_stage_of((branch,))
    state = ModelQueueState()
    generation = build_generation(
        config,
        behavior_tree=branch.tree,
        scene=scene,
        kind_store=branch.kind_store,
        closed_days=runner.closed_days,
        foresight=foresight,
        marks=marks,
        dirty_days=runner.dirty_days,
        retry=retry,
        scene_lock=scene_lock,
        closed_at=lambda day: _closed_at(runner.day_ledger.state().closed.get(day)),
        unreduced_gaps=lambda cutoff: unreduced_gaps(
            branch.judgements, runner.ledger, cutoff=cutoff, zone=config.locale.zone()
        ),
        queue=state,
    )
    look = build_generation_runner(config, generation, root=root, traces=traces, observer=observer)
    queue = build_model_queue(
        config, root=root, scene=scene, generation=generation, branch=branch, state=state, retry=retry,
        unsealed=unsealed, traces=traces, lock_store=lock_store, observer=observer, scene_lock=scene_lock,
    )  # fmt: skip
    return ScenePipeline(
        behavior=branch, scene=scene, generation=look, queue=queue, foresight=foresight, tick=tick,
        lease=ledger_lease, unsealed=unsealed,
    )  # fmt: skip


def build_workers(
    config: HabitusConfig, pipelines: tuple[ScenePipeline, ...], *, observer: Observer | None = None
) -> tuple[ForesightWorker, ModelQueueWorker]:
    """两支共用的两个常驻 worker（worker 不翻倍）：预测 worker 每个槽边界按支依次（会话在前）持这一支的预测根租约先看一眼换代、
    再做这一支的预测拍；模型队列 worker 每一轮按支依次做各支的队列。一支的失败、锁忙、退避只算这一支的。"""

    slots = ForesightWorker(
        tuple(
            SlotBranch(
                item.lane,
                generation=item.generation,
                tick=item.tick,
                # 拍点对齐槽边界（``foresight.tick_interval_seconds`` 已由配置根校验等于槽宽）；没有预测拍时那一槽自己持租约
                lease=None if item.tick is not None else item.lease,
            )
            for item in pipelines
        ),
        interval_seconds=config.temporal.slot_minutes * 60.0,
        shutdown_timeout_seconds=config.foresight.worker_shutdown_timeout_seconds,
        zone=config.locale.zone(),
        observer=observer,
    )
    queues = ModelQueueWorker(
        tuple(item.queue for item in pipelines),
        shutdown_timeout_seconds=config.scene.model_queue_shutdown_timeout_seconds,
        observer=observer,
    )
    return slots, queues


def build_model_queue(
    config: HabitusConfig,
    *,
    root: Path,
    scene: SceneNightlyRun,
    generation: Generation,
    branch: BehaviorRuntimeComponents,
    state: ModelQueueState,
    retry: RetryStage | None,
    unsealed: UnsealedFromJudgements | None,
    traces: TraceFiles | None,
    lock_store: LockStore,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    scene_lock: SceneWriteLock | None = None,
) -> ModelQueue:
    """一支的模型队列（实时化方案第 4 节）：与这一支的换代读同一棵行为树、同一份语义树存储、同一个重试堆编排；"已封口的日子的活
    能不能做"问这一支的换代（``Generation.sealed_day``）。序列读者是它自己的一份（读者名 ``model_queue``，脏日水位各记各的）；
    队列租约与语义树写锁不同键、都由这一支的根派生，租期与续约间隔共用 ``scene.generation_lease_*`` 那两个数。"""

    runner = branch.reduction_runner
    scene_config = config.scene
    return ModelQueue(
        scene=scene,
        series=SeriesReader(
            branch.tree,
            () if runner.dirty_days is None else (runner.dirty_days,),
            name="model_queue",
            kinds=kind_resolution(branch.kind_store),
        ),
        today=generation.today,
        sealed_day=generation.sealed_day,
        vocabulary=generation.vocabulary.revision,
        # 账的主人带支（生命周期按主人清两支的重试账，同 ``behavior-semantic-<lane>``）
        retries=retry_ledger(
            root, owner=f"model-queue-{branch.lane.value}", config=config.lifecycle.retry, clock=clock
        ),
        lease=Lease(
            lock_store,
            model_queue_lease_key(root),
            ttl_seconds=scene_config.generation_lease_ttl_seconds,
            heartbeat_seconds=scene_config.generation_heartbeat_seconds,
        ),
        state=state,
        scene_lock=scene_lock,
        marks=generation.marks,
        retry=retry,
        unsealed=unsealed,
        settings=model_queue_settings(config),
        traces=traces,
        journal=QueueJournal(root / "model_queue", artifact_root=root, zone=config.locale.zone()),
        observer=observer,
        clock=clock,
        lane=branch.lane.value,
    )


def model_queue_settings(config: HabitusConfig) -> QueueSettings:
    scene = config.scene
    return QueueSettings(
        poll_seconds=scene.model_queue_poll_seconds,
        backoff_base_seconds=scene.model_queue_backoff_base_seconds,
        backoff_max_seconds=scene.model_queue_backoff_max_seconds,
        unreachable_after_failures=scene.model_queue_unreachable_after_failures,
        warning_items=scene.model_queue_backlog_warning_items,
        warning_wait_seconds=scene.model_queue_backlog_warning_seconds,
        kept_days=scene.generation_kept_days,
    )


def _closed_at(stamp: str | None) -> datetime | None:
    """定稿日账上那一天的定稿时刻（ISO 文本）；没有或认不出是 None。"""

    if stamp is None:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return parsed if parsed.utcoffset() is not None else None


def generation_marks(root: Path) -> GenerationMarks:
    """一支换代的完成标记（``<这一支的根>/generation/<日子>.json``）。组合根一支建一份：换代写它，这一支的结算读它认没有这一代的那天。"""

    return GenerationMarks(root / "generation", artifact_root=root)


def scene_write_lock(config: HabitusConfig, root: Path, lock_store: LockStore) -> SceneWriteLock:
    """一支的语义树写锁：这一支换代的语义树一步整步持有，这一支的模型队列写语义树的每一项在它里面、调模型的那几段除外。
    同一支的换代与队列共用这一个实例；租期与续租间隔是三把锁共用的那两个数。"""

    return SceneWriteLock(
        Lease(
            lock_store,
            scene_write_lock_key(root),
            ttl_seconds=config.scene.generation_lease_ttl_seconds,
            heartbeat_seconds=config.scene.generation_heartbeat_seconds,
        )
    )


def scene_write_lock_key(root: Path) -> str:
    """语义树写锁的键：由这一支的语义树根派生，换代的语义树一步与同一支模型队列的落盘互斥；两支不同键。"""

    digest = hashlib.sha256(str(root.expanduser().absolute()).encode("utf-8")).hexdigest()[:24]
    return f"scene:generation:{digest}"


def model_queue_lease_key(root: Path) -> str:
    """队列租约的键：由这一支的语义树根派生（与语义树写锁不同键），同一个根同一时刻只有一个进程在做这一支队列的活。"""

    digest = hashlib.sha256(str(root.expanduser().absolute()).encode("utf-8")).hexdigest()[:24]
    return f"scene:model-queue:{digest}"


def generation_settings(config: HabitusConfig) -> GenerationSettings:
    scene = config.scene
    return GenerationSettings(
        kept_days=scene.generation_kept_days,
        stuck_ticks=scene.generation_stuck_ticks,
        retry_base_seconds=scene.generation_retry_base_seconds,
        retry_max_seconds=scene.generation_retry_max_seconds,
    )


def generation_traces(root: Path, backend: SpanBackend | None = None) -> TraceFiles:
    """一支换代与预测拍的本地 trace（``<这一支的根>/generation/traces/<日期>.jsonl``）；``backend`` 是 OTel 那一侧。"""

    return TraceFiles(root / "generation" / "traces", artifact_root=root, backend=backend)


def model_tracer(metrics: Observer) -> Observer:
    """观测后端：调用链上有 span 时（换代、预测拍），每一次模型调用补一个 ``model.call`` 子 span，按（换代的那一步, 触点）计数进指标。"""

    return ModelCallTracer(metrics=metrics)


def build_generation_runner(
    config: HabitusConfig,
    generation: Generation,
    *,
    root: Path,
    traces: TraceFiles,
    observer: Observer | None = None,
) -> GenerationRunner:
    """一支换代的那一眼：连续定稿到昨天、预测拍水位过了余量，就做今天这一代（实时化方案第 3 节）。预测 worker 每槽调它。
    报告写在这一支的根下，关系表的双向变化读这一支的关系表。"""

    return GenerationRunner(
        generation,
        settings=generation_settings(config),
        reports=GenerationReports(
            root / "generation" / "reports",
            artifact_root=root,
            relations=RelationStore(root),
            lane=generation.lane,
        ),
        traces=traces,
        observer=observer,
    )


__all__ = [
    "assemble_branch",
    "build_generation",
    "build_generation_runner",
    "build_model_queue",
    "build_scene_nightly",
    "build_workers",
    "generation_marks",
    "generation_settings",
    "generation_traces",
    "model_queue_lease_key",
    "model_queue_settings",
    "model_tracer",
    "scene_branch_root",
    "scene_write_lock_key",
]
