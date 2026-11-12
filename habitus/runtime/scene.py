"""语义树与换代在组合根的组装（语义树新方案 ``13`` 第 11 步另一半；实时化方案阶段 1）。

一支一份 ``SceneNightlyRun``：词表口是行为树那一支的词表（``VocabularyCatalog``），概念集、映射、关系表、先验建议都住
``config.scene_root``；模型触点（映射器、概念作者、先验、调节条件）全部走同一个 ``StructuredChatClient``；关系检验的
时间尺子只认 ``config.temporal``（``relation_config``）、常态分份的容差是 ``config.scene.baseline_slack_slots``。未启用时返回 None。

换代的串法（续做词表迁移 → 重读放假数据 → 重试收尾 → 结算 → 读序列 → 语义树 → 机会表 → 单元历史 → 压实 → 当天的日子预期 →
完成标记）在 ``runtime.generation.Generation``，由 ``build_generation`` 装好、``build_generation_runner`` 包成换代的那一眼
（``runtime.generation_runner.GenerationRunner``：连续定稿到昨天、预测拍水位过了余量后做今天这一代），交给预测 worker
每槽先调（``runtime.foresight_worker``，同一次预测根租约之内先换代、再做预测拍）。完成标记、每代报告与
trace 都在 ``<scene_root>/generation`` 下。换代里没有模型调用：调模型的活（出错重试、词表的每晚新增与定期拆改、
同步词表牵到的重映射、写概念、映射、候选条件与先验的补问）由模型队列做（``runtime.model_queue``，``build_model_queue``
装好、交给 ``ModelQueueWorker``）；两边共用一份 ``ModelQueueState``——换代读"队列是不是在整体退避""各类还欠着几项"，
队列读"换代是不是在做"——与一把语义树写锁（``runtime.scene_lock``）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import date, datetime, timedelta

from habitus.behavior.kinds.reader import VocabularyReader
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.retry import RetryStage
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.conditions import DayTypeCalendar
from habitus.config import HabitusConfig
from habitus.foundation.observability import Observer
from habitus.infrastructure.store.contracts import LockStore
from habitus.model_client import StructuredChatClient
from habitus.runtime.behavior import BehaviorBranches, retry_stage_of
from habitus.runtime.conditions import SharedCalendar, local_calendar
from habitus.runtime.evidence import evidence_of
from habitus.runtime.foresight import ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.foresight_worker import ForesightWorker
from habitus.runtime.generation import Generation
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_marks import GenerationMarkError, GenerationMarks
from habitus.runtime.generation_report import GenerationReports
from habitus.runtime.generation_runner import GenerationRunner, GenerationSettings
from habitus.runtime.generation_trace import ModelCallTracer, SpanBackend, TraceFiles
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.model_queue import ModelQueue, ModelQueueState, QueueJournal, QueueSettings, VocabularyBranch
from habitus.runtime.model_queue_worker import ModelQueueWorker
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_config import relation_config
from habitus.runtime.scene_lock import SceneWriteLock, unlocked
from habitus.runtime.scene_night import SceneNightConfig, SceneNightlyRun
from habitus.runtime.scene_rhythms import RhythmFiles
from habitus.runtime.scene_vocabulary import VocabularyCatalog
from habitus.runtime.unsealed import UnsealedFromJudgements, unreduced_gaps
from habitus.scene.advice.conditions import ConditionProposer
from habitus.scene.advice.prior import PriorAdvisor
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts.author import ConceptAuthor
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.mapper import ConceptMapper
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.relations.store import RelationStore
from habitus.series.reader import SeriesReader


def build_scene_nightly(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    kind_store: BehaviorKindStore,
    structured_chat: StructuredChatClient,
    clock: Callable[[], datetime] | None = None,
    calendar: DayTypeCalendar | None = None,
    marks: GenerationMarks | None = None,
) -> SceneNightlyRun | None:
    """一支的语义树（换代里语义树那一段）；``config.scene.enabled`` 关着就是 None。``calendar`` 是组合根建好、与预测层共用的那一份
    （不给就自己建）。``marks`` 是换代的完成标记：最近一代做完时的词表版本从它读（映射取"近几天同类记录"时认类是不是那之后才变的）。"""

    if not config.scene.enabled:
        return None
    scene_root = config.scene_root
    subject = config.behavior.primary_subject

    def mapper_for(concepts: ConceptSet) -> ConceptMapper:
        return ConceptMapper(structured_chat, concepts, subject=subject, clock=clock)

    return SceneNightlyRun(
        behavior_tree=behavior_tree,
        catalog=VocabularyCatalog(VocabularyReader(kind_store)),
        concepts=ConceptStore(scene_root),
        hits=ConceptHitStore(scene_root),
        calendar=calendar if calendar is not None else local_calendar(config),
        timezone=config.locale.zone(),
        mapper_for=mapper_for,
        relations=RelationStore(
            scene_root,
            rotate_files=config.lifecycle.transitions_rotate_files,
            fallback_generations=config.lifecycle.fallback_generations,
        ),
        relation_config=relation_config(config),
        authoring=ConceptAuthoring(ConceptAuthor(structured_chat, clock=clock)),
        advice=RelationAdvice(
            store=AdviceStore(scene_root),
            prior=PriorAdvisor(structured_chat),
            conditions=ConditionProposer(structured_chat),
            zone=config.locale.zone(),
        ),
        subject=subject,
        config=SceneNightConfig(slack_slots=config.scene.baseline_slack_slots),
        rulers=rhythm_rulers(config),
        # 每一代的节律落盘一份（映射读它），与完成标记放在一起、一起留存与清理
        rhythm_files=RhythmFiles(scene_root / "generation" / "rhythms", artifact_root=scene_root),
        frozen_vocabulary=None if marks is None else lambda: _frozen_vocabulary(marks),
    )


def _frozen_vocabulary(marks: GenerationMarks) -> int | None:
    """最近一代做完时的词表版本（完成标记上记的）；一代都还没有、或最新的标记读不了是 None（读不了由健康面另报）。"""

    try:
        latest = marks.latest()
    except GenerationMarkError:
        return None
    return None if latest is None else latest.vocabulary_version


def rhythm_rulers(config: HabitusConfig) -> RhythmRulers:
    """算节律的尺子只认 ``config.temporal``（裁定 59）；时段读数看最近几个发生日，取证据的窗口 M（裁定 57）。"""

    temporal = config.temporal
    return RhythmRulers(
        slot_minutes=temporal.slot_minutes,
        merge_window_seconds=temporal.merge_window_seconds,
        pool_half_width=temporal.pool_half_width,
        shrink_slot_to_pool=temporal.shrink_slot_to_pool,
        shrink_pool_to_all_day=temporal.shrink_pool_to_all_day,
        min_peak_mass=temporal.min_peak_mass,
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
    migration_jobs: Sequence[Callable[[], Awaitable[object]]] = (),
    scene_lock: SceneWriteLock | None = None,
    closed_at: Callable[[date], datetime | None] | None = None,
    unreduced_gaps: Callable[[date], int] | None = None,
    queue: ModelQueueState | None = None,
    revisions: Callable[[], int] | None = None,
) -> Generation:
    """换代的串法（``runtime.generation``；没有模型调用）。``dirty_days`` 是这一支归约记的脏日日志（序列按天缓存、只重读变过
    的日子）；``migration_jobs`` 续做迁移（最前）；``queue`` 是模型队列与换代共用的那一面；预测层接着时触发还要看预测拍水位
    过了余量（``foresight.generation_margin_slots`` 槽）。"""

    reader = VocabularyReader(kind_store)
    marker = None if foresight is None else foresight.watermark
    return Generation(
        behavior_tree=behavior_tree,
        series=SeriesReader(behavior_tree, () if dirty_days is None else (dirty_days,), name="generation"),
        scene=scene,
        zone=config.locale.zone(),
        # 每一代开头重读放假数据（用户 10-18：节假日在换代时判断）；语义树与预测层读的是同一份，换了两边一起换
        refresh_calendar=scene.calendar.refresh if isinstance(scene.calendar, SharedCalendar) else None,
        closed_days=closed_days,
        vocabulary=lambda: (reader.version(), kind_store.read_migration() is not None),
        marks=marks,
        clock=clock,
        foresight=(
            None
            if foresight is None
            else build_foresight_nightly(
                foresight,
                relations=RelationStore(config.scene_root),
                relation_config=relation_config(config),
                unseen_day_types=config.foresight.unseen_day_types,
            )
        ),
        settlement=None if foresight is None else foresight.settlement,
        retry=retry,
        days=None if foresight is None else foresight.days,
        migration_jobs=migration_jobs,
        models_down=None if queue is None else queue.down,
        queue_owed=None if queue is None else queue.owed_now,
        watermark=None if marker is None else marker.read,
        margin=timedelta(minutes=config.foresight.generation_margin_slots * config.temporal.slot_minutes),
        scene_lock=scene_lock,
        generating=None if queue is None else queue.set_generating,
        revisions=revisions,
        closed_at=closed_at,
        unreduced_gaps=unreduced_gaps,
    )


def assemble_generation(
    config: HabitusConfig,
    *,
    behavior: BehaviorBranches,
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
) -> tuple[SceneNightlyRun | None, GenerationRunner | None, ModelQueueWorker | None, ForesightWorker | None]:
    """组合根的那一段：语义树 + 换代 + 模型队列与它的 worker + 每槽先看一眼换代、再做预测拍的那个 worker（实时化方案第 3、
    4 节；阶段 2 第二小步）。只接会话那一支（见 assembly 的 TODO）。``unsealed`` 是预测层那座未封口的桥（换代里的日子预期只读
    它的缓存，归类由模型队列先做）；``tick`` 是预测拍（没开预测层是 None：那个 worker 只持 ``ledger_lease`` 看换代）。"""

    session = behavior.session
    scene_lock = scene_write_lock(config, lock_store)
    scene = build_scene_nightly(
        config, behavior_tree=session.tree, calendar=calendar, kind_store=session.kind_store,
        structured_chat=unlocked(structured_chat, scene_lock), marks=marks,
    )  # fmt: skip
    if scene is None:
        return None, None, None, None
    runner = session.reduction_runner
    retry = retry_stage_of(behavior)
    state = ModelQueueState()
    generation = build_generation(
        config,
        behavior_tree=session.tree,
        scene=scene,
        kind_store=session.kind_store,
        closed_days=runner.closed_days,
        foresight=foresight,
        marks=marks,
        dirty_days=runner.dirty_days,
        retry=retry,
        migration_jobs=tuple(branch.reduction_runner.vocabulary.resume for branch in behavior),
        revisions=lambda: sum(1 for branch in behavior if branch.reduction_runner.vocabulary.revision_pending()),
        scene_lock=scene_lock,
        closed_at=lambda day: _closed_at(runner.day_ledger.state().closed.get(day)),
        unreduced_gaps=lambda cutoff: unreduced_gaps(
            session.judgements, runner.ledger, cutoff=cutoff, zone=config.locale.zone()
        ),
        queue=state,
    )
    look = build_generation_runner(config, generation, traces=traces, observer=observer)
    queue = build_model_queue(
        config, scene=scene, generation=generation, behavior=behavior, state=state, retry=retry, unsealed=unsealed,
        traces=traces, lock_store=lock_store, observer=observer, scene_lock=scene_lock,
    )  # fmt: skip
    queue_worker = ModelQueueWorker(
        queue, shutdown_timeout_seconds=config.scene.model_queue_shutdown_timeout_seconds, observer=observer
    )
    return scene, look, queue_worker, build_slot_worker(config, tick, look, ledger_lease, observer=observer)


def build_slot_worker(
    config: HabitusConfig,
    tick: ForesightTick | None,
    generation: GenerationRunner,
    ledger_lease: Lease,
    *,
    observer: Observer | None = None,
) -> ForesightWorker:
    """每个槽边界持预测根租约先看一眼换代、再做预测拍的常驻 worker（拍点对齐槽边界：``foresight.tick_interval_seconds`` 已由
    配置根校验等于槽宽）。没开预测层（``tick`` 是 None）时没有预测拍，它自己持 ``ledger_lease`` 只看换代。"""

    return ForesightWorker(
        tick,
        generation=generation,
        lease=None if tick is not None else ledger_lease,
        interval_seconds=config.temporal.slot_minutes * 60.0,
        shutdown_timeout_seconds=config.foresight.worker_shutdown_timeout_seconds,
        zone=config.locale.zone(),
        observer=observer,
    )


def build_model_queue(
    config: HabitusConfig,
    *,
    scene: SceneNightlyRun,
    generation: Generation,
    behavior: BehaviorBranches,
    state: ModelQueueState,
    retry: RetryStage | None,
    unsealed: UnsealedFromJudgements | None,
    traces: TraceFiles | None,
    lock_store: LockStore,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    scene_lock: SceneWriteLock | None = None,
) -> ModelQueue:
    """模型队列（实时化方案第 4 节）：与换代读同一棵行为树、同一份语义树存储、同一个重试堆编排；"已封口的日子的活能不能做"
    问换代（``Generation.sealed_day``）。序列读者是它自己的一份（读者名 ``model_queue``，脏日水位各记各的）；队列租约与语义树
    写锁不同键，租期与续约间隔共用 ``scene.generation_lease_*`` 那两个数。``scene_lock`` 是与换代共用的语义树写锁。"""

    runner = behavior.session.reduction_runner
    scene_config = config.scene
    return ModelQueue(
        scene=scene,
        series=SeriesReader(
            behavior.session.tree, () if runner.dirty_days is None else (runner.dirty_days,), name="model_queue"
        ),
        today=generation.today,
        sealed_day=generation.sealed_day,
        vocabulary=generation.vocabulary,
        retries=retry_ledger(config.scene_root, owner="model-queue", config=config.lifecycle.retry, clock=clock),
        lease=Lease(
            lock_store,
            model_queue_lease_key(config),
            ttl_seconds=scene_config.generation_lease_ttl_seconds,
            heartbeat_seconds=scene_config.generation_heartbeat_seconds,
        ),
        state=state,
        scene_lock=scene_lock,
        marks=generation.marks,
        retry=retry,
        vocabulary_branches=tuple(
            VocabularyBranch(
                branch.lane.value,
                branch.reduction_runner.vocabulary.jobs.due,
                branch.reduction_runner.nightly_vocabulary,
                branch.reduction_runner.vocabulary.revision_pending,
            )
            for branch in behavior
        ),
        unsealed=unsealed,
        settings=model_queue_settings(config),
        traces=traces,
        journal=QueueJournal(
            config.scene_root / "model_queue", artifact_root=config.scene_root, zone=config.locale.zone()
        ),
        observer=observer,
        clock=clock,
        watermark=generation.watermark,
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
        mapping_margin_slots=scene.mapping_margin_slots,
        slot_minutes=config.temporal.slot_minutes,
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


def generation_marks(config: HabitusConfig) -> GenerationMarks:
    """换代的完成标记（``<scene_root>/generation/<日子>.json``）。组合根只建一份：换代写它，结算读它认没有这一代的那天。"""

    return GenerationMarks(config.scene_root / "generation", artifact_root=config.scene_root)


def scene_write_lock(config: HabitusConfig, lock_store: LockStore) -> SceneWriteLock:
    """语义树写锁：换代的语义树一步整步持有，模型队列写语义树的每一项在它里面、调模型的那几段除外。换代与队列共用这一个
    实例；租期与续租间隔是三把锁共用的那两个数。"""

    return SceneWriteLock(
        Lease(
            lock_store,
            scene_write_lock_key(config),
            ttl_seconds=config.scene.generation_lease_ttl_seconds,
            heartbeat_seconds=config.scene.generation_heartbeat_seconds,
        )
    )


def scene_write_lock_key(config: HabitusConfig) -> str:
    """语义树写锁的键：由语义树根派生（沿用原换代租约的键），换代的语义树一步与模型队列的落盘互斥。"""

    digest = hashlib.sha256(str(config.scene_root.expanduser().absolute()).encode("utf-8")).hexdigest()[:24]
    return f"scene:generation:{digest}"


def model_queue_lease_key(config: HabitusConfig) -> str:
    """队列租约的键：由语义树根派生（与语义树写锁不同键），同一个根同一时刻只有一个进程在做队列的活。"""

    digest = hashlib.sha256(str(config.scene_root.expanduser().absolute()).encode("utf-8")).hexdigest()[:24]
    return f"scene:model-queue:{digest}"


def generation_settings(config: HabitusConfig) -> GenerationSettings:
    scene = config.scene
    return GenerationSettings(
        kept_days=scene.generation_kept_days,
        stuck_ticks=scene.generation_stuck_ticks,
        retry_base_seconds=scene.generation_retry_base_seconds,
        retry_max_seconds=scene.generation_retry_max_seconds,
    )


def generation_traces(config: HabitusConfig, backend: SpanBackend | None = None) -> TraceFiles:
    """换代与预测拍的本地 trace（``generation/traces/<日期>.jsonl``）；``backend`` 是 OTel 那一侧（开了 OTLP 时同时进去）。"""

    return TraceFiles(config.scene_root / "generation" / "traces", artifact_root=config.scene_root, backend=backend)


def model_tracer(metrics: Observer) -> Observer:
    """观测后端：调用链上有 span 时（换代、预测拍），每一次模型调用补一个 ``model.call`` 子 span，按（换代的那一步, 触点）计数进指标。"""

    return ModelCallTracer(metrics=metrics)


def build_generation_runner(
    config: HabitusConfig,
    generation: Generation,
    *,
    traces: TraceFiles,
    observer: Observer | None = None,
) -> GenerationRunner:
    """换代的那一眼：连续定稿到昨天、预测拍水位过了余量，就做今天这一代（实时化方案第 3 节）。预测 worker 每槽调它。"""

    return GenerationRunner(
        generation,
        settings=generation_settings(config),
        reports=GenerationReports(
            config.scene_root / "generation" / "reports",
            artifact_root=config.scene_root,
            relations=RelationStore(config.scene_root),
        ),
        traces=traces,
        observer=observer,
    )


__all__ = [
    "assemble_generation",
    "build_generation",
    "build_generation_runner",
    "build_model_queue",
    "build_scene_nightly",
    "generation_marks",
    "generation_settings",
    "generation_traces",
    "model_queue_lease_key",
    "model_queue_settings",
    "model_tracer",
    "scene_write_lock_key",
]
