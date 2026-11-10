"""语义树与换代在组合根的组装（语义树新方案 ``13`` 第 11 步另一半；实时化方案阶段 1）。

一支一份 ``SceneNightlyRun``：词表口是行为树那一支的词表（``VocabularyCatalog``），概念集、映射、关系表、先验建议都住
``config.scene_root``；模型触点（映射器、概念作者、先验、调节条件）全部走同一个 ``StructuredChatClient``；关系检验的
时间尺子只认 ``config.temporal``（``relation_config``）、常态分份的容差是 ``config.scene.baseline_slack_slots``。未启用时返回 None。

换代的串法（探模型 → 重读放假数据 → 重试收尾 → 结算 → 词表 → 读序列 → 语义树 → 机会表 → 单元历史 → 压实 → 当天的日子预期 →
完成标记）在 ``runtime.generation.Generation``，由 ``build_generation`` 装好、``build_generation_worker`` 交给换代 worker
（``runtime.generation_worker.GenerationWorker``：连续定稿到昨天、预测拍水位过了余量后做今天这一代）。完成标记、每代报告与
trace 都在 ``<scene_root>/generation`` 下。
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
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
from habitus.model_client import ModelUnreachable, StructuredChatClient
from habitus.runtime.behavior import BehaviorBranches, retry_stage_of
from habitus.runtime.conditions import SharedCalendar, local_calendar
from habitus.runtime.evidence import evidence_of
from habitus.runtime.foresight import ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.generation import Generation
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_marks import GenerationMarks
from habitus.runtime.generation_report import GenerationReports
from habitus.runtime.generation_trace import ModelCallTracer, SpanBackend, TraceFiles
from habitus.runtime.generation_worker import GenerationSettings, GenerationWorker
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_config import relation_config
from habitus.runtime.scene_night import SceneNightConfig, SceneNightlyRun
from habitus.runtime.scene_vocabulary import VocabularyCatalog
from habitus.runtime.unsealed import unreduced_gaps
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
) -> SceneNightlyRun | None:
    """一支的语义树（换代里语义树那一段）；``config.scene.enabled`` 关着就是 None。``calendar`` 是组合根建好、与预测层共用的那一份（不给就自己建）。"""

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
        ),
        subject=subject,
        mapping_retries=retry_ledger(scene_root, owner="scene-mapper", config=config.lifecycle.retry, clock=clock),
        config=SceneNightConfig(slack_slots=config.scene.baseline_slack_slots),
        rulers=rhythm_rulers(config),
    )


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
    structured_chat: StructuredChatClient | None = None,
    vocabulary_jobs: Sequence[Callable[[date], Awaitable[object]]] = (),
    migration_jobs: Sequence[Callable[[], Awaitable[object]]] = (),
    ledger_lease: Lease | None = None,
    closed_at: Callable[[date], datetime | None] | None = None,
    unreduced_gaps: Callable[[date], int] | None = None,
) -> Generation:
    """换代的串法（实时化方案第 3 节）：重试白天读不懂的（裁定 67）→ 结算（预测层接着时）→ 词表 → 行为树读一次，语义树、机会表
    用同一份序列 → 当天的日子预期（预测层的 ``days`` 接着时）→ 完成标记。
    ``dirty_days`` 是这一支归约记的脏日日志：序列按天缓存、只重读变过的日子（26 号清单 6.7；读者名 ``generation``）。
    ``migration_jobs`` 续做迁移（最前）、``vocabulary_jobs`` 每晚新增与定期拆改（结算之后）；``ledger_lease`` 是预测根租约；
    预测层接着时触发还要看预测拍水位过了余量（``foresight.generation_margin_slots`` 槽）。"""

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
        probe=None if structured_chat is None else model_probe(structured_chat),
        vocabulary_jobs=vocabulary_jobs,
        migration_jobs=migration_jobs,
        watermark=None if marker is None else marker.read,
        margin=timedelta(minutes=config.foresight.generation_margin_slots * config.temporal.slot_minutes),
        ledger_lease=ledger_lease,
        closed_at=closed_at,
        unreduced_gaps=unreduced_gaps,
    )


def model_probe(structured_chat: StructuredChatClient) -> Callable[[], Awaitable[None]]:
    """换代最先那一探（裁定 94 第 4 条）：问一次供应商的健康探针（网络抖动按普通调用的规矩先重试几次，
    ``ChatClient.health_check_async``），答不上（连不上、鉴权、权限、服务端坏、没配好）就抛 ``ModelUnreachable``——换代记下
    "模型调不通"、照做（模型调不通不停一天）。供应商没有异步探针的不探（探不了不等于调不通）：额度满这类探针看不出来的，由之后
    各处调模型碰到时各自报（重试的会话轮、归约归类、每晚新增词表、语义树的各触点）。"""

    async def probe() -> None:
        check: Callable[[], Awaitable[Mapping[str, object]]] | None = getattr(
            structured_chat.client, "health_check_async", None
        )
        if not callable(check):
            return
        result = await check()
        code = str(result.get("error_code") or "health check failed")
        if result.get("ok") is not True and code != "async_health_check_unsupported":
            raise ModelUnreachable(code)

    return probe


def assemble_generation(
    config: HabitusConfig,
    *,
    behavior: BehaviorBranches,
    calendar: DayTypeCalendar | None,
    structured_chat: StructuredChatClient,
    foresight: ForesightRuntimeComponents | None,
    marks: GenerationMarks,
    traces: TraceFiles,
    ledger_lease: Lease | None,
    lock_store: LockStore,
    observer: Observer,
) -> tuple[SceneNightlyRun | None, GenerationWorker | None]:
    """组合根的那一段：语义树 + 换代 + 换代 worker（实时化方案第 3 节）。只接会话那一支（见 assembly 的 TODO）。"""

    session = behavior.session
    scene = build_scene_nightly(
        config,
        behavior_tree=session.tree,
        calendar=calendar,
        kind_store=session.kind_store,
        structured_chat=structured_chat,
    )
    if scene is None:
        return None, None
    runner = session.reduction_runner
    generation = build_generation(
        config,
        behavior_tree=session.tree,
        scene=scene,
        kind_store=session.kind_store,
        closed_days=runner.closed_days,
        foresight=foresight,
        marks=marks,
        dirty_days=runner.dirty_days,
        retry=retry_stage_of(behavior),
        structured_chat=structured_chat,
        vocabulary_jobs=tuple(branch.reduction_runner.nightly_vocabulary for branch in behavior),
        migration_jobs=tuple(branch.reduction_runner.vocabulary.resume for branch in behavior),
        ledger_lease=ledger_lease,
        closed_at=lambda day: _closed_at(runner.day_ledger.state().closed.get(day)),
        unreduced_gaps=lambda cutoff: unreduced_gaps(
            session.judgements, runner.ledger, cutoff=cutoff, zone=config.locale.zone()
        ),
    )
    worker = build_generation_worker(config, generation, traces=traces, lock_store=lock_store, observer=observer)
    return scene, worker


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


def generation_lease_key(config: HabitusConfig) -> str:
    """换代租约的键：由语义树根派生，同一个根同一天只换代一次。"""

    digest = hashlib.sha256(str(config.scene_root.expanduser().absolute()).encode("utf-8")).hexdigest()[:24]
    return f"scene:generation:{digest}"


def generation_settings(config: HabitusConfig) -> GenerationSettings:
    scene = config.scene
    return GenerationSettings(
        poll_seconds=scene.generation_poll_seconds,
        kept_days=scene.generation_kept_days,
        lease_ttl_seconds=scene.generation_lease_ttl_seconds,
        heartbeat_seconds=scene.generation_heartbeat_seconds,
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


def build_generation_worker(
    config: HabitusConfig,
    generation: Generation,
    *,
    traces: TraceFiles,
    lock_store: LockStore,
    observer: Observer | None = None,
) -> GenerationWorker:
    """换代 worker：连续定稿到昨天、预测拍水位过了余量，就做今天这一代（实时化方案第 3 节）。"""

    return GenerationWorker(
        generation,
        lock_store=lock_store,
        lease_key=generation_lease_key(config),
        shutdown_timeout_seconds=config.scene.generation_shutdown_timeout_seconds,
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
    "build_generation_worker",
    "build_scene_nightly",
    "generation_lease_key",
    "generation_marks",
    "generation_settings",
    "generation_traces",
    "model_probe",
    "model_tracer",
]
