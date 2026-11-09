"""语义树夜批在组合根的组装（语义树新方案 ``13`` 第 11 步另一半）。

一支一份 ``SceneNightlyRun``：词表口是行为树那一支的词表（``VocabularyCatalog``），概念集、映射、关系表、先验建议都住
``config.scene_root``；模型触点（映射器、概念作者、先验、调节条件）全部走同一个 ``StructuredChatClient``；关系检验的
时间尺子只认 ``config.temporal``（``relation_config``）、常态分份的容差是 ``config.scene.baseline_slack_slots``。未启用时返回 None。

整晚的串法（前置处理：探模型、重试 → 结算 → 词表（每晚新增、定期拆改）→ 行为树读一次 → 语义树 → 机会表 → 单元 → 当天的日子预期）在 ``runtime.nightly.Nightly``，由
``build_nightly`` 装好、``build_nightly_worker`` 交给夜线的 Job 执行器（``runtime.nightly_worker.NightlyWorker``：前一天定稿后给这一晚
入队 5 段，裁定 94 第 1 条；预测树删了，裁定 59）；结算排在语义树之前（找原因要在同一晚读到它结出的落空）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

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
from habitus.runtime.conditions import SharedCalendar, local_calendar
from habitus.runtime.evidence import evidence_of
from habitus.runtime.foresight import ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.nightly import Nightly
from habitus.runtime.nightly_jobs import NightJobs
from habitus.runtime.nightly_report import NightReports
from habitus.runtime.nightly_settings import NightlySettings
from habitus.runtime.nightly_trace import ModelCallTracer, SpanBackend, TraceFiles
from habitus.runtime.nightly_worker import NightlyWorker
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_config import relation_config
from habitus.runtime.scene_night import SceneNightConfig, SceneNightlyRun
from habitus.runtime.scene_vocabulary import VocabularyCatalog
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
    """一支的语义树夜批；``config.scene.enabled`` 关着就是 None。``calendar`` 是组合根建好、与预测层共用的那一份（不给就自己建）。"""

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


def build_nightly(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    scene: SceneNightlyRun,
    kind_store: BehaviorKindStore,
    closed_days: Callable[[], Iterable[date]],
    foresight: ForesightRuntimeComponents | None,
    clock: Callable[[], datetime] | None = None,
    dirty_days: DirtyDayLog | None = None,
    retry: RetryStage | None = None,
    structured_chat: StructuredChatClient | None = None,
    vocabulary_jobs: Sequence[Callable[[date], Awaitable[object]]] = (),
) -> Nightly:
    """整晚的串法：重试白天读不懂的（裁定 67）→ 结算（预测层接着时）→ 行为树读一次，语义树、机会表与找原因（预测层接着时）用同一份序列
    → 当天的日子预期（预测层的 ``days`` 接着时，裁定 94 第 1 条）。
    ``dirty_days`` 是这一支归约记的脏日日志：序列按天缓存、只重读变过的日子（26 号清单 6.7）。``structured_chat`` 给了就在线的
    最前面探一次模型（``model_probe``，裁定 94 第 4 条）。``vocabulary_jobs`` 是各支的词表整理活（每晚新增、定期拆改），在线上结构那一段
    之前跑（裁定 94 第 8、9 条）。"""

    reader = VocabularyReader(kind_store)
    return Nightly(
        behavior_tree=behavior_tree,
        series=SeriesReader(behavior_tree, () if dirty_days is None else (dirty_days,)),
        scene=scene,
        zone=config.locale.zone(),
        # 每晚开头重读放假数据（用户 10-18：节假日在夜批时判断）；语义树与预测层读的是同一份，换了两边一起换
        refresh_calendar=scene.calendar.refresh if isinstance(scene.calendar, SharedCalendar) else None,
        closed_days=closed_days,
        vocabulary=lambda: (reader.version(), kind_store.read_migration() is not None),
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
    )


def model_probe(structured_chat: StructuredChatClient) -> Callable[[], Awaitable[None]]:
    """前置处理最先那一探（裁定 94 第 4、8 条）：问一次供应商的健康探针（网络抖动按普通调用的规矩先重试几次，
    ``ChatClient.health_check_async``），答不上（连不上、鉴权、权限、服务端坏、没配好）就抛 ``ModelUnreachable``，这一晚停下。
    供应商没有异步探针的不探（探不了不等于调不通）：额度满这类探针看不出来的，由线上之后任何一处调模型碰到时停线（重试的会话轮、
    归约归类、每晚新增词表、语义树的各触点）。"""

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


def nightly_jobs(config: HabitusConfig) -> NightJobs:
    """夜线的 Job 账（每晚 5 段各一个 Job），落在语义树根下 ``nightly/jobs``。组合根只建一份：执行器写它，结算读它认停掉的那天、
    白天那一拍读它决定今天建不建预期（裁定 94 第 1、9 条）。"""

    return NightJobs(
        config.scene_root / "nightly" / "jobs",
        artifact_root=config.scene_root,
        kept_nights=config.scene.nightly_kept_nights,
    )


def nightly_lease_key(config: HabitusConfig) -> str:
    """夜线全局租约的键：由语义树根派生，同一个根只有一个执行者。"""

    digest = hashlib.sha256(str(config.scene_root.expanduser().absolute()).encode("utf-8")).hexdigest()[:24]
    return f"scene:nightly:{digest}"


def nightly_settings(config: HabitusConfig) -> NightlySettings:
    scene = config.scene
    return NightlySettings(
        poll_seconds=scene.nightly_poll_seconds,
        kept_nights=scene.nightly_kept_nights,
        lease_ttl_seconds=scene.nightly_lease_ttl_seconds,
        heartbeat_seconds=scene.nightly_heartbeat_seconds,
        max_attempts=scene.nightly_max_attempts,
        retry_base_seconds=scene.nightly_retry_base_seconds,
        retry_max_seconds=scene.nightly_retry_max_seconds,
    )


def nightly_traces(config: HabitusConfig, backend: SpanBackend | None = None) -> TraceFiles:
    """夜线与白天那一拍的本地 trace（``nightly/traces/<日期>.jsonl``）；``backend`` 是 OTel 那一侧（开了 OTLP 时同时进去）。"""

    return TraceFiles(config.scene_root / "nightly" / "traces", artifact_root=config.scene_root, backend=backend)


@dataclass(frozen=True)
class NightLine:
    """夜线在组合根只建一份的两样：Job 账（执行器写，结算与白天那一拍读）与本地 trace（夜线与白天那一拍都写）。"""

    jobs: NightJobs
    traces: TraceFiles


def nightly_line(config: HabitusConfig, backend: SpanBackend | None = None) -> NightLine:
    return NightLine(nightly_jobs(config), nightly_traces(config, backend))


def model_tracer(metrics: Observer) -> Observer:
    """观测后端：调用链上有 span 时（夜线、白天那一拍），每一次模型调用补一个 ``model.call`` 子 span，按（段, 触点）计数进指标。"""

    return ModelCallTracer(metrics=metrics)


def build_nightly_worker(
    config: HabitusConfig,
    nightly: Nightly,
    *,
    line: NightLine,
    lock_store: LockStore,
    observer: Observer | None = None,
) -> NightlyWorker:
    """夜线的 Job 执行器：前一天定稿完成就给这一晚入队 5 段、按先后跑（裁定 94 第 1 条）。"""

    return NightlyWorker(
        nightly,
        jobs=line.jobs,
        lock_store=lock_store,
        lease_key=nightly_lease_key(config),
        shutdown_timeout_seconds=config.scene.nightly_shutdown_timeout_seconds,
        settings=nightly_settings(config),
        reports=NightReports(
            config.scene_root / "nightly" / "reports",
            artifact_root=config.scene_root,
            relations=RelationStore(config.scene_root),
        ),
        traces=line.traces,
        observer=observer,
    )


__all__ = [
    "build_nightly",
    "build_nightly_worker",
    "build_scene_nightly",
    "model_probe",
    "model_tracer",
    "NightLine",
    "nightly_jobs",
    "nightly_line",
    "nightly_lease_key",
    "nightly_settings",
    "nightly_traces",
]
