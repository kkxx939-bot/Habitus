"""语义树夜批在组合根的组装（语义树新方案 ``13`` 第 11 步另一半）。

一支一份 ``SceneNightlyRun``：词表口是行为树那一支的词表（``VocabularyCatalog``），概念集、映射、关系表、先验建议都住
``config.scene_root``；模型触点（映射器、概念作者、先验、调节条件）全部走同一个 ``StructuredChatClient``；关系检验的
时间尺子只认 ``config.temporal``（``relation_config``）、常态分份的容差是 ``config.scene.baseline_slack_slots``。未启用时返回 None。

整晚的串法（结算 → 行为树读一次 → 语义树 → 机会表）在 ``runtime.nightly.Nightly``，由 ``build_nightly`` 装好、
``build_nightly_worker`` 交给它自己的常驻节拍（``runtime.nightly_worker.NightlyWorker``；预测树删了，裁定 59）；
结算是整晚的第一步（找原因要在同一晚读到它结出的落空）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime

from habitus.behavior.kinds.reader import VocabularyReader
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.retry import RetryStage
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.conditions import DayTypeCalendar
from habitus.config import HabitusConfig
from habitus.foundation.observability import Observer
from habitus.model_client import StructuredChatClient
from habitus.runtime.conditions import SharedCalendar, local_calendar
from habitus.runtime.evidence import evidence_of
from habitus.runtime.foresight import ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.nightly import Nightly
from habitus.runtime.nightly_worker import LastRun, NightlyWorker
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_night import SceneNightConfig, SceneNightlyRun, relation_config
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
) -> Nightly:
    """整晚的串法：重试白天读不懂的（裁定 67）→ 结算（预测层接着时）→ 行为树读一次，语义树、机会表与找原因（预测层接着时）用同一份序列。
    ``dirty_days`` 是这一支归约记的脏日日志：序列按天缓存、只重读变过的日子（26 号清单 6.7）。"""

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
    )


def build_nightly_worker(config: HabitusConfig, nightly: Nightly, *, observer: Observer | None = None) -> NightlyWorker:
    """整晚那一趟的常驻节拍：每天主体本地的 ``scene.nightly_at``；上一趟开跑的时刻落在语义树根下（裁定 67 第 7 条）。"""

    return NightlyWorker(
        nightly,
        at=config.scene.nightly_time,
        zone=config.locale.zone(),
        shutdown_timeout_seconds=config.scene.nightly_shutdown_timeout_seconds,
        last_run=LastRun(config.scene_root / "nightly" / "last_run.json", root=config.scene_root),
        observer=observer,
    )


__all__ = ["build_nightly", "build_nightly_worker", "build_scene_nightly"]
