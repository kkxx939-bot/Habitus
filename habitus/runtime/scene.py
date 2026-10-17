"""语义树夜批在组合根的组装（语义树新方案 ``13`` 第 11 步另一半）。

一支一份 ``SceneNightlyRun``：词表口是行为树那一支的词表（``VocabularyCatalog``），概念集、映射、关系表、先验建议都住
``config.scene_root``；模型触点（映射器、概念作者、先验、调节条件）全部走同一个 ``StructuredChatClient``；关系检验的时间
参数只认预测树（``relation_config``）、常态分份的容差复用预测树的 ``pool_half_width``。未启用时返回 None。

整晚的串法（行为树读一次 → 预测树 → 语义树 → 机会表）在 ``runtime.nightly.Nightly``，由 ``build_nightly`` 装好交给
预测树那一支的常驻 worker（``PredictionRebuildWorker.attach_nightly``）；结算仍挂在它后面的 ``after_rebuild``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime

from habitus.behavior.kinds.reader import VocabularyReader
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.model_client import StructuredChatClient
from habitus.prediction.config import PredictionTreeConfig
from habitus.runtime.foresight import ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.nightly import Nightly
from habitus.runtime.prediction import PredictionRebuilder
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_authoring import ConceptAuthoring
from habitus.runtime.scene_night import SceneNightConfig, SceneNightlyRun, relation_config
from habitus.runtime.scene_vocabulary import VocabularyCatalog
from habitus.scene import NominalCalendar
from habitus.scene.advice.conditions import ConditionProposer
from habitus.scene.advice.prior import PriorAdvisor
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts.author import ConceptAuthor
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.mapper import ConceptMapper
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.relations.store import RelationStore


def build_scene_nightly(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    kind_store: BehaviorKindStore,
    structured_chat: StructuredChatClient,
    tree_config: PredictionTreeConfig,
    clock: Callable[[], datetime] | None = None,
) -> SceneNightlyRun | None:
    """一支的语义树夜批；``config.scene.enabled`` 关着就是 None。"""

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
        calendar=NominalCalendar(),
        timezone=config.locale.zone(),
        mapper_for=mapper_for,
        relations=RelationStore(scene_root),
        relation_config=relation_config(config),
        authoring=ConceptAuthoring(ConceptAuthor(structured_chat, clock=clock)),
        advice=RelationAdvice(
            store=AdviceStore(scene_root),
            prior=PriorAdvisor(structured_chat),
            conditions=ConditionProposer(structured_chat),
        ),
        subject=subject,
        config=SceneNightConfig(slack_slots=tree_config.pool_half_width),
    )


def build_nightly(
    config: HabitusConfig,
    *,
    behavior_tree: BehaviorTree,
    prediction: PredictionRebuilder,
    scene: SceneNightlyRun,
    kind_store: BehaviorKindStore,
    closed_days: Callable[[], Iterable[date]],
    foresight: ForesightRuntimeComponents | None,
    clock: Callable[[], datetime] | None = None,
) -> Nightly:
    """整晚的串法：行为树读一次，预测树、语义树、机会表（预测层接着时）用同一份序列。"""

    reader = VocabularyReader(kind_store)
    return Nightly(
        behavior_tree=behavior_tree,
        prediction=prediction,
        scene=scene,
        closed_days=closed_days,
        vocabulary=lambda: (reader.version(), kind_store.read_migration() is not None),
        clock=clock,
        foresight=(
            None
            if foresight is None
            else build_foresight_nightly(
                foresight, relations=RelationStore(config.scene_root), relation_config=relation_config(config)
            )
        ),
    )


__all__ = ["build_nightly", "build_scene_nightly"]
