"""概念层测试共用的材料：一小组自洽的概念、一个按文本查表的假嵌入器。

概念集刻意覆盖三种判法：「晚睡」是机械判据（相对常态的偏移，算法判、不问模型），「早餐」要看当天时间线
（``context=DAY``），「打球」是纯语义（问模型）。「运动」是「打球」的上级——有子概念的行为概念不参与映射，
读侧沿 parent 链聚合。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from habitus.model_client import EmbeddingVector
from habitus.scene.concepts import (
    ConceptDefinition,
    ConceptGrade,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
    ContextScope,
    GradeMeasure,
    MechanicalRule,
)
from habitus.scene.concepts.situation import SituationBasis, SituationRule

CREATED = datetime(2026, 9, 26, 0, 0, tzinfo=UTC)
BASELINE = ConceptSource(ConceptOrigin.BASELINE)
DIMENSION = 4
BEDTIME_KEY = "就寝:usual_start:recent"  # 键的第三段是窗；判据一律比近期常态（2026-09-27 裁定）
BEDTIME_KEY_ALL = "就寝:usual_start:all"  # 历来常态只用来读漂移，不能当判据


def concept(
    name: str,
    definition: str,
    *,
    role: ConceptRole = ConceptRole.BEHAVIOR,
    parent: str | None = None,
    grades: tuple[ConceptGrade, ...] = (),
    rule: MechanicalRule | None = None,
    context: ContextScope = ContextScope.OCCURRENCE,
    baseline_keys: tuple[str, ...] = (),
    situation: SituationRule | None = None,
    source: ConceptSource = BASELINE,
) -> ConceptDefinition:
    return ConceptDefinition(
        name=name,
        definition=definition,
        role=role,
        parent=parent,
        grades=grades,
        rule=rule,
        context=context,
        baseline_keys=baseline_keys,
        situation=situation,
        source=source,
        created_at=CREATED,
    )


#: 晚睡 = 入睡比常态晚两小时以上；档按偏移分：轻 2–4 小时，重 4–12 小时。
LATE_RULE = MechanicalRule(GradeMeasure.START_MINUTE_OF_DAY, lower=120, upper=None, relative_to=BEDTIME_KEY)
LATE_GRADES = (
    ConceptGrade("轻", GradeMeasure.START_MINUTE_OF_DAY, 120, 240, relative=True),
    ConceptGrade("重", GradeMeasure.START_MINUTE_OF_DAY, 240, 720, relative=True),
)

SLEEP_LATE = concept("晚睡", "入睡时刻晚于常态两小时以上", rule=LATE_RULE, grades=LATE_GRADES)
BREAKFAST = concept("早餐", "起床后两小时内的第一次进食", context=ContextScope.DAY)
EXERCISE = concept("运动", "以活动身体为目的、持续十分钟以上的行为")
BALL = concept("打球", "参与一场球类运动", parent="运动")
#: 「出差中」**故意不带情境说明**：这类状态要等事实门（``scene/facts.py``）接上真实数据源才算得出，
#: 在那之前它进得了概念集、也能被假设引用，只是永远不命中、分不出层。
TRAVELLING = concept("出差中", "人在常住地之外过夜", role=ConceptRole.STATE)
#: 「周末」是纯日历的，算法算得出：名义上的周六/周日。
WEEKEND = concept(
    "周末",
    "当地日历上的周六或周日",
    role=ConceptRole.DAY_TYPE,
    situation=SituationRule(SituationBasis.WEEKDAYS, weekdays=(5, 6)),
)

#: 父在子前：``ConceptStore.write`` 要求上级先落盘。
ALL_CONCEPTS = (EXERCISE, BALL, SLEEP_LATE, BREAKFAST, TRAVELLING, WEEKEND)


def concept_set() -> ConceptSet:
    return ConceptSet(ALL_CONCEPTS)


class TableEmbedder:
    """按文本查表的嵌入器；查不到的文本给一个远离所有人的方向。记下每次查询。"""

    provider_name = "fake"
    model = "fake-embed"
    is_remote = False

    def __init__(self, table: Mapping[str, Sequence[float]]) -> None:
        self.table = dict(table)
        self.queries: list[str] = []
        self.documents: list[Sequence[str]] = []

    def _vector(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(tuple(self.table.get(text, (0.0, 0.0, 0.0, 1.0))))

    async def embed_query(self, text: str) -> EmbeddingVector:
        self.queries.append(text)
        return self._vector(text)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        self.documents.append(tuple(texts))
        return tuple(self._vector(text) for text in texts)

    async def aclose(self) -> None:
        return None


__all__ = [
    "ALL_CONCEPTS",
    "BALL",
    "BASELINE",
    "BEDTIME_KEY",
    "BEDTIME_KEY_ALL",
    "BREAKFAST",
    "CREATED",
    "DIMENSION",
    "EXERCISE",
    "LATE_GRADES",
    "LATE_RULE",
    "SLEEP_LATE",
    "TRAVELLING",
    "WEEKEND",
    "TableEmbedder",
    "concept",
    "concept_set",
]
