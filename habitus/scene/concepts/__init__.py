"""① ``scene/concepts/`` 概念定义：语义树的词汇层。基础概念跟着词表的类自动生成，细分 / 汇总 / 情境概念由模型写；无版本。

``author`` 刻意**不从这里导出**：它触达 ``model_client``，包根一 import 就把整支拖进模型依赖
（架构测试按传递闭包判）。要用就 ``from habitus.scene.concepts.author import ConceptAuthor``。
"""

from habitus.scene.concepts.model import (
    BASELINE_KEY_SEPARATOR,
    IDENTITY_SEPARATORS,
    BaselineKey,
    BaselineStatistic,
    BaselineWindow,
    ConceptDefinition,
    ConceptError,
    ConceptGrade,
    ConceptKind,
    ConceptLookupError,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
    ContextScope,
    GradeMeasure,
    MechanicalDecision,
    MechanicalRule,
    circular_offset,
    concept_identity,
    parse_baseline_value,
)
from habitus.scene.concepts.rhythm import (
    MAX_RHYTHM_PEAKS,
    RHYTHM_RENDER_VERSION,
    Rhythm,
    RhythmPeak,
    RhythmRulers,
    rhythms_of,
)

__all__ = [
    "rhythms_of",
    "BASELINE_KEY_SEPARATOR",
    "IDENTITY_SEPARATORS",
    "MAX_RHYTHM_PEAKS",
    "RHYTHM_RENDER_VERSION",
    "BaselineKey",
    "BaselineStatistic",
    "BaselineWindow",
    "ConceptDefinition",
    "ConceptError",
    "ConceptGrade",
    "ConceptKind",
    "ConceptLookupError",
    "ConceptOrigin",
    "ConceptRole",
    "ConceptSet",
    "ConceptSource",
    "ContextScope",
    "GradeMeasure",
    "MechanicalDecision",
    "MechanicalRule",
    "Rhythm",
    "RhythmPeak",
    "RhythmRulers",
    "circular_offset",
    "concept_identity",
    "parse_baseline_value",
]
