"""① ``scene/concepts/`` 概念定义：语义树的词汇层。基准（LLM）写、按残差升级增补；无版本。

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
    MIN_RHYTHM_DAYS,
    Rhythm,
    RhythmPeak,
    RhythmProvider,
    rhythm_of,
)
from habitus.scene.concepts.store import CONCEPTS_SEGMENT, ConceptStore, ConceptStoreError
from habitus.scene.concepts.vectors import (
    ConceptVectorError,
    ConceptVectorIndex,
    ConceptVectorRefreshReport,
    ConceptVectorStore,
    embedding_text,
    refresh_concept_vectors,
)

__all__ = [
    "BASELINE_KEY_SEPARATOR",
    "CONCEPTS_SEGMENT",
    "IDENTITY_SEPARATORS",
    "MAX_RHYTHM_PEAKS",
    "MIN_RHYTHM_DAYS",
    "BaselineKey",
    "BaselineStatistic",
    "BaselineWindow",
    "ConceptDefinition",
    "ConceptError",
    "ConceptGrade",
    "ConceptLookupError",
    "ConceptOrigin",
    "ConceptRole",
    "ConceptSet",
    "ConceptSource",
    "ConceptStore",
    "ConceptStoreError",
    "ConceptVectorError",
    "ConceptVectorIndex",
    "ConceptVectorRefreshReport",
    "ConceptVectorStore",
    "ContextScope",
    "GradeMeasure",
    "MechanicalDecision",
    "MechanicalRule",
    "Rhythm",
    "RhythmPeak",
    "RhythmProvider",
    "circular_offset",
    "concept_identity",
    "embedding_text",
    "parse_baseline_value",
    "refresh_concept_vectors",
    "rhythm_of",
]
