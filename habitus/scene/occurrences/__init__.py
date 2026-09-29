"""③ ``scene/occurrences/`` 概念命中：与行为树同构、叶名相同，按天完成标记。

包根只导出模型与存储。映射器（``habitus.scene.occurrences.mapper``）是模型触点，**不从包根转手**——
否则任何 ``from habitus.scene.occurrences import ConceptHitStore`` 都会把 ``model_client`` 拖进来，账本读
命中就传递性地依赖了模型客户端。组合根按模块路径直接 import 它。
"""

from habitus.scene.occurrences.baselines import (
    DRIFT_MINUTES,
    MIN_BASELINE_SAMPLES,
    RECENT_WINDOW_DAYS,
    BaselineDrift,
    BaselineSnapshot,
    baseline_table,
    declared_keys,
)
from habitus.scene.occurrences.model import ConceptHit, ConceptHits, ConceptHitsError
from habitus.scene.occurrences.store import (
    DONE_FILENAME,
    OCCURRENCES_SEGMENT,
    ConceptHitStore,
    ConceptHitStoreError,
    DayMarker,
)

__all__ = [
    "DONE_FILENAME",
    "DRIFT_MINUTES",
    "MIN_BASELINE_SAMPLES",
    "OCCURRENCES_SEGMENT",
    "RECENT_WINDOW_DAYS",
    "BaselineDrift",
    "BaselineSnapshot",
    "ConceptHit",
    "ConceptHitStore",
    "ConceptHitStoreError",
    "ConceptHits",
    "ConceptHitsError",
    "DayMarker",
    "baseline_table",
    "declared_keys",
]
