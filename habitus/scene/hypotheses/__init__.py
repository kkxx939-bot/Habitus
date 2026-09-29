"""② ``scene/hypotheses/`` 假设：基准（LLM）写的待核对关系，按后件分目录；无版本、无 status、无强度数值。

``author`` 刻意**不从这里导出**（同 ``concepts``）：它触达 ``model_client``。
"""

from habitus.scene.hypotheses.model import (
    ASPECT_LABELS,
    MAX_ANTECEDENTS,
    MAX_OPPORTUNITIES,
    Antecedent,
    Aspect,
    Direction,
    Hypothesis,
    HypothesisError,
    HypothesisOrigin,
    HypothesisSource,
    TypePrior,
)
from habitus.scene.hypotheses.store import HYPOTHESES_SEGMENT, HypothesisStore, HypothesisStoreError

__all__ = [
    "ASPECT_LABELS",
    "HYPOTHESES_SEGMENT",
    "MAX_ANTECEDENTS",
    "MAX_OPPORTUNITIES",
    "Antecedent",
    "Aspect",
    "Direction",
    "Hypothesis",
    "HypothesisError",
    "HypothesisOrigin",
    "HypothesisSource",
    "HypothesisStore",
    "HypothesisStoreError",
    "TypePrior",
]
