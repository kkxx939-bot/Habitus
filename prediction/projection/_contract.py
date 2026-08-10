"""确定性投影合同的版本身份；样本物化身份由它派生。"""

from __future__ import annotations

from foundation.integrity import canonical_digest
from prediction.model import PredictionKind

PROJECTION_VERSION = "behavior-prediction-v3"
PROJECTOR_DIGEST = canonical_digest(
    {
        "projection": PROJECTION_VERSION,
        "source": "behavior-semantic-tree-v1",
        "samples": [kind.value for kind in PredictionKind],
        "cutoff_rule": "input_contains_only-observable-prefix",
    }
)

__all__ = ["PROJECTION_VERSION", "PROJECTOR_DIGEST"]
