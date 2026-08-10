"""预测事实样本的逻辑身份与物化身份。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from foundation.integrity import canonical_digest, canonicalize
from prediction.model import PredictionKind, prediction_sample_id

LOGICAL_IDENTITY_CONTRACT = "prediction-logical-sample-v1"
MATERIALIZATION_IDENTITY_CONTRACT = "prediction-materialization-v1"


@dataclass(frozen=True)
class PredictionSampleIdentity:
    """同一预测问题的稳定身份，以及特定投影合同下的物化身份。"""

    logical_sample_id: str
    materialization_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_sample_id", prediction_sample_id(self.logical_sample_id))
        object.__setattr__(self, "materialization_id", prediction_sample_id(self.materialization_id))


def derive_sample_identity(
    kind: PredictionKind | str,
    identity_material: Mapping[str, Any],
    projection_version: str,
    materialization_context: Mapping[str, Any] | None = None,
) -> PredictionSampleIdentity:
    """由可审计材料确定性派生逻辑身份和版本物化身份。"""

    normalized_kind = PredictionKind(kind)
    if not isinstance(identity_material, Mapping) or not identity_material:
        raise ValueError("prediction identity material must be a non-empty mapping")
    if not isinstance(projection_version, str) or not projection_version.strip():
        raise ValueError("prediction projection version must be non-empty text")
    normalized_material = canonicalize(dict(identity_material))
    logical_sample_id = canonical_digest(
        {
            "contract": LOGICAL_IDENTITY_CONTRACT,
            "kind": normalized_kind.value,
            "identity": normalized_material,
        }
    )
    materialization_id = canonical_digest(
        {
            "contract": MATERIALIZATION_IDENTITY_CONTRACT,
            "logical_sample_id": logical_sample_id,
            "projection_version": projection_version.strip(),
            "context": canonicalize({} if materialization_context is None else dict(materialization_context)),
        }
    )
    return PredictionSampleIdentity(logical_sample_id, materialization_id)


__all__ = ["PredictionSampleIdentity", "derive_sample_identity"]
