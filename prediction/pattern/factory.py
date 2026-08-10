"""模式图节点的确定性身份工厂。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foundation.integrity import canonical_digest
from prediction.model import PredictionPatternKind
from prediction.pattern.document import PredictionPatternDocument, PredictionPatternDocumentCodec


class PredictionPatternFactory:
    def __init__(self, codec: PredictionPatternDocumentCodec | None = None) -> None:
        self.codec = codec or PredictionPatternDocumentCodec()

    def build_state(
        self,
        *,
        pattern_generation: str,
        identity: Mapping[str, Any],
        fields: Mapping[str, Any],
    ) -> PredictionPatternDocument:
        if {
            "state_key",
            "logical_state_key",
            "pattern_generation",
            "branch_key",
            "identity_material",
        } & set(fields):
            raise ValueError("prediction pattern identity fields are factory-owned")
        logical_state_key = canonical_digest(
            {"contract": "prediction-logical-state-pattern-v1", "identity": identity}
        )
        state_key = canonical_digest(
            {
                "contract": "prediction-state-pattern-materialization-v1",
                "logical_state_key": logical_state_key,
                "pattern_generation": pattern_generation,
            }
        )
        return self._build(
            PredictionPatternKind.STATE,
            {
                "state_key": state_key,
                "logical_state_key": logical_state_key,
                "pattern_generation": pattern_generation,
                "identity_material": identity,
                **fields,
            },
        )

    def build_branch(
        self,
        *,
        state: PredictionPatternDocument,
        identity: Mapping[str, Any],
        fields: Mapping[str, Any],
    ) -> PredictionPatternDocument:
        if not isinstance(state, PredictionPatternDocument) or state.kind is not PredictionPatternKind.STATE:
            raise TypeError("state must be a StatePattern document")
        if {
            "state_key",
            "branch_key",
            "logical_branch_key",
            "pattern_generation",
            "identity_material",
        } & set(fields):
            raise ValueError("prediction pattern identity fields are factory-owned")
        branch_identity_material = {
            "logical_state_key": state.fields["logical_state_key"],
            "branch_identity": identity,
        }
        logical_branch_key = canonical_digest(
            {
                "contract": "prediction-logical-branch-pattern-v1",
                **branch_identity_material,
            }
        )
        branch_key = canonical_digest(
            {
                "contract": "prediction-branch-pattern-materialization-v1",
                "logical_branch_key": logical_branch_key,
                "pattern_generation": state.fields["pattern_generation"],
            }
        )
        return self._build(
            PredictionPatternKind.BRANCH,
            {
                "state_key": state.address.state_key,
                "branch_key": branch_key,
                "logical_branch_key": logical_branch_key,
                "pattern_generation": state.fields["pattern_generation"],
                "identity_material": branch_identity_material,
                **fields,
            },
        )

    def _build(
        self,
        kind: PredictionPatternKind,
        payload: Mapping[str, Any],
    ) -> PredictionPatternDocument:
        return self.codec.build(kind, payload)


__all__ = ["PredictionPatternFactory"]
