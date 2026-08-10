"""基于语义和结构条件检索并展开 Prediction Pattern。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from foundation.integrity import text_digest
from prediction.model import PredictionPatternAddress, PredictionPatternKind, prediction_sample_id
from prediction.pattern.contracts import PredictionPatternRepository
from prediction.pattern.document import PredictionPatternDocument
from prediction.pattern.generation import (
    PredictionPatternGenerationEntry,
    PredictionPatternGenerationManifest,
    PredictionPatternGenerationState,
    PredictionPatternGenerationStore,
    PredictionPatternGenerationStoreError,
    default_generation_store_root,
)
from prediction.uri import PredictionURI


class PredictionPatternGraphIntegrityError(ValueError):
    """激活清单与不可变 Pattern 文档不一致。"""


@dataclass(frozen=True)
class PredictionStateExpansion:
    """一个 StatePattern 及其全部候选行为分支。"""

    state: PredictionPatternDocument
    branches: tuple[PredictionPatternDocument, ...]
    pattern_generation: str


class PredictionPatternGraph:
    """Pattern 文档之上的只读检索视图。

    一致性单元是单个 State：每个逻辑 State 独立激活自己的 generation，
    展开一个 State 只解析它所在的分片和它自己的清单，成本与状态总量无关。
    多步推演不属于本层——本层不保存也不遍历状态之间的边。
    """

    def __init__(
        self,
        tree: PredictionPatternRepository,
        *,
        generation_store: PredictionPatternGenerationStore | None = None,
    ) -> None:
        if not isinstance(tree, PredictionPatternRepository):
            raise TypeError("tree must satisfy PredictionPatternRepository")
        self.tree = tree
        self.generation_store = generation_store or PredictionPatternGenerationStore(
            default_generation_store_root(tree.root)
        )

    def expand(self, logical_state_key: str) -> PredictionStateExpansion:
        """展开一个逻辑 State 当前激活的那一代；只读它自己的清单。"""

        manifest = self._active_manifest(prediction_sample_id(logical_state_key))
        assert manifest is not None
        entries = {
            PredictionURI.parse(entry.uri).to_pattern_address(): entry for entry in manifest.entries
        }
        self._verify_materialization(manifest, entries)
        state = self.tree.read_pattern(
            PredictionPatternAddress(PredictionPatternKind.STATE, manifest.state_key)
        )
        branches = tuple(
            self.tree.read_pattern(address)
            for address in sorted(
                (address for address in entries if address.kind is PredictionPatternKind.BRANCH),
                key=lambda address: address.branch_key or "",
            )
        )
        if sum(branch.fields["conditional_probability"] for branch in branches) > 1.000000001:
            raise PredictionPatternGraphIntegrityError(
                "StatePattern outgoing branch probabilities cannot exceed one"
            )
        ordered = tuple(
            sorted(
                branches,
                key=lambda item: (
                    -item.fields["conditional_probability"],
                    -item.fields["support_count"],
                    item.address.branch_key or "",
                ),
            )
        )
        return PredictionStateExpansion(state, ordered, manifest.generation_id)

    def search_states(
        self,
        *,
        semantic_terms: Sequence[str] = (),
        observed_values: Mapping[str, Any] | None = None,
        active_goals: Sequence[str] = (),
        candidate_logical_state_keys: Sequence[str] | None = None,
        limit: int = 20,
    ) -> tuple[PredictionPatternDocument, ...]:
        """本地确定性召回；向量检索可先返回 logical_state_key，再复用 ``expand``。"""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise ValueError("pattern search limit must be between 1 and 1000")
        terms = tuple(_text(item, "semantic term").casefold() for item in semantic_terms)
        goals = {_text(item, "active goal") for item in active_goals}
        values = {} if observed_values is None else dict(observed_values)
        skip_missing = candidate_logical_state_keys is not None
        if candidate_logical_state_keys is None:
            logical_keys = self.generation_store.active_logical_state_keys()
        else:
            logical_keys = tuple(sorted({prediction_sample_id(key) for key in candidate_logical_state_keys}))
        ranked: list[tuple[float, str, PredictionPatternDocument]] = []
        for logical_key in logical_keys:
            manifest = self._active_manifest(logical_key, missing_ok=skip_missing)
            if manifest is None:
                continue
            state = self.tree.read_pattern(
                PredictionPatternAddress(PredictionPatternKind.STATE, manifest.state_key)
            )
            summary = state.fields["semantic_summary"].casefold()
            term_score = sum(term in summary for term in terms)
            predicate_score = sum(_predicate_matches(predicate, values) for predicate in state.fields["predicates"])
            goal_score = len(goals & set(state.fields["active_goals"]))
            if terms and term_score == 0 and not predicate_score and not goal_score:
                continue
            support_score = min(state.fields["support_count"], 1_000) / 1_000_000
            score = term_score * 4.0 + predicate_score * 3.0 + goal_score * 2.0 + support_score
            ranked.append((score, logical_key, state))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in ranked[:limit])

    def _active_manifest(
        self,
        logical_state_key: str,
        *,
        missing_ok: bool = False,
    ) -> PredictionPatternGenerationManifest | None:
        try:
            record = self.generation_store.read_active_state(logical_state_key)
        except PredictionPatternGenerationStoreError:
            if missing_ok:
                return None
            raise
        if record.state is not PredictionPatternGenerationState.READY:
            raise PredictionPatternGraphIntegrityError("active Pattern generation is not READY")
        return record.manifest

    def _verify_materialization(
        self,
        manifest: PredictionPatternGenerationManifest,
        entries: Mapping[PredictionPatternAddress, PredictionPatternGenerationEntry],
    ) -> None:
        for address, entry in entries.items():
            document = self.tree.read_pattern(address)
            if document.fields["pattern_generation"] != manifest.generation_id:
                raise PredictionPatternGraphIntegrityError(
                    "Pattern document generation does not match the active manifest"
                )
            if document.fields["projection_version"] != manifest.projection_version:
                raise PredictionPatternGraphIntegrityError(
                    "Pattern document projection version does not match the active manifest"
                )
            if text_digest(self.tree.pattern_codec.encode(document)) != entry.digest:
                raise PredictionPatternGraphIntegrityError(
                    "Pattern document digest does not match the active manifest"
                )


def _predicate_matches(predicate: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    field = predicate["field"]
    operator = predicate["operator"]
    if operator == "exists":
        return field in values
    if field not in values:
        return False
    actual = values[field]
    expected = predicate["value"]
    try:
        if operator == "eq":
            return actual == expected
        if operator == "neq":
            return actual != expected
        if operator == "gt":
            return bool(actual > expected)
        if operator == "gte":
            return bool(actual >= expected)
        if operator == "lt":
            return bool(actual < expected)
        if operator == "lte":
            return bool(actual <= expected)
        if operator == "contains":
            return bool(expected in actual)
    except (TypeError, ValueError):
        return False
    return False


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


__all__ = [
    "PredictionPatternGraph",
    "PredictionPatternGraphIntegrityError",
    "PredictionStateExpansion",
]
