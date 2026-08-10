"""LLM 语义先验蒸馏成的版本化伪计数表。

冷上下文里计数为零，回退只能给出根分布；语义先验回答"这样的人在这个
情境下倾向做什么"，以等效样本量 ``strength`` 的伪计数进入估计——两三条
真实观测即可压过先验。表由离线 LLM 蒸馏产出（组合根批处理），本对象
只承载确定性内容：全部条目进入版本摘要，换表即换代，学习聚合零 LLM。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from foundation.integrity import canonical_digest, canonical_json, canonicalize
from prediction.learning.config import PredictionLearningError
from prediction.learning.keys import logical_state_key

_PRIOR_CONTRACT = "prediction-behavior-prior-v1"


@dataclass(frozen=True)
class PredictionPriorEntry:
    """一个状态的先验分布：完整逻辑身份加各行为的份额。"""

    identity: Mapping[str, Any]
    branches: tuple[tuple[Mapping[str, Any], float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Mapping) or not self.identity:
            raise PredictionLearningError("prior entry identity must be a non-empty mapping")
        object.__setattr__(self, "identity", dict(canonicalize(self.identity)))
        branches: list[tuple[dict[str, Any], float]] = []
        total = 0.0
        for target, share in self.branches:
            if not isinstance(target, Mapping) or not target:
                raise PredictionLearningError("prior branch target must be a non-empty mapping")
            if isinstance(share, bool) or not isinstance(share, (int, float)):
                raise PredictionLearningError("prior branch share must be numeric")
            value = float(share)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise PredictionLearningError("prior branch share must be within (0, 1]")
            total += value
            branches.append((dict(canonicalize(target)), value))
        if total > 1.000000001:
            raise PredictionLearningError("prior branch shares cannot exceed one")
        keys = [canonical_json(target) for target, _share in branches]
        if len(keys) != len(set(keys)):
            raise PredictionLearningError("prior entry must not repeat one branch target")
        object.__setattr__(self, "branches", tuple(branches))


@dataclass(frozen=True)
class PredictionBehaviorPrior:
    """按状态逻辑键索引的伪计数先验；空表即无先验。"""

    entries: tuple[PredictionPriorEntry, ...] = ()
    strength: float = 2.0
    _index: dict[str, dict[str, float]] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(not isinstance(item, PredictionPriorEntry) for item in entries):
            raise PredictionLearningError("prior entries must be PredictionPriorEntry values")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not math.isfinite(float(self.strength))
            or float(self.strength) <= 0
        ):
            raise PredictionLearningError("prior strength must be a positive finite number")
        object.__setattr__(self, "strength", float(self.strength))
        object.__setattr__(self, "entries", entries)
        index: dict[str, dict[str, float]] = {}
        for entry in entries:
            key = logical_state_key(entry.identity)
            if key in index:
                raise PredictionLearningError("prior entries must not repeat one state identity")
            index[key] = {
                canonical_json(target): share * self.strength for target, share in entry.branches
            }
        object.__setattr__(self, "_index", index)

    @property
    def version(self) -> str:
        """先验表内容的稳定版本摘要，进入 generation 身份。"""

        return canonical_digest(
            {
                "contract": _PRIOR_CONTRACT,
                "strength": self.strength,
                "entries": [
                    {
                        "identity": entry.identity,
                        "branches": [
                            {"target": target, "share": share} for target, share in entry.branches
                        ],
                    }
                    for entry in self.entries
                ],
            }
        )

    def pseudo_counts(self, state_logical_key: str) -> Mapping[str, float]:
        """返回一个状态的伪计数（分支键 → 伪计数），无先验时为空。"""

        return self._index.get(state_logical_key, {})


def prior_entry(
    identity: Mapping[str, Any],
    branches: Sequence[tuple[Mapping[str, Any], float]],
) -> PredictionPriorEntry:
    """构造一个先验条目的便捷入口。"""

    return PredictionPriorEntry(identity=identity, branches=tuple(branches))


__all__ = ["PredictionBehaviorPrior", "PredictionPriorEntry", "prior_entry"]
