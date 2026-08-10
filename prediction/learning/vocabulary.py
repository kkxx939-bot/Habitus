"""聚合键的确定性行为词表归一层。"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from foundation.integrity import canonical_digest
from prediction.learning.config import PredictionLearningError


@dataclass(frozen=True)
class PredictionBehaviorVocabulary:
    """把自由文本行为词归一为稳定聚合键的确定性词表。

    分裂的聚合键会把有效样本量除以同义表达数，因此归一是准确率的第一杠杆。
    确定性部分做 NFKC、空白折叠与大小写归一；同义合并由显式别名表表达。
    别名表内容进入词表版本摘要，换表即换代，学习聚合本身保持可确定性重放；
    表的生产方式（人工维护或离线 LLM 归并）不属于本层。
    """

    aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.aliases, Mapping):
            raise PredictionLearningError("vocabulary aliases must be a mapping")
        normalized: dict[str, str] = {}
        for raw_key, raw_value in self.aliases.items():
            if not isinstance(raw_key, str) or not isinstance(raw_value, str):
                raise PredictionLearningError("vocabulary aliases must map text to text")
            key = _normalize(raw_key)
            value = _normalize(raw_value)
            if not key or not value:
                raise PredictionLearningError("vocabulary aliases must not be empty after normalization")
            if normalized.get(key, value) != value:
                raise PredictionLearningError("vocabulary aliases collide after normalization")
            normalized[key] = value
        for target in normalized.values():
            if normalized.get(target, target) != target:
                raise PredictionLearningError("vocabulary aliases must not form chains")
        object.__setattr__(
            self,
            "aliases",
            MappingProxyType({key: normalized[key] for key in sorted(normalized)}),
        )

    @property
    def version(self) -> str:
        """词表内容的稳定版本摘要，进入 generation 身份材料。"""

        return canonical_digest(
            {
                "contract": "prediction-behavior-vocabulary-v1",
                "aliases": dict(self.aliases),
            }
        )

    def canonical_token(self, value: object, label: str) -> str:
        """归一一个行为词；空白或非文本输入直接失败，不静默清洗成空键。"""

        if not isinstance(value, str):
            raise PredictionLearningError(f"{label} must be text")
        token = _normalize(value)
        if not token:
            raise PredictionLearningError(f"{label} must not be empty")
        return self.aliases.get(token, token)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


__all__ = ["PredictionBehaviorVocabulary"]
