"""通用向量记录、过滤条件、状态和搜索结果。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from habitus.foundation.integrity import canonicalize
from habitus.model_client import EmbeddingVector

VectorScalar = str | int | float | bool
VectorValue = VectorScalar | tuple[VectorScalar, ...]


class VectorStoreError(RuntimeError):
    """向量存储无法在完整性边界内完成操作。"""


class VectorStoreConflictError(VectorStoreError):
    """索引代次或检查点已经被其他写入推进。"""


class VectorStoreBusyError(VectorStoreError):
    """向量存储暂时无法取得写锁。"""


class VectorStoreIntegrityError(VectorStoreError):
    """耐久向量记录、元数据或物理布局损坏。"""


class VectorStoreUnsupportedTopologyError(VectorStoreError):
    """当前发布协议无法为目标物理拓扑提供正确性保证。"""


@dataclass(frozen=True)
class VectorStoreRecord:
    """一个后端无关的完整向量记录。"""

    identity: str
    vector: EmbeddingVector
    content: str
    content_digest: str
    attributes: Mapping[str, VectorValue]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity or self.identity != self.identity.strip():
            raise ValueError("vector record identity must be normalized non-empty text")
        if any(ord(character) < 32 for character in self.identity):
            raise ValueError("vector record identity contains control characters")
        if not isinstance(self.vector, EmbeddingVector):
            raise TypeError("vector record vector must be EmbeddingVector")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("vector record content must be non-empty text")
        expected_digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_digest != expected_digest:
            raise ValueError("vector record content_digest does not match its content")
        normalized = _attributes(self.attributes)
        object.__setattr__(self, "attributes", MappingProxyType(normalized))


@dataclass(frozen=True)
class VectorStoreFilter:
    """跨 Adapter 统一的等值和集合交集过滤。"""

    equals: Mapping[str, VectorScalar]
    one_of: Mapping[str, tuple[VectorScalar, ...]]

    def __post_init__(self) -> None:
        equals_values = _attributes(self.equals)
        if any(isinstance(value, tuple) for value in equals_values.values()):
            raise TypeError("vector filter equals values must be scalars")
        equals = {field: cast(VectorScalar, value) for field, value in equals_values.items()}
        if not isinstance(self.one_of, Mapping):
            raise TypeError("vector filter one_of must be an object")
        one_of: dict[str, tuple[VectorScalar, ...]] = {}
        for field, raw_values in self.one_of.items():
            _field(field)
            if not isinstance(raw_values, tuple) or not raw_values:
                raise ValueError("vector filter one_of values must be non-empty tuples")
            values = tuple(_scalar(value, "vector filter value") for value in raw_values)
            if len(values) != len(set(values)):
                raise ValueError("vector filter one_of values must be unique")
            one_of[field] = values
        overlap = set(equals) & set(one_of)
        if overlap:
            raise ValueError("one vector filter field cannot use multiple operators")
        object.__setattr__(self, "equals", MappingProxyType(equals))
        object.__setattr__(self, "one_of", MappingProxyType(one_of))

    def matches(self, attributes: Mapping[str, VectorValue]) -> bool:
        """定义所有 Adapter 必须保持一致的确定性过滤语义。"""

        return bool(
            all(attributes.get(field) == value for field, value in self.equals.items())
            and all(_matches_any(attributes.get(field), values) for field, values in self.one_of.items())
        )


@dataclass(frozen=True)
class VectorStoreMatch:
    """向量存储返回的完整记录和有限余弦分数。"""

    record: VectorStoreRecord
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.record, VectorStoreRecord):
            raise TypeError("vector match record must be VectorStoreRecord")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("vector match score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError("vector match score must be a finite cosine value")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class VectorStoreState:
    """一个集合当前已经原子发布的索引代次。"""

    schema_version: str
    embedding_fingerprint: str
    dimension: int
    checkpoint: int
    generation: int
    record_count: int
    ready: bool = True

    def __post_init__(self) -> None:
        for name in ("schema_version", "embedding_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"vector state {name} must be normalized non-empty text")
        for name, minimum in (("dimension", 1), ("checkpoint", 0), ("generation", 1), ("record_count", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"vector state {name} must be at least {minimum}")
        if not isinstance(self.ready, bool):
            raise TypeError("vector state ready must be boolean")


@dataclass(frozen=True)
class VectorPublicationSnapshot:
    """同时表达已发布状态、所有权标记和正在构建标记。"""

    state: VectorStoreState | None = None
    claim_exists: bool = False
    building: bool = False
    coordination_domain: str | None = None

    def __post_init__(self) -> None:
        if self.state is not None and not isinstance(self.state, VectorStoreState):
            raise TypeError("vector publication state must be VectorStoreState or None")
        if not isinstance(self.claim_exists, bool) or not isinstance(self.building, bool):
            raise TypeError("vector publication flags must be boolean")
        if self.building and not self.claim_exists:
            raise ValueError("a building vector publication must have an ownership claim")
        if self.coordination_domain is not None and (
            not isinstance(self.coordination_domain, str)
            or not self.coordination_domain
            or self.coordination_domain != self.coordination_domain.strip()
        ):
            raise ValueError("vector publication coordination domain must be normalized text")
        if self.coordination_domain is not None and not self.claim_exists:
            raise ValueError("a vector coordination domain requires an ownership claim")


def _attributes(value: object) -> dict[str, VectorValue]:
    if not isinstance(value, Mapping):
        raise TypeError("vector attributes must be an object")
    normalized = canonicalize(value)
    if not isinstance(normalized, dict):
        raise TypeError("vector attributes must be an object")
    result: dict[str, VectorValue] = {}
    for field, item in normalized.items():
        _field(field)
        if isinstance(item, list):
            values = tuple(_scalar(child, "vector attribute item") for child in item)
            if not values:
                raise ValueError("vector attribute arrays must not be empty")
            if len(values) != len(set(values)):
                raise ValueError("vector attribute arrays must contain unique values")
            result[field] = values
        else:
            result[field] = _scalar(item, "vector attribute")
    return result


def _matches_any(actual: VectorValue | None, expected: tuple[VectorScalar, ...]) -> bool:
    if isinstance(actual, tuple):
        return bool(set(actual) & set(expected))
    return actual in expected


def _field(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("vector attribute field must be normalized non-empty text")
    if any(ord(character) < 32 for character in value):
        raise ValueError("vector attribute field contains control characters")
    return value


def _scalar(value: object, label: str) -> VectorScalar:
    if not isinstance(value, str | int | float | bool):
        raise TypeError(f"{label} must be a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


__all__ = [
    "VectorScalar",
    "VectorValue",
    "VectorStoreBusyError",
    "VectorStoreConflictError",
    "VectorStoreError",
    "VectorStoreFilter",
    "VectorStoreIntegrityError",
    "VectorStoreMatch",
    "VectorPublicationSnapshot",
    "VectorStoreRecord",
    "VectorStoreState",
    "VectorStoreUnsupportedTopologyError",
]
