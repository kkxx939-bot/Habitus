"""词表的向量旁册与候选检索：只管**召回**，不下语义结论。

embedding 实验（BHV-KINDS-002，豆包 1024 维、七天 2,956 个名字）：最近邻 top-30 对"别名→正名"
召回 95%（字面重合 85%），但正例与最难负例的相似度分布重叠（上楼/下楼 0.77、拿起/放下 0.64），
**没有可用的自动合并阈值**——所以向量只用来把可能相关的 kind 摆到模型面前，"是不是同一件事"
仍由模型判。

旁册 ``kinds.vectors.json`` 是派生物：每个 kind 存 token 与 label 两条向量（float16 + base64），
记录出自哪个 embedding 模型与维度；模型或维度不符即整表作废重算。丢了不影响正确性，只影响召回
方式（退回字面重合并留信号）。
"""

from __future__ import annotations

import base64
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from operator import mul
from pathlib import Path
from types import MappingProxyType

from behavior.kinds.model import BehaviorKindRegistry
from behavior.model import semantic_name
from foundation.ids import canonical_path_identity
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    read_regular_bytes,
)

KINDS_VECTORS_FILENAME = "kinds.vectors.json"
KINDS_VECTORS_SCHEMA_VERSION = "behavior_kind_vectors_v1"
_KEYS = {"schema_version", "model", "dimension", "vectors"}


class BehaviorKindVectorError(ValueError):
    """向量旁册内容与登记约束不一致。"""


def normalized(values: Sequence[float]) -> tuple[float, ...]:
    """单位化；零向量原样返回（余弦为 0）。"""

    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return tuple(float(v) for v in values)
    return tuple(float(v) / norm for v in values)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """两个**已单位化**向量的余弦（点积）。"""

    return sum(map(mul, a, b))


@dataclass(frozen=True)
class BehaviorKindVectorIndex:
    """名字 → 单位化向量；``model``/``dimension`` 钉死它出自哪套 embedding。"""

    model: str
    dimension: int
    vectors: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise BehaviorKindVectorError("vector index model must be non-empty text")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise BehaviorKindVectorError("vector index dimension must be a positive integer")
        if not isinstance(self.vectors, Mapping):
            raise BehaviorKindVectorError("vector index vectors must be a mapping")
        cleaned: dict[str, tuple[float, ...]] = {}
        for name, values in self.vectors.items():
            key = semantic_name(name, "behavior kind vector name")
            if isinstance(values, str) or not isinstance(values, Sequence) or len(values) != self.dimension:
                raise BehaviorKindVectorError(f"vector for {key!r} must have {self.dimension} values")
            cleaned[key] = tuple(float(v) for v in values)
        object.__setattr__(self, "vectors", MappingProxyType(cleaned))

    def with_vectors(self, values: Mapping[str, Sequence[float]]) -> BehaviorKindVectorIndex:
        """写入即按落盘精度（float16）保存：同一进程内与重启读回的候选排序一致。"""

        merged = dict(self.vectors)
        for name, vector in values.items():
            merged[name] = _unpack(_pack(normalized(vector)), self.dimension)
        return BehaviorKindVectorIndex(self.model, self.dimension, merged)

    def retain(self, names: Iterable[str]) -> BehaviorKindVectorIndex:
        """只保留仍被词表引用的名字——旁册按名字键，删条目时不能按 (token, label) 直删
        （别的条目可能同名），而要按"谁还在用"来收。"""

        keep = set(names)
        return BehaviorKindVectorIndex(
            self.model, self.dimension, {k: v for k, v in self.vectors.items() if k in keep}
        )

    def has(self, name: str) -> bool:
        return name in self.vectors


class BehaviorKindVectorStore:
    """在行为树根下保存向量旁册；读到模型/维度不符的文件按空索引处理（派生物，可重算）。"""

    def __init__(
        self,
        behavior_root: str | Path,
        *,
        model: str,
        dimension: int,
        max_encoded_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.path = self.root / KINDS_VECTORS_FILENAME
        self.model = model
        self.dimension = dimension
        self.max_encoded_bytes = max_encoded_bytes

    def empty(self) -> BehaviorKindVectorIndex:
        return BehaviorKindVectorIndex(self.model, self.dimension)

    def read(self) -> BehaviorKindVectorIndex:
        try:
            encoded = read_regular_bytes(
                self.path, artifact_root=self.root, max_bytes=self.max_encoded_bytes
            )
        except FileNotFoundError:
            return self.empty()
        except DurablePathIntegrityError as exc:
            raise BehaviorKindVectorError("behavior kind vectors cannot be read safely") from exc
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BehaviorKindVectorError("behavior kind vectors are corrupt") from exc
        if not isinstance(payload, dict) or set(payload) != _KEYS:
            raise BehaviorKindVectorError("behavior kind vectors shape is invalid")
        if (
            payload["schema_version"] != KINDS_VECTORS_SCHEMA_VERSION
            or payload["model"] != self.model
            or payload["dimension"] != self.dimension
        ):
            # 出自另一套 embedding 或旧格式：作废，调用方按空索引补算（旁册是派生物，同一口径）。
            return self.empty()
        raw = payload["vectors"]
        if not isinstance(raw, dict):
            raise BehaviorKindVectorError("behavior kind vectors must be a mapping")
        vectors = {name: _unpack(text, self.dimension) for name, text in raw.items()}
        return BehaviorKindVectorIndex(self.model, self.dimension, vectors)

    def replace(self, index: BehaviorKindVectorIndex) -> None:
        if not isinstance(index, BehaviorKindVectorIndex):
            raise TypeError("index must be BehaviorKindVectorIndex")
        if index.model != self.model or index.dimension != self.dimension:
            raise BehaviorKindVectorError("vector index does not match this store's embedding")
        payload = {
            "schema_version": KINDS_VECTORS_SCHEMA_VERSION,
            "model": self.model,
            "dimension": self.dimension,
            "vectors": {name: _pack(vector) for name, vector in sorted(index.vectors.items())},
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        if len(encoded) > self.max_encoded_bytes:
            raise BehaviorKindVectorError("behavior kind vectors exceed their byte bound")
        try:
            atomic_replace_bytes(self.path, encoded, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise BehaviorKindVectorError("behavior kind vectors cannot be written safely") from exc


def _pack(vector: Sequence[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}e", *vector)).decode("ascii")


def _unpack(text: object, dimension: int) -> tuple[float, ...]:
    if not isinstance(text, str):
        raise BehaviorKindVectorError("behavior kind vector must be base64 text")
    try:
        raw = base64.b64decode(text, validate=True)
        values = struct.unpack(f"<{dimension}e", raw)
    except (ValueError, struct.error) as exc:
        raise BehaviorKindVectorError("behavior kind vector is not decodable") from exc
    return tuple(float(v) for v in values)


# ── 候选检索（只管召回）───────────────────────────────────────────────────────────


def nearest_kinds(
    query: Sequence[float],
    registry: BehaviorKindRegistry,
    index: BehaviorKindVectorIndex,
    *,
    limit: int,
) -> tuple[str, ...]:
    """按每个 kind 的 (token, label) 向量取最大余弦，返回最近的 ``limit`` 个 token。

    没有向量的 kind 不参与（旁册是派生物，缺的由补算兜底）。
    """

    unit = normalized(query)
    scored: list[tuple[float, str, str]] = []
    for token, entry in registry.entries.items():
        best: float | None = None
        for name in (entry.token, entry.label):
            vector = index.vectors.get(name)
            if vector is None:
                continue
            score = cosine(unit, vector)
            if best is None or score > best:
                best = score
        if best is not None:
            scored.append((-best, canonical_path_identity(token, "behavior kind token"), token))
    scored.sort()
    return tuple(token for _, _, token in scored[: max(0, limit)])


def _grams(text: str) -> set[str]:
    compact = "".join(text.split())
    return set(compact) | {compact[i : i + 2] for i in range(len(compact) - 1)}


def literal_kinds(name: str, registry: BehaviorKindRegistry, *, limit: int) -> tuple[str, ...]:
    """字面重合（字符 + 二元组 Jaccard）最近的 ``limit`` 个 token——embedding 不可用时的退路。"""

    query = _grams(name)
    scored: list[tuple[float, str, str]] = []
    for token, entry in registry.entries.items():
        best = 0.0
        for candidate in (entry.token, entry.label, *entry.aliases):
            grams = _grams(candidate)
            union = len(query | grams)
            score = len(query & grams) / union if union else 0.0
            best = max(best, score)
        scored.append((-best, canonical_path_identity(token, "behavior kind token"), token))
    scored.sort()
    return tuple(token for _, _, token in scored[: max(0, limit)])


def names_missing_vectors(registry: BehaviorKindRegistry, index: BehaviorKindVectorIndex) -> tuple[str, ...]:
    """还没有向量的 token/label（补算清单）。"""

    missing: list[str] = []
    for entry in registry.entries.values():
        for name in (entry.token, entry.label):
            if name not in index.vectors and name not in missing:
                missing.append(name)
    return tuple(missing)


__all__ = [
    "KINDS_VECTORS_FILENAME",
    "KINDS_VECTORS_SCHEMA_VERSION",
    "BehaviorKindVectorError",
    "BehaviorKindVectorIndex",
    "BehaviorKindVectorStore",
    "cosine",
    "literal_kinds",
    "names_missing_vectors",
    "nearest_kinds",
    "normalized",
]
