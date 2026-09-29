"""概念定义的向量旁册 ``concepts/.vectors.json``：只管**召回**，不下语义结论。

映射器拿一条 occurrence 的名字与概要做 embedding，对叶子行为概念的定义向量取 top-K 当候选，再由 LLM
逐个判是/否——召回只影响漏，不影响错。旁册是派生物：按概念身份存一条单位化向量（float16 + base64）
和被嵌入的那句话的摘要，定义改了摘要就对不上、这条要重算；换 embedding 模型或维度整表作废。

"丢了不影响正确性"只在**有人拦着不让映射**的前提下成立：旁册空着就跑映射，会得到一整天"零命中"并盖上
完成标记。所以映射器构造时核对旁册覆盖了全部叶子行为概念，缺就拒绝——那道闸在 ``mapper.py``。

这里不 import ``model_client``：嵌入器按结构协议注入，模型触点收敛在映射器一处。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Protocol

from habitus.foundation.integrity import canonical_digest
from habitus.foundation.vectors import VectorCodecError, cosine, normalized, pack_float16, unpack_float16
from habitus.scene.concepts.model import ConceptDefinition, ConceptError, ConceptSet, concept_identity
from habitus.scene.concepts.store import CONCEPT_VECTORS_FILENAME, CONCEPTS_SEGMENT
from habitus.scene.storage import SceneStore

CONCEPT_VECTORS_SCHEMA_VERSION = "scene_concept_vectors_v1"
MAX_VECTORS_BYTES = 64 * 1024 * 1024
_KEYS = {"schema_version", "model", "dimension", "vectors"}
_ENTRY_KEYS = {"digest", "vector"}


class ConceptVectorError(ValueError):
    """向量旁册内容与它声明的形状不一致。"""


def embedding_text(definition: ConceptDefinition) -> str:
    """被嵌入的那句话：名字 + 判据句。映射器与刷新用同一个函数，摘要才对得上。"""

    return f"{definition.name}：{definition.definition}"


def text_digest(text: str) -> str:
    return canonical_digest(text)[:16]


@dataclass(frozen=True)
class ConceptVector:
    digest: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not self.digest:
            raise ConceptVectorError("a concept vector must carry the digest of its text")
        if any(not math.isfinite(value) for value in self.values):
            raise ConceptVectorError("a concept vector must hold finite values")


@dataclass(frozen=True)
class ConceptVectorIndex:
    """身份 → (被嵌入文本的摘要, 单位化向量)。``model`` / ``dimension`` 钉死它出自哪套 embedding。"""

    model: str
    dimension: int
    vectors: Mapping[str, ConceptVector] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ConceptVectorError("vector index model must be non-empty text")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ConceptVectorError("vector index dimension must be a positive integer")
        if not isinstance(self.vectors, Mapping):
            raise ConceptVectorError("vector index vectors must be a mapping")
        cleaned: dict[str, ConceptVector] = {}
        for name, entry in self.vectors.items():
            try:
                key = concept_identity(name)
            except ConceptError as exc:
                raise ConceptVectorError(str(exc)) from exc
            if not isinstance(entry, ConceptVector) or len(entry.values) != self.dimension:
                raise ConceptVectorError(f"vector for {key!r} must be a ConceptVector of {self.dimension} values")
            cleaned[key] = entry
        object.__setattr__(self, "vectors", MappingProxyType(cleaned))

    def with_vector(self, definition: ConceptDefinition, values: Sequence[float]) -> ConceptVectorIndex:
        """写入即按落盘精度（float16）保存，同一进程内与重启读回的排序一致。"""

        if len(values) != self.dimension:
            raise ConceptVectorError(f"vector for {definition.name!r} must have {self.dimension} values")
        try:
            stored = unpack_float16(pack_float16(normalized(values)), self.dimension)
        except VectorCodecError as exc:
            raise ConceptVectorError(f"vector for {definition.name!r} cannot be stored: {exc}") from exc
        merged = dict(self.vectors)
        merged[definition.identity] = ConceptVector(text_digest(embedding_text(definition)), stored)
        return ConceptVectorIndex(self.model, self.dimension, merged)

    def retain(self, identities: Iterable[str]) -> ConceptVectorIndex:
        keep = set(identities)
        return ConceptVectorIndex(self.model, self.dimension, {k: v for k, v in self.vectors.items() if k in keep})

    def is_current(self, definition: ConceptDefinition) -> bool:
        """有向量、且是对**现在这句定义**算的。定义改了摘要就变，这条要重算。"""

        entry = self.vectors.get(definition.identity)
        return entry is not None and entry.digest == text_digest(embedding_text(definition))

    def stale(self, concepts: ConceptSet, *, among: Iterable[str] | None = None) -> tuple[str, ...]:
        """没有当期向量的概念身份（默认查全部；``among`` 限定范围）。"""

        pool = sorted(concepts) if among is None else sorted(concept_identity(name) for name in among)
        return tuple(identity for identity in pool if identity in concepts and not self.is_current(concepts[identity]))

    def nearest(
        self, query: Sequence[float], *, limit: int, among: Iterable[str] | None = None
    ) -> tuple[tuple[str, float], ...]:
        """余弦最近的 ``limit`` 个 (身份, 余弦)，降序、同分按身份。``among`` 限定只在这些身份里找。"""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ConceptVectorError("limit must be a positive integer")
        unit = normalized(query)
        if len(unit) != self.dimension:
            raise ConceptVectorError(f"query must have {self.dimension} values")
        if any(not math.isfinite(value) for value in unit):
            raise ConceptVectorError("query must hold finite values")
        pool = self.vectors.keys() if among is None else {concept_identity(name) for name in among} & self.vectors.keys()
        scored = sorted(
            ((identity, cosine(unit, self.vectors[identity].values)) for identity in pool),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(scored[:limit])


class ConceptVectorStore(SceneStore):
    """在 ``scene/concepts/.vectors.json`` 保存旁册；模型/维度不符的文件按空索引处理（派生物，可重算）。"""

    error_type: ClassVar[type[ValueError]] = ConceptVectorError

    def __init__(self, scene_root: str | Path, *, model: str, dimension: int, max_encoded_bytes: int = MAX_VECTORS_BYTES) -> None:
        super().__init__(scene_root)
        self.path = self._inside(Path(CONCEPTS_SEGMENT, CONCEPT_VECTORS_FILENAME))
        self.model = model
        self.dimension = dimension
        self.max_encoded_bytes = max_encoded_bytes

    def empty(self) -> ConceptVectorIndex:
        return ConceptVectorIndex(self.model, self.dimension)

    def read(self) -> ConceptVectorIndex:
        encoded = self._read_bytes(self.path, self.max_encoded_bytes)
        if encoded is None:
            return self.empty()
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConceptVectorError("concept vectors are corrupt") from exc
        if not isinstance(payload, dict) or set(payload) != _KEYS:
            raise ConceptVectorError("concept vectors shape is invalid")
        if (
            payload["schema_version"] != CONCEPT_VECTORS_SCHEMA_VERSION
            or payload["model"] != self.model
            or payload["dimension"] != self.dimension
        ):
            return self.empty()
        raw = payload["vectors"]
        if not isinstance(raw, dict):
            raise ConceptVectorError("concept vectors must be a mapping")
        return ConceptVectorIndex(self.model, self.dimension, {name: self._entry(name, item) for name, item in raw.items()})

    def replace(self, index: ConceptVectorIndex) -> None:
        if not isinstance(index, ConceptVectorIndex):
            raise TypeError("index must be ConceptVectorIndex")
        if index.model != self.model or index.dimension != self.dimension:
            raise ConceptVectorError("vector index does not match this store's embedding")
        payload = {
            "schema_version": CONCEPT_VECTORS_SCHEMA_VERSION,
            "model": self.model,
            "dimension": self.dimension,
            "vectors": {
                name: {"digest": entry.digest, "vector": pack_float16(entry.values)}
                for name, entry in sorted(index.vectors.items())
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self._atomic_write(self.path, encoded, maximum=self.max_encoded_bytes)

    def _entry(self, name: str, item: object) -> ConceptVector:
        if not isinstance(item, dict) or set(item) != _ENTRY_KEYS or not isinstance(item["digest"], str):
            raise ConceptVectorError(f"concept vector entry for {name!r} is malformed")
        try:
            return ConceptVector(item["digest"], unpack_float16(item["vector"], self.dimension))
        except VectorCodecError as exc:
            raise ConceptVectorError(f"concept vector for {name!r} is not decodable") from exc


class _Vector(Protocol):
    @property
    def values(self) -> Sequence[float]: ...


class _Embedder(Protocol):
    """只用到 ``embed_documents``；不 import ModelClient。"""

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[_Vector]: ...


@dataclass(frozen=True)
class ConceptVectorRefreshReport:
    """重算了哪些、收掉了哪些——**按身份**列出，夜批决定重算范围时要用这份清单，不是一个总数。"""

    embedded: tuple[str, ...]
    dropped: tuple[str, ...]


async def refresh_concept_vectors(
    store: ConceptVectorStore, concepts: ConceptSet, embedder: _Embedder
) -> tuple[ConceptVectorIndex, ConceptVectorRefreshReport]:
    """补算缺的或定义改过的，收掉不再存在的概念；落盘并返回新索引。"""

    index = store.read()
    stale = index.stale(concepts)
    if stale:
        definitions = [concepts[identity] for identity in stale]
        vectors = await embedder.embed_documents([embedding_text(definition) for definition in definitions])
        if len(vectors) != len(definitions):
            raise ConceptVectorError("embedder returned a different number of vectors than texts")
        for definition, vector in zip(definitions, vectors, strict=True):
            index = index.with_vector(definition, tuple(vector.values))
    dropped = tuple(sorted(set(index.vectors) - set(concepts)))
    index = index.retain(concepts.keys())
    store.replace(index)
    return index, ConceptVectorRefreshReport(embedded=stale, dropped=dropped)


__all__ = [
    "CONCEPT_VECTORS_SCHEMA_VERSION",
    "ConceptVector",
    "ConceptVectorError",
    "ConceptVectorIndex",
    "ConceptVectorRefreshReport",
    "ConceptVectorStore",
    "embedding_text",
    "refresh_concept_vectors",
    "text_digest",
]
