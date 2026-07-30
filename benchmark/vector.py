"""数据集驱动的真实 VectorStore 构建、过滤检索与增量发布基准。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from benchmark.isolation import isolated_config, require_empty_directory
from benchmark.metrics import latency_distribution
from Config import M2BOSConfig
from infrastructure.vector import (
    VectorStore,
    VectorStoreFilter,
    VectorStoreRecord,
    VectorStoreState,
)
from ModelClient import Embedder, EmbeddingVector
from Runtime import build_runtime


class VectorBenchmarkError(RuntimeError):
    """向量 benchmark 数据、远程状态或结果不满足可比性要求。"""


@dataclass(frozen=True)
class VectorBenchmarkDocument:
    """向量数据集中的一份文档。"""

    document_id: str
    content: str
    scope: str

    def __post_init__(self) -> None:
        for name in ("document_id", "content", "scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"vector document {name} must be non-empty text")


@dataclass(frozen=True)
class VectorBenchmarkQuery:
    """携带目录过滤范围和相关文档标注的一条查询。"""

    query_id: str
    query: str
    scope: str
    relevant_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("query_id", "query", "scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"vector query {name} must be non-empty text")
        if not isinstance(self.relevant_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.relevant_ids
        ):
            raise ValueError("vector query relevant_ids must contain non-empty text")
        if len(self.relevant_ids) != len(set(self.relevant_ids)):
            raise ValueError("vector query relevant_ids must be unique")


@dataclass(frozen=True)
class VectorBenchmarkDataset:
    """独立于具体数据库厂商的文档、查询和过滤标注。"""

    name: str
    source_path: str
    source_sha256: str
    documents: tuple[VectorBenchmarkDocument, ...]
    queries: tuple[VectorBenchmarkQuery, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("vector dataset name must be non-empty text")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("vector dataset source_path must be non-empty text")
        if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64:
            raise ValueError("vector dataset source_sha256 must be a SHA-256 hex digest")
        if not self.documents or any(not isinstance(item, VectorBenchmarkDocument) for item in self.documents):
            raise ValueError("vector dataset requires documents")
        if not self.queries or any(not isinstance(item, VectorBenchmarkQuery) for item in self.queries):
            raise ValueError("vector dataset requires queries")
        identities = {item.document_id for item in self.documents}
        if len(identities) != len(self.documents):
            raise ValueError("vector document IDs must be unique")
        if len({item.query_id for item in self.queries}) != len(self.queries):
            raise ValueError("vector query IDs must be unique")
        for query in self.queries:
            unknown = set(query.relevant_ids) - identities
            if unknown:
                raise ValueError(f"vector query references unknown document IDs: {sorted(unknown)}")
            wrong_scope = {
                item.document_id
                for item in self.documents
                if item.document_id in query.relevant_ids and item.scope != query.scope
            }
            if wrong_scope:
                raise ValueError(f"vector query relevant documents violate its scope: {sorted(wrong_scope)}")


def load_vector_dataset(path: str | Path) -> VectorBenchmarkDataset:
    """读取严格 JSON 数据集；不从测试代码生成预期命中。"""

    source = Path(path).expanduser().resolve(strict=True)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VectorBenchmarkError(f"invalid vector benchmark JSON: {source}") from exc
    root = _mapping(raw, "vector dataset")
    if set(root) != {"schema_version", "name", "documents", "queries"}:
        raise VectorBenchmarkError("vector dataset contains missing or unknown top-level fields")
    if root["schema_version"] != "m2bos_vector_benchmark_v1":
        raise VectorBenchmarkError("unsupported vector benchmark schema_version")
    documents = tuple(_document(value, index) for index, value in enumerate(_array(root["documents"], "documents")))
    queries = tuple(_query(value, index) for index, value in enumerate(_array(root["queries"], "queries")))
    return VectorBenchmarkDataset(
        name=_text(root["name"], "name"),
        source_path=str(source),
        source_sha256=_file_sha256(source),
        documents=documents,
        queries=queries,
    )


class VectorBenchmarkRunner:
    """通过 m2bOS 正式 Embedder 与 VectorStore 协议执行完整远程基准。"""

    def __init__(
        self,
        config: M2BOSConfig,
        dataset: VectorBenchmarkDataset,
        *,
        output_directory: str | Path,
        work_directory: str | Path,
        top_k: int = 10,
        concurrency: int = 8,
        repeats: int = 1,
        update_fraction: float = 0.05,
        delete_fraction: float = 0.05,
        warmup_seconds: float = 0.0,
        phase_seconds: float | None = None,
        concurrency_levels: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(config, M2BOSConfig):
            raise TypeError("config must be M2BOSConfig")
        if not isinstance(dataset, VectorBenchmarkDataset):
            raise TypeError("dataset must be VectorBenchmarkDataset")
        if not 1 <= top_k <= config.memory.vector_index.max_search_hits:
            raise ValueError("top_k exceeds configured vector search capacity")
        if not 1 <= concurrency <= 256 or not 1 <= repeats <= 10_000:
            raise ValueError("vector concurrency or repeats is outside its bound")
        for name, value in (("update_fraction", update_fraction), ("delete_fraction", delete_fraction)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= float(value) < 1:
                raise ValueError(f"{name} must be between zero and one")
        if float(update_fraction) + float(delete_fraction) >= 1:
            raise ValueError("update and delete fractions must leave retained records")
        if len(dataset.documents) > config.memory.vector_index.max_records:
            raise ValueError("vector dataset exceeds configured record capacity")
        if (
            isinstance(warmup_seconds, bool)
            or not isinstance(warmup_seconds, int | float)
            or not 0 <= float(warmup_seconds) <= 3_600
        ):
            raise ValueError("warmup_seconds must be between zero and 3600")
        if phase_seconds is not None and (
            isinstance(phase_seconds, bool)
            or not isinstance(phase_seconds, int | float)
            or not 1 <= float(phase_seconds) <= 86_400
        ):
            raise ValueError("phase_seconds must be between one and 86400 or None")
        selected_concurrency = (concurrency,) if concurrency_levels is None else tuple(concurrency_levels)
        if (
            not selected_concurrency
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256
                for value in selected_concurrency
            )
            or tuple(sorted(set(selected_concurrency))) != selected_concurrency
        ):
            raise ValueError("concurrency_levels must contain sorted unique integers between one and 256")
        self.config = config
        self.dataset = dataset
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.top_k = top_k
        self.concurrency = concurrency
        self.repeats = repeats
        self.update_fraction = float(update_fraction)
        self.delete_fraction = float(delete_fraction)
        self.warmup_seconds = float(warmup_seconds)
        self.phase_seconds = None if phase_seconds is None else float(phase_seconds)
        self.concurrency_levels = selected_concurrency

    async def run(self) -> Mapping[str, object]:
        """执行全量发布、并发过滤检索、增量更新删除和最终一致性检查。"""

        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise VectorBenchmarkError("vector benchmark output directory is not empty")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        require_empty_directory(self.work_directory, label="vector benchmark work directory")
        scope = hashlib.sha256(str(self.output_directory).encode("utf-8")).hexdigest()
        runtime = build_runtime(
            isolated_config(
                self.config,
                storage_root=self.work_directory,
                collection_scope=f"vector:{scope}",
            )
        )
        runtime.initialize()
        store = runtime.components.memory.vector_index.store
        embedder = runtime.components.models.embedder
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            await store.initialize()
            previous = await store.state()
            embed_started = time.perf_counter()
            vectors = await embedder.embed_documents(tuple(item.content for item in self.dataset.documents))
            embedding_latency_ms = (time.perf_counter() - embed_started) * 1_000
            records = tuple(_record(item, vector) for item, vector in zip(self.dataset.documents, vectors, strict=True))
            build_started = time.perf_counter()
            state = await store.replace_all(
                records,
                schema_version="m2bos_vector_benchmark_v1",
                embedding_fingerprint=_embedding_fingerprint(self.config),
                dimension=self.config.models.embedding.dimension,
                checkpoint=1,
                expected_generation=None if previous is None else previous.generation,
            )
            build_latency_ms = (time.perf_counter() - build_started) * 1_000
            query_vectors = await asyncio.gather(*(embedder.embed_query(query.query) for query in self.dataset.queries))
            search_points: list[dict[str, object]] = []
            query_events: list[tuple[int, tuple[dict[str, object], ...]]] = []
            for concurrency in self.concurrency_levels:
                await self._warmup(store, tuple(query_vectors), concurrency=concurrency)
                events, search_wall_ms = await self._search(
                    store,
                    tuple(query_vectors),
                    concurrency=concurrency,
                )
                search_points.append(
                    {
                        "concurrency": concurrency,
                        "query_execution_count": len(events),
                        "search": _search_summary(events, wall_latency_ms=search_wall_ms),
                    }
                )
                query_events.append((concurrency, events))
            mutation = await self._mutate(store, state, records, embedder)
            primary_search = search_points[0]
            result: dict[str, object] = {
                "schema_version": "m2bos_vector_benchmark_result_v1",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "dataset": self.dataset.name,
                "dataset_source": self.dataset.source_path,
                "dataset_sha256": self.dataset.source_sha256,
                "provider": store.provider_name,
                "adapter": store.adapter_name,
                "collection": store.collection,
                "document_count": len(records),
                "query_count": len(self.dataset.queries),
                "query_execution_count": sum(len(events) for _, events in query_events),
                "dimension": self.config.models.embedding.dimension,
                "embedding": _embedding_identity(self.config),
                "top_k": self.top_k,
                "concurrency": self.concurrency,
                "concurrency_levels": list(self.concurrency_levels),
                "repeats": self.repeats,
                "search_mode": "fixed_repeats" if self.phase_seconds is None else "time_based_steady_state",
                "warmup_seconds": self.warmup_seconds,
                "target_phase_seconds": self.phase_seconds,
                "embedding_latency_ms": embedding_latency_ms,
                "full_publish_latency_ms": build_latency_ms,
                "full_publish_records_per_second": len(records) / max(build_latency_ms / 1_000, 1e-9),
                "search": primary_search["search"],
                "search_points": search_points,
                "incremental": mutation,
                "environment": _environment(),
            }
            _write_json(self.output_directory / "summary.json", result)
            if len(query_events) == 1:
                _write_jsonl(self.output_directory / "queries.jsonl", query_events[0][1])
            else:
                for concurrency, events in query_events:
                    _write_jsonl(self.output_directory / f"queries-c{concurrency:03d}.jsonl", events)
            return result
        finally:
            await runtime.close()

    async def _search(
        self,
        store: VectorStore,
        query_vectors: tuple[EmbeddingVector, ...],
        *,
        concurrency: int,
    ) -> tuple[tuple[dict[str, object], ...], float]:
        semaphore = asyncio.Semaphore(concurrency)

        async def execute(
            query: VectorBenchmarkQuery,
            vector: EmbeddingVector,
            repeat: int,
        ) -> dict[str, object]:
            async with semaphore:
                started = time.perf_counter()
                try:
                    matches = await store.search(
                        vector,
                        filters=VectorStoreFilter(
                            equals={"kind": "benchmark"},
                            one_of={"scope_roots": (query.scope,)},
                        ),
                        limit=self.top_k,
                    )
                    latency = (time.perf_counter() - started) * 1_000
                    identities = tuple(match.record.identity for match in matches)
                    leaked = tuple(
                        match.record.identity
                        for match in matches
                        if not _scope_contains(match.record.attributes.get("scope_roots"), query.scope)
                    )
                    relevant = set(query.relevant_ids)
                    hits = len(relevant & set(identities))
                    return {
                        "query_id": query.query_id,
                        "repeat": repeat,
                        "latency_ms": latency,
                        "success": True,
                        "returned_ids": list(identities),
                        "filter_leak_ids": list(leaked),
                        "recall_at_k": hits / len(relevant) if relevant else None,
                        "precision_at_k": hits / len(identities) if identities and relevant else None,
                        "ndcg_at_k": _ndcg(identities, relevant),
                        "error": "",
                    }
                except Exception as exc:
                    return {
                        "query_id": query.query_id,
                        "repeat": repeat,
                        "latency_ms": (time.perf_counter() - started) * 1_000,
                        "success": False,
                        "returned_ids": [],
                        "filter_leak_ids": [],
                        "recall_at_k": None,
                        "precision_at_k": None,
                        "ndcg_at_k": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        started = time.perf_counter()
        if self.phase_seconds is None:
            events = tuple(
                await asyncio.gather(
                    *(
                        execute(query, query_vectors[index], repeat)
                        for repeat in range(self.repeats)
                        for index, query in enumerate(self.dataset.queries)
                    )
                )
            )
        else:
            collected: list[dict[str, object]] = []
            operation_ids = count()
            stop = asyncio.Event()
            ready = asyncio.Event()

            async def worker() -> None:
                await ready.wait()
                while not stop.is_set():
                    operation_id = next(operation_ids)
                    query_index = operation_id % len(self.dataset.queries)
                    collected.append(
                        await execute(
                            self.dataset.queries[query_index],
                            query_vectors[query_index],
                            operation_id,
                        )
                    )

            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            ready.set()
            await asyncio.sleep(self.phase_seconds)
            stop.set()
            await asyncio.gather(*tasks)
            events = tuple(collected)
        return events, (time.perf_counter() - started) * 1_000

    async def _warmup(
        self,
        store: VectorStore,
        query_vectors: tuple[EmbeddingVector, ...],
        *,
        concurrency: int,
    ) -> None:
        if self.warmup_seconds <= 0:
            return
        deadline = time.monotonic() + self.warmup_seconds

        async def worker(offset: int) -> None:
            query_index = offset
            while time.monotonic() < deadline:
                query = self.dataset.queries[query_index % len(self.dataset.queries)]
                await store.search(
                    query_vectors[query_index % len(query_vectors)],
                    filters=VectorStoreFilter(
                        equals={"kind": "benchmark"},
                        one_of={"scope_roots": (query.scope,)},
                    ),
                    limit=self.top_k,
                )
                query_index += concurrency

        await asyncio.gather(*(worker(index) for index in range(concurrency)))

    async def _mutate(
        self,
        store: VectorStore,
        state: VectorStoreState,
        records: tuple[VectorStoreRecord, ...],
        embedder: Embedder,
    ) -> dict[str, object]:
        update_count = _fraction_count(len(records), self.update_fraction)
        delete_count = _fraction_count(len(records), self.delete_fraction)
        if update_count + delete_count >= len(records):
            raise VectorBenchmarkError("incremental fractions leave no retained vector record")
        updates_source = records[:update_count]
        deletes = tuple(record.identity for record in records[update_count : update_count + delete_count])
        updated_contents = tuple(f"{record.content}\n[benchmark revision]" for record in updates_source)
        updated_vectors = await embedder.embed_documents(updated_contents) if updated_contents else ()
        upserts = tuple(
            VectorStoreRecord(
                identity=record.identity,
                vector=vector,
                content=content,
                content_digest=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                attributes=record.attributes,
            )
            for record, content, vector in zip(updates_source, updated_contents, updated_vectors, strict=True)
        )
        started = time.perf_counter()
        next_state = await store.apply(
            upserts,
            deletes,
            checkpoint=2,
            expected_generation=state.generation,
            expected_checkpoint=state.checkpoint,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        changed = await store.read(tuple(record.identity for record in upserts) + deletes)
        observed = {record.identity: record for record in changed}
        update_visible = sum(
            observed.get(record.identity) is not None
            and observed[record.identity].content_digest == record.content_digest
            for record in upserts
        )
        delete_visible = sum(identity not in observed for identity in deletes)
        return {
            "update_count": len(upserts),
            "delete_count": len(deletes),
            "latency_ms": latency_ms,
            "operations_per_second": (len(upserts) + len(deletes)) / max(latency_ms / 1_000, 1e-9),
            "updates_visible": update_visible,
            "deletes_visible": delete_visible,
            "generation": next_state.generation,
            "checkpoint": next_state.checkpoint,
            "record_count": next_state.record_count,
        }


def _document(value: object, index: int) -> VectorBenchmarkDocument:
    item = _mapping(value, f"documents[{index}]")
    if set(item) != {"id", "content", "scope"}:
        raise VectorBenchmarkError(f"documents[{index}] contains missing or unknown fields")
    return VectorBenchmarkDocument(
        document_id=_text(item["id"], "document id"),
        content=_text(item["content"], "document content"),
        scope=_text(item["scope"], "document scope"),
    )


def _query(value: object, index: int) -> VectorBenchmarkQuery:
    item = _mapping(value, f"queries[{index}]")
    if set(item) != {"id", "query", "scope", "relevant_ids"}:
        raise VectorBenchmarkError(f"queries[{index}] contains missing or unknown fields")
    return VectorBenchmarkQuery(
        query_id=_text(item["id"], "query id"),
        query=_text(item["query"], "query text"),
        scope=_text(item["scope"], "query scope"),
        relevant_ids=tuple(_text(child, "relevant id") for child in _array(item["relevant_ids"], "relevant_ids")),
    )


def _record(document: VectorBenchmarkDocument, vector: EmbeddingVector) -> VectorStoreRecord:
    return VectorStoreRecord(
        identity=document.document_id,
        vector=vector,
        content=document.content,
        content_digest=hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
        attributes={
            "uri": f"benchmark://{document.document_id}",
            "level": 2,
            "scope_roots": (document.scope,),
            "kind": "benchmark",
            "revision": 1,
        },
    )


def _search_summary(
    events: Sequence[Mapping[str, object]],
    *,
    wall_latency_ms: float,
) -> dict[str, object]:
    latencies = [_number(item["latency_ms"], "query latency") for item in events]
    succeeded = [item for item in events if item["success"] is True]
    recalls = [_number(item["recall_at_k"], "recall_at_k") for item in succeeded if item["recall_at_k"] is not None]
    precisions = [
        _number(item["precision_at_k"], "precision_at_k") for item in succeeded if item["precision_at_k"] is not None
    ]
    ndcg = [_number(item["ndcg_at_k"], "ndcg_at_k") for item in succeeded if item["ndcg_at_k"] is not None]
    return {
        "success_count": len(succeeded),
        "error_count": len(events) - len(succeeded),
        "latency_ms": latency_distribution(latencies),
        "wall_latency_ms": wall_latency_ms,
        "aggregate_queries_per_second": len(events) / max(wall_latency_ms / 1_000, 1e-9),
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "mean_precision_at_k": sum(precisions) / len(precisions) if precisions else None,
        "mean_ndcg_at_k": sum(ndcg) / len(ndcg) if ndcg else None,
        "filter_leak_count": sum(len(_string_list(item["filter_leak_ids"])) for item in succeeded),
    }


def _ndcg(identities: Sequence[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    dcg = sum(1 / math.log2(index + 2) for index, identity in enumerate(identities) if identity in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), len(identities))))
    return dcg / ideal if ideal else 0.0


def _fraction_count(size: int, fraction: float) -> int:
    if fraction == 0:
        return 0
    return max(1, math.floor(size * fraction))


def _embedding_fingerprint(config: M2BOSConfig) -> str:
    route = config.models.embedding.route
    return f"benchmark:{route.provider}:{route.adapter}:{route.model}:{config.models.embedding.dimension}"


def _embedding_identity(config: M2BOSConfig) -> Mapping[str, object]:
    route = config.models.embedding.route
    return {
        "provider": route.provider,
        "adapter": route.adapter,
        "model": route.model,
        "dimension": config.models.embedding.dimension,
    }


def _environment() -> Mapping[str, object]:
    memory_bytes: int | None = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(page_count, int):
            memory_bytes = page_size * page_count
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "byte_order": sys.byteorder,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _scope_contains(value: object, scope: str) -> bool:
    return isinstance(value, tuple) and scope in value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VectorBenchmarkError(f"{label} must be numeric")
    return float(value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise VectorBenchmarkError("filter_leak_ids must be a string array")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise VectorBenchmarkError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise VectorBenchmarkError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VectorBenchmarkError(f"{label} must be non-empty text")
    return value.strip()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


__all__ = [
    "VectorBenchmarkDataset",
    "VectorBenchmarkDocument",
    "VectorBenchmarkError",
    "VectorBenchmarkQuery",
    "VectorBenchmarkRunner",
    "load_vector_dataset",
]
