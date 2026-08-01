"""按索引容量比例和并发阶梯寻找 m2bOS VectorStore 边界。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from benchmark.boundary import (
    BoundaryPolicy,
    BoundaryProfile,
    config_digest,
    environment_metadata,
    process_rss_bytes,
    write_boundary_outputs,
)
from benchmark.isolation import require_empty_directory
from benchmark.vector import (
    VectorBenchmarkDataset,
    VectorBenchmarkDocument,
    VectorBenchmarkRunner,
)
from Config import M2BOSConfig


class VectorBoundaryBenchmark:
    """在真实 Embedder 与 VectorStore 上执行规模和并发二维矩阵。"""

    def __init__(
        self,
        config: M2BOSConfig,
        dataset: VectorBenchmarkDataset,
        *,
        profile: BoundaryProfile,
        policy: BoundaryPolicy,
        output_directory: str | Path,
        work_directory: str | Path,
        top_k: int = 10,
        update_fraction: float = 0.05,
        delete_fraction: float = 0.05,
        maximum_documents: int | None = None,
    ) -> None:
        if not isinstance(config, M2BOSConfig) or not isinstance(dataset, VectorBenchmarkDataset):
            raise TypeError("vector boundary benchmark requires config and vector dataset")
        if not isinstance(profile, BoundaryProfile) or not isinstance(policy, BoundaryPolicy):
            raise TypeError("vector boundary benchmark requires a profile and policy")
        configured_maximum = config.memory.vector_index.max_records
        if maximum_documents is not None:
            if (
                isinstance(maximum_documents, bool)
                or not isinstance(maximum_documents, int)
                or not len(dataset.documents) <= maximum_documents <= configured_maximum
            ):
                raise ValueError("maximum_documents must be within dataset and configured capacity")
            configured_maximum = maximum_documents
        self.config = config
        self.dataset = dataset
        self.profile = profile
        self.policy = policy
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.top_k = top_k
        self.update_fraction = update_fraction
        self.delete_fraction = delete_fraction
        self.maximum_documents = configured_maximum

    async def run(self) -> Mapping[str, object]:
        require_empty_directory(self.output_directory, label="vector boundary output directory")
        require_empty_directory(self.work_directory, label="vector boundary work directory")
        raw_directory = self.output_directory / "raw"
        raw_directory.mkdir()
        targets = tuple(
            sorted(
                {
                    max(len(self.dataset.documents), round(self.maximum_documents * fraction))
                    for fraction in self.profile.vector_capacity_fractions
                }
            )
        )
        points: list[Mapping[str, object]] = []
        for target in targets:
            expanded = expand_vector_dataset(self.dataset, target)
            for repetition in range(1, self.profile.repetitions + 1):
                run_id = f"d{target:09d}-r{repetition:02d}"
                result_output = raw_directory / f"{run_id}-source"
                result_work = self.work_directory / run_id
                rss_before = process_rss_bytes()
                result = await VectorBenchmarkRunner(
                    self.config,
                    expanded,
                    output_directory=result_output,
                    work_directory=result_work,
                    top_k=self.top_k,
                    concurrency=self.profile.concurrency_levels[0],
                    concurrency_levels=self.profile.concurrency_levels,
                    repeats=1,
                    update_fraction=self.update_fraction,
                    delete_fraction=self.delete_fraction,
                    warmup_seconds=self.profile.warmup_seconds,
                    phase_seconds=self.profile.phase_seconds,
                ).run()
                rss_after = process_rss_bytes()
                for search_point in _mapping_array(result.get("search_points"), "vector search points"):
                    concurrency = _positive_integer(search_point.get("concurrency"), "vector concurrency")
                    point_id = f"d{target:09d}-c{concurrency:03d}-r{repetition:02d}"
                    point = _point(
                        result,
                        search_point,
                        target=target,
                        concurrency=concurrency,
                        repetition=repetition,
                        rss_growth_bytes=max(0, rss_after - rss_before),
                    )
                    points.append(point)
                    (raw_directory / f"{point_id}.json").write_text(
                        json.dumps(point, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
        metadata = environment_metadata(
            Path(__file__).resolve().parents[1],
            config_digest=config_digest(self.config),
            dataset_digest=self.dataset.source_sha256,
        )
        metadata.update(
            {
                "dataset": self.dataset.name,
                "provider": self.config.memory.vector_store.provider,
                "adapter": self.config.memory.vector_store.adapter,
                "model": self.config.models.embedding.route.model,
                "dimension": self.config.models.embedding.dimension,
                "configured_max_records": self.config.memory.vector_index.max_records,
                "matrix_maximum_documents": self.maximum_documents,
                "search_mode": "time_based_steady_state",
                "scale_method": "base_quality_anchors_plus_deterministic_distractors",
            }
        )
        return write_boundary_outputs(
            self.output_directory,
            lane="vector",
            profile=self.profile,
            points=points,
            policy=self.policy,
            metadata=metadata,
        )


def expand_vector_dataset(dataset: VectorBenchmarkDataset, document_count: int) -> VectorBenchmarkDataset:
    """保留质量锚点，并用确定性六类记忆背景记录把数据扩展到目标规模。"""

    if not isinstance(dataset, VectorBenchmarkDataset):
        raise TypeError("dataset must be VectorBenchmarkDataset")
    if (
        isinstance(document_count, bool)
        or not isinstance(document_count, int)
        or document_count < len(dataset.documents)
    ):
        raise ValueError("document_count cannot be smaller than the source dataset")
    documents = list(dataset.documents)
    scopes = tuple(dict.fromkeys(item.scope for item in dataset.documents))
    while len(documents) < document_count:
        index = len(documents) - len(dataset.documents)
        scope = scopes[index % len(scopes)]
        documents.append(
            VectorBenchmarkDocument(
                document_id=f"boundary-background-{index:012d}",
                content=(
                    f"边界容量背景记录 {index:012d}，分组 {index % 997:03d}，批次 {index // 997:06d}。"
                    "该记录用于测量大规模记忆索引中的过滤、吞吐与长尾延迟。"
                ),
                scope=scope,
            )
        )
    digest = hashlib.sha256(f"{dataset.source_sha256}\0{document_count}".encode()).hexdigest()
    return VectorBenchmarkDataset(
        name=f"{dataset.name}-expanded-{document_count}",
        source_path=f"{dataset.source_path}#expanded={document_count}",
        source_sha256=digest,
        documents=tuple(documents),
        queries=dataset.queries,
    )


def _point(
    result: Mapping[str, object],
    search_point: Mapping[str, object],
    *,
    target: int,
    concurrency: int,
    repetition: int,
    rss_growth_bytes: int,
) -> Mapping[str, object]:
    search = _mapping(search_point.get("search"), "vector search summary")
    latency = _mapping(search.get("latency_ms"), "vector latency summary")
    query_count = _number(search_point.get("query_execution_count"), "query execution count")
    errors = _number(search.get("error_count"), "query error count")
    incremental = _mapping(result.get("incremental"), "vector incremental summary")
    changed_count = _number(incremental.get("update_count"), "vector update count") + _number(
        incremental.get("delete_count"),
        "vector delete count",
    )
    visible_count = _number(incremental.get("updates_visible"), "visible vector updates") + _number(
        incremental.get("deletes_visible"),
        "visible vector deletes",
    )
    metrics = {
        "operation_count": query_count,
        "throughput_per_second": _number(
            search.get("aggregate_queries_per_second"),
            "aggregate queries per second",
        ),
        "error_rate": errors / max(query_count, 1.0),
        "p50_ms": _number(latency.get("p50"), "vector p50"),
        "p95_ms": _number(latency.get("p95"), "vector p95"),
        "p99_ms": _number(latency.get("p99"), "vector p99"),
        "recall_at_k": _number(search.get("mean_recall_at_k"), "mean recall at k"),
        "filter_leak_count": _number(search.get("filter_leak_count"), "filter leak count"),
        "mutation_visibility_rate": visible_count / max(changed_count, 1.0),
        "rss_growth_bytes": rss_growth_bytes,
        "full_publish_records_per_second": _number(
            result.get("full_publish_records_per_second"),
            "full publish records per second",
        ),
    }
    return {
        "schema_version": "m2bos_boundary_point_v1",
        "lane": "vector",
        "scenario": "filtered_search_after_full_publish",
        "scale_name": "vector_documents",
        "scale_value": target,
        "concurrency": concurrency,
        "write_fraction": 0.0,
        "repetition": repetition,
        "metrics": metrics,
        "incremental": incremental,
        "source_summary": str(result.get("dataset_source", "")),
    }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _mapping_array(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"{label} must be an object array")
    return list(value)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    return float(value)


__all__ = ["VectorBoundaryBenchmark", "expand_vector_dataset"]
