"""m2bOS 产品边界矩阵、预算判定与独立运行聚合。"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast


class BoundaryBenchmarkError(RuntimeError):
    """边界计划、原始结果或预算无法形成可信结论。"""


class BoundaryProfileName(str, Enum):
    """与运行成本和目标相对应的正式压力画像。"""

    SMOKE = "smoke"
    STANDARD = "standard"
    STRESS = "stress"
    SOAK = "soak"


@dataclass(frozen=True)
class BoundaryProfile:
    """描述规模、并发、读写比例、稳态时长和独立运行次数。"""

    name: BoundaryProfileName
    conversation_scales: tuple[int, ...]
    vector_capacity_fractions: tuple[float, ...]
    concurrency_levels: tuple[int, ...]
    write_fractions: tuple[float, ...]
    warmup_seconds: float
    phase_seconds: float
    repetitions: int
    sample_interval_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", BoundaryProfileName(self.name))
        for name in ("conversation_scales", "concurrency_levels"):
            values = tuple(getattr(self, name))
            if not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
            ):
                raise ValueError(f"{name} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)
        fractions = tuple(float(value) for value in self.vector_capacity_fractions)
        if not fractions or any(not math.isfinite(value) or not 0 < value <= 1 for value in fractions):
            raise ValueError("vector_capacity_fractions must be between zero and one")
        if tuple(sorted(set(fractions))) != fractions:
            raise ValueError("vector_capacity_fractions must be sorted and unique")
        object.__setattr__(self, "vector_capacity_fractions", fractions)
        writes = tuple(float(value) for value in self.write_fractions)
        if not writes or any(not math.isfinite(value) or not 0 <= value <= 1 for value in writes):
            raise ValueError("write_fractions must be between zero and one")
        if tuple(sorted(set(writes))) != writes:
            raise ValueError("write_fractions must be sorted and unique")
        object.__setattr__(self, "write_fractions", writes)
        for name, minimum, maximum in (
            ("warmup_seconds", 0.0, 3_600.0),
            ("phase_seconds", 1.0, 86_400.0),
            ("sample_interval_seconds", 0.05, 60.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
                raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
            object.__setattr__(self, name, normalized)
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or not 1 <= self.repetitions <= 100
        ):
            raise ValueError("repetitions must be between one and 100")

    @classmethod
    def named(cls, name: str | BoundaryProfileName) -> BoundaryProfile:
        selected = BoundaryProfileName(name)
        profiles = {
            BoundaryProfileName.SMOKE: cls(
                selected,
                conversation_scales=(1,),
                vector_capacity_fractions=(0.001,),
                concurrency_levels=(1, 4),
                write_fractions=(0.0, 0.2),
                warmup_seconds=1.0,
                phase_seconds=5.0,
                repetitions=1,
                sample_interval_seconds=0.25,
            ),
            BoundaryProfileName.STANDARD: cls(
                selected,
                conversation_scales=(1, 10, 50),
                vector_capacity_fractions=(0.01, 0.1),
                concurrency_levels=(1, 2, 4, 8, 16, 32),
                write_fractions=(0.0, 0.1, 0.3, 0.5),
                warmup_seconds=5.0,
                phase_seconds=30.0,
                repetitions=3,
                sample_interval_seconds=0.5,
            ),
            BoundaryProfileName.STRESS: cls(
                selected,
                conversation_scales=(10, 100, 500),
                vector_capacity_fractions=(0.1, 0.5, 1.0),
                concurrency_levels=(1, 4, 8, 16, 32, 64, 128),
                write_fractions=(0.0, 0.1, 0.3, 0.5, 0.8),
                warmup_seconds=10.0,
                phase_seconds=120.0,
                repetitions=3,
                sample_interval_seconds=0.5,
            ),
            BoundaryProfileName.SOAK: cls(
                selected,
                conversation_scales=(100,),
                vector_capacity_fractions=(1.0,),
                concurrency_levels=(16, 32),
                write_fractions=(0.1, 0.3),
                warmup_seconds=30.0,
                phase_seconds=3_600.0,
                repetitions=1,
                sample_interval_seconds=1.0,
            ),
        }
        return profiles[selected]

    def override(
        self,
        *,
        conversation_scales: Sequence[int] | None = None,
        vector_capacity_fractions: Sequence[float] | None = None,
        concurrency_levels: Sequence[int] | None = None,
        write_fractions: Sequence[float] | None = None,
        warmup_seconds: float | None = None,
        phase_seconds: float | None = None,
        repetitions: int | None = None,
    ) -> BoundaryProfile:
        return replace(
            self,
            conversation_scales=(
                self.conversation_scales if conversation_scales is None else tuple(conversation_scales)
            ),
            vector_capacity_fractions=(
                self.vector_capacity_fractions
                if vector_capacity_fractions is None
                else tuple(vector_capacity_fractions)
            ),
            concurrency_levels=(self.concurrency_levels if concurrency_levels is None else tuple(concurrency_levels)),
            write_fractions=(self.write_fractions if write_fractions is None else tuple(write_fractions)),
            warmup_seconds=self.warmup_seconds if warmup_seconds is None else warmup_seconds,
            phase_seconds=self.phase_seconds if phase_seconds is None else phase_seconds,
            repetitions=self.repetitions if repetitions is None else repetitions,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "conversation_scales": list(self.conversation_scales),
            "vector_capacity_fractions": list(self.vector_capacity_fractions),
            "concurrency_levels": list(self.concurrency_levels),
            "write_fractions": list(self.write_fractions),
            "warmup_seconds": self.warmup_seconds,
            "phase_seconds": self.phase_seconds,
            "repetitions": self.repetitions,
            "sample_interval_seconds": self.sample_interval_seconds,
        }


@dataclass(frozen=True)
class BoundaryPolicy:
    """由产品负责人显式给出的可持续运行预算，不在代码中臆造 SLA。"""

    policy_id: str
    max_error_rate: float
    max_p95_ms: float
    max_p99_ms: float
    max_degraded_read_rate: float | None = None
    min_throughput_per_second: float | None = None
    min_recall_at_k: float | None = None
    min_mutation_visibility_rate: float | None = None
    max_filter_leak_count: float | None = None
    max_metrics_observation_error_count: float | None = None
    max_queue_depth: float | None = None
    max_queue_oldest_age_seconds: float | None = None
    max_drain_seconds: float | None = None
    max_rss_growth_bytes: float | None = None
    plateau_gain_fraction: float = 0.05
    knee_latency_multiplier: float = 2.0

    SCHEMA_VERSION = "m2bos_boundary_policy_v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy_id, str)
            or not self.policy_id
            or self.policy_id != self.policy_id.strip()
            or len(self.policy_id) > 128
        ):
            raise ValueError("policy_id must be non-empty normalized text")
        rates = (
            "max_error_rate",
            "max_degraded_read_rate",
            "min_recall_at_k",
            "min_mutation_visibility_rate",
            "plateau_gain_fraction",
        )
        for item in fields(self):
            if item.name in {"SCHEMA_VERSION", "policy_id"}:
                continue
            value = getattr(self, item.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{item.name} must be numeric or null")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{item.name} must be finite and non-negative")
            if item.name in rates and number > 1:
                raise ValueError(f"{item.name} must be between zero and one")
            if item.name == "knee_latency_multiplier" and number < 1:
                raise ValueError("knee_latency_multiplier must be at least one")
            object.__setattr__(self, item.name, number)
        if self.max_p95_ms <= 0 or self.max_p99_ms <= 0:
            raise ValueError("latency budgets must be positive")
        if self.max_p95_ms > self.max_p99_ms:
            raise ValueError("max_p95_ms cannot exceed max_p99_ms")

    @classmethod
    def from_file(cls, path: str | Path) -> BoundaryPolicy:
        source = Path(path).expanduser().resolve(strict=True)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BoundaryBenchmarkError("boundary policy must be valid UTF-8 JSON") from exc
        if not isinstance(raw, Mapping):
            raise BoundaryBenchmarkError("boundary policy must be a JSON object")
        value = dict(raw)
        if value.pop("schema_version", None) != cls.SCHEMA_VERSION:
            raise BoundaryBenchmarkError("unsupported boundary policy schema_version")
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise BoundaryBenchmarkError(f"unknown boundary policy fields: {sorted(unknown)}")
        missing = {"policy_id", "max_error_rate", "max_p95_ms", "max_p99_ms"} - set(value)
        if missing:
            raise BoundaryBenchmarkError(f"boundary policy is missing fields: {sorted(missing)}")
        return cls(**cast(Any, value))

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.SCHEMA_VERSION, **asdict(self)}

    def evaluate(self, metrics: Mapping[str, object]) -> tuple[dict[str, object], ...]:
        definitions = (
            ("error_rate", "<=", self.max_error_rate),
            ("degraded_read_rate", "<=", self.max_degraded_read_rate),
            ("p95_ms", "<=", self.max_p95_ms),
            ("p99_ms", "<=", self.max_p99_ms),
            ("throughput_per_second", ">=", self.min_throughput_per_second),
            ("recall_at_k", ">=", self.min_recall_at_k),
            ("mutation_visibility_rate", ">=", self.min_mutation_visibility_rate),
            ("filter_leak_count", "<=", self.max_filter_leak_count),
            (
                "metrics_observation_error_count",
                "<=",
                self.max_metrics_observation_error_count,
            ),
            ("queue_depth", "<=", self.max_queue_depth),
            ("queue_oldest_age_seconds", "<=", self.max_queue_oldest_age_seconds),
            ("drain_seconds", "<=", self.max_drain_seconds),
            ("rss_growth_bytes", "<=", self.max_rss_growth_bytes),
        )
        checks: list[dict[str, object]] = []
        for metric, relation, threshold in definitions:
            if threshold is None:
                continue
            actual = _number_or_none(metrics.get(metric))
            passed = actual is not None and (actual <= threshold if relation == "<=" else actual >= threshold)
            checks.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "relation": relation,
                    "threshold": threshold,
                    "passed": passed,
                }
            )
        return tuple(checks)


_GROUP_FIELDS = ("lane", "scenario", "scale_name", "scale_value", "concurrency", "write_fraction")


def aggregate_boundary_points(
    points: Sequence[Mapping[str, object]],
    *,
    policy: BoundaryPolicy,
) -> dict[str, object]:
    """按同一压力点聚合独立运行，并识别预算边界与性能拐点。"""

    if not points:
        raise BoundaryBenchmarkError("boundary aggregation requires points")
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for point in points:
        _validate_boundary_point(point)
        groups[tuple(point[name] for name in _GROUP_FIELDS)].append(point)

    aggregated: list[dict[str, object]] = []
    for key, variants in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        metric_names = sorted(
            {
                str(name)
                for variant in variants
                for name, value in cast(Mapping[str, object], variant["metrics"]).items()
                if _number_or_none(value) is not None
            }
        )
        metric_summary = {
            name: _median_mad(
                tuple(
                    value
                    for variant in variants
                    if (value := _number_or_none(cast(Mapping[str, object], variant["metrics"]).get(name))) is not None
                )
            )
            for name in metric_names
        }
        medians = {name: value["median"] for name, value in metric_summary.items()}
        checks = policy.evaluate(medians)
        aggregated.append(
            {
                **dict(zip(_GROUP_FIELDS, key, strict=True)),
                "independent_run_count": len(variants),
                "metrics": metric_summary,
                "checks": list(checks),
                "sustainable": bool(checks) and all(bool(item["passed"]) for item in checks),
            }
        )

    boundaries = _boundaries(aggregated, policy)
    return {
        "schema_version": "m2bos_boundary_aggregate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy.to_dict(),
        "point_count": len(points),
        "aggregated_point_count": len(aggregated),
        "points": aggregated,
        "boundaries": boundaries,
    }


def write_boundary_outputs(
    output_directory: str | Path,
    *,
    lane: str,
    profile: BoundaryProfile,
    points: Sequence[Mapping[str, object]],
    policy: BoundaryPolicy,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """同时保存原始压力点、聚合结论和可读边界报告。"""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_boundary_points(points, policy=policy)
    document: dict[str, object] = {
        "schema_version": "m2bos_boundary_run_v1",
        "lane": lane,
        "profile": profile.to_dict(),
        "metadata": dict(metadata),
        "points": [dict(point) for point in points],
        "aggregate": aggregate,
    }
    _write_json(output / "run_summary.json", document)
    _write_jsonl(output / "points.jsonl", points)
    (output / "summary.md").write_text(_markdown_report(document), encoding="utf-8")
    return document


def aggregate_boundary_runs(
    paths: Sequence[str | Path],
    *,
    policy: BoundaryPolicy,
    output_directory: str | Path,
) -> dict[str, object]:
    """聚合由不同进程产生的可比 run_summary，避免单次结果偶然性。"""

    if len(paths) < 2:
        raise BoundaryBenchmarkError("at least two independent boundary runs are required")
    documents = [_read_json_object(Path(path), label="boundary run") for path in paths]
    reference = documents[0]
    for document in documents:
        if document.get("schema_version") != "m2bos_boundary_run_v1":
            raise BoundaryBenchmarkError("unsupported boundary run schema_version")
        if document.get("lane") != reference.get("lane") or document.get("profile") != reference.get("profile"):
            raise BoundaryBenchmarkError("boundary runs use different lanes or profiles")
        if _comparison_metadata(document) != _comparison_metadata(reference):
            raise BoundaryBenchmarkError("boundary runs use different datasets or configurations")
    points = [point for document in documents for point in _object_array(document.get("points"), "boundary points")]
    aggregate = aggregate_boundary_points(points, policy=policy)
    result: dict[str, object] = {
        "schema_version": "m2bos_boundary_multi_run_v1",
        "lane": reference["lane"],
        "profile": reference["profile"],
        "source_runs": [str(Path(path).expanduser().resolve()) for path in paths],
        "metadata": reference.get("metadata", {}),
        "aggregate": aggregate,
    }
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "run_summary.json", result)
    (output / "summary.md").write_text(_markdown_report(result), encoding="utf-8")
    return result


def environment_metadata(repo_root: str | Path, *, config_digest: str, dataset_digest: str) -> dict[str, object]:
    """记录决定可比性的代码、配置、数据和机器身份。"""

    root = Path(repo_root).expanduser().resolve()
    revision = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "git_dirty": dirty,
        "config_digest": config_digest,
        "dataset_digest": dataset_digest,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "process_max_rss_bytes": process_max_rss_bytes(),
        "byte_order": sys.byteorder,
    }


def process_max_rss_bytes() -> int:
    """把 Unix 平台不一致的 ru_maxrss 单位统一成字节。"""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def process_rss_bytes() -> int:
    """读取当前进程 RSS；不可用时退回平台最大 RSS。"""

    if sys.platform.startswith("linux"):
        try:
            pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(page_size, int):
                return pages * page_size
        except (OSError, ValueError, IndexError):
            pass
    try:
        result = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(os.getpid())),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip()) * 1_024
    except (OSError, ValueError, subprocess.SubprocessError):
        return process_max_rss_bytes()


def directory_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def config_digest(value: object) -> str:
    """对配置的公开结构形成稳定摘要，结果中不复制凭据值。"""

    import hashlib

    if is_dataclass(value) and not isinstance(value, type):
        payload = _jsonable({item.name: getattr(value, item.name) for item in fields(value)})
    elif hasattr(value, "__dict__"):
        payload = _jsonable(vars(value))
    else:
        payload = _jsonable(value)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _boundaries(points: Sequence[Mapping[str, object]], policy: BoundaryPolicy) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    identity = ("lane", "scenario", "scale_name", "scale_value", "write_fraction")
    for point in points:
        grouped[tuple(point[name] for name in identity)].append(point)
    results: list[dict[str, object]] = []
    for key, variants in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        ordered = sorted(variants, key=lambda item: int(cast(int, item["concurrency"])))
        frontier, first_failed, non_monotonic = _contiguous_frontier(ordered)
        knee = _knee(frontier, policy)
        results.append(
            {
                **dict(zip(identity, key, strict=True)),
                "boundary_dimension": "concurrency",
                "highest_sustainable_value": (
                    max(int(cast(int, item["concurrency"])) for item in frontier) if frontier else None
                ),
                "first_failed_value": None if first_failed is None else first_failed["concurrency"],
                "empirical_knee_value": knee,
                "highest_sustainable_concurrency": (
                    max(int(cast(int, item["concurrency"])) for item in frontier) if frontier else None
                ),
                "first_failed_concurrency": None if first_failed is None else first_failed["concurrency"],
                "empirical_knee_concurrency": knee,
                "tested_maximum_value": ordered[-1]["concurrency"],
                "boundary_observed": first_failed is not None,
                "non_monotonic_recovery": non_monotonic,
            }
        )
    scale_identity = ("lane", "scenario", "concurrency", "write_fraction", "scale_name")
    scale_groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for point in points:
        scale_groups[tuple(point[name] for name in scale_identity)].append(point)
    for key, variants in sorted(scale_groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        ordered = sorted(variants, key=lambda item: float(cast(int | float, item["scale_value"])))
        frontier, first_failed, non_monotonic = _contiguous_frontier(ordered)
        results.append(
            {
                **dict(zip(scale_identity, key, strict=True)),
                "boundary_dimension": "scale",
                "highest_sustainable_value": (
                    max(float(cast(int | float, item["scale_value"])) for item in frontier) if frontier else None
                ),
                "first_failed_value": None if first_failed is None else first_failed["scale_value"],
                "empirical_knee_value": None,
                "tested_maximum_value": ordered[-1]["scale_value"],
                "boundary_observed": first_failed is not None,
                "non_monotonic_recovery": non_monotonic,
            }
        )
    return results


def _contiguous_frontier(
    points: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], Mapping[str, object] | None, bool]:
    frontier: list[Mapping[str, object]] = []
    first_failed: Mapping[str, object] | None = None
    non_monotonic = False
    for point in points:
        if first_failed is None and point.get("sustainable") is True:
            frontier.append(point)
        elif first_failed is None:
            first_failed = point
        elif point.get("sustainable") is True:
            non_monotonic = True
    return frontier, first_failed, non_monotonic


def _knee(points: Sequence[Mapping[str, object]], policy: BoundaryPolicy) -> int | None:
    previous: Mapping[str, object] | None = None
    for point in points:
        if previous is not None:
            current_metrics = cast(Mapping[str, Mapping[str, object]], point["metrics"])
            previous_metrics = cast(Mapping[str, Mapping[str, object]], previous["metrics"])
            qps = _nested_median(current_metrics, "throughput_per_second")
            previous_qps = _nested_median(previous_metrics, "throughput_per_second")
            p95 = _nested_median(current_metrics, "p95_ms")
            previous_p95 = _nested_median(previous_metrics, "p95_ms")
            if (
                qps is not None
                and previous_qps is not None
                and p95 is not None
                and previous_p95 is not None
                and previous_qps > 0
                and previous_p95 > 0
            ):
                gain = (qps - previous_qps) / previous_qps
                latency_ratio = p95 / previous_p95
                if gain <= policy.plateau_gain_fraction and latency_ratio >= policy.knee_latency_multiplier:
                    return int(cast(int, point["concurrency"]))
        previous = point
    return None


def _nested_median(metrics: Mapping[str, Mapping[str, object]], name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else _number_or_none(value.get("median"))


def _median_mad(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "median": None, "mad": None, "minimum": None, "maximum": None}
    median = statistics.median(values)
    return {
        "count": len(values),
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _comparison_metadata(document: Mapping[str, object]) -> dict[str, object]:
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        raise BoundaryBenchmarkError("boundary run metadata must be an object")
    return {
        name: metadata.get(name)
        for name in (
            "config_digest",
            "dataset_digest",
            "provider",
            "adapter",
            "model",
            "dimension",
            "server_identity",
        )
    }


def _markdown_report(document: Mapping[str, object]) -> str:
    aggregate = document.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise BoundaryBenchmarkError("boundary report requires aggregate results")
    boundaries = _object_array(aggregate.get("boundaries"), "boundary summaries")
    points = _object_array(aggregate.get("points"), "aggregated points")
    lines = [
        "# m2bOS 产品边界报告",
        "",
        f"- Lane: `{document.get('lane', 'multi')}`",
        f"- Raw points: `{aggregate.get('point_count', 0)}`",
        f"- Aggregated points: `{aggregate.get('aggregated_point_count', 0)}`",
        "",
        "## 边界结论",
        "",
        "| scenario | dimension | fixed scale/concurrency | write | highest sustainable | first failed | knee |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in boundaries:
        lines.append(
            "| {scenario} | {dimension} | {fixed} | {write} | {highest} | {failed} | {knee} |".format(
                scenario=item.get("scenario"),
                dimension=item.get("boundary_dimension"),
                fixed=(
                    item.get("scale_value")
                    if item.get("boundary_dimension") == "concurrency"
                    else item.get("concurrency")
                ),
                write=item.get("write_fraction"),
                highest=item.get("highest_sustainable_value"),
                failed=item.get("first_failed_value"),
                knee=item.get("empirical_knee_value"),
            )
        )
    lines.extend(
        [
            "",
            "## 压力点",
            "",
            "| scenario | scale | concurrency | write | runs | sustainable |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in points:
        lines.append(
            "| {scenario} | {scale} | {concurrency} | {write} | {runs} | {sustainable} |".format(
                scenario=item.get("scenario"),
                scale=item.get("scale_value"),
                concurrency=item.get("concurrency"),
                write=item.get("write_fraction"),
                runs=item.get("independent_run_count"),
                sustainable=item.get("sustainable"),
            )
        )
    return "\n".join(lines) + "\n"


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundaryBenchmarkError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BoundaryBenchmarkError(f"{label} must be an object")
    return value


def _object_array(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise BoundaryBenchmarkError(f"{label} must be an object array")
    return cast(list[Mapping[str, object]], value)


def _number_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_boundary_point(point: Mapping[str, object]) -> None:
    if not isinstance(point, Mapping):
        raise BoundaryBenchmarkError("boundary point must be an object")
    if point.get("schema_version") != "m2bos_boundary_point_v1":
        raise BoundaryBenchmarkError("unsupported boundary point schema_version")
    missing = set(_GROUP_FIELDS) - set(point)
    if missing:
        raise BoundaryBenchmarkError(f"boundary point is missing fields: {sorted(missing)}")
    for name in ("lane", "scenario", "scale_name"):
        value = point.get(name)
        if not isinstance(value, str) or not value.strip():
            raise BoundaryBenchmarkError(f"boundary point {name} must be non-empty text")
    scale = _number_or_none(point.get("scale_value"))
    if scale is None or scale < 0:
        raise BoundaryBenchmarkError("boundary point scale_value must be finite and non-negative")
    concurrency = point.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise BoundaryBenchmarkError("boundary point concurrency must be a positive integer")
    write_fraction = _number_or_none(point.get("write_fraction"))
    if write_fraction is None or not 0 <= write_fraction <= 1:
        raise BoundaryBenchmarkError("boundary point write_fraction must be between zero and one")
    repetition = point.get("repetition")
    if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition <= 0:
        raise BoundaryBenchmarkError("boundary point repetition must be a positive integer")
    metrics = point.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise BoundaryBenchmarkError("boundary point metrics must be a non-empty object")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise BoundaryBenchmarkError("boundary metric names must be non-empty text")
        if value is not None and _number_or_none(value) is None:
            raise BoundaryBenchmarkError(f"boundary metric {name} must be finite numeric or null")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items() if not _secret_key(str(key))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _secret_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return any(marker in normalized for marker in ("credential", "api_key", "password", "secret", "token"))


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * page_count if isinstance(page_size, int) and isinstance(page_count, int) else None


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


__all__ = [
    "BoundaryBenchmarkError",
    "BoundaryPolicy",
    "BoundaryProfile",
    "BoundaryProfileName",
    "aggregate_boundary_points",
    "aggregate_boundary_runs",
    "config_digest",
    "directory_bytes",
    "environment_metadata",
    "process_max_rss_bytes",
    "process_rss_bytes",
    "write_boundary_outputs",
]
