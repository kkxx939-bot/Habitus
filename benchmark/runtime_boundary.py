"""通过唯一 Runtime 主链寻找检索、写入和 MemoryJob 队列边界。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from benchmark.boundary import (
    BoundaryPolicy,
    BoundaryProfile,
    config_digest,
    directory_bytes,
    environment_metadata,
    process_rss_bytes,
    write_boundary_outputs,
)
from benchmark.isolation import isolated_config, require_empty_directory
from benchmark.metrics import evidence_recall, latency_distribution
from benchmark.model import BenchmarkDataset, BenchmarkQuestion, BenchmarkSample
from benchmark.runner import conversation_batch
from Config import M2BOSConfig
from memory.conversation import ConversationAddress
from pre.conversation import ConversationBatch
from Runtime import Runtime, build_runtime


class RuntimeBoundaryBenchmark:
    """在规模、并发和读写比例矩阵上重复构造真实 Runtime。"""

    def __init__(
        self,
        config: M2BOSConfig,
        dataset: BenchmarkDataset,
        *,
        profile: BoundaryProfile,
        policy: BoundaryPolicy,
        output_directory: str | Path,
        work_directory: str | Path,
        top_k: int = 10,
        drain_timeout_seconds: float = 1_800.0,
    ) -> None:
        if not isinstance(config, M2BOSConfig) or not isinstance(dataset, BenchmarkDataset):
            raise TypeError("runtime boundary benchmark requires config and dataset")
        if len(dataset.samples) != 1:
            raise ValueError("runtime boundary benchmark requires exactly one selected sample")
        if not isinstance(profile, BoundaryProfile) or not isinstance(policy, BoundaryPolicy):
            raise TypeError("runtime boundary benchmark requires a profile and policy")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= config.memory.search_service.max_limit
        ):
            raise ValueError("top_k exceeds SearchService limits")
        if (
            isinstance(drain_timeout_seconds, bool)
            or not isinstance(drain_timeout_seconds, int | float)
            or not 1 <= float(drain_timeout_seconds) <= 86_400
        ):
            raise ValueError("drain_timeout_seconds must be between one and 86400")
        self.config = config
        self.dataset = dataset
        self.sample = dataset.samples[0]
        self.profile = profile
        self.policy = policy
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.top_k = top_k
        self.drain_timeout_seconds = float(drain_timeout_seconds)

    async def run(self) -> Mapping[str, object]:
        """每个压力点使用独立本地根和远程集合，保留原始结果后统一聚合。"""

        require_empty_directory(self.output_directory, label="runtime boundary output directory")
        require_empty_directory(self.work_directory, label="runtime boundary work directory")
        raw_directory = self.output_directory / "raw"
        raw_directory.mkdir()
        points: list[Mapping[str, object]] = []
        for scale in self.profile.conversation_scales:
            for write_fraction in self.profile.write_fractions:
                for concurrency in self.profile.concurrency_levels:
                    for repetition in range(1, self.profile.repetitions + 1):
                        point_id = _point_id(scale, write_fraction, concurrency, repetition)
                        point = await self._run_point(
                            scale=scale,
                            write_fraction=write_fraction,
                            concurrency=concurrency,
                            repetition=repetition,
                            point_id=point_id,
                        )
                        points.append(point)
                        (raw_directory / f"{point_id}.json").write_text(
                            json.dumps(point, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8",
                        )
        metadata = environment_metadata(
            Path(__file__).resolve().parents[1],
            config_digest=config_digest(self.config),
            dataset_digest=_dataset_digest(self.dataset),
        )
        metadata.update(
            {
                "dataset": self.dataset.name.value,
                "sample_id": self.sample.sample_id,
                "top_k": self.top_k,
                "provider": self.config.memory.vector_store.provider,
                "adapter": self.config.memory.vector_store.adapter,
                "model": self.config.models.embedding.route.model,
                "dimension": self.config.models.embedding.dimension,
            }
        )
        return write_boundary_outputs(
            self.output_directory,
            lane="runtime",
            profile=self.profile,
            points=points,
            policy=self.policy,
            metadata=metadata,
        )

    async def _run_point(
        self,
        *,
        scale: int,
        write_fraction: float,
        concurrency: int,
        repetition: int,
        point_id: str,
    ) -> Mapping[str, object]:
        work = self.work_directory / point_id
        work.mkdir()
        runtime = build_runtime(
            isolated_config(
                self.config,
                storage_root=work,
                collection_scope=f"runtime-boundary:{self.output_directory}:{point_id}",
            )
        )
        runtime.initialize()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            await runtime.components.memory.vector_index.rebuild(checkpoint=0)
            await runtime.components.conversation.summary_vector_index.rebuild(checkpoint=0)
            ingest_started = time.perf_counter()
            await _ingest_scale(runtime, self.sample, scale)
            ingest_seconds = time.perf_counter() - ingest_started
            await _warmup(runtime, self.sample.questions, self.top_k, concurrency, self.profile.warmup_seconds)
            await runtime.components.workflow.worker.start()
            rss_before = process_rss_bytes()
            storage_before = directory_bytes(work)
            events, queue_samples, phase_seconds = await _run_phase(
                runtime,
                self.sample,
                top_k=self.top_k,
                concurrency=concurrency,
                write_fraction=write_fraction,
                duration_seconds=self.profile.phase_seconds,
                sample_interval_seconds=self.profile.sample_interval_seconds,
            )
            drain_started = time.perf_counter()
            drained, blocked = await _wait_for_drain(runtime, timeout_seconds=self.drain_timeout_seconds)
            drain_seconds = time.perf_counter() - drain_started
            await runtime.components.workflow.worker.stop()
            rss_after = process_rss_bytes()
            storage_after = directory_bytes(work)
            metrics = _point_metrics(
                events,
                queue_samples,
                phase_seconds=phase_seconds,
                drain_seconds=drain_seconds,
                drained=drained,
                rss_growth_bytes=max(0, rss_after - rss_before),
                storage_growth_bytes=max(0, storage_after - storage_before),
            )
            addresses = runtime.components.memory.tree.list_addresses(limit=10_000)
            return {
                "schema_version": "m2bos_boundary_point_v1",
                "lane": "runtime",
                "scenario": "recall" if write_fraction == 0 else "mixed",
                "scale_name": "baseline_conversations",
                "scale_value": scale,
                "concurrency": concurrency,
                "write_fraction": write_fraction,
                "repetition": repetition,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "baseline_ingest_seconds": ingest_seconds,
                "observed_memory_documents": len(addresses),
                "drained": drained,
                "blocked_job_status": None if blocked is None else blocked,
                "metrics": metrics,
            }
        finally:
            await runtime.close()


async def _ingest_scale(runtime: Runtime, sample: BenchmarkSample, scale: int) -> None:
    for copy_index in range(scale):
        for session in sample.sessions:
            conversation_id = f"boundary-seed-{copy_index:06d}-{session.session_id}"
            address = ConversationAddress(conversation_id, session.started_at.date())
            _append_and_flush(runtime, address, conversation_batch(conversation_id, session))
            while True:
                result = await runtime.run_next()
                if result.job is None:
                    break


async def _warmup(
    runtime: Runtime,
    questions: Sequence[BenchmarkQuestion],
    top_k: int,
    concurrency: int,
    duration_seconds: float,
) -> None:
    if duration_seconds <= 0:
        return
    deadline = time.monotonic() + duration_seconds

    async def worker(offset: int) -> None:
        index = offset
        while time.monotonic() < deadline:
            await runtime.search_memory(questions[index % len(questions)].question, limit=top_k)
            index += concurrency

    await asyncio.gather(*(worker(index) for index in range(concurrency)))


async def _run_phase(
    runtime: Runtime,
    sample: BenchmarkSample,
    *,
    top_k: int,
    concurrency: int,
    write_fraction: float,
    duration_seconds: float,
    sample_interval_seconds: float,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], float]:
    events: list[dict[str, object]] = []
    queue_samples: list[dict[str, object]] = []
    operation_ids = count()
    stop = asyncio.Event()
    ready = asyncio.Event()

    async def execute_recall(worker_id: int, operation_id: int) -> None:
        question = sample.questions[operation_id % len(sample.questions)]
        started = time.perf_counter()
        try:
            result = await runtime.search_memory(question.question, limit=top_k)
            quality = evidence_recall(result.context, question.evidence_texts)
            events.append(
                {
                    "operation": "recall",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": True,
                    "recall_at_k": quality,
                    "degradation_count": len(result.degradations),
                    "error_type": "",
                }
            )
        except Exception as exc:
            events.append(
                {
                    "operation": "recall",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": False,
                    "recall_at_k": None,
                    "degradation_count": 0,
                    "error_type": type(exc).__name__,
                }
            )

    async def execute_remember(worker_id: int, operation_id: int) -> None:
        session = sample.sessions[operation_id % len(sample.sessions)]
        conversation_id = f"boundary-write-{worker_id:03d}-{operation_id:012d}"
        address = ConversationAddress(conversation_id, session.started_at.date())
        batch = conversation_batch(conversation_id, session)
        started = time.perf_counter()
        try:
            jobs = await asyncio.to_thread(_append_and_flush, runtime, address, batch)
            runtime.components.workflow.worker.wake()
            events.append(
                {
                    "operation": "remember",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": True,
                    "recall_at_k": None,
                    "degradation_count": 0,
                    "job_count": jobs,
                    "error_type": "",
                }
            )
        except Exception as exc:
            events.append(
                {
                    "operation": "remember",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": False,
                    "recall_at_k": None,
                    "degradation_count": 0,
                    "job_count": 0,
                    "error_type": type(exc).__name__,
                }
            )

    async def operation_worker(worker_id: int) -> None:
        await ready.wait()
        while not stop.is_set():
            operation_id = next(operation_ids)
            if _is_write_operation(operation_id, write_fraction):
                await execute_remember(worker_id, operation_id)
            else:
                await execute_recall(worker_id, operation_id)

    async def sample_queue() -> None:
        await ready.wait()
        while not stop.is_set():
            snapshot = await asyncio.to_thread(runtime.components.workflow.jobs.observability_snapshot)
            queue_samples.append(
                {
                    "elapsed_seconds": max(0.0, time.perf_counter() - phase_started),
                    "staged": snapshot.staged,
                    "queued": snapshot.queued,
                    "running": snapshot.running,
                    "failed": snapshot.failed,
                    "committed": snapshot.committed,
                    "depth": snapshot.staged + snapshot.queued + snapshot.running,
                    "oldest_age_seconds": snapshot.oldest_age_seconds,
                    "rss_bytes": process_rss_bytes(),
                }
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=sample_interval_seconds)
            except TimeoutError:
                pass

    tasks = [asyncio.create_task(operation_worker(index)) for index in range(concurrency)]
    phase_started = time.perf_counter()
    sampler = asyncio.create_task(sample_queue())
    ready.set()
    await asyncio.sleep(duration_seconds)
    stop.set()
    await asyncio.gather(*tasks)
    await sampler
    return tuple(events), tuple(queue_samples), time.perf_counter() - phase_started


async def _wait_for_drain(runtime: Runtime, *, timeout_seconds: float) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = await asyncio.to_thread(runtime.components.workflow.jobs.observability_snapshot)
        if snapshot.failed:
            job = await asyncio.to_thread(runtime.components.workflow.jobs.oldest_uncommitted)
            return False, None if job is None else job.status.value
        if snapshot.staged + snapshot.queued + snapshot.running == 0 and not runtime.components.workflow.worker.busy:
            return True, None
        await asyncio.sleep(0.05)
    job = await asyncio.to_thread(runtime.components.workflow.jobs.oldest_uncommitted)
    return False, None if job is None else job.status.value


def _append_and_flush(runtime: Runtime, address: ConversationAddress, batch: ConversationBatch) -> int:
    ingest = runtime.components.workflow.enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)
    flushed = runtime.components.workflow.enqueuer.flush(address)
    job_count = len(ingest.jobs) + len(flushed.jobs)
    if job_count <= 0:
        raise RuntimeError("benchmark remember operation produced no MemoryJob")
    return job_count


def _point_metrics(
    events: Sequence[Mapping[str, object]],
    queue_samples: Sequence[Mapping[str, object]],
    *,
    phase_seconds: float,
    drain_seconds: float,
    drained: bool,
    rss_growth_bytes: int,
    storage_growth_bytes: int,
) -> dict[str, object]:
    latencies = [_number(event["latency_ms"], "operation latency") for event in events]
    failures = sum(event.get("success") is not True for event in events) + (0 if drained else 1)
    denominator = len(events) + (0 if drained else 1)
    distribution = latency_distribution(latencies)
    recalls = [
        value
        for event in events
        if (value := _optional_number(event.get("recall_at_k"))) is not None and event.get("success") is True
    ]
    depths = [_number(item["depth"], "queue depth") for item in queue_samples]
    ages = [_number(item["oldest_age_seconds"], "queue age") for item in queue_samples]
    write_count = sum(event.get("operation") == "remember" for event in events)
    read_count = len(events) - write_count
    degraded_read_count = sum(
        event.get("operation") == "recall" and _non_negative_integer(event.get("degradation_count", 0)) > 0
        for event in events
    )
    return {
        "operation_count": len(events),
        "read_operation_count": read_count,
        "write_operation_count": write_count,
        "achieved_write_fraction": write_count / max(len(events), 1),
        "degraded_read_rate": degraded_read_count / max(read_count, 1),
        "throughput_per_second": len(events) / max(phase_seconds, 1e-9),
        "error_rate": failures / max(denominator, 1),
        "p50_ms": distribution["p50"],
        "p95_ms": distribution["p95"],
        "p99_ms": distribution["p99"],
        "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "degradation_count": sum(_non_negative_integer(event.get("degradation_count", 0)) for event in events),
        "queue_depth": max(depths, default=0.0),
        "queue_oldest_age_seconds": max(ages, default=0.0),
        "failed_job_count": max((_number(item["failed"], "failed jobs") for item in queue_samples), default=0.0),
        "drain_seconds": drain_seconds,
        "rss_growth_bytes": rss_growth_bytes,
        "storage_growth_bytes": storage_growth_bytes,
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _optional_number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _non_negative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("degradation count must be a non-negative integer")
    return value


def _is_write_operation(operation_id: int, write_fraction: float) -> bool:
    if write_fraction <= 0:
        return False
    if write_fraction >= 1:
        return True
    return math.floor((operation_id + 1) * write_fraction) > math.floor(operation_id * write_fraction)


def _point_id(scale: int, write_fraction: float, concurrency: int, repetition: int) -> str:
    write_percent = round(write_fraction * 100)
    return f"s{scale:06d}-w{write_percent:03d}-c{concurrency:03d}-r{repetition:02d}"


def _dataset_digest(dataset: BenchmarkDataset) -> str:
    payload = "\0".join((dataset.name.value, dataset.source_path, *(sample.digest for sample in dataset.samples)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["RuntimeBoundaryBenchmark"]
