"""以真实 Runtime 执行检索、Conversation 写入和 MemoryJob 的混合负载基准。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from habitus.benchmark.isolation import isolated_config, require_empty_directory
from habitus.benchmark.metrics import latency_distribution
from habitus.benchmark.model import BenchmarkDataset, BenchmarkQuestion, BenchmarkSample, BenchmarkSession
from habitus.benchmark.runner import BenchmarkRunError, conversation_batch
from habitus.config import HabitusConfig
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import MemoryJobStatus
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import Runtime, build_runtime


class RuntimeLoadBenchmark:
    """对一份真实会话样本施加只读和读写混合负载。"""

    def __init__(
        self,
        config: HabitusConfig,
        dataset: BenchmarkDataset,
        *,
        output_directory: str | Path,
        work_directory: str | Path,
        search_operations: int = 100,
        write_operations: int = 20,
        concurrency: int = 16,
        top_k: int = 10,
        drain_timeout_seconds: float = 1_800.0,
    ) -> None:
        if not isinstance(config, HabitusConfig) or not isinstance(dataset, BenchmarkDataset):
            raise TypeError("runtime load benchmark requires config and dataset")
        if len(dataset.samples) != 1:
            raise ValueError("runtime load benchmark requires exactly one selected sample")
        for name, value, maximum in (
            ("search_operations", search_operations, 1_000_000),
            ("write_operations", write_operations, 100_000),
            ("concurrency", concurrency, 512),
            ("top_k", top_k, config.memory.search_service.max_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between one and {maximum}")
        if (
            isinstance(drain_timeout_seconds, bool)
            or not isinstance(drain_timeout_seconds, int | float)
            or not 1 <= float(drain_timeout_seconds) <= 86_400
        ):
            raise ValueError("drain_timeout_seconds must be between one and 86400")
        self.config = config
        self.dataset = dataset
        self.sample = dataset.samples[0]
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.search_operations = search_operations
        self.write_operations = write_operations
        self.concurrency = concurrency
        self.top_k = top_k
        self.drain_timeout_seconds = float(drain_timeout_seconds)

    async def run(self) -> Mapping[str, object]:
        """先建立稳定基线，再分别测量只读与 MemoryJob 提交期间的混合负载。"""

        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise BenchmarkRunError("runtime load benchmark output directory is not empty")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        require_empty_directory(self.work_directory, label="runtime load work directory")
        scope = hashlib.sha256(str(self.output_directory).encode("utf-8")).hexdigest()
        runtime = build_runtime(
            isolated_config(
                self.config,
                storage_root=self.work_directory,
                collection_scope=f"runtime-load:{scope}",
            )
        )
        runtime.initialize()
        events: list[dict[str, object]] = []
        started_at = datetime.now(UTC).isoformat()
        try:
            await runtime.components.memory.vector_index.rebuild(checkpoint=0)
            await runtime.components.conversation.summary_vector_index.rebuild(checkpoint=0)
            baseline_started = time.perf_counter()
            await _ingest_baseline(runtime, self.sample)
            baseline_latency_ms = (time.perf_counter() - baseline_started) * 1_000

            read_wall_started = time.perf_counter()
            read_events = await self._search_phase(runtime, phase="read_only")
            read_wall_ms = (time.perf_counter() - read_wall_started) * 1_000
            events.extend(read_events)

            await runtime.components.workflow.worker.start()
            mixed_wall_started = time.perf_counter()
            write_task = asyncio.create_task(self._enqueue_writes(runtime), name="benchmark-runtime-writes")
            mixed_search_task = asyncio.create_task(
                self._search_phase(runtime, phase="mixed"),
                name="benchmark-runtime-searches",
            )
            write_events, mixed_search_events = await asyncio.gather(write_task, mixed_search_task)
            active_mixed_wall_ms = (time.perf_counter() - mixed_wall_started) * 1_000
            events.extend(write_events)
            events.extend(mixed_search_events)
            drain_started = time.perf_counter()
            await self._wait_for_drain(runtime)
            drain_latency_ms = (time.perf_counter() - drain_started) * 1_000
            mixed_total_wall_ms = (time.perf_counter() - mixed_wall_started) * 1_000
            await runtime.components.workflow.worker.stop()

            jobs = runtime.components.workflow.jobs.high_watermark()
            memories = runtime.components.memory.tree.list_addresses(limit=10_000)
            summary: dict[str, object] = {
                "schema_version": "habitus_runtime_load_benchmark_v1",
                "started_at": started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "dataset": self.dataset.name.value,
                "sample_id": self.sample.sample_id,
                "search_operations_per_phase": self.search_operations,
                "write_operations": self.write_operations,
                "concurrency": self.concurrency,
                "top_k": self.top_k,
                "baseline_ingest_latency_ms": baseline_latency_ms,
                "read_only": _phase_summary(read_events, wall_latency_ms=read_wall_ms),
                "mixed": _phase_summary(mixed_search_events, wall_latency_ms=active_mixed_wall_ms),
                "writes": _phase_summary(write_events, wall_latency_ms=active_mixed_wall_ms),
                "mixed_total_wall_latency_ms": mixed_total_wall_ms,
                "job_drain_latency_ms": drain_latency_ms,
                "memory_job_high_watermark": jobs,
                "memory_document_count": len(memories),
                "blocked_job": _blocked_job(runtime),
            }
            _write_json(self.output_directory / "summary.json", summary)
            _write_jsonl(self.output_directory / "events.jsonl", events)
            return summary
        finally:
            await runtime.close()

    async def _search_phase(self, runtime: Runtime, *, phase: str) -> tuple[dict[str, object], ...]:
        semaphore = asyncio.Semaphore(self.concurrency)
        questions = self.sample.questions

        async def execute(index: int, question: BenchmarkQuestion) -> dict[str, object]:
            async with semaphore:
                started = time.perf_counter()
                try:
                    result = await runtime.search_memory(question.question, limit=self.top_k)
                    return {
                        "phase": phase,
                        "operation": "search",
                        "operation_id": index,
                        "latency_ms": (time.perf_counter() - started) * 1_000,
                        "success": True,
                        "direct_memory_count": len(result.memories),
                        "summary_fallback_count": len(result.summary_fallbacks),
                        "error": "",
                    }
                except Exception as exc:
                    return {
                        "phase": phase,
                        "operation": "search",
                        "operation_id": index,
                        "latency_ms": (time.perf_counter() - started) * 1_000,
                        "success": False,
                        "direct_memory_count": 0,
                        "summary_fallback_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        return tuple(
            await asyncio.gather(
                *(execute(index, questions[index % len(questions)]) for index in range(self.search_operations))
            )
        )

    async def _enqueue_writes(self, runtime: Runtime) -> tuple[dict[str, object], ...]:
        semaphore = asyncio.Semaphore(self.concurrency)
        sessions = self.sample.sessions

        async def execute(index: int, session: BenchmarkSession) -> dict[str, object]:
            conversation_id = f"load-{index:08d}-{session.session_id}"
            address = ConversationAddress(
                conversation_id=conversation_id,
                started_on=session.started_at.date(),
            )
            batch = conversation_batch(conversation_id, session)
            async with semaphore:
                started = time.perf_counter()
                try:
                    job_count = await asyncio.to_thread(_append_and_flush, runtime, address, batch)
                    runtime.components.workflow.worker.wake()
                    return {
                        "phase": "mixed",
                        "operation": "conversation_commit",
                        "operation_id": index,
                        "latency_ms": (time.perf_counter() - started) * 1_000,
                        "success": True,
                        "job_count": job_count,
                        "error": "",
                    }
                except Exception as exc:
                    return {
                        "phase": "mixed",
                        "operation": "conversation_commit",
                        "operation_id": index,
                        "latency_ms": (time.perf_counter() - started) * 1_000,
                        "success": False,
                        "job_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        return tuple(
            await asyncio.gather(
                *(execute(index, sessions[index % len(sessions)]) for index in range(self.write_operations))
            )
        )

    async def _wait_for_drain(self, runtime: Runtime) -> None:
        deadline = time.monotonic() + self.drain_timeout_seconds
        while time.monotonic() < deadline:
            oldest = await asyncio.to_thread(runtime.components.workflow.jobs.oldest_uncommitted)
            if oldest is None and not runtime.components.workflow.worker.busy:
                return
            if oldest is not None and oldest.status is MemoryJobStatus.FAILED:
                raise BenchmarkRunError(
                    f"runtime load benchmark blocked by FAILED job {oldest.memory_sequence}: {oldest.last_error}"
                )
            await asyncio.sleep(0.05)
        raise TimeoutError("runtime load benchmark did not drain MemoryJobs before timeout")


async def _ingest_baseline(runtime: Runtime, sample: BenchmarkSample) -> None:
    for session in sample.sessions:
        conversation_id = f"baseline-{session.session_id}"
        address = ConversationAddress(
            conversation_id=conversation_id,
            started_on=session.started_at.date(),
        )
        _append_and_flush(runtime, address, conversation_batch(conversation_id, session))
        while True:
            result = await runtime.run_next()
            if result.job is None:
                break


def _append_and_flush(runtime: Runtime, address: ConversationAddress, batch: ConversationBatch) -> int:
    ingest = runtime.components.workflow.enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)
    flushed = runtime.components.workflow.enqueuer.flush(address)
    return len(ingest.jobs) + len(flushed.jobs)


def _phase_summary(events: Sequence[Mapping[str, object]], *, wall_latency_ms: float) -> dict[str, object]:
    latencies = [_number(item["latency_ms"], "operation latency") for item in events]
    successes = [item for item in events if item["success"] is True]
    return {
        "operation_count": len(events),
        "success_count": len(successes),
        "error_count": len(events) - len(successes),
        "wall_latency_ms": wall_latency_ms,
        "operations_per_second": len(events) / max(wall_latency_ms / 1_000, 1e-9),
        "latency_ms": latency_distribution(latencies),
    }


def _blocked_job(runtime: Runtime) -> Mapping[str, object] | None:
    job = runtime.components.workflow.jobs.oldest_uncommitted()
    if job is None:
        return None
    return {
        "memory_sequence": job.memory_sequence,
        "status": job.status.value,
        "attempts": job.attempts,
        "last_error": job.last_error,
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkRunError(f"{label} must be numeric")
    return float(value)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


__all__ = ["RuntimeLoadBenchmark"]
