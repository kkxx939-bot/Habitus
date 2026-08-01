"""测量连续瞬态发布故障达到何种程度会耗尽 MemoryJob 恢复预算。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from benchmark.boundary import (
    BoundaryPolicy,
    BoundaryProfile,
    BoundaryProfileName,
    config_digest,
    environment_metadata,
    write_boundary_outputs,
)
from benchmark.isolation import isolated_config, require_empty_directory
from benchmark.metrics import latency_distribution
from benchmark.model import BenchmarkDataset
from benchmark.reliability import TransientFailureVectorStore
from benchmark.runner import conversation_batch
from Config import M2BOSConfig
from memory.conversation import ConversationAddress
from memory.workflow import MemoryJobExecutionError, MemoryJobStatus
from Runtime import build_runtime


class RecoveryBoundaryBenchmark:
    """逐级增加连续故障次数，直到正式 Job 重试预算耗尽。"""

    def __init__(
        self,
        config: M2BOSConfig,
        dataset: BenchmarkDataset,
        *,
        profile: BoundaryProfile,
        policy: BoundaryPolicy,
        output_directory: str | Path,
        work_directory: str | Path,
        fault_counts: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(config, M2BOSConfig) or not isinstance(dataset, BenchmarkDataset):
            raise TypeError("recovery boundary benchmark requires config and dataset")
        if len(dataset.samples) != 1:
            raise ValueError("recovery boundary benchmark requires exactly one selected sample")
        if not isinstance(profile, BoundaryProfile) or not isinstance(policy, BoundaryPolicy):
            raise TypeError("recovery boundary benchmark requires a profile and policy")
        selected = (
            _default_fault_counts(profile.name, config.workflow.jobs.max_attempts)
            if fault_counts is None
            else tuple(fault_counts)
        )
        if (
            not selected
            or any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100 for value in selected)
            or tuple(sorted(set(selected))) != selected
        ):
            raise ValueError("fault_counts must be sorted unique integers between one and 100")
        self.config = config
        self.dataset = dataset
        self.sample = dataset.samples[0]
        self.profile = profile
        self.policy = policy
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.fault_counts = selected

    async def run(self) -> Mapping[str, object]:
        require_empty_directory(self.output_directory, label="recovery boundary output directory")
        require_empty_directory(self.work_directory, label="recovery boundary work directory")
        raw_directory = self.output_directory / "raw"
        raw_directory.mkdir()
        points: list[Mapping[str, object]] = []
        for fault_count in self.fault_counts:
            for repetition in range(1, self.profile.repetitions + 1):
                point_id = f"f{fault_count:03d}-r{repetition:02d}"
                point = await self._run_point(fault_count, repetition, point_id)
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
                "max_attempts": self.config.workflow.jobs.max_attempts,
                "retry_base_delay_seconds": self.config.workflow.jobs.retry_base_delay_seconds,
                "provider": self.config.memory.vector_store.provider,
                "adapter": self.config.memory.vector_store.adapter,
                "model": self.config.models.embedding.route.model,
                "dimension": self.config.models.embedding.dimension,
            }
        )
        return write_boundary_outputs(
            self.output_directory,
            lane="recovery",
            profile=self.profile,
            points=points,
            policy=self.policy,
            metadata=metadata,
        )

    async def _run_point(self, fault_count: int, repetition: int, point_id: str) -> Mapping[str, object]:
        work = self.work_directory / point_id
        work.mkdir()
        runtime = build_runtime(
            isolated_config(
                self.config,
                storage_root=work,
                collection_scope=f"recovery-boundary:{self.output_directory}:{point_id}",
            )
        )
        runtime.initialize()
        try:
            await runtime.components.memory.vector_index.rebuild(checkpoint=0)
            await runtime.components.conversation.summary_vector_index.rebuild(checkpoint=0)
            vector_index = runtime.components.memory.vector_index
            fault_store = TransientFailureVectorStore(vector_index.store, operation="apply", failures=fault_count)
            vector_index.store = fault_store
            session = self.sample.sessions[0]
            conversation_id = f"recovery-boundary-{point_id}"
            address = ConversationAddress(conversation_id, session.started_at.date())
            batch = conversation_batch(conversation_id, session)
            runtime.components.workflow.enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)
            runtime.components.workflow.enqueuer.flush(address)
            started = time.perf_counter()
            attempt_latencies: list[float] = []
            final_status = "unknown"
            final_attempts = 0
            resumed = False
            while True:
                attempt_started = time.perf_counter()
                try:
                    result = await runtime.run_next()
                except MemoryJobExecutionError as exc:
                    attempt_latencies.append((time.perf_counter() - attempt_started) * 1_000)
                    if exc.job is None:
                        raise
                    final_status = exc.job.status.value
                    final_attempts = exc.job.attempts
                    if exc.job.status is MemoryJobStatus.FAILED:
                        break
                    if exc.job.next_attempt_at is not None:
                        delay = (exc.job.next_attempt_at - datetime.now(timezone.utc)).total_seconds()
                        if delay > 0:
                            await asyncio.sleep(delay)
                    continue
                attempt_latencies.append((time.perf_counter() - attempt_started) * 1_000)
                if result.job is None:
                    break
                final_status = result.job.status.value
                final_attempts = result.job.attempts
                resumed = result.recovered
                if result.job.status is MemoryJobStatus.COMMITTED:
                    break
            elapsed = time.perf_counter() - started
            committed = final_status == MemoryJobStatus.COMMITTED.value
            distribution = latency_distribution(attempt_latencies)
            state = await vector_index.store.state()
            metrics = {
                "operation_count": len(attempt_latencies),
                "throughput_per_second": 1 / max(elapsed, 1e-9),
                "error_rate": 0.0 if committed else 1.0,
                "p50_ms": distribution["p50"],
                "p95_ms": distribution["p95"],
                "p99_ms": distribution["p99"],
                "queue_depth": 0.0 if committed else 1.0,
                "queue_oldest_age_seconds": elapsed,
                "drain_seconds": elapsed,
            }
            return {
                "schema_version": "m2bos_boundary_point_v1",
                "lane": "recovery",
                "scenario": "consecutive_vector_apply_failures",
                "scale_name": "consecutive_failures",
                "scale_value": fault_count,
                "concurrency": 1,
                "write_fraction": 1.0,
                "repetition": repetition,
                "injected_failure_count": fault_store.injected_count,
                "final_status": final_status,
                "final_attempts": final_attempts,
                "resumed_committed_transaction": resumed,
                "vector_checkpoint": None if state is None else state.checkpoint,
                "metrics": metrics,
            }
        finally:
            await runtime.close()


def _default_fault_counts(profile: BoundaryProfileName, max_attempts: int) -> tuple[int, ...]:
    if profile is BoundaryProfileName.SMOKE:
        return (1,)
    if profile is BoundaryProfileName.STANDARD:
        return tuple(sorted({1, max(1, max_attempts // 2), max_attempts, max_attempts + 1}))
    if profile is BoundaryProfileName.STRESS:
        return tuple(range(1, min(100, max_attempts + 3) + 1))
    return (1, max_attempts, min(100, max_attempts + 1))


def _dataset_digest(dataset: BenchmarkDataset) -> str:
    payload = "\0".join((dataset.name.value, dataset.source_path, *(sample.digest for sample in dataset.samples)))
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = ["RecoveryBoundaryBenchmark"]
