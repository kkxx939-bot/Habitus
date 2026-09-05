"""使用正式生命周期策略测量 Conversation 压缩、归档和清理边界。"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from habitus.benchmark.boundary import (
    BoundaryPolicy,
    BoundaryProfile,
    config_digest,
    directory_bytes,
    environment_metadata,
    process_rss_bytes,
    write_boundary_outputs,
)
from habitus.benchmark.isolation import isolated_config, require_empty_directory
from habitus.benchmark.metrics import latency_distribution
from habitus.benchmark.model import BenchmarkDataset, BenchmarkMessage, BenchmarkSample, BenchmarkSession
from habitus.benchmark.runner import conversation_batch
from habitus.config import HabitusConfig
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import ConversationLifecycleMaintenanceResult
from habitus.pre.conversation import ConversationRangeSummaryStage
from habitus.runtime import Runtime, build_runtime


class LifecycleBoundaryBenchmark:
    """扩大同一 Conversation 的真实 Session 数量，不篡改正式压缩阈值。"""

    def __init__(
        self,
        config: HabitusConfig,
        dataset: BenchmarkDataset,
        *,
        profile: BoundaryProfile,
        policy: BoundaryPolicy,
        output_directory: str | Path,
        work_directory: str | Path,
        age_days: int = 400,
        max_cycles: int = 10_000,
    ) -> None:
        if not isinstance(config, HabitusConfig) or not isinstance(dataset, BenchmarkDataset):
            raise TypeError("lifecycle boundary benchmark requires config and dataset")
        if len(dataset.samples) != 1:
            raise ValueError("lifecycle boundary benchmark requires exactly one selected sample")
        if not isinstance(profile, BoundaryProfile) or not isinstance(policy, BoundaryPolicy):
            raise TypeError("lifecycle boundary benchmark requires a profile and policy")
        if isinstance(age_days, bool) or not isinstance(age_days, int) or not 0 <= age_days <= 36_500:
            raise ValueError("age_days must be between zero and 36500")
        if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or not 1 <= max_cycles <= 100_000:
            raise ValueError("max_cycles must be between one and 100000")
        self.config = config
        self.dataset = dataset
        self.sample = dataset.samples[0]
        self.profile = profile
        self.policy = policy
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.age_days = age_days
        self.max_cycles = max_cycles

    async def run(self) -> Mapping[str, object]:
        require_empty_directory(self.output_directory, label="lifecycle boundary output directory")
        require_empty_directory(self.work_directory, label="lifecycle boundary work directory")
        raw_directory = self.output_directory / "raw"
        raw_directory.mkdir()
        points: list[Mapping[str, object]] = []
        for session_count in self.profile.conversation_scales:
            for repetition in range(1, self.profile.repetitions + 1):
                point_id = f"s{session_count:07d}-r{repetition:02d}"
                point = await self._run_point(session_count, repetition, point_id)
                points.append(point)
                (raw_directory / f"{point_id}.json").write_text(
                    json.dumps(point, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
        metadata = environment_metadata(
            Path(__file__).resolve().parents[2],  # 仓库根：本文件在 habitus/benchmark/ 下
            config_digest=config_digest(self.config),
            dataset_digest=_dataset_digest(self.dataset),
        )
        metadata.update(
            {
                "dataset": self.dataset.name.value,
                "sample_id": self.sample.sample_id,
                "policy_mode": "production",
                "age_days": self.age_days,
                "provider": self.config.memory.vector_store.provider,
                "adapter": self.config.memory.vector_store.adapter,
                "model": self.config.models.embedding.route.model,
                "dimension": self.config.models.embedding.dimension,
            }
        )
        return write_boundary_outputs(
            self.output_directory,
            lane="lifecycle",
            profile=self.profile,
            points=points,
            policy=self.policy,
            metadata=metadata,
        )

    async def _run_point(self, session_count: int, repetition: int, point_id: str) -> Mapping[str, object]:
        work = self.work_directory / point_id
        work.mkdir()
        runtime = build_runtime(
            isolated_config(
                self.config,
                storage_root=work,
                collection_scope=f"lifecycle-boundary:{self.output_directory}:{point_id}",
            )
        )
        runtime.initialize()
        try:
            await runtime.components.memory.vector_index.rebuild(checkpoint=0)
            await runtime.components.conversation.summary_vector_index.rebuild(checkpoint=0)
            sessions = _scaled_sessions(self.sample, session_count)
            address = ConversationAddress(
                f"lifecycle-boundary-{point_id}",
                sessions[0].started_at.date(),
            )
            sequence = 0
            rss_before = process_rss_bytes()
            ingest_started = time.perf_counter()
            for session in sessions:
                batch = conversation_batch(address.conversation_id, session, start_sequence=sequence)
                sequence += len(batch.messages)
                runtime.components.workflow.enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)
                runtime.components.workflow.enqueuer.flush(address)
                while True:
                    result = await runtime.run_next()
                    if result.job is None:
                        break
            ingest_seconds = time.perf_counter() - ingest_started
            before = _snapshot(runtime, address)
            storage_before = directory_bytes(work)
            maintenance_time = sessions[-1].messages[-1].occurred_at + timedelta(days=self.age_days)
            latencies: list[float] = []
            changed_cycles = 0
            drained = False
            failure_type = ""
            for _cycle in range(self.max_cycles):
                started = time.perf_counter()
                try:
                    maintenance_result = await runtime.maintain_conversation(address, now=maintenance_time)
                except Exception as exc:
                    latencies.append((time.perf_counter() - started) * 1_000)
                    failure_type = type(exc).__name__
                    break
                latencies.append((time.perf_counter() - started) * 1_000)
                if not _changed(maintenance_result):
                    drained = True
                    break
                changed_cycles += 1
            after = _snapshot(runtime, address)
            storage_after = directory_bytes(work)
            rss_after = process_rss_bytes()
            distribution = latency_distribution(latencies)
            metrics = {
                "operation_count": len(latencies),
                "throughput_per_second": session_count / max(ingest_seconds, 1e-9),
                "error_rate": 0.0 if drained else 1.0,
                "p50_ms": distribution["p50"],
                "p95_ms": distribution["p95"],
                "p99_ms": distribution["p99"],
                "drain_seconds": sum(latencies) / 1_000,
                "rss_growth_bytes": max(0, rss_after - rss_before),
                "storage_growth_bytes": max(0, storage_after - storage_before),
            }
            return {
                "schema_version": "habitus_boundary_point_v1",
                "lane": "lifecycle",
                "scenario": "production_policy_maintenance",
                "scale_name": "conversation_sessions",
                "scale_value": session_count,
                "concurrency": 1,
                "write_fraction": 1.0,
                "repetition": repetition,
                "message_count": sequence,
                "ingest_seconds": ingest_seconds,
                "maintenance_cycle_count": len(latencies),
                "changed_cycle_count": changed_cycles,
                "drained": drained,
                "failure_type": failure_type,
                "storage_bytes_before": storage_before,
                "storage_bytes_after": storage_after,
                "before": before,
                "after": after,
                "metrics": metrics,
            }
        finally:
            await runtime.close()


def _scaled_sessions(sample: BenchmarkSample, target_count: int) -> tuple[BenchmarkSession, ...]:
    source = sample.sessions
    first = min(session.started_at for session in source)
    last = max(message.occurred_at for session in source for message in session.messages)
    span = max(timedelta(days=1), last - first + timedelta(days=1))
    sessions: list[BenchmarkSession] = []
    for index in range(target_count):
        original = source[index % len(source)]
        cycle = index // len(source)
        shift = span * cycle
        messages = tuple(_shifted_message(message, index=index, shift=shift) for message in original.messages)
        sessions.append(
            BenchmarkSession(
                session_id=_scaled_id(original.session_id, index),
                started_at=original.started_at + shift,
                messages=messages,
                source_label=original.source_label,
            )
        )
    return tuple(sessions)


def _shifted_message(message: BenchmarkMessage, *, index: int, shift: timedelta) -> BenchmarkMessage:
    return replace(
        message,
        message_id=_scaled_id(message.message_id, index),
        occurred_at=message.occurred_at + shift,
    )


def _scaled_id(source: str, index: int) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()[:24]
    return f"boundary-{index:07d}-{digest}"


def _snapshot(runtime: Runtime, address: ConversationAddress) -> dict[str, int]:
    conversation = runtime.components.conversation
    compactor = conversation.summary_compactor
    return {
        "history_segments": len(conversation.journal.list_history(address)),
        "segment_summaries": len(conversation.summaries.store.list(address)),
        "range_summaries": len(compactor.range_store.list(address, ConversationRangeSummaryStage.RANGE)),
        "archive_summaries": len(compactor.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE)),
        "jobs": len(runtime.components.workflow.jobs.list_for_conversation(address)),
        "receipts": len(runtime.components.workflow.receipts.list_for_conversation(address)),
    }


def _changed(result: ConversationLifecycleMaintenanceResult) -> bool:
    compaction = result.compaction
    if bool(compaction.created):
        return True
    return any(
        (
            result.purged_history_segment_ids,
            result.released_history_segment_ids,
            result.deleted_segment_summary_ids,
            result.deleted_range_summary_ids,
            result.deleted_memory_job_sequences,
            result.deleted_memory_receipt_ids,
        )
    )


def _dataset_digest(dataset: BenchmarkDataset) -> str:
    payload = "\0".join((dataset.name.value, dataset.source_path, *(sample.digest for sample in dataset.samples)))
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = ["LifecycleBoundaryBenchmark"]
