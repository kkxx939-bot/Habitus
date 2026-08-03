"""基于连续真实 ConversationSegment 的 Summary 压缩与清理基准。"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from benchmark.isolation import isolated_config, require_empty_directory
from benchmark.metrics import latency_distribution
from benchmark.model import BenchmarkDataset
from benchmark.runner import BenchmarkRunError, conversation_batch
from Config import M2BOSConfig
from memory.conversation import ConversationAddress
from pre.conversation import ConversationRangeSummaryStage
from Runtime import Runtime, build_runtime


class LifecycleBenchmark:
    """测量 History、Segment Summary、Range 与 Archive 的真实演进成本。"""

    def __init__(
        self,
        config: M2BOSConfig,
        dataset: BenchmarkDataset,
        *,
        output_directory: str | Path,
        work_directory: str | Path,
        age_days: int = 400,
        max_cycles: int = 100,
        stage_source_count: int = 2,
    ) -> None:
        if not isinstance(config, M2BOSConfig) or not isinstance(dataset, BenchmarkDataset):
            raise TypeError("lifecycle benchmark requires config and dataset")
        if len(dataset.samples) != 1:
            raise ValueError("lifecycle benchmark requires exactly one selected sample")
        if len(dataset.samples[0].sessions) < 2:
            raise ValueError("lifecycle benchmark requires at least two ordered sessions")
        if isinstance(age_days, bool) or not isinstance(age_days, int) or not 0 <= age_days <= 36_500:
            raise ValueError("age_days must be between zero and 36500")
        if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or not 1 <= max_cycles <= 10_000:
            raise ValueError("max_cycles must be between one and 10000")
        if (
            isinstance(stage_source_count, bool)
            or not isinstance(stage_source_count, int)
            or not 2 <= stage_source_count <= 100
        ):
            raise ValueError("stage_source_count must be between two and 100")
        if len(dataset.samples[0].sessions) < stage_source_count * 2:
            raise ValueError("lifecycle sample needs at least twice stage_source_count sessions")
        self.config = _lifecycle_benchmark_config(config, stage_source_count)
        self.dataset = dataset
        self.sample = dataset.samples[0]
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.age_days = age_days
        self.max_cycles = max_cycles
        self.stage_source_count = stage_source_count

    async def run(self) -> Mapping[str, object]:
        """顺序生成多个 Segment，再按配置反复维护到当前前沿稳定。"""

        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise BenchmarkRunError("lifecycle benchmark output directory is not empty")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        require_empty_directory(self.work_directory, label="lifecycle benchmark work directory")
        scope = hashlib.sha256(str(self.output_directory).encode("utf-8")).hexdigest()
        runtime = build_runtime(
            isolated_config(
                self.config,
                storage_root=self.work_directory,
                collection_scope=f"lifecycle:{scope}",
            )
        )
        runtime.initialize()
        events: list[dict[str, object]] = []
        try:
            await runtime.components.memory.vector_index.rebuild(checkpoint=0)
            await runtime.components.conversation.summary_vector_index.rebuild(checkpoint=0)
            address = ConversationAddress(
                conversation_id=f"lifecycle-{self.sample.sample_id}",
                started_on=self.sample.sessions[0].started_at.date(),
            )
            sequence = 0
            ingest_started = time.perf_counter()
            for session in self.sample.sessions:
                batch = conversation_batch(
                    address.conversation_id,
                    session,
                    start_sequence=sequence,
                )
                sequence += len(batch.messages)
                runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
                    address,
                    batch,
                    after_turn=True,
                )
                runtime.components.workflow.enqueuer.flush(address)
                while True:
                    job_result = await runtime.run_next()
                    if job_result.job is None:
                        break
            ingest_latency_ms = (time.perf_counter() - ingest_started) * 1_000
            before = _snapshot(runtime, address)
            before_bytes = _tree_bytes(runtime.config.storage_root)
            latest = max(message.occurred_at for session in self.sample.sessions for message in session.messages)
            maintenance_time = latest.astimezone(timezone.utc) + timedelta(days=self.age_days)
            for cycle in range(1, self.max_cycles + 1):
                started = time.perf_counter()
                maintenance = await runtime.maintain_conversation(address, now=maintenance_time)
                event = {
                    "cycle": cycle,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "compaction_created": maintenance.compaction.created,
                    "compaction_reason": maintenance.compaction.reason,
                    "compaction_stage": (
                        maintenance.compaction.summary.stage.value
                        if maintenance.compaction.summary is not None
                        else None
                    ),
                    "purged_history": len(maintenance.purged_history_segment_ids),
                    "released_history": len(maintenance.released_history_segment_ids),
                    "deleted_segment_summaries": len(maintenance.deleted_segment_summary_ids),
                    "deleted_range_summaries": len(maintenance.deleted_range_summary_ids),
                    "deleted_jobs": len(maintenance.deleted_memory_job_sequences),
                    "deleted_receipts": len(maintenance.deleted_memory_receipt_ids),
                }
                events.append(event)
                if not _changed(event):
                    break
            after = _snapshot(runtime, address)
            after_bytes = _tree_bytes(runtime.config.storage_root)
            summary: dict[str, object] = {
                "schema_version": "m2bos_lifecycle_benchmark_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": self.dataset.name.value,
                "sample_id": self.sample.sample_id,
                "session_count": len(self.sample.sessions),
                "message_count": sequence,
                "maintenance_time": maintenance_time.isoformat(),
                "age_days": self.age_days,
                "stage_source_count": self.stage_source_count,
                "policy_mode": "accelerated_mechanics",
                "ingest_latency_ms": ingest_latency_ms,
                "maintenance_cycle_count": len(events),
                "maintenance_latency_ms": latency_distribution(
                    [_number(item["latency_ms"], "maintenance latency") for item in events]
                ),
                "before": before,
                "after": after,
                "storage_bytes_before": before_bytes,
                "storage_bytes_after": after_bytes,
                "storage_reduction_ratio": ((before_bytes - after_bytes) / before_bytes if before_bytes else None),
            }
            _write_json(self.output_directory / "summary.json", summary)
            _write_jsonl(self.output_directory / "cycles.jsonl", events)
            return summary
        finally:
            await runtime.close()


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


def _changed(event: Mapping[str, object]) -> bool:
    return bool(
        event["compaction_created"]
        or any(
            _integer(event[name], name) > 0
            for name in (
                "purged_history",
                "released_history",
                "deleted_segment_summaries",
                "deleted_range_summaries",
                "deleted_jobs",
                "deleted_receipts",
            )
        )
    )


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _lifecycle_benchmark_config(config: M2BOSConfig, source_count: int) -> M2BOSConfig:
    """缩短阶段门槛但保留同一生产执行器，使一次基准可覆盖 Range 与 Archive。"""

    compaction = config.conversation.lifecycle.summary_compaction
    accelerated = replace(
        compaction,
        segment_to_range=replace(
            compaction.segment_to_range,
            min_age_days=0,
            min_source_count=source_count,
            max_wait_days=0,
            max_source_count=source_count,
        ),
        range_to_archive=replace(
            compaction.range_to_archive,
            min_age_days=0,
            min_source_count=2,
            max_source_count=max(2, source_count),
        ),
        recent_use_protection_days=1,
        archive_retire_days=1,
        archive_retire_grace_days=1,
    )
    conversation = replace(
        config.conversation,
        lifecycle=replace(
            config.conversation.lifecycle,
            summary_compaction=accelerated,
        ),
    )
    workflow = replace(
        config.workflow,
        lifecycle=replace(
            config.workflow.lifecycle,
            committed_job_retention_days=0,
            committed_receipt_retention_days=1,
        ),
    )
    return replace(config, conversation=conversation, workflow=workflow)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkRunError(f"{label} must be numeric")
    return float(value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkRunError(f"{label} must be an integer")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


__all__ = ["LifecycleBenchmark"]
