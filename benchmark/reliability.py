"""以真实模型和远程存储为主链，只注入一次明确瞬态故障的恢复基准。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchmark.isolation import isolated_config, require_empty_directory
from benchmark.model import BenchmarkDataset
from benchmark.runner import BenchmarkRunError, conversation_batch
from Config import HabitusConfig
from infrastructure.vector import (
    VectorStore,
    VectorStoreBusyError,
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
)
from memory.conversation import ConversationAddress
from memory.workflow import MemoryJobExecutionError, MemoryJobStatus
from ModelClient import EmbeddingVector
from Runtime import build_runtime


class TransientFailureVectorStore:
    """在指定发布操作连续失败有限次数，其余请求全部委托真实 VectorStore。"""

    def __init__(self, delegate: VectorStore, *, operation: str = "apply", failures: int = 1) -> None:
        if operation not in {"apply", "replace_all"}:
            raise ValueError("fault operation must be apply or replace_all")
        if isinstance(failures, bool) or not isinstance(failures, int) or not 1 <= failures <= 100:
            raise ValueError("failures must be between one and 100")
        self.delegate = delegate
        self.operation = operation
        self.failures = failures
        self.injected_count = 0

    @property
    def injected(self) -> bool:
        return self.injected_count > 0

    @property
    def adapter_name(self) -> str:
        return self.delegate.adapter_name

    @property
    def provider_name(self) -> str:
        return self.delegate.provider_name

    @property
    def collection(self) -> str:
        return self.delegate.collection

    async def initialize(self) -> None:
        await self.delegate.initialize()

    async def state(self) -> VectorStoreState | None:
        return await self.delegate.state()

    async def read(self, identities: tuple[str, ...]) -> Sequence[VectorStoreRecord]:
        return await self.delegate.read(identities)

    async def replace_all(
        self,
        records: tuple[VectorStoreRecord, ...],
        *,
        schema_version: str,
        embedding_fingerprint: str,
        dimension: int,
        checkpoint: int,
        expected_generation: int | None,
    ) -> VectorStoreState:
        self._fail("replace_all")
        return await self.delegate.replace_all(
            records,
            schema_version=schema_version,
            embedding_fingerprint=embedding_fingerprint,
            dimension=dimension,
            checkpoint=checkpoint,
            expected_generation=expected_generation,
        )

    async def apply(
        self,
        upserts: tuple[VectorStoreRecord, ...],
        deletes: tuple[str, ...],
        *,
        checkpoint: int,
        expected_generation: int,
        expected_checkpoint: int,
    ) -> VectorStoreState:
        self._fail("apply")
        return await self.delegate.apply(
            upserts,
            deletes,
            checkpoint=checkpoint,
            expected_generation=expected_generation,
            expected_checkpoint=expected_checkpoint,
        )

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> Sequence[VectorStoreMatch]:
        return await self.delegate.search(query_vector, filters=filters, limit=limit)

    async def scan(
        self,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> Sequence[VectorStoreRecord]:
        return await self.delegate.scan(filters=filters, limit=limit)

    async def close(self) -> None:
        await self.delegate.close()

    def _fail(self, operation: str) -> None:
        if self.injected_count < self.failures and operation == self.operation:
            self.injected_count += 1
            raise VectorStoreBusyError(
                f"benchmark injected transient {operation} failure {self.injected_count}/{self.failures}"
            )


class RecoveryBenchmark:
    """测量 L2 已提交而向量发布瞬态失败后的耐久重放。"""

    def __init__(
        self,
        config: HabitusConfig,
        dataset: BenchmarkDataset,
        *,
        output_directory: str | Path,
        work_directory: str | Path,
    ) -> None:
        if not isinstance(config, HabitusConfig) or not isinstance(dataset, BenchmarkDataset):
            raise TypeError("recovery benchmark requires config and dataset")
        if len(dataset.samples) != 1:
            raise ValueError("recovery benchmark requires exactly one selected sample")
        self.config = config
        self.dataset = dataset
        self.sample = dataset.samples[0]
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()

    async def run(self) -> Mapping[str, object]:
        """注入一次 VectorStoreBusyError，并确认同一 Job 从提交日志恢复完成。"""

        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise BenchmarkRunError("recovery benchmark output directory is not empty")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        require_empty_directory(self.work_directory, label="recovery benchmark work directory")
        scope = hashlib.sha256(str(self.output_directory).encode("utf-8")).hexdigest()
        runtime = build_runtime(
            isolated_config(
                self.config,
                storage_root=self.work_directory,
                collection_scope=f"recovery:{scope}",
            )
        )
        runtime.initialize()
        try:
            await runtime.components.memory.vector_index.rebuild(checkpoint=0)
            await runtime.components.conversation.summary_vector_index.rebuild(checkpoint=0)
            vector_index = runtime.components.memory.vector_index
            fault_store = TransientFailureVectorStore(vector_index.store, operation="apply")
            vector_index.store = fault_store

            session = self.sample.sessions[0]
            conversation_id = f"recovery-{self.sample.sample_id}"
            address = ConversationAddress(conversation_id, session.started_at.date())
            batch = conversation_batch(conversation_id, session)
            runtime.components.workflow.enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)
            runtime.components.workflow.enqueuer.flush(address)

            started = time.perf_counter()
            failure: MemoryJobExecutionError | None = None
            try:
                await runtime.run_next()
            except MemoryJobExecutionError as exc:
                failure = exc
            if failure is None or failure.job is None:
                raise BenchmarkRunError("transient vector failure did not reach the MemoryJob retry boundary")
            failed_job = failure.job
            if failed_job.status is MemoryJobStatus.FAILED:
                raise BenchmarkRunError("transient vector failure was incorrectly classified as terminal")
            if failed_job.next_attempt_at is not None:
                delay = (failed_job.next_attempt_at - datetime.now(UTC)).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)

            recovered = await runtime.run_next()
            if recovered.job is None or recovered.job.status is not MemoryJobStatus.COMMITTED:
                raise BenchmarkRunError("retry did not commit the interrupted MemoryJob")
            if not recovered.recovered:
                raise BenchmarkRunError("retry replanned instead of resuming the committed transaction")
            final_state = await vector_index.store.state()
            if final_state is None or final_state.checkpoint != recovered.job.memory_sequence:
                raise BenchmarkRunError("recovered vector checkpoint does not match the MemoryJob sequence")
            elapsed_ms = (time.perf_counter() - started) * 1_000
            summary: dict[str, object] = {
                "schema_version": "habitus_recovery_benchmark_v1",
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset": self.dataset.name.value,
                "sample_id": self.sample.sample_id,
                "fault_operation": fault_store.operation,
                "fault_injected": fault_store.injected,
                "first_attempt_status": failed_job.status.value,
                "first_attempt_count": failed_job.attempts,
                "final_status": recovered.job.status.value,
                "final_attempt_count": recovered.job.attempts,
                "resumed_committed_transaction": recovered.recovered,
                "summary_indexed": recovered.summary_indexed,
                "memory_vector_indexed": recovered.vector_indexed,
                "receipt_committed": recovered.change_receipt is not None,
                "vector_checkpoint": final_state.checkpoint,
                "recovery_latency_ms": elapsed_ms,
            }
            _write_json(self.output_directory / "summary.json", summary)
            return summary
        finally:
            await runtime.close()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


__all__ = ["RecoveryBenchmark", "TransientFailureVectorStore"]
