"""把数据集完整会话送入真实 Habitus Runtime 并执行记忆问答。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from benchmark.context import select_answer_context
from benchmark.isolation import isolated_config
from benchmark.metrics import evidence_recall
from benchmark.model import (
    BenchmarkAnswerRecord,
    BenchmarkDataset,
    BenchmarkMessage,
    BenchmarkQuestion,
    BenchmarkSample,
    BenchmarkSession,
)
from benchmark.prompts import answer_prompt
from benchmark.protocol import OPENVIKING_REFERENCE_REVISION
from Config import HabitusConfig
from memory.conversation import ConversationAddress
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from memory.uri import MemoryURI
from ModelClient import ChatCallContext, ChatMessage, ChatRequest
from pre.conversation import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from Runtime import Runtime, build_runtime


class BenchmarkRunError(RuntimeError):
    """基准无法保证样本隔离、完整导入或结果耐久性。"""


class BenchmarkRunner:
    """按样本隔离记忆根，顺序导入 Session，并并发回答该样本的问题。"""

    def __init__(
        self,
        config: HabitusConfig,
        dataset: BenchmarkDataset,
        *,
        output_directory: str | Path,
        work_directory: str | Path,
        top_k: int = 10,
        max_answer_context_chars: int = 4_000,
        question_concurrency: int = 8,
        resume: bool = False,
    ) -> None:
        if not isinstance(config, HabitusConfig):
            raise TypeError("benchmark config must be HabitusConfig")
        if not isinstance(dataset, BenchmarkDataset):
            raise TypeError("benchmark dataset must be BenchmarkDataset")
        if not 1 <= top_k <= config.memory.search_service.max_limit:
            raise ValueError("benchmark top_k exceeds SearchService limits")
        if not 1 <= question_concurrency <= 256:
            raise ValueError("question_concurrency must be between one and 256")
        if (
            isinstance(max_answer_context_chars, bool)
            or not isinstance(max_answer_context_chars, int)
            or max_answer_context_chars <= 0
        ):
            raise ValueError("max_answer_context_chars must be a positive integer")
        self.base_config = config
        self.dataset = dataset
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.top_k = top_k
        self.max_answer_context_chars = max_answer_context_chars
        self.question_concurrency = question_concurrency
        self.resume = resume
        self.answers_path = self.output_directory / "answers.jsonl"
        self.ingest_path = self.output_directory / "ingest.jsonl"
        self.manifest_path = self.output_directory / "run.json"
        self._collection_scope = hashlib.sha256(f"{dataset.name.value}\0{self.output_directory}".encode()).hexdigest()[
            :16
        ]

    async def run(self) -> tuple[BenchmarkAnswerRecord, ...]:
        """执行全部选定样本，逐条耐久写出答案以支持中断续跑。"""

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.work_directory.mkdir(parents=True, exist_ok=True)
        if not self.resume and (self.answers_path.exists() or self.ingest_path.exists()):
            raise BenchmarkRunError("benchmark output exists; use --resume or a new output directory")
        identity = self._run_identity()
        started_at = datetime.now(UTC).isoformat()
        if self.manifest_path.exists():
            previous = _read_object(self.manifest_path)
            if not self.resume:
                raise BenchmarkRunError("benchmark manifest exists; use --resume or a new output directory")
            if previous.get("identity") != identity:
                raise BenchmarkRunError("benchmark resume settings differ from the existing run")
            raw_started_at = previous.get("started_at")
            if isinstance(raw_started_at, str) and raw_started_at:
                started_at = raw_started_at
        _write_json(
            self.manifest_path,
            {
                "schema_version": "habitus_benchmark_run_v2",
                "status": "running",
                "started_at": started_at,
                "updated_at": datetime.now(UTC).isoformat(),
                "identity": identity,
            },
        )
        existing = self._existing_answers() if self.resume else {}
        completed: list[BenchmarkAnswerRecord] = []
        try:
            for sample in self.dataset.samples:
                sample_records = await self._run_sample(sample, existing)
                completed.extend(sample_records)
        except BaseException as exc:
            _write_json(
                self.manifest_path,
                {
                    "schema_version": "habitus_benchmark_run_v2",
                    "status": "failed",
                    "started_at": started_at,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "identity": identity,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        _write_json(
            self.manifest_path,
            {
                "schema_version": "habitus_benchmark_run_v2",
                "status": "completed",
                "started_at": started_at,
                "updated_at": datetime.now(UTC).isoformat(),
                "identity": identity,
                "answer_count": len(completed),
                "answer_error_count": sum(bool(record.error) for record in completed),
            },
        )
        return tuple(completed)

    def _run_identity(self) -> Mapping[str, object]:
        chat = self.base_config.models.chat.route
        embedding = self.base_config.models.embedding
        rerank = self.base_config.models.rerank
        revision = _source_revision()
        return {
            "dataset": self.dataset.name.value,
            "dataset_source": self.dataset.source_path,
            "dataset_sha256": _file_sha256(Path(self.dataset.source_path)),
            "selected_sample_digests": [sample.digest for sample in self.dataset.samples],
            "sample_count": len(self.dataset.samples),
            "question_count": sum(len(sample.questions) for sample in self.dataset.samples),
            "top_k": self.top_k,
            "max_answer_context_chars": self.max_answer_context_chars,
            "benchmark_protocol": {
                "name": "openviking-public-memory-compatible-v1",
                "openviking_revision": OPENVIKING_REFERENCE_REVISION,
            },
            "question_concurrency": self.question_concurrency,
            "config_fingerprint": hashlib.sha256(repr(self.base_config).encode()).hexdigest(),
            "chat": {
                "provider": chat.provider,
                "adapter": chat.adapter,
                "model": chat.model,
            },
            "embedding": {
                "provider": embedding.route.provider,
                "adapter": embedding.route.adapter,
                "model": embedding.route.model,
                "dimension": embedding.dimension,
            },
            "rerank": (
                None
                if rerank is None
                else {
                    "provider": rerank.route.provider,
                    "adapter": rerank.route.adapter,
                    "model": rerank.route.model,
                }
            ),
            "vector_store": {
                "provider": self.base_config.memory.vector_store.provider,
                "adapter": self.base_config.memory.vector_store.adapter,
            },
            "code_revision": revision,
            "source_tree_sha256": _source_tree_sha256(),
        }

    async def _run_sample(
        self,
        sample: BenchmarkSample,
        existing: Mapping[tuple[str, str], BenchmarkAnswerRecord],
    ) -> tuple[BenchmarkAnswerRecord, ...]:
        sample_root = self.work_directory / self.dataset.name.value / sample.sample_id
        sample_config = isolated_config(
            self.base_config,
            storage_root=sample_root,
            collection_scope=self._collection_scope,
        )
        runtime = build_runtime(sample_config)
        try:
            runtime.initialize()
            # 两个 run-scoped Collection 在样本之间复用；每个样本先从自己的真相源完整重建，
            # 避免为 LongMemEval 的每个问题创建远程 Collection，也避免跨样本污染。
            await runtime.components.memory.vector_index.rebuild(checkpoint=0)
            await runtime.components.conversation.summary_vector_index.rebuild(checkpoint=0)
            await self._ingest_sample(runtime, sample, sample_root)
            pending = [
                question for question in sample.questions if (sample.sample_id, question.question_id) not in existing
            ]
            semaphore = asyncio.Semaphore(self.question_concurrency)

            async def answer(question: BenchmarkQuestion) -> BenchmarkAnswerRecord:
                async with semaphore:
                    return await self._answer(runtime, sample, question)

            generated = tuple(await asyncio.gather(*(answer(question) for question in pending)))
            for record in generated:
                _append_jsonl(self.answers_path, record.to_dict())
            retained = tuple(
                existing[(sample.sample_id, question.question_id)]
                for question in sample.questions
                if (sample.sample_id, question.question_id) in existing
            )
            return (*retained, *generated)
        finally:
            await runtime.close()

    async def _ingest_sample(
        self,
        runtime: Runtime,
        sample: BenchmarkSample,
        sample_root: Path,
    ) -> None:
        manifest_path = sample_root / "benchmark_ingest.json"
        if manifest_path.exists():
            manifest = _read_object(manifest_path)
            if manifest.get("sample_digest") != sample.digest:
                raise BenchmarkRunError(f"sample work tree differs from dataset: {sample.sample_id}")
            if not self.resume:
                raise BenchmarkRunError(f"sample work tree already contains an import: {sample.sample_id}")
            return
        has_conversation_data = runtime.config.conversation_root.exists() and any(
            runtime.config.conversation_root.rglob("*.jsonl")
        )
        if (
            runtime.components.workflow.jobs.high_watermark() != 0
            or runtime.components.memory.tree.list_addresses(limit=10_000)
            or has_conversation_data
        ):
            raise BenchmarkRunError(f"sample work tree contains an incomplete prior import: {sample.sample_id}")

        started = time.perf_counter()
        job_count = 0
        message_count = 0
        committed_receipts = 0
        for session in sample.sessions:
            address = ConversationAddress(
                conversation_id=f"{sample.sample_id}-{session.session_id}",
                started_on=session.started_at.date(),
            )
            batch = conversation_batch(address.conversation_id, session)
            message_count += len(batch.messages)
            ingest = runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
                address,
                batch,
                after_turn=True,
            )
            flushed = runtime.components.workflow.enqueuer.flush(address)
            job_count += len(ingest.jobs) + len(flushed.jobs)
            live = runtime.components.conversation.journal.read_live(address)
            if live is not None:
                raise BenchmarkRunError(
                    f"dataset session is not a complete flushable conversation: {sample.sample_id}/{session.session_id}"
                )
            while True:
                result = await runtime.run_next()
                if result.job is None:
                    break
                if result.change_receipt is not None:
                    committed_receipts += 1

        addresses = runtime.components.memory.tree.list_addresses(limit=10_000)
        kinds: dict[str, int] = {}
        for memory_address in addresses:
            kinds[memory_address.kind.value] = kinds.get(memory_address.kind.value, 0) + 1
        ingest_record: dict[str, object] = {
            "schema_version": "habitus_benchmark_ingest_v1",
            "dataset": self.dataset.name.value,
            "sample_id": sample.sample_id,
            "source_id": sample.source_id,
            "sample_digest": sample.digest,
            "session_count": len(sample.sessions),
            "message_count": message_count,
            "job_count": job_count,
            "committed_receipt_count": committed_receipts,
            "memory_document_count": len(addresses),
            "memory_documents_by_kind": dict(sorted(kinds.items())),
            "ingest_latency_ms": (time.perf_counter() - started) * 1_000,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json(manifest_path, ingest_record)
        _append_jsonl(self.ingest_path, ingest_record)

    async def _answer(
        self,
        runtime: Runtime,
        sample: BenchmarkSample,
        question: BenchmarkQuestion,
    ) -> BenchmarkAnswerRecord:
        retrieval_started = time.perf_counter()
        intention_scope, kinds, expected_summary, expected_related = _retrieval_options(question)
        try:
            search = await runtime.search_memory(
                question.question,
                limit=self.top_k,
                kinds=kinds,
                intention_scope=intention_scope,
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1_000
            answer_context = select_answer_context(
                search,
                max_content_chars=self.max_answer_context_chars,
            )
            context = answer_context.text
            retrieved_uris = tuple(str(item.uri) for item in search.memories)
            related_uris = tuple(
                sorted(
                    {
                        str(MemoryURI.from_address(related.document.address))
                        for memory in search.memories
                        for related in memory.related
                    }
                )
            )
            summary_ids = tuple(item.reference.identity for item in search.summary_fallbacks)
            assessment = search.retrieval_assessment
            sufficient = None if assessment is None else assessment.decision.value == "sufficient"
        except Exception as exc:
            return _answer_error(
                self.dataset.name.value,
                sample,
                question,
                stage="retrieval",
                error=exc,
                retrieval_latency_ms=(time.perf_counter() - retrieval_started) * 1_000,
                answer_context_limit_chars=self.max_answer_context_chars,
                expected_summary_fallback=expected_summary,
                expected_related_memory=expected_related,
            )

        answer_started = time.perf_counter()
        try:
            response = await runtime.components.models.chat.complete_async(
                ChatRequest(
                    messages=(
                        ChatMessage(
                            role="user",
                            content=answer_prompt(self.dataset.name, question, context),
                        ),
                    ),
                    temperature=0.0,
                    max_output_tokens=2_000,
                ),
                context=ChatCallContext(
                    prompt_version="habitus_benchmark_answer_v2",
                    metadata={
                        "dataset": self.dataset.name.value,
                        "sample_id": sample.sample_id,
                        "question_id": question.question_id,
                    },
                ),
            )
            if not response.content or not response.content.strip():
                raise BenchmarkRunError("answer model returned empty content")
        except Exception as exc:
            return _answer_error(
                self.dataset.name.value,
                sample,
                question,
                stage="answer",
                error=exc,
                retrieval_latency_ms=retrieval_latency_ms,
                context_chars=len(context),
                retrieved_uris=retrieved_uris,
                related_uris=related_uris,
                summary_fallback_ids=summary_ids,
                answer_memory_uris=answer_context.memory_uris,
                answer_related_uris=answer_context.related_uris,
                answer_summary_ids=answer_context.summary_ids,
                skipped_context_ids=answer_context.skipped_ids,
                answer_context_limit_chars=self.max_answer_context_chars,
                summary_fallback_attempted=search.summary_fallback_attempted,
                retrieval_sufficient=sufficient,
                expected_summary_fallback=expected_summary,
                expected_related_memory=expected_related,
                answer_latency_ms=(time.perf_counter() - answer_started) * 1_000,
            )
        usage = response.usage
        prompt_evidence_recall = evidence_recall(context, question.evidence_texts)
        return BenchmarkAnswerRecord(
            dataset=self.dataset.name.value,
            sample_id=sample.sample_id,
            source_id=sample.source_id,
            question_id=question.question_id,
            question_type=question.question_type,
            question=question.question,
            reference_answer=question.reference_answer,
            response=response.content.strip(),
            question_time=(question.question_time.isoformat() if question.question_time else ""),
            retrieved_uris=retrieved_uris,
            related_uris=related_uris,
            summary_fallback_ids=summary_ids,
            answer_memory_uris=answer_context.memory_uris,
            answer_related_uris=answer_context.related_uris,
            answer_summary_ids=answer_context.summary_ids,
            skipped_context_ids=answer_context.skipped_ids,
            answer_context_limit_chars=self.max_answer_context_chars,
            summary_fallback_attempted=search.summary_fallback_attempted,
            retrieval_sufficient=sufficient,
            evidence_count=len(question.evidence_texts),
            evidence_recall=prompt_evidence_recall,
            expected_summary_fallback=expected_summary,
            expected_related_memory=expected_related,
            summary_fallback_route_correct=(
                None if expected_summary is None else search.summary_fallback_attempted is expected_summary
            ),
            related_memory_route_correct=(None if expected_related is None else bool(related_uris) is expected_related),
            context_chars=len(context),
            context_tokens_approx=max(0, len(context) // 4),
            answer_input_tokens=usage.input_tokens,
            answer_output_tokens=usage.output_tokens,
            answer_total_tokens=usage.total_tokens,
            retrieval_latency_ms=retrieval_latency_ms,
            answer_latency_ms=(time.perf_counter() - answer_started) * 1_000,
        )

    def _existing_answers(self) -> dict[tuple[str, str], BenchmarkAnswerRecord]:
        if not self.answers_path.exists():
            return {}
        result: dict[tuple[str, str], BenchmarkAnswerRecord] = {}
        for value in _read_jsonl(self.answers_path):
            record = answer_record_from_mapping(value)
            key = (record.sample_id, record.question_id)
            if key in result:
                raise BenchmarkRunError("answers.jsonl contains duplicate question results")
            result[key] = record
        return result


def answer_record_from_mapping(value: Mapping[str, object]) -> BenchmarkAnswerRecord:
    """从耐久 JSONL 恢复回答记录，拒绝缺失的关键字段。"""

    required = {
        "dataset",
        "sample_id",
        "source_id",
        "question_id",
        "question_type",
        "question",
        "reference_answer",
        "response",
        "question_time",
        "retrieved_uris",
        "related_uris",
        "summary_fallback_ids",
        "answer_memory_uris",
        "answer_related_uris",
        "answer_summary_ids",
        "skipped_context_ids",
        "answer_context_limit_chars",
        "summary_fallback_attempted",
        "retrieval_sufficient",
        "evidence_count",
        "evidence_recall",
        "expected_summary_fallback",
        "expected_related_memory",
        "summary_fallback_route_correct",
        "related_memory_route_correct",
        "context_chars",
        "context_tokens_approx",
        "answer_input_tokens",
        "answer_output_tokens",
        "answer_total_tokens",
        "retrieval_latency_ms",
        "answer_latency_ms",
        "error",
    }
    if not required <= set(value):
        raise BenchmarkRunError("answer record is missing required fields")
    return BenchmarkAnswerRecord(
        dataset=str(value["dataset"]),
        sample_id=str(value["sample_id"]),
        source_id=str(value["source_id"]),
        question_id=str(value["question_id"]),
        question_type=str(value["question_type"]),
        question=str(value["question"]),
        reference_answer=str(value["reference_answer"]),
        response=str(value["response"]),
        question_time=str(value["question_time"]),
        retrieved_uris=tuple(str(item) for item in _list_field(value, "retrieved_uris")),
        related_uris=tuple(str(item) for item in _list_field(value, "related_uris")),
        summary_fallback_ids=tuple(str(item) for item in _list_field(value, "summary_fallback_ids")),
        answer_memory_uris=tuple(str(item) for item in _list_field(value, "answer_memory_uris")),
        answer_related_uris=tuple(str(item) for item in _list_field(value, "answer_related_uris")),
        answer_summary_ids=tuple(str(item) for item in _list_field(value, "answer_summary_ids")),
        skipped_context_ids=tuple(str(item) for item in _list_field(value, "skipped_context_ids")),
        answer_context_limit_chars=_int_value(
            value["answer_context_limit_chars"],
            "answer_context_limit_chars",
        ),
        summary_fallback_attempted=_required_bool(
            value["summary_fallback_attempted"],
            "summary_fallback_attempted",
        ),
        retrieval_sufficient=_optional_bool(value["retrieval_sufficient"], "retrieval_sufficient"),
        evidence_count=_int_value(value["evidence_count"], "evidence_count"),
        evidence_recall=(
            None if value["evidence_recall"] is None else _float_value(value["evidence_recall"], "evidence_recall")
        ),
        expected_summary_fallback=_optional_bool(
            value["expected_summary_fallback"],
            "expected_summary_fallback",
        ),
        expected_related_memory=_optional_bool(
            value["expected_related_memory"],
            "expected_related_memory",
        ),
        summary_fallback_route_correct=_optional_bool(
            value["summary_fallback_route_correct"],
            "summary_fallback_route_correct",
        ),
        related_memory_route_correct=_optional_bool(
            value["related_memory_route_correct"],
            "related_memory_route_correct",
        ),
        context_chars=_int_value(value["context_chars"], "context_chars"),
        context_tokens_approx=_int_value(value["context_tokens_approx"], "context_tokens_approx"),
        answer_input_tokens=_int_value(value["answer_input_tokens"], "answer_input_tokens"),
        answer_output_tokens=_int_value(value["answer_output_tokens"], "answer_output_tokens"),
        answer_total_tokens=_int_value(value["answer_total_tokens"], "answer_total_tokens"),
        retrieval_latency_ms=_float_value(value["retrieval_latency_ms"], "retrieval_latency_ms"),
        answer_latency_ms=_float_value(value["answer_latency_ms"], "answer_latency_ms"),
        error=str(value["error"]),
    )


def load_answer_records(path: str | Path) -> tuple[BenchmarkAnswerRecord, ...]:
    """读取 answers.jsonl，并确保题目身份不重复。"""

    records = tuple(answer_record_from_mapping(value) for value in _read_jsonl(Path(path)))
    identities = tuple((record.sample_id, record.question_id) for record in records)
    if len(set(identities)) != len(identities):
        raise BenchmarkRunError("answers JSONL contains duplicate question identities")
    return records


def conversation_batch(
    conversation_id: str,
    session: BenchmarkSession,
    *,
    start_sequence: int = 0,
) -> ConversationBatch:
    """把数据集 Session 转成严格保留角色和工具边界的 ConversationBatch。"""

    if isinstance(start_sequence, bool) or not isinstance(start_sequence, int) or start_sequence < 0:
        raise ValueError("start_sequence must be a non-negative integer")
    messages = tuple(
        _conversation_message(message, start_sequence + offset) for offset, message in enumerate(session.messages)
    )
    return ConversationBatch(conversation_id=conversation_id, messages=messages)


def _conversation_message(message: BenchmarkMessage, sequence: int) -> ConversationMessage:
    role = ConversationMessageRole(message.role)
    if role is ConversationMessageRole.TOOL_RESULT:
        if message.tool_status is None:
            raise BenchmarkRunError("tool_result benchmark message lost its terminal status")
        return ConversationMessage(
            message_id=message.message_id,
            sequence=sequence,
            role=role,
            occurred_at=message.occurred_at,
            content=message.content,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_status=ConversationToolResultStatus(message.tool_status),
            content_mode=ConversationToolResultContentMode.INLINE,
            source_ref=message.source_ref,
        )
    return ConversationMessage(
        message_id=message.message_id,
        sequence=sequence,
        role=role,
        occurred_at=message.occurred_at,
        content=message.content,
        tool_call_id=message.tool_call_id,
        tool_name=message.tool_name,
    )


def _answer_error(
    dataset: str,
    sample: BenchmarkSample,
    question: BenchmarkQuestion,
    *,
    stage: str,
    error: Exception,
    retrieval_latency_ms: float,
    context_chars: int = 0,
    retrieved_uris: tuple[str, ...] = (),
    related_uris: tuple[str, ...] = (),
    summary_fallback_ids: tuple[str, ...] = (),
    answer_memory_uris: tuple[str, ...] = (),
    answer_related_uris: tuple[str, ...] = (),
    answer_summary_ids: tuple[str, ...] = (),
    skipped_context_ids: tuple[str, ...] = (),
    answer_context_limit_chars: int = 0,
    summary_fallback_attempted: bool = False,
    retrieval_sufficient: bool | None = None,
    expected_summary_fallback: bool | None = None,
    expected_related_memory: bool | None = None,
    answer_latency_ms: float = 0.0,
) -> BenchmarkAnswerRecord:
    return BenchmarkAnswerRecord(
        dataset=dataset,
        sample_id=sample.sample_id,
        source_id=sample.source_id,
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        reference_answer=question.reference_answer,
        response="",
        question_time=question.question_time.isoformat() if question.question_time else "",
        retrieved_uris=retrieved_uris,
        related_uris=related_uris,
        summary_fallback_ids=summary_fallback_ids,
        answer_memory_uris=answer_memory_uris,
        answer_related_uris=answer_related_uris,
        answer_summary_ids=answer_summary_ids,
        skipped_context_ids=skipped_context_ids,
        answer_context_limit_chars=answer_context_limit_chars,
        summary_fallback_attempted=summary_fallback_attempted,
        retrieval_sufficient=retrieval_sufficient,
        evidence_count=len(question.evidence_texts),
        evidence_recall=None,
        expected_summary_fallback=expected_summary_fallback,
        expected_related_memory=expected_related_memory,
        summary_fallback_route_correct=None,
        related_memory_route_correct=None,
        context_chars=context_chars,
        context_tokens_approx=max(0, context_chars // 4),
        answer_input_tokens=0,
        answer_output_tokens=0,
        answer_total_tokens=0,
        retrieval_latency_ms=retrieval_latency_ms,
        answer_latency_ms=answer_latency_ms,
        error=f"{stage}: {type(error).__name__}: {error}",
    )


def _list_field(value: Mapping[str, object], name: str) -> list[object]:
    selected = value[name]
    if not isinstance(selected, list):
        raise BenchmarkRunError(f"answer record {name} must be an array")
    return selected


def _int_value(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkRunError(f"answer record {label} must be an integer")
    return value


def _float_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkRunError(f"answer record {label} must be numeric")
    return float(value)


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise BenchmarkRunError(f"answer record {label} must be boolean or null")
    return value


def _required_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkRunError(f"answer record {label} must be boolean")
    return value


def _retrieval_options(
    question: BenchmarkQuestion,
) -> tuple[MemoryIntentionRecallScope, tuple[MemoryKind, ...], bool | None, bool | None]:
    raw_scope = question.metadata.get("intention_scope", MemoryIntentionRecallScope.ACTIVE.value)
    if not isinstance(raw_scope, str):
        raise BenchmarkRunError("question metadata intention_scope must be text")
    try:
        intention_scope = MemoryIntentionRecallScope(raw_scope)
    except ValueError as exc:
        raise BenchmarkRunError("question metadata intention_scope is unsupported") from exc
    raw_kinds = question.metadata.get("kinds", ())
    if not isinstance(raw_kinds, list | tuple) or any(not isinstance(item, str) for item in raw_kinds):
        raise BenchmarkRunError("question metadata kinds must be a string array")
    try:
        kinds = tuple(MemoryKind(item) for item in raw_kinds)
    except ValueError as exc:
        raise BenchmarkRunError("question metadata kinds contains an unsupported memory kind") from exc
    if len(kinds) != len(set(kinds)):
        raise BenchmarkRunError("question metadata kinds must be unique")
    return (
        intention_scope,
        kinds,
        _metadata_optional_bool(question.metadata, "expect_summary_fallback"),
        _metadata_optional_bool(question.metadata, "expect_related_memory"),
    )


def _metadata_optional_bool(metadata: Mapping[str, object], name: str) -> bool | None:
    value = metadata.get(name)
    if value is not None and not isinstance(value, bool):
        raise BenchmarkRunError(f"question metadata {name} must be boolean")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    result: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise BenchmarkRunError(f"JSONL contains an empty line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkRunError(f"JSONL is invalid at line {line_number}") from exc
        if not isinstance(value, Mapping):
            raise BenchmarkRunError(f"JSONL line {line_number} must be an object")
        result.append(value)
    return tuple(result)


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkRunError(f"invalid benchmark manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise BenchmarkRunError(f"benchmark manifest must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return revision or "unknown"


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files: list[Path] = []
    for name in (
        "Config",
        "ModelClient",
        "Runtime",
        "benchmark",
        "foundation",
        "infrastructure",
        "memory",
        "pre",
    ):
        directory = root / name
        if directory.exists():
            benchmark_results = root / "benchmark" / "results"
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not path.is_relative_to(benchmark_results)
                and path.suffix in {".json", ".md", ".py", ".yaml", ".yml"}
            )
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "BenchmarkRunError",
    "BenchmarkRunner",
    "answer_record_from_mapping",
    "conversation_batch",
    "load_answer_records",
]
