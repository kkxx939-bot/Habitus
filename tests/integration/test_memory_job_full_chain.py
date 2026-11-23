"""从完整 ConversationSegment 到 Job/Receipt/双向量 checkpoint 的主链验收测试。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.foundation.observability import ObservationEvent
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.memory.editor import MemoryTransactionJournalState
from habitus.memory.workflow import (
    MemoryChangeReceiptState,
    MemoryChangeSource,
    MemoryJobExecutionError,
    MemoryJobStatus,
)
from habitus.model_client import (
    ModelResponse,
    ProviderCapabilities,
    ProviderFactory,
)
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import build_runtime
from tests.helpers import BASE_TIME, closed_turn, summary_content, tool_turn
from tests.lock_store import ProcessLocalLockStore
from tests.model_helpers import prepare_chat_request

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def runtime_config(tmp_path: Path, *, max_attempts: int = 3) -> HabitusConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    # 单测不往 127.0.0.1:4318 真导出：没关掉的 Runtime 会留下每 10 秒导出一次的线程，串进后面设了代理环境变量的测试
    payload["observability"]["tracing"]["enabled"] = False
    payload["models"]["chat"]["route"].update(
        provider="fake",
        adapter="fake_chat",
        credential_ref="",
    )
    payload["workflow"]["jobs"]["max_attempts"] = max_attempts
    return HabitusConfig.from_mapping(payload)


@dataclass
class DispatchingChatProvider:
    provider_name: str
    model: str
    response_names: list[str] = field(default_factory=list)
    capabilities: ProviderCapabilities = ProviderCapabilities()
    is_remote: bool = False
    preferences: list[dict[str, object]] = field(default_factory=list)
    retrieval_script: list[dict[str, object]] = field(default_factory=list)
    candidate_script: list[Callable[[dict[str, object]], dict[str, object]]] = field(default_factory=list)
    candidate_requests: list[object] = field(default_factory=list)

    prepare = staticmethod(prepare_chat_request)

    def complete(self, prepared):
        request = prepared.request
        assert request.response_format is not None
        response_name = request.response_format.name
        self.response_names.append(response_name)
        summary = summary_content()
        payloads = {
            "conversation_segment_summary_field_operations": {
                "operations": [
                    {"field": "overview", "operation": "update", "content": summary.overview},
                    {
                        "field": "chronology",
                        "operation": "append",
                        "items": list(summary.chronology),
                    },
                    {
                        "field": "ending_state",
                        "operation": "update",
                        "content": summary.ending_state,
                    },
                ]
            },
            "memory_retrieval_decision": {
                "status": "irrelevant",
                "action": "finish",
                "directory": None,
                "uris": [],
                "keywords": [],
                "reason": "当前没有相关旧记忆。",
            },
            "memory_candidate_batch": {
                "profile": [],
                "preferences": self.preferences,
                "entities": [],
                "tools": [],
                "identity_proposals": [],
                "relations": [],
            },
        }
        if response_name == "memory_retrieval_decision" and self.retrieval_script:
            payloads[response_name] = self.retrieval_script.pop(0)
        if response_name == "memory_candidate_batch":
            self.candidate_requests.append(request)
            # 修补轮：原请求之后接着上一次的答复与错误清单；剧本用完时模型照旧答上一次的内容
            repair = len(request.messages) >= 3 and request.messages[-2].role == "assistant"
            facts = request.messages[-3] if repair else request.messages[-1]
            if self.candidate_script:
                visible = json.loads(str(facts.content).split("\n", 1)[1])
                payloads[response_name] = self.candidate_script.pop(0)(visible)
            elif repair:
                return ModelResponse(str(request.messages[-2].content), self.model, self.provider_name)
        payload = payloads[response_name]
        return ModelResponse(json.dumps(payload, ensure_ascii=False), self.model, self.provider_name)

    async def complete_async(self, request):
        return self.complete(request)

    def stream(self, _request):
        return iter(())

    async def stream_async(self, _request):
        if False:
            yield None

    def health_check(self):
        return {"ok": True}

    async def aclose(self):
        return None


def dependencies():
    providers = ProviderFactory()
    chat_instances: list[DispatchingChatProvider] = []

    def chat_builder(context):
        provider = DispatchingChatProvider(context.route.provider, context.route.model)
        chat_instances.append(provider)
        return provider

    providers.register_adapter("chat", "fake_chat", chat_builder)
    return providers, chat_instances


def test_full_job_chain_commits_summary_receipt_and_job_before_cleaning_journal(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path)
    providers, chats = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-1", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]

    result = asyncio.run(runtime.run_next())

    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.job.memory_sequence == queued.memory_sequence == 1
    assert result.change_receipt is not None
    assert result.change_receipt.state is MemoryChangeReceiptState.COMMITTED
    assert result.change_receipt.node_changes == ()
    assert result.summary_generated
    assert result.journal_cleaned
    assert runtime.components.memory.editor.transaction.journal.try_read(queued.transaction_id) is None
    summary = runtime.components.conversation.summaries.store.read(
        address,
        runtime.components.conversation.journal.read_segment(address, queued.segment_id),
    )
    assert summary.source_message_digest == queued.source_segment_digest
    assert set(chats[0].response_names) == {
        "conversation_segment_summary_field_operations",
        "memory_retrieval_decision",
        "memory_candidate_batch",
    }


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def test_empty_tree_final_round_still_insufficient_commits_job_and_audits_limit(
    tmp_path: Path,
) -> None:
    """主链端到端实测（Job #1 首次）：空树上两轮列目录都为空，终轮如实答不足，Job 仍要写完。"""

    config = runtime_config(tmp_path, max_attempts=1)
    providers, chats = dependencies()
    observer = RecordingObserver()
    runtime = build_runtime(
        config,
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
        observer=observer,
    )
    runtime.initialize()
    chats[0].retrieval_script.extend(
        [
            _list("memory://preferences"),
            _list("memory://entities"),
            {
                "status": "insufficient",
                "action": "finish",
                "directory": None,
                "uris": [],
                "keywords": [],
                "reason": "完整旧 L2 快照均缺失，两个目录都没有条目。",
            },
        ]
    )
    chats[0].preferences.append({"page_id": 100, "topic": "回答风格", "content": "- 偏好简洁回答"})
    address = ConversationAddress("conversation-limit", date(2026, 7, 4))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-limit", closed_turn()),
        after_turn=True,
    )
    runtime.components.workflow.enqueuer.flush(address)

    result = asyncio.run(runtime.run_next())

    assert chats[0].retrieval_script == []
    assert chats[0].response_names.count("memory_retrieval_decision") == 3
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt is not None
    assert [str(item.uri) for item in result.change_receipt.node_changes] == ["memory://preferences/回答风格.md"]
    plans = [event for event in observer.events if (event.category, event.operation) == ("workflow", "memory_plan")]
    assert len(plans) == 1
    assert plans[0].attributes["retrieval_ended_at_limit"] is True
    assert plans[0].context.memory_sequence == result.job.memory_sequence


def _list(directory: str) -> dict[str, object]:
    return {
        "status": "insufficient",
        "action": "memory_list",
        "directory": directory,
        "uris": [],
        "keywords": [],
        "reason": "旧 L2 快照为空，需要列目录。",
    }


def _read_unexposed() -> dict[str, object]:
    """读一个名字清单、目录列表与一跳关系里都没出现过的地址。"""

    return {
        "status": "insufficient",
        "action": "memory_read",
        "directory": None,
        "uris": ["memory://preferences/磁盘清理.md"],
        "keywords": [],
        "reason": "猜一个可能存在的旧记忆。",
    }


def _read_seeded_entity() -> dict[str, object]:
    """名字清单里有第一场会话种下的实体，模型先把它读进来。"""

    return {
        "status": "insufficient",
        "action": "memory_read",
        "directory": None,
        "uris": ["memory://entities/软件项目/habitus.md"],
        "keywords": [],
        "reason": "名字清单里有相关实体，读完整旧记忆。",
    }


def _empty_candidates() -> dict[str, list[object]]:
    return {
        name: []
        for name in (
            "profile",
            "preferences",
            "entities",
            "tools",
            "identity_proposals",
            "relations",
        )
    }


def test_reading_an_unexposed_uri_still_fails_job_instead_of_ending_at_limit(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, max_attempts=1)
    providers, chats = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    chats[0].retrieval_script.append(_read_unexposed())
    address = ConversationAddress("conversation-search-down", date(2026, 7, 5))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-search-down", closed_turn()),
        after_turn=True,
    )
    runtime.components.workflow.enqueuer.flush(address)

    try:
        asyncio.run(runtime.run_next())
    except MemoryJobExecutionError as exc:
        failed = exc.job
    else:
        raise AssertionError("reading an unexposed URI must fail the job")

    assert failed is not None and failed.status is MemoryJobStatus.FAILED
    assert failed.last_error is not None and "memory_read URI was not exposed" in failed.last_error
    assert "memory_candidate_batch" not in chats[0].response_names


def _fail_semantic_refresh_once(runtime) -> None:
    """L2 提交之后的目录 L0/L1 刷新失败一次：L2 不回滚，Job 保持未完成。"""

    refresher = runtime.components.workflow.runner.committed_finalizer.semantic_refresher
    refresh = refresher.refresh_for_many
    failures = [OSError("simulated L0/L1 refresh failure")]

    def fail_once(addresses):
        if failures:
            raise failures.pop()
        return refresh(addresses)

    refresher.refresh_for_many = fail_once


def test_committed_l2_job_resumes_after_semantic_refresh_failure_without_replanning(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, max_attempts=1)
    providers, chats = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    address = ConversationAddress("conversation-recovery", date(2026, 7, 2))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-recovery", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]
    _fail_semantic_refresh_once(runtime)

    try:
        asyncio.run(runtime.run_next())
    except MemoryJobExecutionError as exc:
        failed = exc.job
    else:
        raise AssertionError("semantic refresh failure must fail the first attempt")

    assert failed is not None and failed.status is MemoryJobStatus.FAILED
    source = MemoryChangeSource.from_job(failed)
    receipt = runtime.components.workflow.receipts.read(source)
    journal = runtime.components.memory.editor.transaction.journal.read(failed.transaction_id)
    assert receipt.state is MemoryChangeReceiptState.COMMITTED
    assert journal.state is MemoryTransactionJournalState.COMMITTED
    assert chats[0].response_names.count("memory_candidate_batch") == 1
    assert asyncio.run(runtime.failed_memory_job()) == failed

    retried = asyncio.run(runtime.retry_failed_memory_job(failed))
    assert retried.reopened_job.status is MemoryJobStatus.QUEUED
    result = asyncio.run(runtime.run_next())

    assert result.recovered
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt is not None
    receipt_source = result.change_receipt.source
    assert receipt_source.same_trigger(source)
    trigger = runtime.components.workflow.runner.executor.conversations.read_segment(
        address,
        failed.segment_id,
    )
    editor_segment = (
        None
        if trigger.ends_mid_turn
        else runtime.components.workflow.runner.executor.conversations.read_editor_turn(address, trigger).whole
    )
    assert editor_segment is not None
    assert receipt_source.editor_segment_id == editor_segment.segment_id
    assert receipt_source.editor_segment_digest == editor_segment.digest
    assert chats[0].response_names.count("memory_candidate_batch") == 1
    assert runtime.components.memory.editor.transaction.journal.try_read(queued.transaction_id) is None
    assert asyncio.run(runtime.failed_memory_job()) is None


def test_l2_written_but_committed_mark_lost_keeps_receipt_and_retry_finishes_job(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, max_attempts=1)
    providers, chats = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    chats[0].preferences.append({"page_id": 100, "topic": "回答风格", "content": "- 偏好简洁回答"})
    address = ConversationAddress("conversation-mark", date(2026, 7, 3))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-mark", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]
    journal = runtime.components.memory.editor.transaction.journal
    mark = journal.mark
    failures = [OSError("simulated COMMITTED mark loss")]

    def lose_first_commit_mark(transaction_id, state, *, timestamp):
        if state is MemoryTransactionJournalState.COMMITTED and failures:
            raise failures.pop()
        return mark(transaction_id, state, timestamp=timestamp)

    journal.mark = lose_first_commit_mark
    try:
        asyncio.run(runtime.run_next())
    except MemoryJobExecutionError as exc:
        failed = exc.job
    else:
        raise AssertionError("lost COMMITTED mark must fail the first attempt")

    assert failed is not None and failed.status is MemoryJobStatus.FAILED
    source = MemoryChangeSource.from_job(failed)
    assert journal.read(failed.transaction_id).state is MemoryTransactionJournalState.PREPARED
    assert runtime.components.workflow.receipts.read(source).state is MemoryChangeReceiptState.PREPARED

    asyncio.run(runtime.retry_failed_memory_job(failed))
    result = asyncio.run(runtime.run_next())

    assert result.recovered
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt is not None
    assert result.change_receipt.state is MemoryChangeReceiptState.COMMITTED
    assert [str(item.uri) for item in result.change_receipt.node_changes] == ["memory://preferences/回答风格.md"]
    assert result.journal_cleaned
    assert journal.try_read(queued.transaction_id) is None
    assert chats[0].response_names.count("memory_candidate_batch") == 1


def test_lifecycle_keeps_history_and_summary_sources_until_terminal_archive_retirement(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path)
    providers, _chats = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    address = ConversationAddress("conversation-lifecycle", date(2026, 7, 3))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-lifecycle", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]
    completed = asyncio.run(runtime.run_next())
    assert completed.change_receipt is not None
    source = completed.change_receipt.source
    maintenance_time = BASE_TIME + timedelta(days=400)

    released = asyncio.run(runtime.maintain_conversation(address, now=maintenance_time))

    assert released.released_history_segment_ids == ()
    assert released.purged_history_segment_ids == ()
    assert released.deleted_memory_job_sequences == ()
    assert released.deleted_memory_receipt_ids == ()
    assert (
        runtime.components.workflow.jobs.try_read_source(
            address,
            queued.segment_id,
            queued.source_segment_digest,
        )
        is not None
    )
    assert runtime.components.workflow.receipts.try_read(source) is not None
    assert (
        runtime.components.conversation.summaries.store.try_read_by_id(
            address,
            queued.segment_id,
        )
        is not None
    )
    history_path = runtime.components.conversation.journal.layout.history_path(
        address,
        queued.segment_id,
    )
    assert history_path.exists()

    purged = asyncio.run(runtime.maintain_conversation(address, now=maintenance_time))

    assert purged.purged_history_segment_ids == ()
    assert purged.released_history_segment_ids == ()
    assert (
        runtime.components.conversation.summaries.store.try_read_by_id(
            address,
            queued.segment_id,
        )
        is not None
    )


_GOOD_PREFERENCE = {"page_id": 150, "topic": "回答风格", "content": "- 偏好简洁回答"}


def _seed_session(_visible: dict[str, object]) -> dict[str, object]:
    """第一场会话种下实体与 tool，供下一场会话的候选引用。"""

    batch = _empty_candidates()
    batch["entities"] = [
        {"page_id": 100, "category": "软件项目", "name": "Habitus", "summary": "Habitus 是长期记忆项目。"}
    ]
    batch["tools"] = [{"page_id": 103, "tool_name": "workspace.inspect", "purpose": "检查工作区状态。"}]
    return batch


def _existing_page_id(visible: dict[str, object], suffix: str) -> int:
    items = visible["existing_page_ids"]
    assert isinstance(items, list)
    return next(int(item["page_id"]) for item in items if str(item["uri"]).endswith(suffix))


def _with_good_preference(**extra: Callable[[dict[str, object]], list[object]]):
    def build(visible: dict[str, object]) -> dict[str, object]:
        batch = _empty_candidates()
        batch["preferences"] = [dict(_GOOD_PREFERENCE)]
        for name, items in extra.items():
            batch[name] = items(visible)
        return batch

    return build


def _seeded_runtime(tmp_path: Path, *, max_attempts: int = 1):
    config = runtime_config(tmp_path, max_attempts=max_attempts)
    providers, chats = dependencies()
    observer = RecordingObserver()
    runtime = build_runtime(
        config,
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
        observer=observer,
    )
    runtime.initialize()
    chats[0].candidate_script.append(_seed_session)
    _enqueue(runtime, "conversation-seed", date(2026, 7, 1), tool_turn())
    seeded = asyncio.run(runtime.run_next())
    assert seeded.job is not None and seeded.job.status is MemoryJobStatus.COMMITTED
    return runtime, chats, observer


def _enqueue(runtime, conversation_id: str, day: date, messages) -> None:
    address = ConversationAddress(conversation_id, day)
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch(conversation_id, messages),
        after_turn=True,
    )
    runtime.components.workflow.enqueuer.flush(address)


def _plan_events(observer: RecordingObserver) -> list[ObservationEvent]:
    return [event for event in observer.events if (event.category, event.operation) == ("workflow", "memory_plan")]


def test_failing_candidate_after_repair_is_dropped_and_the_rest_of_the_batch_and_later_jobs_commit(
    tmp_path: Path,
) -> None:
    """裁定 82：合格偏好 + 一条用新编号沿用已有实体地址的候选，修补后仍不合格 → 只丢那一条，Job 照常提交，
    回执记剔除 1 条及原因，另一场会话的下一个 Job 随后提交。"""

    runtime, chats, observer = _seeded_runtime(tmp_path)
    same_entity_new_page = _with_good_preference(
        entities=lambda _visible: [
            {"page_id": 160, "category": "软件项目", "name": "Habitus", "summary": "Habitus 补充对比结论。"}
        ]
    )
    chats[0].candidate_script.append(same_entity_new_page)
    chats[0].retrieval_script.append(_read_seeded_entity())
    _enqueue(runtime, "conversation-x", date(2026, 7, 1), closed_turn())
    _enqueue(runtime, "conversation-y", date(2026, 7, 2), closed_turn(prompt="另一场会话：我喜欢中文回答"))
    calls_before = chats[0].response_names.count("memory_candidate_batch")

    second = asyncio.run(runtime.run_next())

    assert second.job is not None and second.job.status is MemoryJobStatus.COMMITTED and second.job.attempts == 1
    assert chats[0].response_names.count("memory_candidate_batch") - calls_before == 2
    memory_root = tmp_path / "data" / "memory"
    assert (memory_root / "preferences" / "回答风格.md").exists()
    assert "补充对比结论" not in (memory_root / "entities" / "软件项目" / "habitus.md").read_text(encoding="utf-8")
    plan = _plan_events(observer)[-1]
    assert plan.context.memory_sequence == second.job.memory_sequence
    assert plan.attributes["source_rejected_count"] == 1
    assert plan.attributes["source_rejected_reasons"] == (
        "entities page_id=160: candidate for an existing memory URI must reuse its existing page_id"
    )

    third = asyncio.run(runtime.run_next())

    assert third.job is not None and third.job.status is MemoryJobStatus.COMMITTED
    assert third.job.conversation_id == "conversation-y"


def test_a_candidate_on_an_unread_old_card_never_fails_the_job_and_later_jobs_commit(tmp_path: Path) -> None:
    """用户 2026-11-23 定：模型没读已有实体就用新编号写它的地址。系统补读那张旧卡片、交回修补轮；修补后仍不合格只剔
    这一条，Job 照常提交，后面的序号照常推进（原来这里字段规划永久失败、堵住队列）。"""

    runtime, chats, observer = _seeded_runtime(tmp_path)
    unread_entity = _with_good_preference(
        entities=lambda _visible: [
            {"page_id": 160, "category": "软件项目", "name": "Habitus", "summary": "Habitus 补充对比结论。"}
        ]
    )
    chats[0].candidate_script.extend([unread_entity, unread_entity])
    _enqueue(runtime, "conversation-x", date(2026, 7, 1), closed_turn())
    _enqueue(runtime, "conversation-y", date(2026, 7, 2), closed_turn(prompt="另一场会话：我喜欢中文回答"))

    second = asyncio.run(runtime.run_next())

    assert second.job is not None and second.job.status is MemoryJobStatus.COMMITTED
    assert (tmp_path / "data" / "memory" / "preferences" / "回答风格.md").exists()
    plan = _plan_events(observer)[-1]
    assert plan.attributes["source_rejected_count"] == 1
    third = asyncio.run(runtime.run_next())
    assert third.job is not None and third.job.status is MemoryJobStatus.COMMITTED
    assert third.job.conversation_id == "conversation-y"


def test_failing_candidate_repaired_by_the_model_is_written_with_the_rest_of_the_batch(tmp_path: Path) -> None:
    """裁定 82 第 1 条：修补轮改对了，两条都写进，模型被问了两次。"""

    runtime, chats, observer = _seeded_runtime(tmp_path)
    renamed = _with_good_preference(
        entities=lambda _visible: [{"page_id": 160, "category": "软件项目", "name": "Habitus", "summary": "补充。"}]
    )
    repaired = _with_good_preference(
        entities=lambda _visible: [
            {"page_id": 160, "category": "软件项目", "name": "OpenViking", "summary": "OpenViking 是会话记忆项目。"}
        ]
    )
    chats[0].candidate_script.extend([renamed, repaired])
    chats[0].retrieval_script.append(_read_seeded_entity())
    _enqueue(runtime, "conversation-x", date(2026, 7, 1), closed_turn())
    calls_before = chats[0].response_names.count("memory_candidate_batch")

    result = asyncio.run(runtime.run_next())

    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert chats[0].response_names.count("memory_candidate_batch") - calls_before == 2
    assert chats[0].candidate_script == []
    assert result.change_receipt is not None
    assert sorted(str(item.uri) for item in result.change_receipt.node_changes) == [
        "memory://entities/软件项目/openviking.md",
        "memory://preferences/回答风格.md",
    ]
    assert _plan_events(observer)[-1].attributes["source_rejected_count"] == 0


@pytest.mark.parametrize(
    ("messages", "bad"),
    [
        # 复用旧编号却改了身份字段（实体名加了"项目"）
        (
            closed_turn,
            {
                "entities": lambda visible: [
                    {
                        "page_id": _existing_page_id(visible, "habitus.md"),
                        "category": "软件项目",
                        "name": "Habitus 项目",
                        "summary": "Habitus 是长期记忆项目。",
                    }
                ]
            },
        ),
        # 已有实体用新编号再出一次
        (
            closed_turn,
            {
                "entities": lambda _visible: [
                    {"page_id": 160, "category": "软件项目", "name": "Habitus", "summary": "参考 OpenViking。"}
                ]
            },
        ),
        # tool 名与本段调用名写法不同
        (
            tool_turn,
            {"tools": lambda _visible: [{"page_id": 160, "tool_name": "workspace_inspect", "purpose": "检查工作区。"}]},
        ),
        # 本段新写 failure_recovery，本段却没有"失败后成功"
        (
            tool_turn,
            {
                "tools": lambda visible: [
                    {
                        "page_id": _existing_page_id(visible, "workspace.inspect.md"),
                        "tool_name": "workspace.inspect",
                        "purpose": "检查工作区状态。",
                        "failure_recovery": "失败就重跑。",
                    }
                ]
            },
        ),
    ],
    ids=[
        "renamed-identity",
        "existing-uri-new-page-id",
        "tool-name-spelling",
        "new-failure-recovery",
    ],
)
def test_normal_model_variation_in_source_checks_no_longer_fails_the_job(
    tmp_path: Path,
    messages: Callable[..., tuple[object, ...]],
    bad: dict[str, Callable[[dict[str, object]], list[object]]],
) -> None:
    """测试工程师 B 列的模型正常波动：修补后仍不过只丢那一条，Job 不再 FAILED。"""

    runtime, chats, observer = _seeded_runtime(tmp_path)
    chats[0].candidate_script.append(_with_good_preference(**bad))
    chats[0].retrieval_script.append(_read_seeded_entity())
    _enqueue(runtime, "conversation-x", date(2026, 7, 1), messages(start_sequence=0))

    result = asyncio.run(runtime.run_next())

    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert (tmp_path / "data" / "memory" / "preferences" / "回答风格.md").exists()
    assert _plan_events(observer)[-1].attributes["source_rejected_count"] == 1


def test_a_running_runtime_refuses_to_retry_or_abandon_while_its_worker_is_not_blocked(tmp_path: Path) -> None:
    """FAILED Job 的人工处置在 RUNNING 时只许在 MemoryWorker 因它进了 BLOCKED 且空闲之后做（AGENTS 运维入口）；
    Worker 没被它挡住时两种处置都拒绝，Job 原样留着。"""

    from habitus.runtime import RuntimeState, RuntimeStateError

    config = runtime_config(tmp_path, max_attempts=1)
    providers, chats = dependencies()
    runtime = build_runtime(config, providers=providers, path_lock=PathLock(ProcessLocalLockStore()))
    runtime.initialize()
    chats[0].retrieval_script.append(_read_unexposed())
    address = ConversationAddress("conversation-gate", date(2026, 7, 5))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address, ConversationBatch("conversation-gate", closed_turn()), after_turn=True
    )
    runtime.components.workflow.enqueuer.flush(address)
    with pytest.raises(MemoryJobExecutionError) as caught:
        asyncio.run(runtime.run_next())
    failed = caught.value.job
    assert failed is not None and failed.status is MemoryJobStatus.FAILED

    runtime._state = RuntimeState.RUNNING  # 只拨状态、不起 Worker：Worker 没进 BLOCKED
    try:
        with pytest.raises(RuntimeStateError, match="retry a failed job after its memory worker is blocked and idle"):
            asyncio.run(runtime.retry_failed_memory_job(failed))
        with pytest.raises(RuntimeStateError, match="abandon a failed job after its memory worker is blocked and idle"):
            asyncio.run(runtime.abandon_failed_memory_job(failed, reason="人工确认无法修复"))
    finally:
        runtime._state = RuntimeState.READY
    assert asyncio.run(runtime.failed_memory_job()) == failed


def test_a_failed_job_whose_change_already_committed_cannot_be_abandoned(tmp_path: Path) -> None:
    """L2 已提交（回执 COMMITTED）之后才失败的 Job 不许显式放弃：放弃会丢掉已发布的副作用，只能重试补完。"""

    from habitus.runtime import RuntimeStateError

    config = runtime_config(tmp_path, max_attempts=1)
    providers, _chats = dependencies()
    runtime = build_runtime(config, providers=providers, path_lock=PathLock(ProcessLocalLockStore()))
    runtime.initialize()
    address = ConversationAddress("conversation-abandon", date(2026, 7, 2))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address, ConversationBatch("conversation-abandon", closed_turn()), after_turn=True
    )
    runtime.components.workflow.enqueuer.flush(address)
    _fail_semantic_refresh_once(runtime)
    with pytest.raises(MemoryJobExecutionError) as caught:
        asyncio.run(runtime.run_next())
    failed = caught.value.job
    assert failed is not None and failed.status is MemoryJobStatus.FAILED

    with pytest.raises(
        RuntimeStateError, match="a failed job with a prepared or committed receipt cannot be abandoned"
    ):
        asyncio.run(runtime.abandon_failed_memory_job(failed, reason="人工确认无法修复"))
    assert asyncio.run(runtime.failed_memory_job()) == failed
