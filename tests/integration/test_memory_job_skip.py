"""裁定 108：长期记忆任务碰到"这一条输入有问题"时跳过这一段、后面的序号照常往下走；其余模型失败照旧挡住后面。"""

from __future__ import annotations

import asyncio
import time
from datetime import date, timedelta
from pathlib import Path

from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import MemoryJobStatus
from habitus.model_client import ModelContentSafetyError, ModelTransportError
from habitus.runtime import MemoryConsistencyState
from habitus.runtime.model_health import model_checks
from tests.helpers import closed_turn
from tests.integration.test_memory_job_full_chain import _enqueue, _seeded_runtime

_SUMMARY = "conversation_segment_summary_field_operations"
_CANDIDATES = "memory_candidate_batch"


def _refusing(chat, refused: set[str], error: type[Exception] = ModelContentSafetyError):  # type: ignore[no-untyped-def]
    """让脚本化的对话模型对 ``refused`` 里的结构化输出名抛 ``error``（只在 ``armed[0]`` 为真时）。"""

    armed = [True]
    real = chat.complete
    chat.refusals = 0

    def complete(prepared):  # type: ignore[no-untyped-def]
        if armed[0] and prepared.request.response_format.name in refused:
            chat.refusals += 1
            raise error("blocked by the provider")
        return real(prepared)

    chat.complete = complete
    return armed


def test_a_refused_segment_is_skipped_with_its_reason_and_the_next_sequence_commits(tmp_path: Path) -> None:
    runtime, chats, _observer = _seeded_runtime(tmp_path)
    _refusing(chats[0], {_CANDIDATES})
    _enqueue(runtime, "conversation-refused", date(2026, 7, 2), closed_turn())
    _enqueue(runtime, "conversation-after", date(2026, 7, 3), closed_turn())

    skipped = asyncio.run(runtime.run_next())
    assert skipped.job is not None and skipped.job.status is MemoryJobStatus.SKIPPED
    assert skipped.job.last_error is not None and skipped.job.last_error.startswith("content_safety:")
    assert chats[0].refusals == 1  # 同一份输入不重发：客户端不重试，任务也不再排回
    after = asyncio.run(runtime.run_next())  # 第二段被同一个开关拒了，也是跳过；序号照常往下走
    assert after.job is not None and after.job.memory_sequence == skipped.job.memory_sequence + 1
    assert after.job.status is MemoryJobStatus.SKIPPED and chats[0].refusals == 2
    jobs = runtime.components.workflow.jobs
    assert jobs.oldest_unfinished() is None and jobs.observability_snapshot().skipped == 2
    # 摘要照常做出来（只有记忆解析被拒），这一段的原文有摘要接替
    summaries = runtime.components.conversation.summaries.store
    address = ConversationAddress("conversation-refused", date(2026, 7, 2))
    assert summaries.list(address)
    check = next(item for item in model_checks(runtime.components) if item.name == "memory_skipped")
    assert "2 段对话没能记进长期记忆" in check.detail and "内容安全" in check.detail
    consistency = asyncio.run(runtime.memory_consistency(skipped.job))
    assert consistency.state is MemoryConsistencyState.SKIPPED


def test_when_the_summary_is_refused_too_the_history_is_still_released_on_schedule(tmp_path: Path) -> None:
    """阶段 3 改了裁定 108 那句：摘要也做不出来的段，原文照同一条规则到期释放（任务跳过后 10 天）。"""

    runtime, chats, _observer = _seeded_runtime(tmp_path)
    armed = _refusing(chats[0], {_SUMMARY, _CANDIDATES})
    _enqueue(runtime, "conversation-refused", date(2026, 7, 2), closed_turn())
    skipped = asyncio.run(runtime.run_next())
    assert skipped.job is not None and skipped.job.status is MemoryJobStatus.SKIPPED
    address = ConversationAddress("conversation-refused", date(2026, 7, 2))
    assert runtime.components.conversation.summaries.store.list(address) == ()
    assert runtime.components.conversation.journal.list_history(address)  # 还没到期，原文留着
    maintained = asyncio.run(
        runtime.maintain_conversation(address, now=skipped.job.updated_at + timedelta(days=11))
    )
    assert maintained.released_history_segment_ids == (skipped.job.segment_id,)
    assert runtime.components.conversation.journal.list_history(address) == ()
    armed[0] = False
    _enqueue(runtime, "conversation-after", date(2026, 7, 3), closed_turn())
    after = asyncio.run(runtime.run_next())
    assert after.job is not None and after.job.status is MemoryJobStatus.COMMITTED


def test_a_service_outage_still_blocks_the_later_sequences_until_recovery(tmp_path: Path) -> None:
    runtime, chats, _observer = _seeded_runtime(tmp_path)
    armed = _refusing(chats[0], {_CANDIDATES}, ModelTransportError)
    _enqueue(runtime, "conversation-outage", date(2026, 7, 2), closed_turn())
    _enqueue(runtime, "conversation-after", date(2026, 7, 3), closed_turn())
    first = runtime.components.workflow.jobs.oldest_unfinished()
    try:
        asyncio.run(runtime.run_next())
    except Exception:  # noqa: BLE001 - 记忆任务失败按设计抛出（MemoryJobExecutionError）
        pass
    waiting = runtime.components.workflow.jobs.oldest_unfinished()
    assert waiting is not None and first is not None and waiting.memory_sequence == first.memory_sequence
    assert waiting.status is MemoryJobStatus.QUEUED and waiting.waiting_for_model
    # 恢复：熔断到了试探的时候、模型好了；排期时刻也到了
    armed[0] = False
    runtime.components.models.chat.breaker._monotonic = lambda: time.monotonic() + 86_400.0
    runtime.components.workflow.jobs.clock = lambda: waiting.next_attempt_at  # type: ignore[assignment,return-value]
    done = asyncio.run(runtime.run_next())
    assert done.job is not None and done.job.status is MemoryJobStatus.COMMITTED
