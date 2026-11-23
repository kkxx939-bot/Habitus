"""记忆任务失败的处置分类，以及只看任务状态的一致性观察。"""

from datetime import date
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.memory.editor import MemoryCommitConflictError, MemoryCommitError
from habitus.memory.editor.extraction import (
    MemoryExtractionCapacityError,
    MemoryExtractionError,
    MemoryExtractionPermanentError,
)
from habitus.memory.editor.retrieval import MemoryRetrievalError
from habitus.memory.workflow.failure import MemoryJobFailureKind, memory_job_failure_kind
from habitus.memory.workflow.jobs import MemoryJobConfig, MemoryJobError, MemoryJobLeaseLostError, MemoryJobStore
from habitus.model_client import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelPermissionError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelResponseError,
    ModelStructuredOutputError,
    ModelTransportError,
)
from habitus.runtime.consistency import MemoryConsistencyService, MemoryConsistencyState
from tests.helpers import closed_turn, segment
from tests.lock_store import ProcessLocalLockStore

WAITS = MemoryJobFailureKind.WAITS_FOR_MODEL
RETRY = MemoryJobFailureKind.RETRYABLE
PERMANENT = MemoryJobFailureKind.PERMANENT


def wrapped(outer: BaseException, cause: BaseException) -> BaseException:
    """照检索、抽取的写法把依赖的失败包一层（``raise outer from cause``）。"""

    outer.__cause__ = cause
    return outer


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (TimeoutError("slow"), RETRY),
        (MemoryCommitConflictError("changed"), RETRY),
        # 写到一半停下：已经写成的留着，整条重做（阶段 3）
        (MemoryCommitError("stopped part-way"), RETRY),
        # 检索、抽取把依赖失败包成自己的 RuntimeError 之后按根因归类
        (wrapped(MemoryRetrievalError("memory read failed"), ModelTransportError("network")), WAITS),
        (wrapped(MemoryExtractionError("memory read failed"), ModelTransportError("network")), WAITS),
        (wrapped(MemoryRetrievalError("memory read failed"), RuntimeError("unknown")), RETRY),
        (wrapped(MemoryRetrievalError("memory read failed"), ValueError("bad hit")), RETRY),
        (wrapped(MemoryExtractionPermanentError("invalid"), ModelTransportError("network")), PERMANENT),
        # 模型此刻用不了：断网 / 5xx、限流、额度或余额、鉴权、权限、服务端答坏 → 等模型恢复（裁定 80 问题 2）
        (ModelTransportError("network"), WAITS),
        (ModelRateLimitError("rate limit"), WAITS),
        (ModelQuotaError("Insufficient Balance"), WAITS),
        (ModelAuthenticationError("invalid key"), WAITS),
        (ModelPermissionError("denied"), WAITS),
        (ModelResponseError("invalid"), WAITS),
        # 答复用完结构纠错仍不合格：Job 层有界重试（M14）
        (ModelStructuredOutputError("model failed json_schema validation after 2 attempt(s)"), RETRY),
        # 同一份输入再试也一样：第一次就 FAILED
        (ModelInputTooLargeError("too large"), PERMANENT),
        (ModelContentSafetyError("blocked"), PERMANENT),
        # 没配好属于"要人处理"（裁定 108）：等人改好、随熔断试探接上，不当场 FAILED
        (ModelConfigurationError("missing dependency"), WAITS),
        (MemoryExtractionCapacityError("too large"), PERMANENT),
        (MemoryExtractionPermanentError("invalid candidates"), PERMANENT),
        (MemoryJobLeaseLostError("lost"), PERMANENT),
        (MemoryJobError("invalid state"), PERMANENT),
        (ValueError("bad input"), PERMANENT),
        (RuntimeError("unknown runtime fault"), RETRY),
    ],
)
def test_failure_policy_waits_for_the_model_retries_transient_faults_and_stops_on_deterministic_ones(
    error: BaseException,
    kind: MemoryJobFailureKind,
) -> None:
    assert memory_job_failure_kind(error) is kind


def test_consistency_follows_job_status_and_reports_unknown_after_job_is_gone(
    tmp_path: Path,
) -> None:
    """没有回执之后一致性只看任务本身：在跑是 PENDING、做完是 COMMITTED、任务被删后是 UNKNOWN。"""

    jobs = MemoryJobStore(
        tmp_path / "workflow",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "memory",
        config=MemoryJobConfig(),
    )
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_segment = segment(
        conversation_id=address.conversation_id,
        segment_id="000000000000-000000000001",
        messages=closed_turn(),
    )
    queued = jobs.activate(jobs.stage(address, source_segment))
    lease = jobs.claim(queued, "worker-1")
    consistency = MemoryConsistencyService(jobs)

    assert consistency.inspect(queued).state is MemoryConsistencyState.PENDING
    committed = jobs.complete(lease)
    assert consistency.inspect(queued).state is MemoryConsistencyState.COMMITTED
    assert jobs.discard_committed(committed)
    assert consistency.inspect(queued).state is MemoryConsistencyState.UNKNOWN
