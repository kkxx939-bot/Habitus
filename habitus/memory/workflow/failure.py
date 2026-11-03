"""MemoryJob 失败的确定性重试分类。"""

from __future__ import annotations

from enum import Enum

from habitus.infrastructure.vector import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreIntegrityError,
)
from habitus.memory.editor.extraction.model import MemoryExtractionPermanentError
from habitus.memory.editor.transaction import MemoryCommitConflictError
from habitus.memory.workflow.jobs import MemoryJobError, MemoryJobLeaseLostError
from habitus.model_client import MODEL_UNAVAILABLE, ModelClientError, ModelStructuredOutputError

#: 依赖此刻用不了：对话或嵌入模型用不了，或向量库忙 / 连不上。
_DEPENDENCY_UNAVAILABLE: tuple[type[BaseException], ...] = (*MODEL_UNAVAILABLE, VectorStoreBusyError)


class MemoryJobFailureKind(str, Enum):
    """一次 Job 失败之后怎么处置。"""

    #: 依赖此刻用不了：模型（``model_client.MODEL_UNAVAILABLE``：连不上、限流、额度或余额、鉴权、权限、服务端答坏，
    #: 对话与嵌入同一组）或向量库忙（``VectorStoreBusyError``：断网、超时、5xx、索引重建中）。同一份输入过一阵再来就能成，
    #: 走单独的长退避、不计入尝试次数，等依赖恢复（裁定 80 问题 2、裁定 82 第 3 条）。
    WAITS_FOR_MODEL = "waits_for_model"
    #: 明确瞬态、答复用完结构纠错仍不合格（模型回答的正常波动，裁定 80 第 4 条）或未知运行时故障：有界退避重试，
    #: 到 ``max_attempts`` 进 FAILED。
    RETRYABLE = "retryable"
    #: 同一份输入再试也一样（输入超长、内容安全、永久抽取错误、完整性错误……）：第一次就 FAILED。
    PERMANENT = "permanent"


def memory_job_failure_kind(error: BaseException) -> MemoryJobFailureKind:
    """把一次执行失败归到三种处置之一。

    顺着 ``__cause__`` 链从外往里看，第一层认得的异常决定处置：检索、抽取把依赖的失败包成自己的 RuntimeError，
    外层认不得时由里面的根因决定，向量库或嵌入服务连不上照样归"等依赖恢复"，向量库 401 / 403 / 400 照样明确失败。
    泛化的 ValueError / TypeError 只在最外层算数，包在里面的不改变外层的处置。整条链都认不得的未知运行时故障有界重试。
    """

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if isinstance(error, MemoryJobLeaseLostError):
        return MemoryJobFailureKind.PERMANENT
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        kind = _known_failure_kind(current, outermost=current is error)
        if kind is not None:
            return kind
        current = current.__cause__
    return MemoryJobFailureKind.RETRYABLE


def _known_failure_kind(error: BaseException, *, outermost: bool) -> MemoryJobFailureKind | None:
    """单层异常认得时给出处置，认不得返回 None。"""

    if isinstance(error, _DEPENDENCY_UNAVAILABLE):
        return MemoryJobFailureKind.WAITS_FOR_MODEL
    if isinstance(error, ModelStructuredOutputError):
        return MemoryJobFailureKind.RETRYABLE
    if isinstance(error, MemoryCommitConflictError | TimeoutError | VectorStoreConflictError):
        return MemoryJobFailureKind.RETRYABLE
    if isinstance(error, VectorStoreIntegrityError | VectorStoreError):
        return MemoryJobFailureKind.PERMANENT
    if isinstance(error, ModelClientError):
        return MemoryJobFailureKind.RETRYABLE if error.retryable else MemoryJobFailureKind.PERMANENT
    if isinstance(error, MemoryExtractionPermanentError):
        return MemoryJobFailureKind.PERMANENT
    if outermost and isinstance(error, MemoryJobError | ValueError | TypeError):
        return MemoryJobFailureKind.PERMANENT
    return None


__all__ = ["MemoryJobFailureKind", "memory_job_failure_kind"]
