"""MemoryJob 失败的确定性重试分类。"""

from __future__ import annotations

from infrastructure.vector import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreIntegrityError,
)
from memory.editor.transaction import MemoryCommitConflictError
from memory.workflow.jobs import MemoryJobError, MemoryJobLeaseLostError
from ModelClient import ModelClientError


def memory_job_failure_is_retryable(error: BaseException) -> bool:
    """只把明确瞬态或未知运行时故障交给有界退避重试。"""

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if isinstance(error, MemoryJobLeaseLostError):
        return False
    if isinstance(error, MemoryCommitConflictError | TimeoutError):
        return True
    if isinstance(error, VectorStoreBusyError | VectorStoreConflictError):
        return True
    if isinstance(error, VectorStoreIntegrityError | VectorStoreError):
        return False
    if isinstance(error, ModelClientError):
        return bool(error.retryable)
    if isinstance(error, MemoryJobError | ValueError | TypeError):
        return False
    return True


__all__ = ["memory_job_failure_is_retryable"]
