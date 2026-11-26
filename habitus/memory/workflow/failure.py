"""MemoryJob 失败的确定性处置分类。模型调用失败只按模型层统一的三条规则分派（``model_client.model_failure``，裁定 108）。"""

from __future__ import annotations

from enum import Enum

from habitus.memory.editor.commit import MemoryCommitError
from habitus.memory.editor.extraction.model import MemoryExtractionPermanentError
from habitus.memory.workflow.jobs import MemoryJobError, MemoryJobLeaseLostError
from habitus.model_client import ModelClientError, ModelErrorRule, ModelStructuredOutputError, model_failure


class MemoryJobSkipped(RuntimeError):  # noqa: N818 - 不是故障，是"这一段跳过"的结论
    """这一段在生成摘要与记忆计划时碰到"这一条输入有问题"（请求不合法、太长、内容安全）：同一份输入不重发，这一段记不进
    长期记忆，任务进 SKIPPED，后面的序号照常往下走（裁定 108）。只在长期记忆还没写之前（规划阶段）认；``reason`` 是
    ``<类别>: <说明>``。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MemoryJobFailureKind(str, Enum):
    """一次 Job 失败之后怎么处置。"""

    #: 依赖此刻用不了：模型（统一规则里的"等一等可能好"用尽、"要人处理"、熔断开着）。
    #: 不计入尝试次数，排回队列等依赖恢复；等的期间挡住后面的序号（裁定 80、82、108）。
    WAITS_FOR_MODEL = "waits_for_model"
    #: 明确瞬态、写之前文件被别人改了、写到一半停下、答复用完结构纠错仍不合格（模型回答的正常波动，裁定 80）
    #: 或未知运行时故障：有界退避整条重做，重试期间挡住后面的序号；到 ``max_attempts`` 进 FAILED，后面照常往下走。
    RETRYABLE = "retryable"
    #: 再试也一样（长期记忆写成之后才碰到的输入问题、永久抽取错误、完整性错误……）：第一次就 FAILED，后面照常往下走，
    #: 等人决定重试还是放弃（裁定 111）。
    PERMANENT = "permanent"


def skip_reason(error: BaseException) -> str | None:
    """规划阶段的失败如果是"这一条输入有问题"，给出跳过的原因；否则 None。"""

    failure = model_failure(error)
    if failure is None or failure.rule is not ModelErrorRule.INPUT or failure.kind is None:
        return None
    return f"{failure.kind.value}: {failure}"


def memory_job_failure_kind(error: BaseException) -> MemoryJobFailureKind:
    """把一次执行失败归到三种处置之一。

    顺着 ``__cause__`` 链从外往里看，第一层认得的异常决定处置；模型调用失败那一层按统一规则（``ModelErrorRule``）认：
    "这一条输入有问题"到了这里（长期记忆已写成之后）是 PERMANENT，其余等模型恢复。泛化的 ValueError / TypeError 只在
    最外层算数。整条链都认不得的未知运行时故障有界重试。
    """

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


def model_wait_seconds(error: BaseException) -> float | None:
    """模型那条能力的熔断给的离下一次试探还有多久（熔断没开是 None）。"""

    failure = model_failure(error)
    return None if failure is None else failure.retry_after_seconds


def _known_failure_kind(error: BaseException, *, outermost: bool) -> MemoryJobFailureKind | None:
    """单层异常认得时给出处置，认不得返回 None。"""

    if isinstance(error, ModelClientError) and error.rule is not None:
        return (
            MemoryJobFailureKind.PERMANENT
            if error.rule is ModelErrorRule.INPUT
            else MemoryJobFailureKind.WAITS_FOR_MODEL
        )
    if isinstance(error, ModelStructuredOutputError):
        return MemoryJobFailureKind.RETRYABLE
    if isinstance(error, MemoryCommitError | TimeoutError):
        return MemoryJobFailureKind.RETRYABLE
    if isinstance(error, MemoryExtractionPermanentError):
        return MemoryJobFailureKind.PERMANENT
    if outermost and isinstance(error, MemoryJobError | ValueError | TypeError):
        return MemoryJobFailureKind.PERMANENT
    return None


__all__ = ["MemoryJobFailureKind", "MemoryJobSkipped", "memory_job_failure_kind", "model_wait_seconds", "skip_reason"]
