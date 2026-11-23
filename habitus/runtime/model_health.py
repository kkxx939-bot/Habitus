"""健康检查里模型调用失败那几项（裁定 108）：给人看的一句话直接写在 ``detail`` 里。

- ``model_chat``：对话能力（唯一的模型能力）的熔断此刻的样子。开着时写打开它的那类错误的那句话、
  多久之后再试一次；"要人处理"的改好配置或额度恢复后，下一次试探成功自己消掉。都是 non-critical：读写各条链自己会等。
- ``memory_skipped``：长期记忆里输入有问题、跳过了的段（仍在保留期里的）：条数、各类几段、最近一段的那句话。
- ``memory_failed_jobs``：重试用完判了失败的记忆任务（裁定 111：不再挡后面的任务）：条数与最早一条，等人决定重试还是放弃。
"""

from __future__ import annotations

from collections import Counter

from habitus.memory.workflow import MemoryJob
from habitus.model_client import ModelErrorKind
from habitus.model_client.breaker import CircuitBreaker
from habitus.runtime.components import RuntimeComponents
from habitus.runtime.health_model import RuntimeHealthCheck, RuntimeHealthStatus


def model_checks(components: RuntimeComponents) -> list[RuntimeHealthCheck]:
    checks = [breaker_check("model_chat", components.models.chat.breaker)]
    try:
        failed = components.workflow.jobs.failed()
    except Exception as exc:  # noqa: BLE001 - 健康检查只报不抛
        checks.append(RuntimeHealthCheck("memory_failed_jobs", RuntimeHealthStatus.DEGRADED, type(exc).__name__, False))
    else:
        components.infrastructure.observability.set_gauge("memory_failed_jobs", len(failed))
        if failed:
            detail = f"failed={len(failed)};oldest={failed[0].memory_sequence}:{failed[0].last_error or ''}"
            checks.append(RuntimeHealthCheck("memory_failed_jobs", RuntimeHealthStatus.DEGRADED, detail[:500], False))
    try:
        skipped = components.workflow.jobs.skipped()
    except Exception as exc:  # noqa: BLE001 - 健康检查只报不抛
        checks.append(RuntimeHealthCheck("memory_skipped", RuntimeHealthStatus.DEGRADED, type(exc).__name__, False))
        return checks
    if skipped:
        checks.append(skipped_check(skipped))
    return checks


def breaker_check(name: str, breaker: CircuitBreaker) -> RuntimeHealthCheck:
    snapshot = breaker.snapshot()
    if not snapshot.open:
        return RuntimeHealthCheck(name, RuntimeHealthStatus.HEALTHY, "available", critical=False)
    reason = snapshot.reason or ModelErrorKind.UNKNOWN
    wait = 0.0 if snapshot.retry_in_seconds is None else snapshot.retry_in_seconds
    detail = f"{reason.brief}（{reason.value}；约 {wait:.0f} 秒后再试一次）最近一次报错：{snapshot.detail}"
    return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, detail[:500], critical=False)


def skipped_check(jobs: tuple[MemoryJob, ...]) -> RuntimeHealthCheck:
    kinds = Counter(_kind(job) for job in jobs)
    latest = max(jobs, key=lambda job: job.memory_sequence)
    counts = "，".join(f"{kind.value}×{count}" for kind, count in sorted(kinds.items(), key=lambda item: item[0].value))
    detail = f"有 {len(jobs)} 段对话没能记进长期记忆（{counts}）。最近一段（序号 {latest.memory_sequence}）：{_kind(latest).brief}"
    return RuntimeHealthCheck("memory_skipped", RuntimeHealthStatus.DEGRADED, detail[:500], critical=False)


def _kind(job: MemoryJob) -> ModelErrorKind:
    """跳过的原因写成 ``<类别>: <说明>``（``memory.workflow.failure.skip_reason``）。"""

    prefix = (job.last_error or "").split(":", 1)[0]
    try:
        return ModelErrorKind(prefix)
    except ValueError:
        return ModelErrorKind.UNKNOWN


__all__ = ["breaker_check", "model_checks", "skipped_check"]
