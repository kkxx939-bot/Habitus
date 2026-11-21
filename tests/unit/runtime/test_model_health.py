"""裁定 108：健康检查直接写出给人看的那句话；"要人处理"的改好后试探成功自己消掉。"""

from __future__ import annotations

from habitus.model_client import BRIEFS, CallFailureConfig, ModelErrorKind, ModelQuotaError
from habitus.model_client.breaker import CircuitBreaker
from habitus.runtime.health_model import RuntimeHealthStatus
from habitus.runtime.model_health import breaker_check


def test_an_open_breaker_shows_the_plain_reason_and_clears_after_a_successful_probe() -> None:
    now = [0.0]
    breaker = CircuitBreaker(CallFailureConfig(), name="chat", monotonic=lambda: now[0])
    assert breaker_check("model_chat", breaker).status is RuntimeHealthStatus.HEALTHY
    breaker.record_failure(ModelQuotaError("insufficient_quota: You exceeded your current quota"))
    check = breaker_check("model_chat", breaker)
    assert check.status is RuntimeHealthStatus.DEGRADED and not check.critical
    assert check.detail.startswith(BRIEFS[ModelErrorKind.QUOTA]) and "60 秒后再试一次" in check.detail
    now[0] = 61.0
    breaker.before_call()
    breaker.record_success()
    assert breaker_check("model_chat", breaker).status is RuntimeHealthStatus.HEALTHY
