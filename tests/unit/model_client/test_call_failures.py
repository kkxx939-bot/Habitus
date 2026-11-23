"""裁定 108：模型调用失败的一套类别、三条规则、一处归类、一层重试与熔断。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from habitus.model_client import (
    BRIEFS,
    MODEL_DETERMINISTIC,
    MODEL_UNAVAILABLE,
    RULES,
    CallFailureConfig,
    ChatClient,
    ModelAuthenticationError,
    ModelCircuitOpenError,
    ModelClientError,
    ModelContentSafetyError,
    ModelErrorKind,
    ModelErrorRule,
    ModelQuotaError,
    ModelTransportError,
    model_failure,
    model_failure_rule,
)
from habitus.model_client.adapters._http import HTTPStatusError, error_code
from habitus.model_client.breaker import BreakerState, CircuitBreaker
from habitus.model_client.failures import classify_failure
from tests.unit.model_client.test_client_runtime_matrix import ScriptedProvider, config, response

SRC = Path(__file__).resolve().parents[3] / "habitus"


def test_every_kind_has_exactly_one_rule_and_a_plain_sentence() -> None:
    assert set(RULES) == set(ModelErrorKind) == set(BRIEFS)
    for kind, brief in BRIEFS.items():
        assert brief.endswith("。") and not re.search(r"[A-Za-z_]{4,}", brief), (kind, brief)
    assert {kind for kind, rule in RULES.items() if rule is ModelErrorRule.INPUT} == {
        ModelErrorKind.INVALID_REQUEST,
        ModelErrorKind.INPUT_TOO_LARGE,
        ModelErrorKind.CONTENT_SAFETY,
    }


def test_the_groups_follow_the_rules() -> None:
    for error_type in MODEL_UNAVAILABLE:
        if error_type is not ModelCircuitOpenError:
            assert error_type.kind is not None and error_type.kind.rule is not ModelErrorRule.INPUT
    for error_type in MODEL_DETERMINISTIC:
        assert error_type.kind is not None and error_type.kind.rule is ModelErrorRule.INPUT


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        # 只看状态码
        (400, None, ModelErrorKind.INVALID_REQUEST),
        (401, None, ModelErrorKind.AUTHENTICATION),
        (402, None, ModelErrorKind.QUOTA),
        (403, None, ModelErrorKind.PERMISSION),
        (404, None, ModelErrorKind.MODEL_NOT_FOUND),
        (405, None, ModelErrorKind.UNSUPPORTED),
        (407, None, ModelErrorKind.NETWORK_SETUP),
        (408, None, ModelErrorKind.NETWORK),
        (409, None, ModelErrorKind.SERVICE),
        (413, None, ModelErrorKind.INPUT_TOO_LARGE),
        (418, None, ModelErrorKind.UNKNOWN),
        (422, None, ModelErrorKind.INVALID_REQUEST),
        (429, None, ModelErrorKind.RATE_LIMIT),
        (500, None, ModelErrorKind.SERVICE),
        (501, None, ModelErrorKind.UNSUPPORTED),
        (504, None, ModelErrorKind.NETWORK),
        (529, None, ModelErrorKind.SERVICE),
        # OpenAI 兼容协议 error.code / error.type
        (429, "insufficient_quota", ModelErrorKind.QUOTA),
        (401, "invalid_api_key", ModelErrorKind.AUTHENTICATION),
        (404, "model_not_found", ModelErrorKind.MODEL_NOT_FOUND),
        (400, "context_length_exceeded", ModelErrorKind.INPUT_TOO_LARGE),
        (400, "content_filter", ModelErrorKind.CONTENT_SAFETY),
        (400, "unsupported_parameter", ModelErrorKind.UNSUPPORTED),
        (429, "rate_limit_exceeded", ModelErrorKind.RATE_LIMIT),
        (500, "server_error", ModelErrorKind.SERVICE),
        (401, "invalid_request_error", ModelErrorKind.AUTHENTICATION),  # 笼统的类型不收，看状态码
        # Anthropic Messages error.type
        (400, "invalid_request_error", ModelErrorKind.INVALID_REQUEST),
        (401, "authentication_error", ModelErrorKind.AUTHENTICATION),
        (402, "billing_error", ModelErrorKind.QUOTA),
        (403, "permission_error", ModelErrorKind.PERMISSION),
        (404, "not_found_error", ModelErrorKind.MODEL_NOT_FOUND),
        (413, "request_too_large", ModelErrorKind.INPUT_TOO_LARGE),
        (429, "rate_limit_error", ModelErrorKind.RATE_LIMIT),
        (500, "api_error", ModelErrorKind.SERVICE),
        (504, "timeout_error", ModelErrorKind.NETWORK),
        (529, "overloaded_error", ModelErrorKind.SERVICE),
        (None, "overloaded_error", ModelErrorKind.SERVICE),  # 流里的错误事件没有状态码
        (None, None, ModelErrorKind.UNKNOWN),
        # 向量、重排走的 OpenAI 兼容服务里常见的代码
        (400, "DataInspectionFailed", ModelErrorKind.CONTENT_SAFETY),
        (400, "Arrearage", ModelErrorKind.QUOTA),
        (429, "Throttling", ModelErrorKind.RATE_LIMIT),
    ],
)
def test_one_classifier_maps_codes_then_statuses(status: int | None, code: str | None, kind: ModelErrorKind) -> None:
    failure = classify_failure(HTTPStatusError("x", status_code=status, retry_after=None, provider_code=code))
    assert failure.kind is kind and failure.status_code == status and failure.provider_code == code


def test_message_text_never_decides_the_kind() -> None:
    for text in ("rate limit", "insufficient_quota", "content policy", "prompt is too long", "timed out"):
        assert classify_failure(RuntimeError(text)).kind is ModelErrorKind.UNKNOWN
    assert classify_failure(TimeoutError("")).kind is ModelErrorKind.NETWORK
    assert classify_failure(ConnectionResetError()).kind is ModelErrorKind.NETWORK


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b'{"error":{"code":"insufficient_quota","type":"insufficient_quota"}}', "insufficient_quota"),
        (b'{"error":{"code":null,"type":"invalid_request_error"}}', "invalid_request_error"),
        (b'{"type":"error","error":{"type":"rate_limit_error"}}', "rate_limit_error"),
        (b'{"code":"Arrearage","message":"x"}', "Arrearage"),
        (b'{"type":"error"}', None),
        (b'{"error":{"code":20015}}', None),
        (b"<html>", None),
    ],
)
def test_the_protocol_error_code_is_read_from_the_body(body: bytes, code: str | None) -> None:
    assert error_code(body) == code


def test_the_only_classifier_lives_in_the_model_layer() -> None:
    """归类只有一处：模型层之外没有按状态码、厂商名或报错文字认错误的分支。"""

    pattern = re.compile(r"status_code\s*(==|in)\s*[\[{(]?\s*4\d\d|insufficient_quota|content_filter|rate_limit_error")
    offenders = [
        f"{path.relative_to(SRC)}:{number}"
        for path in SRC.rglob("*.py")
        if "__pycache__" not in path.parts and path.name not in {"failures.py"}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line) and not path.relative_to(SRC).as_posix().startswith("model_client/adapters/")
    ]
    assert offenders == []


def test_model_failure_finds_the_classified_error_through_wrappers() -> None:
    try:
        try:
            raise ModelContentSafetyError("blocked")
        except ModelClientError as inner:
            raise RuntimeError("extraction failed") from inner
    except RuntimeError as outer:
        assert isinstance(model_failure(outer), ModelContentSafetyError)
        assert model_failure_rule(outer) is ModelErrorRule.INPUT
    assert model_failure(ValueError("ours")) is None


# ── 熔断 ──────────────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _breaker(threshold: int = 3) -> tuple[CircuitBreaker, _Clock]:
    clock = _Clock()
    config = CallFailureConfig(breaker_failure_threshold=threshold, breaker_reset_base_seconds=60, breaker_reset_max_seconds=200)
    return CircuitBreaker(config, monotonic=clock), clock


def test_waiting_failures_open_the_breaker_at_the_threshold_and_a_success_clears_the_count() -> None:
    breaker, _clock = _breaker()
    for _ in range(2):
        breaker.before_call()
        breaker.record_failure(ModelTransportError("down"))
    breaker.record_success()
    for _ in range(2):
        breaker.record_failure(ModelTransportError("down"))
    assert breaker.snapshot().state is BreakerState.CLOSED
    error = ModelTransportError("down")
    breaker.record_failure(error)
    assert breaker.snapshot().state is BreakerState.OPEN and error.retry_after_seconds == 60
    with pytest.raises(ModelCircuitOpenError) as caught:
        breaker.before_call()
    assert caught.value.opened_by is ModelErrorKind.NETWORK and caught.value.rule is ModelErrorRule.WAIT


@pytest.mark.parametrize("error", [ModelAuthenticationError("bad key"), ModelQuotaError("no credit")])
def test_operator_failures_open_the_breaker_at_once(error: ModelClientError) -> None:
    breaker, _clock = _breaker()
    breaker.record_failure(error)
    snapshot = breaker.snapshot()
    assert snapshot.open and snapshot.reason is error.kind and snapshot.retry_in_seconds == 60


def test_input_problems_never_count() -> None:
    breaker, _clock = _breaker(threshold=1)
    for _ in range(5):
        breaker.record_failure(ModelContentSafetyError("blocked"))
    assert breaker.snapshot().state is BreakerState.CLOSED


def test_one_probe_after_the_wait_and_a_failed_probe_doubles_the_wait_up_to_the_cap() -> None:
    breaker, clock = _breaker(threshold=1)
    breaker.record_failure(ModelAuthenticationError("bad key"))
    clock.now = 60
    assert breaker.before_call() is True  # 放一个试探
    with pytest.raises(ModelCircuitOpenError):
        breaker.before_call()  # 试探在路上，其余照旧挡住
    breaker.record_failure(ModelAuthenticationError("still bad"))
    assert breaker.snapshot().retry_in_seconds == 120
    clock.now = 180
    breaker.before_call()
    breaker.record_failure(ModelAuthenticationError("still bad"))
    assert breaker.snapshot().retry_in_seconds == 200  # 封顶
    clock.now = 380
    breaker.before_call()
    breaker.record_success()  # 人改好了：试探成功就关上，等待回到起点
    assert breaker.snapshot().state is BreakerState.CLOSED and breaker.before_call() is False


def test_a_probe_without_a_verdict_releases_its_slot() -> None:
    breaker, clock = _breaker(threshold=1)
    breaker.record_failure(ModelTransportError("down"))
    clock.now = 60
    assert breaker.before_call() is True
    breaker.release_probe()
    assert breaker.before_call() is True


# ── 客户端：一层重试 + 熔断 ─────────────────────────────────────────────────────


def test_chat_client_retries_only_waiting_failures_and_records_one_failure_per_call() -> None:
    provider = ScriptedProvider(completions=[ModelTransportError("a"), ModelTransportError("b"), response()])
    client = ChatClient(config(retries=2), provider, sleep=lambda _delay: None)
    assert client.complete("hi") == response()
    assert len(provider.requests) == 3 and client.breaker.snapshot().consecutive_failures == 0


def test_an_open_chat_breaker_sends_nothing_until_the_probe_and_resumes_by_itself() -> None:
    provider = ScriptedProvider(completions=[ModelAuthenticationError("bad key"), response()])
    client = ChatClient(config(retries=2), provider, sleep=lambda _delay: None)
    with pytest.raises(ModelAuthenticationError):
        client.complete("hi")
    assert len(provider.requests) == 1  # 要人处理：不重试
    with pytest.raises(ModelCircuitOpenError):
        client.complete("hi")
    assert len(provider.requests) == 1  # 熔断开着：不发请求
    client.breaker._monotonic = lambda: 10.0**9  # 到了试探的时候（人已经把密钥改好）
    assert client.complete("hi") == response()
    assert not client.breaker.is_open()
