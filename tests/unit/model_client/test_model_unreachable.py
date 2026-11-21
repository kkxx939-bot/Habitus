"""裁定 94 第 8 条：模型"整个用不了"不吞成"这一处没答成"——``ask`` 抛 ``ModelUnreachable``；夜里的探针遇网络抖动按普通调用的
规矩先重试几次再判。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from habitus.model_client import (
    ChatClient,
    ChatMessage,
    ChatRequest,
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelQuotaError,
    ModelTransportError,
    ModelUnreachable,
    StructuredChatClient,
)
from habitus.model_client.asking import Failed, ask
from tests.unit.model_client.test_client_runtime_matrix import ScriptedProvider, config


def _ask(*errors: BaseException) -> object:
    provider = ScriptedProvider(async_completions=list(errors))
    client = StructuredChatClient(ChatClient(config(retries=0), provider), validation_retries=0)
    request = ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
    return asyncio.run(
        ask(
            client,
            request,
            schema={"type": "object"},
            name="probe",
            validator=lambda value: value,
        )
    )


@pytest.mark.parametrize(
    "errors",
    [
        (ModelQuotaError("quota"),),
        (ModelAuthenticationError("denied"),),
        (ModelConfigurationError("no key"),),
        (ModelTransportError("down"),),  # "等一等可能好"在客户端一层重试用完（这里 retries=0），ask 不再重试
    ],
)
def test_an_unusable_model_is_raised_not_swallowed_as_a_failed_answer(errors: tuple[BaseException, ...]) -> None:
    with pytest.raises(ModelUnreachable):
        _ask(*errors)


def test_other_failures_stay_a_failed_answer_at_this_touchpoint() -> None:
    answer = _ask(ModelContentSafetyError("blocked"))
    assert isinstance(answer, Failed) and "ModelContentSafetyError" in answer.signal and answer.input_problem


class _Probed(ScriptedProvider):
    def __init__(self, *results: object) -> None:
        super().__init__()
        self.results = list(results)
        self.probes = 0

    async def health_check_async(self) -> Mapping[str, object]:
        self.probes += 1
        current = self.results.pop(0)
        if isinstance(current, BaseException):
            raise current
        return current  # type: ignore[return-value]


def test_the_probe_retries_a_network_blip_like_an_ordinary_call() -> None:
    provider = _Probed(ModelTransportError("blip"), {"ok": True})
    assert asyncio.run(ChatClient(config(retries=2), provider).health_check_async()) == {"ok": True}
    assert provider.probes == 2


def test_the_probe_does_not_retry_what_retrying_cannot_fix() -> None:
    provider = _Probed(ModelAuthenticationError("denied"), {"ok": True})
    result = asyncio.run(ChatClient(config(retries=2), provider).health_check_async())
    assert result["ok"] is False and result["error_code"] == "MODEL_AUTHENTICATION" and provider.probes == 1
