"""词表唯一的模型触点：一次结构化调用（重试只在模型客户端一层，裁定 108）。

选词经这里调模型（``pick``），只交出系统提示、用户消息、schema 与形状校验。
这一次调用本身答不了时（校验用尽，或这一条输入有问题）抛 ``KindModelOutputError``，调用方按"没答好"降级；
模型此刻用不了（全流程统一规则里的"等一等可能好"用尽、"要人处理"、熔断开着）原样抛出。

调用方要按错误分类分开处理时（裁定 94）用 ``call_failure``：它只按模型层统一的判定入口
（``model_client.model_failure``）分派，行为侧别处不直接认模型客户端的异常类型。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, TypeVar, cast

from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.model_client import (
    MODEL_REFUSED,
    ChatMessage,
    ChatRequest,
    ModelErrorRule,
    ModelStructuredOutputError,
    StructuredChatClient,
    model_failure,
)

T = TypeVar("T")


class KindModelOutputError(RuntimeError):
    """模型的输出在校验重试之后仍然不合形状。"""


class CallFailure(str, Enum):
    """一次词表调模型失败属于哪一类（裁定 94）。"""

    INVALID = "invalid"  # 答复重问之后仍不合格：选词记「待定」，不硬归
    UNAVAILABLE = "unavailable"  # 模型此刻用不了（"等一等可能好"用尽、"要人处理"、熔断开着）：放一边等下一次
    REFUSED = "refused"  # 这一条输入有问题里的内容安全：不重发，单列
    TOO_LARGE = "too_large"  # 这一条输入有问题里的其余两种（太长、请求不合法）：不重发
    BUG = "bug"  # 别的：我们自己的代码出错，当场报


def call_failure(exc: BaseException) -> CallFailure:
    """按模型层统一的判定入口（``model_failure``）定这一次失败属于哪一类；``KindModelOutputError`` 把原异常挂在 ``__cause__`` 上。"""

    failure = model_failure(exc)
    if failure is not None:
        if failure.rule is ModelErrorRule.INPUT:
            return CallFailure.REFUSED if isinstance(failure, MODEL_REFUSED) else CallFailure.TOO_LARGE
        return CallFailure.UNAVAILABLE
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ModelStructuredOutputError):
            return CallFailure.INVALID
        current = current.__cause__
    return CallFailure.BUG


class KindModelCaller:
    def __init__(self, client: StructuredChatClient, *, config: BehaviorKindConfig) -> None:
        self.client = client
        self.config = config

    async def call(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        name: str,
        validator: Callable[[object], T],
    ) -> T:
        request = ChatRequest(
            messages=(ChatMessage(role="system", content=system), ChatMessage(role="user", content=user))
        )
        try:
            response = await self.client.complete_json_async(request, schema=schema, name=name, validator=validator)
        except Exception as exc:
            if call_failure(exc) in (CallFailure.INVALID, CallFailure.REFUSED, CallFailure.TOO_LARGE):
                # 这一次调用本身答不了（输出不合形状、这一条输入有问题）：再问也一样，按"没答好"交调用方降级。
                raise KindModelOutputError(f"{type(exc).__name__}: {exc}") from exc
            raise
        return cast("T", response.value)


__all__ = ["CallFailure", "KindModelCaller", "KindModelOutputError", "call_failure"]
