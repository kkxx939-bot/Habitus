"""词表唯一的模型触点：一次结构化调用 + 瞬态错误的有界重试。

归类、每晚新增、定期拆改都经这里调模型；它们只交出系统提示、用户消息、schema 与形状校验。
这一次调用本身答不了时（校验用尽、输入过长、内容安全拦截）抛 ``KindModelOutputError``，调用方按"没答好"降级；
瞬态错误重试用尽、配额与鉴权这类全局故障原样抛出。

调用方要按错误分类分开处理时（裁定 94 错误分类：归约的白天归类）用 ``call_failure``：模型整个用不了 / 这一份触发内容安全 /
输入超长 / 别的（我们自己的 bug）。分类只在这里认模型客户端的异常，行为侧别处不直接依赖它。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, TypeVar, cast

from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.model_client import (
    MODEL_DETERMINISTIC,
    MODEL_REFUSED,
    MODEL_UNUSABLE,
    ChatMessage,
    ChatRequest,
    StructuredChatClient,
)
from habitus.model_client.contracts import ModelResponseError, ModelStructuredOutputError, ModelTransportError

T = TypeVar("T")

#: 这一次调用本身答不了：输出校验用尽，或同一份输入再问也一样（``MODEL_DETERMINISTIC``：输入过长、内容安全）。
_UNANSWERABLE: tuple[type[BaseException], ...] = (ModelStructuredOutputError, *MODEL_DETERMINISTIC)


class KindModelOutputError(RuntimeError):
    """模型的输出在校验重试之后仍然不合形状。"""


class CallFailure(str, Enum):
    """一次词表调模型失败属于哪一类（裁定 94 错误分类）。"""

    UNAVAILABLE = "unavailable"  # 模型整个用不了（此刻用不了、没配好）：夜里停线，白天放一边等当晚
    REFUSED = "refused"  # 这一份触发内容安全：不重试
    TOO_LARGE = "too_large"  # 输入超长：材料上限与模型窗口没对上，是 bug，不重试
    BUG = "bug"  # 别的：我们自己的代码出错，当场报


def call_failure(exc: BaseException) -> CallFailure:
    """沿 ``__cause__`` 链找模型客户端的异常，定它属于哪一类（``KindModelOutputError`` 把原异常挂在 ``__cause__`` 上）。"""

    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, MODEL_UNUSABLE):
            return CallFailure.UNAVAILABLE
        if isinstance(current, MODEL_REFUSED):
            return CallFailure.REFUSED
        if isinstance(current, MODEL_DETERMINISTIC):
            return CallFailure.TOO_LARGE
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
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(request, schema=schema, name=name, validator=validator)
            except _UNANSWERABLE as exc:
                # 这一次调用本身答不了（输出不合形状、输入过长、被内容安全拦下）：重试也一样，按"没答好"交调用方降级。
                raise KindModelOutputError(f"{type(exc).__name__}: {exc}") from exc
            except (ModelTransportError, ModelResponseError):
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
                continue
            return cast("T", response.value)
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["CallFailure", "KindModelCaller", "KindModelOutputError", "call_failure"]
