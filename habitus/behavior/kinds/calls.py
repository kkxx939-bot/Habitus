"""词表唯一的模型触点：一次结构化调用 + 瞬态错误的有界重试。

归类、每晚新增、定期拆改都经这里调模型；它们只交出系统提示、用户消息、schema 与形状校验。
这一次调用本身答不了时（校验用尽、输入过长、内容安全拦截）抛 ``KindModelOutputError``，调用方按"没答好"降级；
瞬态错误重试用尽、配额与鉴权这类全局故障原样抛出。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.model_client import ChatMessage, ChatRequest, StructuredChatClient
from habitus.model_client.contracts import (
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelResponseError,
    ModelStructuredOutputError,
    ModelTransportError,
)

T = TypeVar("T")


class KindModelOutputError(RuntimeError):
    """模型的输出在校验重试之后仍然不合形状。"""


class KindModelCaller:
    def __init__(self, client: StructuredChatClient, *, config: BehaviorKindConfig) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        if not isinstance(config, BehaviorKindConfig):
            raise TypeError("config must be BehaviorKindConfig")
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
            except (ModelStructuredOutputError, ModelInputTooLargeError, ModelContentSafetyError) as exc:
                # 这一次调用本身答不了（输出不合形状、输入过长、被内容安全拦下）：重试也一样，按"没答好"交调用方降级。
                raise KindModelOutputError(f"{type(exc).__name__}: {exc}") from exc
            except (ModelTransportError, ModelResponseError):
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
                continue
            return cast("T", response.value)
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["KindModelCaller", "KindModelOutputError"]
