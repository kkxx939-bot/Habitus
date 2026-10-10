"""语义树各模型触点共用的那一段：调一次结构化模型、传输层瞬态错有界重试、失败与结构化留痕说清楚。

触点①写概念、映射器各有自己的提示词、schema 与核对；调模型这一段几处一样，收在这里
（第二轮评审 C12）。分工：

- **传输层**（断网、超时、配额这类 ``ModelTransportError``）在这里按退避重试 ``retries`` 次；
- **答复不合格**（形状不对、核对不过）由结构层（``StructuredChatClient`` 的 ``validation_retries``）带着错误再问，这里不管；
- 都没成 → ``Failed``，信号要说出**哪条规则不过**：结构层把核对的报错挂在 ``__cause__`` 上，只报"failed after 2 attempts"
  的话，看日志的人不知道该改提示词的哪一句。

模型层的异常不出这里：调用方按 ``Failed`` 决定"这一批不写 / 记未决"。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from habitus.foundation.text import clean_line
from habitus.model_client import ChatRequest, ModelClientError, ModelTransportError, StructuredChatClient


@dataclass(frozen=True)
class Answered:
    """模型答成了：核对后的值 + 结构化留痕（第几轮答对、JSON 是不是修出来的）+ 实际作答的模型（答复里报的，不是配置里写的：
    回退到别的后端时两者不同；第四轮评审 E15）。"""

    value: Any
    notes: tuple[str, ...] = ()
    model: str = ""


@dataclass(frozen=True)
class Failed:
    """模型这一轮没答成（传输重试用完，或结构 / 核对两轮都不过）。"""

    signal: str


async def ask(
    client: StructuredChatClient,
    request: ChatRequest,
    *,
    schema: Mapping[str, Any],
    name: str,
    validator: Callable[[object], Any],
    retries: int,
    delay_seconds: float,
) -> Answered | Failed:
    for attempt in range(retries + 1):
        try:
            response = await client.complete_json_async(request, schema=schema, name=name, validator=validator)
        except ModelTransportError as exc:
            if attempt >= retries:
                return Failed(f"model: transport failed after {retries + 1} attempts: {clean_line(str(exc))[:120]}")
            await asyncio.sleep(delay_seconds * (attempt + 1))
            continue
        except ModelClientError as exc:
            return Failed(failure_signal(exc))
        notes: list[str] = []
        if response.validation_attempts > 1:
            notes.append(f"structured: answered on attempt {response.validation_attempts}")
        if response.parse_mode != "strict":
            notes.append(f"structured: json parsed via {response.parse_mode}")
        return Answered(response.value, tuple(notes), response.response.model)
    raise AssertionError("unreachable")  # pragma: no cover


def failure_signal(exc: ModelClientError) -> str:
    """失败信号：异常类型 + 消息 + 结构层挂在 ``__cause__`` 上的那条核对报错。"""

    cause = exc.__cause__
    detail = f"；{clean_line(str(cause))[:220]}" if cause is not None else ""
    return f"model: {type(exc).__name__}: {clean_line(str(exc))[:120]}{detail}"


__all__ = ["Answered", "Failed", "ask", "failure_signal"]
