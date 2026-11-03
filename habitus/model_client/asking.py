"""各模型触点共用的那一段：调一次结构化模型、传输层瞬态错有界重试、失败与结构化留痕说清楚。

语义树的触点（写概念、映射器、先验、候选调节条件）与预测层的判决各有自己的提示词、schema 与核对；调模型这一段几处一样，
收在这里一份（第二轮评审 C12；预测层读不了语义树，所以放在两边都能 import 的模型层）。分工：

- **传输层**（断网、超时、配额这类 ``ModelTransportError``）在这里按退避重试 ``retries`` 次；
- **答复不合格**（形状不对、核对不过）由结构层（``StructuredChatClient`` 的 ``validation_retries``）带着错误再问，这里不管；
- 传输重试用完、或碰到"整个用不了"的那一类（额度、鉴权、权限、服务端答坏、没配好，``MODEL_UNUSABLE``）→ 抛
  ``ModelUnreachable``：不是"这一处没答成"，夜里那条线据它整晚停（裁定 94 第 8 条）；白天的调用方（预测层的判决）自己接住、
  按模型不可用降级；
- 别的没答成（答复不合格、内容安全、输入超长）→ ``Failed``，信号要说出**哪条规则不过**：结构层把核对的报错挂在 ``__cause__``
  上，只报"failed after 2 attempts"的话，看日志的人不知道该改提示词的哪一句。调用方按 ``Failed`` 决定"这一批不写 / 记未决"。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from habitus.foundation.text import clean_line
from habitus.model_client.contracts import (
    MODEL_UNUSABLE,
    ChatRequest,
    ModelClientError,
    ModelTransportError,
    ModelUnreachable,
)
from habitus.model_client.structured import StructuredChatClient


@dataclass(frozen=True)
class Answered:
    """模型答成了：核对后的值 + 结构化留痕（第几轮答对、JSON 是不是修出来的）+ 实际作答的模型（答复里报的，不是配置里写的：
    回退到别的后端时两者不同；第四轮评审 E15）。"""

    value: Any
    notes: tuple[str, ...] = ()
    model: str = ""


@dataclass(frozen=True)
class Failed:
    """模型这一轮没答成（结构 / 核对两轮都不过、内容安全、输入超长）；整个用不了的不在这里，抛 ``ModelUnreachable``。"""

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
                raise ModelUnreachable(failure_signal(exc)) from exc
            await asyncio.sleep(delay_seconds * (attempt + 1))
            continue
        except MODEL_UNUSABLE as exc:
            raise ModelUnreachable(failure_signal(exc)) from exc
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
