"""各模型触点共用的那一段：调一次结构化模型，按全流程统一的三条规则（``model_client.errors``）交出结果。

语义树的触点（先验）与预测层的判决各有自己的提示词、schema 与核对；调模型这一段几处一样，收在这里一份。分工：

- 重试只在客户端一层（"等一等可能好"的按路由有界重试），这里不再重试；
- **答复不合格**由结构层（``StructuredChatClient`` 的 ``validation_retries``）带着错误再问，这里不管；
- 模型此刻用不了（"等一等可能好"重试用尽、"要人处理"、熔断开着，``MODEL_UNAVAILABLE``）→ 抛 ``ModelUnreachable``：换代里的
  那一步据它走退路，白天的调用方（预测层的判决）自己接住、按模型不可用降级；
- 别的没答成 → ``Failed``：答复不合格，或这一条输入有问题（``input_problem``：同一份输入不重发，调用方据它不再问）。信号要说出
  **哪条规则不过**：结构层把核对的报错挂在 ``__cause__`` 上。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from habitus.foundation.text import clean_line
from habitus.model_client.contracts import ChatRequest
from habitus.model_client.errors import MODEL_UNAVAILABLE, ModelClientError, ModelErrorRule, ModelUnreachable
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
    """模型这一轮没答成（结构 / 核对两轮都不过，或这一条输入有问题）；此刻用不了的不在这里，抛 ``ModelUnreachable``。"""

    signal: str
    #: 这一条输入有问题（请求不合法、太长、内容安全）：同一份输入不重发。
    input_problem: bool = False


async def ask(
    client: StructuredChatClient,
    request: ChatRequest,
    *,
    schema: Mapping[str, Any],
    name: str,
    validator: Callable[[object], Any],
) -> Answered | Failed:
    try:
        response = await client.complete_json_async(request, schema=schema, name=name, validator=validator)
    except MODEL_UNAVAILABLE as exc:
        raise ModelUnreachable(failure_signal(exc)) from exc
    except ModelClientError as exc:
        return Failed(failure_signal(exc), input_problem=exc.rule is ModelErrorRule.INPUT)
    notes: list[str] = []
    if response.validation_attempts > 1:
        notes.append(f"structured: answered on attempt {response.validation_attempts}")
    if response.parse_mode != "strict":
        notes.append(f"structured: json parsed via {response.parse_mode}")
    return Answered(response.value, tuple(notes), response.response.model)


def failure_signal(exc: ModelClientError) -> str:
    """失败信号：异常类型 + 消息 + 结构层挂在 ``__cause__`` 上的那条核对报错。"""

    cause = exc.__cause__
    detail = f"；{clean_line(str(cause))[:220]}" if cause is not None else ""
    return f"model: {type(exc).__name__}: {clean_line(str(exc))[:120]}{detail}"


__all__ = ["Answered", "Failed", "ask", "failure_signal"]
