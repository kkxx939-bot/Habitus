"""会话 lane 的模型触点：把一轮的材料交给模型，拿回这一轮的描述。

本条 lane 只有这里调模型；取材料、校验、写判断都是确定性的。重试一轮时失败怎么归类（``RETRY_*``）也定在这里——
它们大多是模型的异常，lane 的重试处理器只按这三组分，不直接认识模型客户端。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import tzinfo

from habitus.behavior.fusion.errors import BehaviorFusionRefusedError
from habitus.behavior.fusion.lanes.session.config import SessionLaneConfig
from habitus.behavior.fusion.lanes.session.material import render_reminders, render_turn
from habitus.behavior.fusion.lanes.session.model import SessionTurn
from habitus.behavior.fusion.lanes.session.prompt import (
    SESSION_SYSTEM_PROMPT,
    TurnDescription,
    assemble_description,
    session_json_schema,
)
from habitus.behavior.fusion.lanes.session.reminders import PendingReminder
from habitus.foundation.lifecycle import DeterministicFailure
from habitus.model_client import (
    MODEL_DETERMINISTIC,
    MODEL_REFUSED,
    MODEL_UNUSABLE,
    ChatMessage,
    ChatRequest,
    ModelStructuredOutputError,
    StructuredChatClient,
)

# 重试一轮（每晚重试、时窗末当场重读，裁定 67、68）失败归哪一类。"模型用不了"与"同一份输入再问也一样"只认
# ``model_client`` 具名导出的那两组（``MODEL_UNUSABLE`` / ``MODEL_DETERMINISTIC``）；模型别的失败也归进三类之一、不往外抛
# （一项抛出去，这一晚剩下的项都试不成、按上一晚剩下的丢掉；白天那一拍的递送全失败）：
# - 答得不合格（结构化输出校验用尽 ``ModelStructuredOutputError``）→ 格式不合；
# - 服务端答坏了（``ModelResponseError``，带 ``failover_eligible``：换一个供应商就可能答好，问题在服务端、不在这份输入）、
#   模型配置用不了（``ModelConfigurationError``，含缺依赖）→ 模型用不了（``MODEL_UNUSABLE``）。
#: 同一份输入再问也一样：这一版处理下必然写不成，或模型对这份输入必然拒绝。
RETRY_DETERMINISTIC: tuple[type[BaseException], ...] = (DeterministicFailure, *MODEL_DETERMINISTIC)
#: 白天第一次描述一轮时模型用不了（额度、鉴权、权限、连不上、没配好）：不抛给会话源等它重做，先落「错误」进夜里待处理的堆，
#: 那天照常按时定稿、结算读到"看不到"而不是"没做"（裁定 80 第 5 条、94 第 3 条）。
DESCRIBE_UNAVAILABLE: tuple[type[BaseException], ...] = MODEL_UNUSABLE
#: 白天描述一轮时模型答了但格式不合（结构化校验用尽）：放一边、夜里再试一次（裁定 94 错误分类）。
DESCRIBE_INVALID: tuple[type[BaseException], ...] = (ModelStructuredOutputError,)
#: 这一条触发了内容安全：只这一条记「错误」，不重试（裁定 94 错误分类）。确定性失败（``BehaviorFusionRefusedError``）的起因。
CONTENT_REFUSED: tuple[type[BaseException], ...] = MODEL_REFUSED
#: 模型整个用不了（此刻用不了、没配好）：夜里那条线停下这一晚（裁定 94 第 4、8 条）、留到当晚（时窗末重读）。
RETRY_UNAVAILABLE: tuple[type[BaseException], ...] = MODEL_UNUSABLE
#: 锁被另一进程拿着（文件锁、sqlite 锁库等不到）：不是错误、更不是模型调不通，还没准备好（裁定 94 第 8、9 条）。
RETRY_BUSY: tuple[type[BaseException], ...] = (TimeoutError,)
#: 格式不合：模型答了、结构化校验用尽仍不合格（裁定 94 错误分类）。我们自己写入出的错（``ValueError``、``BehaviorFusionError``
#: 这类校验不过）不在这里：那是 bug，由重试编排按"我们自己处理不了"拦住这一项、报出（裁定 94 第 9 条）。
RETRY_INVALID: tuple[type[BaseException], ...] = (ModelStructuredOutputError,)


@dataclass(frozen=True)
class DescribedTurn:
    """一次描述的结果与它用了几次校验才答对。"""

    description: TurnDescription
    validation_attempts: int


class SessionTurnDescriber:
    """每轮一次模型调用：材料进，描述出。材料只在这一次调用里用，不落盘（裁定 33）。"""

    def __init__(self, client: StructuredChatClient, *, zone: tzinfo, config: SessionLaneConfig | None = None) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        if not isinstance(zone, tzinfo):
            raise TypeError("zone must be a tzinfo")
        self.client = client
        self.zone = zone
        self.config = config or SessionLaneConfig()

    async def describe(self, turn: SessionTurn, reminders: Sequence[PendingReminder] = ()) -> DescribedTurn:
        """``reminders`` 是这一轮开口时还开着的提醒（预测层接上时才有）：带进材料，模型多答这一轮回应了哪几条、正 / 中 / 负 / 已做。"""

        local = turn.instructed_at.astimezone(self.zone)
        content = f"【这一轮】{local:%Y-%m-%d %H:%M}\n{render_turn(turn, config=self.config)}"
        if reminders:
            content = f"{content}\n\n{render_reminders(reminders, zone=self.zone)}"
        numbers = tuple(range(1, len(reminders) + 1))
        try:
            response = await self.client.complete_json_async(
                ChatRequest(
                    messages=(
                        ChatMessage(role="system", content=SESSION_SYSTEM_PROMPT),
                        ChatMessage(role="user", content=content),
                    )
                ),
                schema=session_json_schema(self.config, reminder_numbers=numbers),
                name="behavior_session_turn",
                validator=lambda parsed: assemble_description(parsed, config=self.config, reminder_numbers=numbers),
            )
        except MODEL_DETERMINISTIC as exc:
            # 同一份材料再问也一样：按确定性失败交给 lane，这一轮当场落「错误」，不连累同一份源里后面的轮
            raise BehaviorFusionRefusedError(f"{type(exc).__name__}: {exc}") from exc
        description = response.value
        if not isinstance(description, TurnDescription):
            raise TypeError("the session turn validator must return TurnDescription")
        return DescribedTurn(description, response.validation_attempts)


__all__ = [
    "CONTENT_REFUSED",
    "DESCRIBE_INVALID",
    "DESCRIBE_UNAVAILABLE",
    "RETRY_BUSY",
    "RETRY_DETERMINISTIC",
    "RETRY_INVALID",
    "RETRY_UNAVAILABLE",
    "DescribedTurn",
    "SessionTurnDescriber",
]
