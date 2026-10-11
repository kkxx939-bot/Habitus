"""会话 lane 的模型触点：把一轮的材料交给模型，拿回这一轮的描述。

本条 lane 只有这里调模型；取材料、校验、写判断都是确定性的。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import tzinfo

from habitus.behavior.fusion.lanes.session.config import SessionLaneConfig
from habitus.behavior.fusion.lanes.session.material import render_turn
from habitus.behavior.fusion.lanes.session.model import SessionTurn
from habitus.behavior.fusion.lanes.session.prompt import (
    SESSION_SYSTEM_PROMPT,
    TurnDescription,
    assemble_description,
    session_json_schema,
)
from habitus.model_client import ChatMessage, ChatRequest, StructuredChatClient


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

    async def describe(self, turn: SessionTurn) -> DescribedTurn:
        local = turn.instructed_at.astimezone(self.zone)
        content = f"【这一轮】{local:%Y-%m-%d %H:%M}\n{render_turn(turn, config=self.config)}"
        response = await self.client.complete_json_async(
            ChatRequest(
                messages=(
                    ChatMessage(role="system", content=SESSION_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=content),
                )
            ),
            schema=session_json_schema(self.config),
            name="behavior_session_turn",
            validator=lambda parsed: assemble_description(parsed, config=self.config),
        )
        description = response.value
        if not isinstance(description, TurnDescription):
            raise TypeError("the session turn validator must return TurnDescription")
        return DescribedTurn(description, response.validation_attempts)


__all__ = ["DescribedTurn", "SessionTurnDescriber"]
