"""会话 lane 测试共用的夹具：造消息与轮、真实的行为侧存储、照脚本回答的模型。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from habitus.behavior.fusion import BehaviorFusionReceiptStore, BehaviorJudgementStore
from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.lanes.session import (
    SessionLane,
    SessionLaneConfig,
    SessionMessage,
    SessionMessageRole,
    SessionTurnDescriber,
    SessionTurnRecorder,
)
from habitus.behavior.observation import BehaviorObservationStore
from habitus.model_client import ChatClient, ChatModelConfig, ProviderConfig, StructuredChatClient
from tests.unit.behavior.test_fusion_jobs import ScriptedProvider

ZONE = ZoneInfo("Asia/Shanghai")
START = datetime(2026, 7, 26, 9, 58, tzinfo=UTC)  # 本地 17:58
SUBJECT = "家庭成员A"


def at(minutes: float) -> datetime:
    return START + timedelta(minutes=minutes)


def prompt(sequence: int, text: str, *, minutes: float = 0.0) -> SessionMessage:
    return SessionMessage(sequence, at(minutes), SessionMessageRole.PROMPT, text)


def reply(sequence: int, text: str, *, minutes: float = 1.0) -> SessionMessage:
    return SessionMessage(sequence, at(minutes), SessionMessageRole.COMPLETION, text)


def call(sequence: int, name: str, arguments: str, *, call_id: str, minutes: float = 0.5) -> SessionMessage:
    return SessionMessage(
        sequence, at(minutes), SessionMessageRole.TOOL_CALL, arguments, tool_name=name, tool_call_id=call_id
    )


def result(
    sequence: int,
    text: str,
    *,
    call_id: str,
    failed: bool = False,
    minutes: float = 0.6,
    name: str | None = None,
) -> SessionMessage:
    return SessionMessage(
        sequence, at(minutes), SessionMessageRole.TOOL_RESULT, text, tool_name=name, tool_call_id=call_id, failed=failed
    )


def description(name: str = "统计本地项目代码行数", **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "summary": "他问项目有多少行代码；助手重新统计后给出行数与文件数。",
        "goal": "了解项目当前的代码规模",
        "steps": ["统计各目录的代码行数", "汇总并与上次对比"],
    }
    body.update(overrides)
    return body


class Stores:
    """一套真实的行为侧存储（观测、判断、回执、覆盖索引），落在临时目录里。"""

    def __init__(self, root: Path) -> None:
        self.observations = BehaviorObservationStore(root)
        self.judgements = BehaviorJudgementStore(root)
        self.receipts = BehaviorFusionReceiptStore(root)
        self.coverage = BehaviorCoverageIndex(root)
        self.now = at(130)

    def recorder(self) -> SessionTurnRecorder:
        return SessionTurnRecorder(
            observations=self.observations,
            judgements=self.judgements,
            receipts=self.receipts,
            coverage=self.coverage,
            primary_subject=SUBJECT,
            zone=ZONE,
            clock=lambda: self.now,
        )


def client(
    bodies: list[dict[str, Any]], *, validation_retries: int = 1
) -> tuple[StructuredChatClient, ScriptedProvider]:
    provider = ScriptedProvider(bodies)
    config = ChatModelConfig(
        route=ProviderConfig(
            provider="fake",
            adapter="openai_compatible_chat",
            model="fake-1",
            base_url="https://example.invalid",
            credential_ref="FAKE_KEY",
        ),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        structured_output_mode="json_schema",
    )
    return StructuredChatClient(ChatClient(config, provider), validation_retries=validation_retries), provider


def lane(
    stores: Stores, bodies: list[dict[str, Any]], *, config: SessionLaneConfig | None = None
) -> tuple[SessionLane, ScriptedProvider]:
    chat, provider = client(bodies)
    return SessionLane(SessionTurnDescriber(chat, zone=ZONE, config=config), stores.recorder()), provider
