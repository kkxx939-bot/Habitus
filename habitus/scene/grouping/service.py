"""归组服务：语义关联层唯一的模型触点（每封口日一次调用）。

与 kinds resolver 同一形态：瞬态传输错误有界重试（路由层自己还有一层重试，这里的次数应
保持很小）；结构化输出经 schema 与装配校验器，装配层对记账疏漏降级而不整批拒，只有穷尽性被
破坏才让结构层重试。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, cast

from habitus.model_client import ModelTransportError, StructuredChatClient
from habitus.scene.grouping.assembly import assemble_grouping
from habitus.scene.grouping.model import GroupingAssembly, SceneGroupingInput
from habitus.scene.grouping.prompt import SCENE_GROUPING_PROMPT_VERSION, build_request
from habitus.scene.grouping.schema import SCHEMA_FINGERPRINT, grouping_json_schema

SCENE_VERSION = f"{SCENE_GROUPING_PROMPT_VERSION}+schema{SCHEMA_FINGERPRINT}"


class SceneGrouper(Protocol):
    """刷新器依赖的归组契约；测试与重放用脚本化实现替换。"""

    @property
    def version(self) -> str: ...

    async def group(self, payload: SceneGroupingInput) -> GroupingAssembly: ...


@dataclass(frozen=True)
class SceneGroupingConfig:
    max_occurrences_per_call: int = 400
    max_prompt_chars: int = 200_000
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in ("max_occurrences_per_call", "max_prompt_chars", "transient_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (name != "transient_retries" and value <= 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.transient_retry_delay_seconds, bool)
            or not isinstance(self.transient_retry_delay_seconds, int | float)
            or self.transient_retry_delay_seconds < 0
        ):
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


class SceneGroupingLimitError(ValueError):
    """一天的行为数或提示词超过本层配置的边界；该日跳过并留信号，不切块（第一期）。"""


class LLMSceneGrouper:
    def __init__(self, client: StructuredChatClient, *, config: SceneGroupingConfig | None = None) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        resolved = config or SceneGroupingConfig()
        if not isinstance(resolved, SceneGroupingConfig):
            raise TypeError("config must be SceneGroupingConfig")
        self.client = client
        self.config = resolved

    @property
    def version(self) -> str:
        return SCENE_VERSION

    async def group(self, payload: SceneGroupingInput) -> GroupingAssembly:
        if not isinstance(payload, SceneGroupingInput):
            raise TypeError("payload must be SceneGroupingInput")
        if len(payload.occurrences) > self.config.max_occurrences_per_call:
            raise SceneGroupingLimitError(
                f"day has {len(payload.occurrences)} occurrences, above max_occurrences_per_call={self.config.max_occurrences_per_call}"
            )
        request = build_request(payload)
        prompt_chars = sum(len(message.content or "") for message in request.messages)
        if prompt_chars > self.config.max_prompt_chars:
            raise SceneGroupingLimitError(f"grouping prompt has {prompt_chars} characters, above max_prompt_chars")
        schema = grouping_json_schema(len(payload.occurrences))
        response = None
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request,
                    schema=schema,
                    name="scene_grouping",
                    validator=lambda parsed: assemble_grouping(parsed, payload),
                )
                break
            except ModelTransportError:
                # 只重试传输层的瞬态错；答复本身不合格（截断、超限）重试也是同一份完整调用，不做。
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
        assert response is not None
        return cast("GroupingAssembly", response.value)


__all__ = ["SCENE_VERSION", "LLMSceneGrouper", "SceneGrouper", "SceneGroupingConfig", "SceneGroupingLimitError"]
