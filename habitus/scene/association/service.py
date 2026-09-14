"""关联服务：规律级语义树的模型触点。

与归组服务同一形态：瞬态传输错误有界重试（路由层自己还有一层，这里的次数要很小）；结构化输出
先过 schema 再过装配校验，装配对记账疏漏降级、只有穷尽性被破坏才让结构层重试。

调用的**单位是一个候选的一批发生**，不是一天：编排层按格子线性推进，把同一个 kind 在同一天的
几次发生凑成一次调用（同一天各次互相是上下文，分开问会得出互相矛盾的说法）。上限按目标数与
提示词字符两道闸，超了就跳过这一批留信号——切块会把"同一天一起看"这个前提破坏掉。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, cast

from habitus.model_client import ModelTransportError, StructuredChatClient
from habitus.scene.association.assembly import assemble_association
from habitus.scene.association.model import AssociationAssembly, AssociationInput
from habitus.scene.association.prompt import ASSOCIATION_PROMPT_VERSION, build_request
from habitus.scene.association.schema import SCHEMA_FINGERPRINT, association_json_schema

ASSOCIATION_VERSION = f"{ASSOCIATION_PROMPT_VERSION}+schema{SCHEMA_FINGERPRINT}"


class Associator(Protocol):
    """编排层依赖的关联契约；测试与重放用脚本化实现替换。"""

    @property
    def version(self) -> str: ...

    async def associate(self, payload: AssociationInput) -> AssociationAssembly: ...


@dataclass(frozen=True)
class AssociationConfig:
    max_targets_per_call: int = 12
    max_prompt_chars: int = 200_000
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in ("max_targets_per_call", "max_prompt_chars", "transient_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if name != "transient_retries" and value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.transient_retry_delay_seconds, bool)
            or not isinstance(self.transient_retry_delay_seconds, int | float)
            or self.transient_retry_delay_seconds < 0
        ):
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


class AssociationLimitError(ValueError):
    """这一批的目标数或提示词超过本层配置的边界；该批跳过并留信号，不切块。"""


class LLMAssociator:
    def __init__(self, client: StructuredChatClient, *, config: AssociationConfig | None = None) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        resolved = config or AssociationConfig()
        if not isinstance(resolved, AssociationConfig):
            raise TypeError("config must be AssociationConfig")
        self.client = client
        self.config = resolved

    @property
    def version(self) -> str:
        return ASSOCIATION_VERSION

    async def associate(self, payload: AssociationInput) -> AssociationAssembly:
        if not isinstance(payload, AssociationInput):
            raise TypeError("payload must be AssociationInput")
        if len(payload.targets) > self.config.max_targets_per_call:
            raise AssociationLimitError(
                f"batch has {len(payload.targets)} targets, above max_targets_per_call={self.config.max_targets_per_call}"
            )
        request = build_request(payload)
        prompt_chars = sum(len(message.content or "") for message in request.messages)
        if prompt_chars > self.config.max_prompt_chars:
            raise AssociationLimitError(f"association prompt has {prompt_chars} characters, above max_prompt_chars")
        schema = association_json_schema(len(payload.targets))
        response = None
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request,
                    schema=schema,
                    name="scene_association",
                    validator=lambda parsed: assemble_association(parsed, payload),
                )
                break
            except ModelTransportError:
                # 只重试传输层的瞬态错；答复本身不合格重试也是同一份完整调用，不做。
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
        assert response is not None
        return cast("AssociationAssembly", response.value)


__all__ = [
    "ASSOCIATION_VERSION",
    "AssociationConfig",
    "AssociationLimitError",
    "Associator",
    "LLMAssociator",
]
