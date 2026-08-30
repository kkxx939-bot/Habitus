"""把融合层的自由文本行为名归属到词表：逐字复用，或新建。

## 模型只选，不造

决策空间被 schema 的 enum 钉死：要么逐字选中清单里的一个正名，要么 null。null 时新类型的
正名就是**原始名字本身**——模型从不发明新名字，词表里的每个字都来自融合层的原话。这与
"写入层零发明"一致：这里不判断"发生了什么"，只登记"这次的名字与历史上哪个名字是同一类"。

## 确定性快路径优先

绝大多数 occurrence 重复已知名字：先按规范身份（NFC + casefold）精确查词表，命中即返回，
不调模型。模型只在真正没见过的名字上被调用——词表收敛后调用会越来越稀。

## 宁分勿并

提示词与 schema 描述都要求拿不准就 null。错误合并会把两类行为的统计搅在一起（预测层
vocabulary_builder 的同一教训：错误合并污染条件分布，比不合并更糟）；错误分裂只是暂时多一个
类型，改词表重打 token 即可修复，原始名一直保留在 occurrence 上。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from behavior.kinds.config import BehaviorKindConfig
from behavior.kinds.model import (
    BehaviorKindError,
    BehaviorKindLimitError,
    BehaviorKindRegistry,
)
from behavior.model import semantic_name
from ModelClient import ChatMessage, ChatRequest, StructuredChatClient
from ModelClient.contracts import ModelResponseError, ModelTransportError

KIND_PROMPT_VERSION = "behavior_kind_prompt_v1"

KIND_SYSTEM_PROMPT = """\
你在为一套行为记忆系统维护「行为类型词表」。

会给你已有的类型清单（一行一类：正名，括号里是它的别名），以及一个新观测到的行为名字。
判断新名字和清单里的哪一类是**同一类行为**：

- 是同一类 → match 填那一类的**正名**，逐字复制清单里的写法。
- 都不是，或拿不准 → match 填 null。

同一类 = 同一件事的不同说法（换措辞、同义词、更具体或更笼统）。
只是发生在相近场合的不同行为不算同一类：「洗手」与「洗碗」是两类。

【重要】宁可 null，不要勉强合并。错误的合并会把两类行为的统计搅在一起，比暂时多出
一个类型更糟——多出的类型以后还能并回去，搅在一起的统计分不开。
"""


def kind_match_schema(tokens: tuple[str, ...]) -> dict[str, Any]:
    """匹配决策的 JSON Schema；enum 把"逐字复用"钉死在格式层。"""

    if not tokens:
        raise BehaviorKindError("kind matching requires at least one registered token")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["match"],
        "properties": {
            "match": {
                "description": (
                    "与新名字属于同一类行为的既有正名，必须逐字复制清单里的写法；"
                    "没有同类或拿不准就填 null——宁可 null，不要勉强合并。"
                ),
                "anyOf": [{"type": "null"}, {"type": "string", "enum": list(tokens)}],
            }
        },
    }


@dataclass(frozen=True)
class BehaviorKindResolution:
    """一次归属的完整结果；``registry`` 是应用后的新词表，落盘由调用方负责。"""

    token: str
    registry: BehaviorKindRegistry
    created: bool
    model_called: bool
    validation_attempts: int
    prompt_version: str = KIND_PROMPT_VERSION


class BehaviorKindResolver:
    """给一个行为名找到（或建立）它的类型归属。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: BehaviorKindConfig | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        resolved = config or BehaviorKindConfig()
        if not isinstance(resolved, BehaviorKindConfig):
            raise TypeError("config must be BehaviorKindConfig")
        self.client = client
        self.config = resolved

    async def resolve(
        self, name: object, registry: BehaviorKindRegistry
    ) -> BehaviorKindResolution:
        """归属一个名字；已知名字零模型调用，未知名字最多一次结构化调用。"""

        if not isinstance(registry, BehaviorKindRegistry):
            raise TypeError("registry must be BehaviorKindRegistry")
        resolved_name = semantic_name(name, "behavior kind name")
        known = registry.token_for(resolved_name)
        if known is not None:
            return BehaviorKindResolution(
                token=known,
                registry=registry,
                created=False,
                model_called=False,
                validation_attempts=0,
            )
        if registry.kind_count == 0:
            return self._created(resolved_name, registry, model_called=False, attempts=0)
        response = None
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    self._request(resolved_name, registry),
                    schema=kind_match_schema(registry.tokens),
                    name="behavior_kind_match",
                    validator=lambda parsed: self._validated(parsed, registry),
                )
                break
            except (ModelTransportError, ModelResponseError):
                # 瞬态错误有界重试；契约类错误（结构化输出违约）照常抛出
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
        assert response is not None
        match = response.value
        if match is None:
            return self._created(
                resolved_name, registry, model_called=True, attempts=response.validation_attempts
            )
        assert isinstance(match, str)  # 校验器保证
        return BehaviorKindResolution(
            token=match,
            registry=self._with_alias(registry, match, resolved_name),
            created=False,
            model_called=True,
            validation_attempts=response.validation_attempts,
        )

    def _created(
        self,
        name: str,
        registry: BehaviorKindRegistry,
        *,
        model_called: bool,
        attempts: int,
    ) -> BehaviorKindResolution:
        if registry.kind_count >= self.config.max_kinds:
            raise BehaviorKindLimitError("behavior kind registry has no remaining kind capacity")
        return BehaviorKindResolution(
            token=name,
            registry=registry.with_new_kind(name),
            created=True,
            model_called=model_called,
            validation_attempts=attempts,
        )

    def _with_alias(
        self, registry: BehaviorKindRegistry, token: str, alias: str
    ) -> BehaviorKindRegistry:
        if len(registry.aliases_of(token)) >= self.config.max_aliases_per_kind:
            raise BehaviorKindLimitError(
                f"behavior kind has no remaining alias capacity: {token}"
            )
        return registry.with_alias(token, alias)

    @staticmethod
    def _validated(parsed: object, registry: BehaviorKindRegistry) -> str | None:
        if not isinstance(parsed, Mapping) or set(parsed) != {"match"}:
            raise BehaviorKindError("kind match output must be an object with exactly `match`")
        match = parsed["match"]
        if match is None:
            return None
        if not isinstance(match, str) or match not in registry.kinds:
            raise BehaviorKindError(
                "kind match must verbatim reuse a registered token or be null"
            )
        return match

    @staticmethod
    def _request(name: str, registry: BehaviorKindRegistry) -> ChatRequest:
        lines = []
        for token in registry.tokens:
            aliases = registry.aliases_of(token)
            lines.append(f"- {token}（别名：{'、'.join(aliases)}）" if aliases else f"- {token}")
        content = "\n".join(
            ("【已有类型清单】", *lines, "", "【新观测到的行为名】", name)
        )
        return ChatRequest(
            messages=(
                ChatMessage(role="system", content=KIND_SYSTEM_PROMPT),
                ChatMessage(role="user", content=content),
            )
        )


__all__ = [
    "KIND_PROMPT_VERSION",
    "KIND_SYSTEM_PROMPT",
    "BehaviorKindResolution",
    "BehaviorKindResolver",
    "kind_match_schema",
]
