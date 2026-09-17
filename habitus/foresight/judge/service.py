"""判断服务：预测层的模型触点。

与关联服务同一形态：瞬态传输错误有界重试（路由层自己还有一层，这里的次数要很小）；结构化输出
先过 schema 再过装配校验，装配对记账疏漏降级、只有输出整个不成形才让结构层重试。

调用的单位是**一包**：一刻的全部摊开候选一次问完，因为它们共用同一个此刻场景，分开问会对同一条线
给出互相矛盾的说法。包里没有摊开的候选时照样问——那时要判的只剩"今天到此刻正不正常"，这也是
一个要读了此刻场景才能答的问题，不由本层替模型填。不按字数截断、不设预算（2026-09-16 定）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol, cast

from habitus.foresight.assemble import EvidencePack
from habitus.foresight.judge.assembly import ASSEMBLY_VERSION, assemble_judgement
from habitus.foresight.judge.model import Judgement
from habitus.foresight.judge.prompt import JUDGE_PROMPT_VERSION, build_request
from habitus.foresight.judge.schema import SCHEMA_FINGERPRINT, judge_json_schema
from habitus.foresight.render import render_pack
from habitus.model_client import ModelTransportError, StructuredChatClient

#: 判断者的版本 = 提示词 + schema 指纹 + 装配纪律。三者任一变了，存下来的判断就不是同一种东西。
JUDGE_VERSION = f"{JUDGE_PROMPT_VERSION}+schema{SCHEMA_FINGERPRINT}+{ASSEMBLY_VERSION}"


class Judge(Protocol):
    """节奏层依赖的判断契约；测试与 DAY1 脚本用脚本化实现替换。"""

    @property
    def version(self) -> str: ...

    async def judge(self, pack: EvidencePack) -> Judgement: ...


@dataclass(frozen=True)
class JudgeConfig:
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.transient_retries, bool)
            or not isinstance(self.transient_retries, int)
            or self.transient_retries < 0
        ):
            raise ValueError("transient_retries must be a non-negative integer")
        if (
            isinstance(self.transient_retry_delay_seconds, bool)
            or not isinstance(self.transient_retry_delay_seconds, int | float)
            or self.transient_retry_delay_seconds < 0
        ):
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


class LLMJudge:
    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: JudgeConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        resolved = config or JudgeConfig()
        if not isinstance(resolved, JudgeConfig):
            raise TypeError("config must be JudgeConfig")
        self.client = client
        self.config = resolved
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    @property
    def version(self) -> str:
        return JUDGE_VERSION

    async def judge(self, pack: EvidencePack) -> Judgement:
        if not isinstance(pack, EvidencePack):
            raise TypeError("pack must be an EvidencePack")
        request = build_request(render_pack(pack))
        schema = judge_json_schema([item.kind_token for item in pack.expanded])
        judged_at = self._clock()
        response = None
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request,
                    schema=schema,
                    name="foresight_judgement",
                    validator=lambda parsed: assemble_judgement(
                        parsed, pack, judged_at=judged_at, version=JUDGE_VERSION
                    ),
                )
                break
            except ModelTransportError:
                # 只重试传输层的瞬态错；答复本身不合格由结构层纠正，这里不再套一层。
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
        assert response is not None
        judgement = cast("Judgement", response.value)
        # 结构层的纠正与修复要留在判断的信号里：第二轮才答对、JSON 是修出来的，读判断的人得知道。
        notes: list[str] = []
        if response.validation_attempts > 1:
            notes.append(f"structured: answered on attempt {response.validation_attempts}")
        if response.parse_mode != "strict":
            notes.append(f"structured: json parsed via {response.parse_mode}")
        if not notes:
            return judgement
        return replace(judgement, signals=(*judgement.signals, *notes))


__all__ = ["JUDGE_VERSION", "Judge", "JudgeConfig", "LLMJudge"]
