"""使用独立模型对数据集回答执行可重试的结构化判定。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from habitus.benchmark.model import BenchmarkAnswerRecord, BenchmarkJudgeRecord
from habitus.benchmark.prompts import judge_prompt
from habitus.benchmark.protocol import BenchmarkJudgePolicy
from habitus.model_client import ChatCallContext, ChatMessage, ChatRequest, StructuredChatClient


@dataclass(frozen=True)
class JudgeDecision:
    """Judge 只能输出二元结果和简短依据。"""

    verdict: Literal["correct", "wrong"]
    reasoning: str

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "reasoning"],
            "properties": {
                "verdict": {"type": "string", "enum": ["correct", "wrong"]},
                "reasoning": {"type": "string", "minLength": 1, "maxLength": 4_000},
            },
        }

    @classmethod
    def model_validate(cls, value: object) -> JudgeDecision:
        if not isinstance(value, Mapping) or set(value) != {"verdict", "reasoning"}:
            raise ValueError("judge result must contain exactly verdict and reasoning")
        verdict = value["verdict"]
        reasoning = value["reasoning"]
        if verdict not in {"correct", "wrong"}:
            raise ValueError("judge verdict must be correct or wrong")
        if not isinstance(reasoning, str) or not reasoning.strip() or len(reasoning) > 4_000:
            raise ValueError("judge reasoning must be bounded non-empty text")
        return cls(
            verdict=cast(Literal["correct", "wrong"], verdict),
            reasoning=reasoning.strip(),
        )


class BenchmarkJudge:
    """与回答器分离的 Judge；错误不会被计入正确或错误分母。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        policy: BenchmarkJudgePolicy = BenchmarkJudgePolicy.STRICT,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("judge client must be StructuredChatClient")
        self.client = client
        self.policy = BenchmarkJudgePolicy(policy)

    async def grade(
        self,
        answer: BenchmarkAnswerRecord,
        *,
        evidence_texts: tuple[str, ...] = (),
    ) -> BenchmarkJudgeRecord:
        started = time.perf_counter()
        try:
            result = await self.client.complete_model_async(
                ChatRequest(
                    messages=(
                        ChatMessage(
                            role="user",
                            content=judge_prompt(
                                answer,
                                evidence_texts=evidence_texts,
                                policy=self.policy,
                            ),
                        ),
                    ),
                    temperature=0.0,
                    max_output_tokens=1_500,
                ),
                model_class=JudgeDecision,
                name="habitus_benchmark_judge_v2",
                context=ChatCallContext(
                    prompt_version=f"habitus_benchmark_judge_v2:{self.policy.value}",
                    metadata={
                        "dataset": answer.dataset,
                        "sample_id": answer.sample_id,
                        "question_id": answer.question_id,
                    },
                ),
            )
        except Exception as exc:
            return BenchmarkJudgeRecord(
                answer=answer,
                verdict="judge_error",
                reasoning=f"{type(exc).__name__}: {exc}",
                judge_latency_ms=(time.perf_counter() - started) * 1_000,
            )
        usage = result.response.usage
        return BenchmarkJudgeRecord(
            answer=answer,
            verdict=result.value.verdict,
            reasoning=result.value.reasoning,
            judge_input_tokens=usage.input_tokens,
            judge_output_tokens=usage.output_tokens,
            judge_total_tokens=usage.total_tokens,
            judge_latency_ms=(time.perf_counter() - started) * 1_000,
        )


__all__ = ["BenchmarkJudge", "JudgeDecision"]
