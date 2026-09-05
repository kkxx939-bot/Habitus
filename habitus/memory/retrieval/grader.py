"""在 Memory 主召回完成后判断是否需要 Conversation Summary 后备。"""

from __future__ import annotations

from typing import cast

from habitus.foundation.integrity import canonical_json
from habitus.memory.retrieval.model import (
    MemoryMatchedMemory,
    MemoryQueryPlan,
    MemoryRetrievalAssessment,
    MemoryRetrievalSufficiency,
    MemorySearchServiceConfig,
)
from habitus.model_client import ChatCallContext, ChatMessage, ChatRequest, StructuredChatClient


class MemoryRetrievalGrader:
    """只判断回答覆盖度，不输出置信度、记忆操作或 Summary 结果。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: MemorySearchServiceConfig | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        if config is not None and not isinstance(config, MemorySearchServiceConfig):
            raise TypeError("config must be MemorySearchServiceConfig")
        self.client = client
        self.config = config or MemorySearchServiceConfig()

    async def assess(
        self,
        plan: MemoryQueryPlan,
        memories: tuple[MemoryMatchedMemory, ...],
        memory_context: str,
    ) -> MemoryRetrievalAssessment:
        """无命中时确定性启用后备；有命中时让模型只审查信息覆盖度。"""

        if not isinstance(plan, MemoryQueryPlan):
            raise TypeError("plan must be MemoryQueryPlan")
        normalized = self._query(plan.original_query)
        if not isinstance(memories, tuple) or any(
            not isinstance(memory, MemoryMatchedMemory) for memory in memories
        ):
            raise TypeError("memories must contain MemoryMatchedMemory values")
        if not isinstance(memory_context, str):
            raise TypeError("memory_context must be text")
        if not memories:
            return MemoryRetrievalAssessment(
                decision=MemoryRetrievalSufficiency.INSUFFICIENT,
                reason="MemoryTree 没有返回可用于回答当前问题的长期记忆。",
                missing_information=("当前问题所需的相关历史信息",),
                summary_query=self._fallback_query(plan),
            )
        if not memory_context:
            raise ValueError("non-empty Memory results require assembled context")
        if len(memory_context) > self.config.retrieval_grader_max_context_chars:
            raise ValueError("Memory retrieval context exceeds the grader input bound")

        payload = canonical_json(
            {
                "current_query": normalized,
                "resolved_memory_queries": tuple(query.to_dict() for query in plan.queries),
                "memory_result_count": len(memories),
                "memory_context": memory_context,
                "summary_query_max_chars": self.config.summary_fallback_max_query_chars,
                "missing_item_limit": self.config.retrieval_grader_max_missing_items,
            }
        )
        response = await self.client.complete_json_async(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "你是只读 Memory 检索充分性审查器。只判断给出的长期 Memory 是否足以支持回答"
                            "当前问题。Memory 已包含完整 L2 与有界 Links/Backlinks 一跳结果。sufficient 的"
                            "标准是 Memory 能够独立、明确地覆盖 current_query 实际要求的信息；相关命中、"
                            "关键词重合或只包含最终结论都不等于充分。当问题询问原因、过程、决策依据、"
                            "时间线、变化、纠正关系，或需要历史上下文才能解释的指代时，只要 Memory 缺少"
                            "对应支撑，就必须判定 insufficient。若问题只要求结论且 Memory 已经完整覆盖，"
                            "不要仅因还能获得额外背景而判定不足。向量分数只代表相关性，不能单独当作事实"
                            "置信度。若 sufficient，missing_information 必须为空且 summary_query 必须为 null。"
                            "若 insufficient，列出"
                            "至多指定数量的真实缺口，并结合 resolved_memory_queries 中已经补全的指代，生成一条"
                            "只用于搜索历史 Conversation Summary 的查询。resolved_memory_queries 只是查询上下文，"
                            "不是历史 Summary 召回结果。"
                            "只能使用输入中出现的信息，不得补造事实。不要输出 URI、记忆类型、分数、写入、"
                            "删除、合并或 Links/Backlinks 操作。"
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content="请输出严格的 Memory 检索充分性 JSON：\n" + payload,
                    ),
                ),
                temperature=0.0,
                max_output_tokens=self.config.retrieval_grader_max_output_tokens,
            ),
            schema=self._schema(),
            validator=self._validate_assessment,
            name="agent_memory_retrieval_assessment",
            context=ChatCallContext(prompt_version="agent_memory_retrieval_grader_v2"),
        )
        return cast(MemoryRetrievalAssessment, response.value)

    def _validate_assessment(self, value: object) -> MemoryRetrievalAssessment:
        assessment = MemoryRetrievalAssessment.model_validate(value)
        if len(assessment.reason) > self.config.retrieval_grader_max_reason_chars:
            raise ValueError("retrieval grader reason exceeds its configured bound")
        if len(assessment.missing_information) > self.config.retrieval_grader_max_missing_items:
            raise ValueError("retrieval grader returned too many missing information items")
        if any(
            len(item) > self.config.retrieval_grader_max_missing_item_chars
            for item in assessment.missing_information
        ):
            raise ValueError("retrieval grader missing information exceeds its configured bound")
        if assessment.summary_query is not None and len(
            assessment.summary_query
        ) > self.config.summary_fallback_max_query_chars:
            raise ValueError("retrieval grader Summary query exceeds its configured bound")
        return assessment

    def _schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "reason", "missing_information", "summary_query"],
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": [item.value for item in MemoryRetrievalSufficiency],
                },
                "reason": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": self.config.retrieval_grader_max_reason_chars,
                },
                "missing_information": {
                    "type": "array",
                    "maxItems": self.config.retrieval_grader_max_missing_items,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": self.config.retrieval_grader_max_missing_item_chars,
                    },
                },
                "summary_query": {
                    "anyOf": [
                        {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": self.config.summary_fallback_max_query_chars,
                        },
                        {"type": "null"},
                    ]
                },
            },
        }

    def _query(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("retrieval grader query must be non-empty text")
        normalized = value.strip()
        if len(normalized) > self.config.max_query_chars:
            raise ValueError("retrieval grader query exceeds its configured bound")
        return normalized

    def _fallback_query(self, plan: MemoryQueryPlan) -> str:
        """在无 Memory 命中时优先使用已补全指代的独立查询。"""

        original_key = " ".join(plan.original_query.casefold().split())
        ordered = sorted(
            enumerate(plan.queries),
            key=lambda item: (
                " ".join(item[1].query.casefold().split()) == original_key,
                item[1].priority,
                item[0],
            ),
        )
        maximum = self.config.summary_fallback_max_query_chars
        selected: list[str] = []
        used = 0
        for _, query in ordered:
            separator = 1 if selected else 0
            if used + separator + len(query.query) > maximum:
                continue
            selected.append(query.query)
            used += separator + len(query.query)
        if selected:
            return "\n".join(selected)
        return plan.queries[0].query[:maximum]


__all__ = ["MemoryRetrievalGrader"]
