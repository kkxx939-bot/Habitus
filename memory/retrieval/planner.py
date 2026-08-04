"""结合 Conversation 上下文生成受控、多查询的 Agent 记忆检索计划。"""

from __future__ import annotations

from typing import cast

from foundation.integrity import canonical_json
from memory.retrieval.context import ConversationSearchContext, render_recent_messages
from memory.retrieval.model import (
    MemoryQueryPlan,
    MemoryQueryPlanContent,
    MemorySearchServiceConfig,
    MemoryTypedQuery,
)
from ModelClient import ChatCallContext, ChatMessage, ChatRequest, StructuredChatClient


class MemorySearchQueryPlanner:
    """分析召回意图，但不允许模型决定 URI 或记忆操作。"""

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

    def direct(self, query: str) -> MemoryQueryPlan:
        """构造不调用 LLM 的直接查询计划。"""

        normalized = self._query(query)
        return MemoryQueryPlan(
            original_query=normalized,
            queries=(MemoryTypedQuery(normalized, "回答当前用户问题", 1),),
        )

    async def plan(
        self,
        query: str,
        context: ConversationSearchContext,
        *,
        target_context: str = "",
    ) -> MemoryQueryPlan:
        """用当前问题、活跃摘要和最近消息生成可独立执行的多查询计划。"""

        normalized = self._query(query)
        if not isinstance(context, ConversationSearchContext):
            raise TypeError("context must be ConversationSearchContext")
        if not isinstance(target_context, str):
            raise TypeError("target_context must be text")
        if context.empty or self.config.max_planned_queries == 1:
            direct = self.direct(normalized)
            return MemoryQueryPlan(
                original_query=direct.original_query,
                queries=direct.queries,
                conversation_id=context.conversation_id,
            )

        recent = render_recent_messages(
            context.recent_messages,
            max_message_chars=self.config.max_recent_message_chars,
        )
        payload = canonical_json(
            {
                "current_query": normalized,
                "conversation_summary": context.summary_context,
                "recent_messages": recent,
                "target_memory_context": target_context,
                "max_queries": self.config.max_planned_queries,
            }
        )
        if len(payload) > self.config.max_planner_context_chars:
            raise ValueError("memory search planner context exceeds its configured bound")

        response = await self.client.complete_json_async(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "你是只读记忆检索查询规划器。根据当前问题、Conversation 历史摘要和最近消息，"
                            "生成能够独立执行的语义查询。查询应补全当前问题中真实存在的省略指代，并在一个"
                            "问题包含多个不同检索意图时拆成多条；不要重复同义查询。只能使用输入里已经出现的"
                            "信息，不得补造事实。不要输出 memory URI、目录、记忆类型、过滤器、分数、"
                            "CREATE/UPDATE/DELETE、Links/Backlinks 或任何写入操作。priority 为 1 到 5，"
                            "数字越小越重要；intent 只简述该查询要找什么。"
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content="请输出严格的记忆查询计划 JSON：\n" + payload,
                    ),
                ),
                temperature=0.0,
                max_output_tokens=self.config.planner_max_output_tokens,
            ),
            schema=self._schema(),
            validator=self._validate_content,
            name="memory_search_query_plan",
            context=ChatCallContext(prompt_version="memory_search_query_plan_v1"),
        )
        content = cast(MemoryQueryPlanContent, response.value)
        planned = self._validated_queries(content.queries)
        queries = self._merge_original(normalized, planned)
        return MemoryQueryPlan(
            original_query=normalized,
            queries=queries,
            conversation_id=context.conversation_id,
        )

    def _validated_queries(
        self,
        queries: tuple[MemoryTypedQuery, ...],
    ) -> tuple[MemoryTypedQuery, ...]:
        if len(queries) > self.config.max_planned_queries:
            raise ValueError("model query plan exceeds the configured query count")
        for query in queries:
            if len(query.query) > self.config.max_planned_query_chars:
                raise ValueError("model planned query exceeds its configured character bound")
            if len(query.intent) > self.config.max_query_intent_chars:
                raise ValueError("model query intent exceeds its configured character bound")
        return queries

    def _validate_content(self, value: object) -> MemoryQueryPlanContent:
        content = MemoryQueryPlanContent.model_validate(value)
        self._validated_queries(content.queries)
        return content

    def _schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": self.config.max_planned_queries,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query", "intent", "priority"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.config.max_planned_query_chars,
                            },
                            "intent": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.config.max_query_intent_chars,
                            },
                            "priority": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                    },
                }
            },
        }

    def _merge_original(
        self,
        original: str,
        planned: tuple[MemoryTypedQuery, ...],
    ) -> tuple[MemoryTypedQuery, ...]:
        values = [MemoryTypedQuery(original, "回答当前用户问题", 1), *planned]
        unique: dict[str, MemoryTypedQuery] = {}
        for value in values:
            key = " ".join(value.query.casefold().split())
            previous = unique.get(key)
            if previous is None or value.priority < previous.priority:
                unique[key] = value
        ordered = sorted(unique.values(), key=lambda value: (value.priority, values.index(value)))
        return tuple(ordered[: self.config.max_planned_queries])

    def _query(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("memory search query must be non-empty text")
        normalized = value.strip()
        if len(normalized) > self.config.max_query_chars:
            raise ValueError("memory search query exceeds its configured character bound")
        return normalized


__all__ = ["MemorySearchQueryPlanner"]
