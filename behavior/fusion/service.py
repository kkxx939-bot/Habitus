"""调用模型完成一次融合判断。

## 领域校验也走模型重试

结构化输出把失败分成 ``json_parse`` / ``json_schema`` / ``domain`` 三个阶段并分别反馈。
这里把**装配与自洽性校验也放进领域校验回调**，于是"引用了不存在的判断编号""两条判断无法区分"
这类错误会被精确告知模型并有界重试，而不是整批失败。

这样安排的理由是它们本来就是可纠正的形式错误：模型看得见片段，只是引用写错了。真正不可纠正的
（片段本身不足以支撑判断）应当由模型主动**把 behavior 留空**，而不是靠重试碰运气。

## 融合是本层唯一调用模型的一环

观测清洗是无模型的确定性归一，装配、派生与落盘也都不含模型。模型只负责"看完这段观测，我判断
这里发生了什么"，其余一律不经过它。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from behavior.fusion.assembly import assemble_judgement_batch
from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionTruncatedError
from behavior.fusion.judgement import BehaviorJudgementBatch
from behavior.fusion.prompt import (
    FUSION_PROMPT_VERSION,
    FUSION_SYSTEM_PROMPT,
    render_context_judgements,
    render_fragments,
    render_primary_subject,
)
from behavior.fusion.result import BehaviorFusionResult
from behavior.fusion.schema import fusion_json_schema
from behavior.fusion.segmentation import BehaviorFusionSegment
from behavior.fusion.validation import validate_judgement_batch
from ModelClient import ChatMessage, ChatRequest, StructuredChatClient
from ModelClient.contracts import ModelStructuredOutputError


class BehaviorJudgementFuser:
    """把一个融合窗口交给模型，并只接受通过全部确定性校验的判断。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: BehaviorFusionConfig | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        resolved = config or BehaviorFusionConfig()
        if not isinstance(resolved, BehaviorFusionConfig):
            raise TypeError("config must be BehaviorFusionConfig")
        self.client = client
        self.config = resolved

    async def fuse(
        self,
        segment: BehaviorFusionSegment,
        *,
        primary_subject: str,
        context_judgements: Sequence[Mapping[str, Any]] = (),
    ) -> BehaviorFusionResult:
        """融合一个窗口；判断未通过确定性校验则明确失败，绝不带病返回。"""

        if not isinstance(segment, BehaviorFusionSegment):
            raise TypeError("segment must be BehaviorFusionSegment")
        request = self._request(segment, primary_subject, context_judgements)
        try:
            response = await self.client.complete_json_async(
                request,
                schema=fusion_json_schema(len(segment.fragments)),
                name="behavior_judgement_batch",
                validator=lambda parsed: self._validated(parsed, segment, context_judgements),
            )
        except ModelStructuredOutputError as exc:
            if _is_truncation(exc):
                # 本层是唯一 import ModelClient 的一环：把"输出截断"翻译成领域错误，runner 据此
                # 降级（整段记为没读懂），不必认识模型客户端的异常谱系。
                raise BehaviorFusionTruncatedError(
                    f"model output truncated for a segment of {len(segment.fragments)} fragments"
                ) from exc
            raise
        batch = response.value
        if not isinstance(batch, BehaviorJudgementBatch):  # pragma: no cover - 校验器保证类型
            raise BehaviorFusionError("structured output did not produce a judgement batch")
        return BehaviorFusionResult(
            segment=segment,
            batch=batch,
            prompt_version=FUSION_PROMPT_VERSION,
            validation_attempts=response.validation_attempts,
        )

    def _validated(
        self,
        parsed: object,
        segment: BehaviorFusionSegment,
        context: Sequence[Mapping[str, Any]],
    ) -> BehaviorJudgementBatch:
        batch = assemble_judgement_batch(
            parsed,
            fragment_count=len(segment.fragments),
            context_states=tuple(item.get("status") for item in context),
            config=self.config,
            # 主体是否在场在装配期降级（剔名/降为没读懂），校验层只做后置断言；否则这类记账
            # 疏漏会走模型重试——真实数据一周 371 次，每次白烧三次调用。
            participants_by_no={
                index: fragment.participants
                for index, fragment in enumerate(segment.fragments, start=1)
            },
        )
        validate_judgement_batch(batch, segment.fragments)
        return batch

    def _request(
        self,
        segment: BehaviorFusionSegment,
        primary_subject: str,
        context_judgements: Sequence[Mapping[str, Any]],
    ) -> ChatRequest:
        content = "\n\n".join(
            (
                "【本次跟踪的主体】",
                render_primary_subject(primary_subject),
                "【先前的判断】",
                render_context_judgements(context_judgements, segment.fragments[0].occurred_at),
                "【本次片段】",
                render_fragments(segment.fragments),
            )
        )
        return ChatRequest(
            messages=(
                ChatMessage(role="system", content=FUSION_SYSTEM_PROMPT),
                ChatMessage(role="user", content=content),
            )
        )


def _is_truncation(error: BaseException) -> bool:
    """结构化客户端把 length 截断包成 ModelStructuredOutputError，原因在 cause 链里。"""

    current: BaseException | None = error
    while current is not None:
        if "truncated" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


__all__ = ["BehaviorJudgementFuser"]
