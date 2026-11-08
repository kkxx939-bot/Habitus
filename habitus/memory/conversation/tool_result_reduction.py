"""写入 Conversation 前对超大工具结果与超限工具交换的确定性降载。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace

from habitus.foundation.integrity import canonical_json
from habitus.memory.conversation.retention import (
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
)
from habitus.memory.conversation.tool_result_extraction import ToolResultExtractor
from habitus.pre.conversation.media import describe_media_only
from habitus.pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
)

#: 抽一组工具交换时，摘录字数从配置值逐次减半，最少到这么多字。
_MIN_EXCHANGE_SUMMARY_CHARS = 128
#: 调用参数是对象时，规范 JSON 不超过这么多字的字段原样保留（路径、命令、开关），只抽更大的字段。
_VERBATIM_ARGUMENT_CHARS = 256


# TODO(conversation-source): 修改此压缩算法语义时，必须同步提升
# memory/workflow/conversation_consumer.py 中 tool_result_reducer_version。
class ConversationToolResultReducer:
    """在写入 Conversation 前压缩超大工具结果与超限工具交换里的调用参数，不建立旁路原文存储。"""

    def __init__(self, config: ConversationSegmentationConfig | None = None) -> None:
        self.config = config or ConversationSegmentationConfig()
        self.extractor = ToolResultExtractor(self.config.max_tool_result_summary_chars)

    def reduce(
        self,
        message: ConversationMessage,
        *,
        force_omit: bool = False,
        force_summarize: bool = False,
        description: str | None = None,
        maximum: int | None = None,
    ) -> ConversationMessage:
        """保留小结果；把大文本抽成关键信息，把媒体或不可保留结果明确省略。

        只有媒体的结果记原载荷的字节数与 sha256（经不经插件占位都相同，见 ``describe_media_only``）；
        其余结果记整份内容规范 JSON 的字节数与 sha256。``maximum`` 是摘录字数上限，缺省取配置值。
        """

        if message.role is not ConversationMessageRole.TOOL_RESULT:
            raise ValueError("only tool_result messages can be reduced")
        if force_omit and force_summarize:
            raise ValueError("tool result cannot be force-omitted and force-summarized together")
        if message.content_mode is not ConversationToolResultContentMode.INLINE:
            return message
        encoded = canonical_json(message.content).encode("utf-8")
        size = len(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        media = describe_media_only(message.content)
        if (
            not force_omit
            and media is None
            and not force_summarize
            and size <= self.config.max_inline_tool_result_bytes
        ):
            return message

        if media is not None:
            # 只有媒体、没有文字的结果按"媒体只存描述、大小与摘要值"处理，不论大小；大小与摘要值指原载荷。
            content = self._description(description, fallback=media.text)
            size, digest = media.original_size_bytes, media.original_sha256
            mode = ConversationToolResultContentMode.OMITTED
        elif force_omit:
            content = self._description(
                description,
                fallback="工具结果属于媒体、二进制、下载载荷或不可长期保存内容，原始载荷未保存。",
            )
            mode = ConversationToolResultContentMode.OMITTED
        else:
            assert message.tool_status is not None
            content = self._extractor(maximum).extract(
                message.content,
                tool_name=message.tool_name,
                tool_status=message.tool_status.value,
                size_bytes=size,
            )
            mode = ConversationToolResultContentMode.SUMMARIZED
        return self._reduced_result(message, content=content, mode=mode, size=size, digest=digest)

    @staticmethod
    def _reduced_result(
        message: ConversationMessage,
        *,
        content: object,
        mode: ConversationToolResultContentMode,
        size: int,
        digest: str,
    ) -> ConversationMessage:
        """用降载后的内容、模式与原载荷的大小、摘要值重建同一条工具结果。"""

        return ConversationMessage(
            message_id=message.message_id,
            sequence=message.sequence,
            role=message.role,
            occurred_at=message.occurred_at,
            content=content,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_status=message.tool_status,
            content_mode=mode,
            source_ref=message.source_ref,
            original_size_bytes=size,
            original_sha256=digest,
        )

    def reduce_tool_call(self, message: ConversationMessage, *, maximum: int | None = None) -> ConversationMessage:
        """把调用参数抽成关键信息：对象参数里小字段原样留、大字段各自抽取，其余情形整份抽取；记原参数的字节数与 sha256。"""

        if message.role is not ConversationMessageRole.TOOL_CALL:
            raise ValueError("only tool_call messages can have their arguments reduced")
        if message.content_mode is not None:
            return message
        encoded = canonical_json(message.content).encode("utf-8")
        return replace(
            message,
            content=self._call_arguments(message.content, message.tool_name, self._maximum(maximum)),
            content_mode=ConversationToolResultContentMode.SUMMARIZED,
            original_size_bytes=len(encoded),
            original_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def fit_exchanges(
        self,
        batch: ConversationBatch,
        planner: ConversationRetentionPlanner,
        *,
        originals: Mapping[str, ConversationMessage],
        omit_tool_call_ids: frozenset[str] = frozenset(),
    ) -> ConversationBatch:
        """让这一批里每组切不开的工具交换都放得进一段。

        超限的一组按当前 Token 从大到小，逐个把工具结果或调用参数从原文（``originals`` 按 message_id 取）
        重新抽取，放得下就停；全抽过仍放不下，摘录字数减半再来，直到下限。只看这一批本身，同一批重放结果相同。
        放不进的（消息条数超限、或切不开的部分不是工具消息）原样留给规划器报错。
        """

        messages = list(batch.messages)
        changed = False
        for start, end in planner.atomic_spans(batch.messages):
            unit = tuple(messages[start:end])
            if planner.fits_segment(unit):
                continue
            fitted = self._fit_unit(unit, planner, originals, omit_tool_call_ids)
            if fitted != unit:
                messages[start:end] = fitted
                changed = True
        return ConversationBatch(batch.conversation_id, tuple(messages)) if changed else batch

    def _fit_unit(
        self,
        unit: tuple[ConversationMessage, ...],
        planner: ConversationRetentionPlanner,
        originals: Mapping[str, ConversationMessage],
        omit_tool_call_ids: frozenset[str],
    ) -> tuple[ConversationMessage, ...]:
        candidates = [
            index
            for index, message in enumerate(unit)
            if self._reducible(originals.get(message.message_id), omit_tool_call_ids)
        ]
        if not candidates or len(unit) > planner.config.max_segment_messages:
            return unit
        current = list(unit)
        maximum = self.config.max_tool_result_summary_chars
        while True:
            for index in sorted(candidates, key=lambda item: (-planner.token_estimator(current[item]), item)):
                if planner.fits_segment(tuple(current)):
                    return tuple(current)
                smaller = self._reduced_member(originals[current[index].message_id], maximum)
                if planner.token_estimator(smaller) < planner.token_estimator(current[index]):
                    current[index] = smaller
            if planner.fits_segment(tuple(current)) or maximum <= _MIN_EXCHANGE_SUMMARY_CHARS:
                return tuple(current)
            maximum = max(_MIN_EXCHANGE_SUMMARY_CHARS, maximum // 2)

    @staticmethod
    def _reducible(original: ConversationMessage | None, omit_tool_call_ids: frozenset[str]) -> bool:
        if original is None:
            return False
        if original.role is ConversationMessageRole.TOOL_CALL:
            return original.content_mode is None
        return (
            original.role is ConversationMessageRole.TOOL_RESULT
            and original.content_mode is ConversationToolResultContentMode.INLINE
            and original.tool_call_id not in omit_tool_call_ids
            and describe_media_only(original.content) is None
        )

    def _reduced_member(self, original: ConversationMessage, maximum: int) -> ConversationMessage:
        if original.role is ConversationMessageRole.TOOL_CALL:
            return self.reduce_tool_call(original, maximum=maximum)
        return self.reduce(original, force_summarize=True, maximum=maximum)

    def _call_arguments(self, content: object, tool_name: str | None, maximum: int) -> object:
        if isinstance(content, Mapping):
            sizes = {key: len(canonical_json(value)) for key, value in content.items()}
            large = {key for key, size in sizes.items() if size > _VERBATIM_ARGUMENT_CHARS}
            kept = sum(size for key, size in sizes.items() if key not in large)
            if large and kept <= maximum // 2:
                extractor = ToolResultExtractor(max(_MIN_EXCHANGE_SUMMARY_CHARS, (maximum - kept) // len(large)))
                return {
                    key: (
                        extractor.extract(
                            value,
                            tool_name=tool_name,
                            size_bytes=len(canonical_json(value).encode("utf-8")),
                            call_argument=str(key),
                        )
                        if key in large
                        else value
                    )
                    for key, value in content.items()
                }
        return ToolResultExtractor(maximum).extract(
            content,
            tool_name=tool_name,
            size_bytes=len(canonical_json(content).encode("utf-8")),
            call_argument="",
        )

    def _maximum(self, maximum: int | None) -> int:
        if maximum is None:
            return self.config.max_tool_result_summary_chars
        if maximum < 1:
            raise ValueError("maximum must be a positive integer or None")
        return maximum

    def _extractor(self, maximum: int | None) -> ToolResultExtractor:
        resolved = self._maximum(maximum)
        return self.extractor if resolved == self.extractor.maximum else ToolResultExtractor(resolved)

    @staticmethod
    def _description(value: str | None, *, fallback: str) -> str:
        if value is None:
            return fallback
        if not value.strip():
            raise ValueError("description must be non-empty text or None")
        return value.strip()


__all__ = ["ConversationToolResultReducer"]
