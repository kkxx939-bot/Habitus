"""测试使用的严格领域对象工厂；不替代任何生产校验。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from habitus.infrastructure.editor.snapshot import SnapshotBatch, SnapshotState, VersionedSnapshot
from habitus.memory.document import MemoryDocument, MemoryDocumentCodec, MemoryDocumentMetadata
from habitus.memory.model import MemoryKind
from habitus.memory.schema import MemorySchemaRegistry
from habitus.pre.conversation.messages import (
    ConversationMessage,
    ConversationMessageRole,
    ConversationSegment,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.pre.conversation.summaries import ConversationSegmentSummary, ConversationSummaryContent

UTC = UTC
BASE_TIME = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)


def message(
    sequence: int,
    role: ConversationMessageRole,
    content: object,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_status: ConversationToolResultStatus | None = None,
    content_mode: ConversationToolResultContentMode | None = None,
) -> ConversationMessage:
    """构造时间和序号严格递增的一条测试消息。"""

    return ConversationMessage(
        message_id=f"message-{sequence}",
        sequence=sequence,
        role=role,
        occurred_at=BASE_TIME + timedelta(seconds=sequence),
        content=content,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_status=tool_status,
        content_mode=content_mode,
    )


def closed_turn(*, start_sequence: int = 0, prompt: str = "请记住我喜欢简洁回答") -> tuple[ConversationMessage, ...]:
    """构造不含工具调用的完整一轮对话。"""

    return (
        message(start_sequence, ConversationMessageRole.PROMPT, prompt),
        message(start_sequence + 1, ConversationMessageRole.COMPLETION, "好的，我会保持简洁。"),
    )


def tool_turn(*, start_sequence: int = 0, status: ConversationToolResultStatus = ConversationToolResultStatus.COMPLETED) -> tuple[ConversationMessage, ...]:
    """构造工具调用与唯一终态结果均闭合的一轮对话。"""

    return (
        message(start_sequence, ConversationMessageRole.PROMPT, "检查项目状态"),
        message(
            start_sequence + 1,
            ConversationMessageRole.TOOL_CALL,
            {"path": "."},
            tool_call_id="call-1",
            tool_name="workspace.inspect",
        ),
        message(
            start_sequence + 2,
            ConversationMessageRole.TOOL_RESULT,
            "工作区正常" if status is ConversationToolResultStatus.COMPLETED else "读取失败",
            tool_call_id="call-1",
            tool_name="workspace.inspect",
            tool_status=status,
            content_mode=ConversationToolResultContentMode.INLINE,
        ),
        message(start_sequence + 3, ConversationMessageRole.COMPLETION, "检查完成。"),
    )


def segment(
    *,
    conversation_id: str = "conversation-1",
    segment_id: str = "segment-1",
    messages: tuple[ConversationMessage, ...] | None = None,
) -> ConversationSegment:
    return ConversationSegment(
        conversation_id=conversation_id,
        segment_id=segment_id,
        messages=closed_turn() if messages is None else messages,
    )


def summary_content() -> ConversationSummaryContent:
    return ConversationSummaryContent(
        overview="用户确认后续回答保持简洁。",
        chronology=("用户提出输出要求。", "助手确认执行。"),
        corrections=(),
        ending_state="偏好已经确认。",
        open_threads=(),
    )


def segment_summary(source: ConversationSegment | None = None) -> ConversationSegmentSummary:
    source = segment() if source is None else source
    content = summary_content()
    return ConversationSegmentSummary(
        conversation_id=source.conversation_id,
        segment_id=source.segment_id,
        source_message_digest=source.digest,
        start_sequence=source.start_sequence,
        end_sequence=source.end_sequence,
        started_at=source.started_at,
        ended_at=source.ended_at,
        generated_at=source.ended_at + timedelta(seconds=1),
        starts_mid_turn=source.starts_mid_turn,
        ends_mid_turn=source.ends_mid_turn,
        **content.to_dict(),
    )


def registry() -> MemorySchemaRegistry:
    return MemorySchemaRegistry.load_default()


def codec() -> MemoryDocumentCodec:
    return MemoryDocumentCodec(registry())


def memory_fields(kind: MemoryKind) -> dict[str, object]:
    return {
        MemoryKind.PROFILE: {"content": "- 用户是视频创作者"},
        MemoryKind.PREFERENCE: {"topic": "回答风格", "content": "- 偏好简洁直接的回答"},
        MemoryKind.ENTITY: {"category": "项目", "name": "Habitus", "summary": "Habitus 是用户正在重构的记忆系统。"},
        MemoryKind.TOOL: {"tool_name": "workspace.inspect", "verification": "返回工作区状态后检查退出状态。"},
        MemoryKind.EVENT: {"event_date": date(2026, 7, 1), "event_name": "确认记忆树", "summary": "用户确认了六类长期记忆树。"},
        MemoryKind.INTENTION: {"intent_name": "完成记忆系统重构", "status": "open", "next_step": "补齐测试体系"},
    }[kind]


def document(
    kind: MemoryKind = MemoryKind.PREFERENCE,
    *,
    fields: dict[str, object] | None = None,
    revision: int = 1,
    timestamp: datetime = BASE_TIME,
) -> MemoryDocument:
    confirmed = kind is MemoryKind.INTENTION
    metadata = MemoryDocumentMetadata(
        revision=revision,
        created_at=timestamp,
        updated_at=timestamp + timedelta(seconds=revision - 1),
        last_confirmed_at=timestamp if confirmed else None,
    )
    return codec().build(kind, memory_fields(kind) if fields is None else fields, metadata=metadata)


def memory_snapshot(value: MemoryDocument) -> VersionedSnapshot[MemoryDocument]:
    """为已校验文档构造完整旧记忆快照。"""

    from habitus.memory.uri import MemoryURI

    return VersionedSnapshot(
        identity=str(MemoryURI.from_address(value.address)),
        state=SnapshotState.FOUND,
        value=value,
        revision=value.metadata.revision,
        source_digest="0" * 64,
        size_bytes=len(value.markdown_body.encode()),
    )


def snapshot_batch(*documents: MemoryDocument) -> SnapshotBatch[MemoryDocument]:
    snapshots = tuple(sorted((memory_snapshot(item) for item in documents), key=lambda item: item.identity))
    return SnapshotBatch(snapshots, sum(item.size_bytes for item in snapshots))
