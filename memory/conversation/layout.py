"""Conversation 原始消息文件的确定性、安全布局。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from foundation.ids import require_safe_path_segment
from pre.conversation import ConversationRangeSummaryStage

_SEGMENT_ID = re.compile(r"^(?P<start>[0-9]{12})-(?P<end>[0-9]{12})$")
_MAX_SEGMENT_SEQUENCE = 999_999_999_999


class ConversationLayoutError(ValueError):
    """Conversation 地址或路径不满足确定性布局。"""


@dataclass(frozen=True)
class ConversationAddress:
    """由可信运行时提供的稳定会话位置。"""

    conversation_id: str
    started_on: date

    def __post_init__(self) -> None:
        try:
            identifier = require_safe_path_segment(self.conversation_id, "conversation_id")
        except ValueError as exc:
            raise ConversationLayoutError(str(exc)) from exc
        if identifier != identifier.strip() or any(ord(character) < 32 for character in identifier):
            raise ConversationLayoutError("conversation_id contains unsafe characters")
        if isinstance(self.started_on, datetime) or not isinstance(self.started_on, date):
            raise ConversationLayoutError("started_on must be a calendar date")
        object.__setattr__(self, "conversation_id", identifier)


class ConversationLayout:
    """只计算路径和锁键，不执行文件操作。"""

    def __init__(self, root: str | Path) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise ConversationLayoutError("conversation root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        self._root_key = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:24]

    def messages_root(self) -> Path:
        """返回按日期组织 Conversation 消息目录的可信根。"""

        return self._inside_root(self.root / "messages")

    def conversation_directory(self, address: ConversationAddress) -> Path:
        resolved = self._address(address)
        path = (
            self.messages_root()
            / f"{resolved.started_on.year:04d}"
            / f"{resolved.started_on.month:02d}"
            / f"{resolved.started_on.day:02d}"
            / resolved.conversation_id
        )
        return self._inside_root(path)

    def live_path(self, address: ConversationAddress) -> Path:
        return self._inside_root(self.conversation_directory(address) / "live.jsonl")

    def state_path(self, address: ConversationAddress) -> Path:
        """返回不依赖 History 保留情况的 Conversation 耐久游标路径。"""

        return self._inside_root(self.conversation_directory(address) / "state.json")

    def history_directory(self, address: ConversationAddress) -> Path:
        return self._inside_root(self.conversation_directory(address) / "history")

    def history_path(self, address: ConversationAddress, segment_id: str) -> Path:
        self.segment_range(segment_id)
        return self._inside_root(self.history_directory(address) / f"{segment_id}.jsonl")

    def summary_directory(self, address: ConversationAddress) -> Path:
        resolved = self._address(address)
        path = (
            self.root
            / "summaries"
            / f"{resolved.started_on.year:04d}"
            / f"{resolved.started_on.month:02d}"
            / f"{resolved.started_on.day:02d}"
            / resolved.conversation_id
        )
        return self._inside_root(path)

    def summary_path(self, address: ConversationAddress, segment_id: str) -> Path:
        """返回与不可变 history 片段一一对应的摘要路径。"""

        self.segment_range(segment_id)
        return self._inside_root(self.summary_directory(address) / f"{segment_id}.json")

    def range_summary_directory(
        self,
        address: ConversationAddress,
        stage: ConversationRangeSummaryStage,
    ) -> Path:
        """返回普通范围摘要或归档范围摘要的独立不可变目录。"""

        if not isinstance(stage, ConversationRangeSummaryStage):
            raise TypeError("range summary stage must be ConversationRangeSummaryStage")
        name = "ranges" if stage is ConversationRangeSummaryStage.RANGE else "archive_ranges"
        return self._inside_root(self.summary_directory(address) / name)

    def range_summary_path(
        self,
        address: ConversationAddress,
        stage: ConversationRangeSummaryStage,
        range_id: str,
    ) -> Path:
        self.segment_range(range_id)
        return self._inside_root(self.range_summary_directory(address, stage) / f"{range_id}.json")

    def lock_key(self, address: ConversationAddress) -> str:
        resolved = self._address(address)
        return f"conversation:{self._root_key}:{resolved.started_on.isoformat()}:{resolved.conversation_id}"

    @staticmethod
    def segment_id(start_sequence: int, end_sequence: int) -> str:
        for label, value in (
            ("start_sequence", start_sequence),
            ("end_sequence", end_sequence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConversationLayoutError(f"{label} must be a non-negative integer")
            if value > _MAX_SEGMENT_SEQUENCE:
                raise ConversationLayoutError(f"{label} exceeds the segment identifier range")
        if start_sequence > end_sequence:
            raise ConversationLayoutError("segment sequence range is invalid")
        return f"{start_sequence:012d}-{end_sequence:012d}"

    @staticmethod
    def segment_range(segment_id: str) -> tuple[int, int]:
        if not isinstance(segment_id, str):
            raise ConversationLayoutError("segment_id must be text")
        match = _SEGMENT_ID.fullmatch(segment_id)
        if match is None:
            raise ConversationLayoutError("segment_id must use 12-digit start-end sequences")
        start_sequence = int(match.group("start"))
        end_sequence = int(match.group("end"))
        if start_sequence > end_sequence:
            raise ConversationLayoutError("segment_id sequence range is invalid")
        return start_sequence, end_sequence

    @staticmethod
    def _address(address: ConversationAddress) -> ConversationAddress:
        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be a ConversationAddress")
        return address

    def _inside_root(self, path: Path) -> Path:
        candidate = path.absolute()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ConversationLayoutError("conversation path escapes its root") from exc
        return candidate


__all__ = ["ConversationAddress", "ConversationLayout", "ConversationLayoutError"]
