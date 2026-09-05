"""记忆变更回执的严格耐久 Schema。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum

from foundation.ids import same_path_identity
from memory.conversation import ConversationLayout
from memory.document import MemoryStoredLink
from memory.editor.candidate import MemoryIdentityProposalBasis
from memory.editor.identity import MemoryNodeDisposition
from memory.model import MemoryKind
from memory.uri import MemoryURI
from memory.workflow.jobs import MemoryJob
from pre.conversation import ConversationSegment
from pre.conversation.messages.model import require_sha256


class MemoryChangeReceiptError(RuntimeError):
    """变更回执与来源、计划或已提交事务不一致。"""


class MemoryChangeReceiptState(str, Enum):
    """回执只允许处于提交前准备态或已验证终态。"""

    PREPARED = "prepared"
    COMMITTED = "committed"


class MemoryNodeChangeAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True)
class MemoryChangeSource:
    """由 MemoryJob 和不可变 Segment 共同绑定的可信来源。"""

    memory_sequence: int
    transaction_id: str
    conversation_id: str
    started_on: date
    segment_id: str
    source_segment_digest: str
    editor_segment_id: str | None = None
    editor_segment_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.memory_sequence, bool)
            or not isinstance(self.memory_sequence, int)
            or self.memory_sequence <= 0
        ):
            raise ValueError("memory_sequence must be a positive integer")
        if not isinstance(self.transaction_id, str) or not self._hex(self.transaction_id, 32):
            raise ValueError("transaction_id must be 32 lowercase hexadecimal characters")
        for name in ("conversation_id", "segment_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty normalized text")
        if isinstance(self.started_on, datetime) or not isinstance(self.started_on, date):
            raise ValueError("started_on must be a calendar date")
        try:
            digest = require_sha256(self.source_segment_digest, "source_segment_digest")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        object.__setattr__(self, "source_segment_digest", digest)
        if (self.editor_segment_id is None) != (self.editor_segment_digest is None):
            raise ValueError("editor segment id and digest must be present together")
        if self.editor_segment_id is not None:
            ConversationLayout.segment_range(self.editor_segment_id)
            editor_segment_digest = self.editor_segment_digest
            assert isinstance(editor_segment_digest, str)
            try:
                editor_digest = require_sha256(
                    editor_segment_digest,
                    "editor_segment_digest",
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            object.__setattr__(self, "editor_segment_digest", editor_digest)

    @classmethod
    def from_job(
        cls,
        job: MemoryJob,
        *,
        editor_segment: ConversationSegment | None = None,
    ) -> MemoryChangeSource:
        """从可信耐久 Job 确定性生成一一对应的变更来源。"""

        if not isinstance(job, MemoryJob):
            raise TypeError("job must be MemoryJob")
        if editor_segment is not None:
            if not isinstance(editor_segment, ConversationSegment):
                raise TypeError("editor_segment must be ConversationSegment or None")
            if not same_path_identity(
                editor_segment.conversation_id,
                job.conversation_id,
                "conversation_id",
            ):
                raise ValueError("editor segment belongs to another conversation")
            trigger_start, trigger_end = ConversationLayout.segment_range(job.segment_id)
            editor_start, editor_end = ConversationLayout.segment_range(editor_segment.segment_id)
            if editor_start > trigger_start or editor_end != trigger_end:
                raise ValueError("editor segment does not end at the triggering segment")
        return cls(
            memory_sequence=job.memory_sequence,
            transaction_id=job.transaction_id,
            conversation_id=job.conversation_id,
            started_on=job.started_on,
            segment_id=job.segment_id,
            source_segment_digest=job.source_segment_digest,
            editor_segment_id=(None if editor_segment is None else editor_segment.segment_id),
            editor_segment_digest=(None if editor_segment is None else editor_segment.digest),
        )

    @property
    def receipt_id(self) -> str:
        return hashlib.sha256(
            (
                f"{self.conversation_id}\0{self.started_on.isoformat()}\0"
                f"{self.segment_id}\0{self.source_segment_digest}"
            ).encode()
        ).hexdigest()

    def require_segment(self, segment: ConversationSegment) -> None:
        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be ConversationSegment")
        if (
            not same_path_identity(
                segment.conversation_id,
                self.conversation_id,
                "conversation_id",
            )
            or segment.segment_id != self.segment_id
            or segment.digest != self.source_segment_digest
        ):
            raise MemoryChangeReceiptError("change source does not match its ConversationSegment")

    def same_trigger(self, other: MemoryChangeSource) -> bool:
        if not isinstance(other, MemoryChangeSource):
            return False
        return (
            self.memory_sequence == other.memory_sequence
            and self.transaction_id == other.transaction_id
            and same_path_identity(
                self.conversation_id,
                other.conversation_id,
                "conversation_id",
            )
            and self.started_on == other.started_on
            and self.segment_id == other.segment_id
            and self.source_segment_digest == other.source_segment_digest
        )

    def matches_lookup(self, lookup: MemoryChangeSource) -> bool:
        return self.same_trigger(lookup) and (lookup.editor_segment_id is None or self == lookup)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "memory_sequence": self.memory_sequence,
            "transaction_id": self.transaction_id,
            "conversation_id": self.conversation_id,
            "started_on": self.started_on.isoformat(),
            "segment_id": self.segment_id,
            "source_segment_digest": self.source_segment_digest,
        }
        if self.editor_segment_id is not None:
            result["editor_segment_id"] = self.editor_segment_id
            result["editor_segment_digest"] = self.editor_segment_digest
        return result

    @classmethod
    def from_dict(cls, value: object) -> MemoryChangeSource:
        expected = {
            "memory_sequence",
            "transaction_id",
            "conversation_id",
            "started_on",
            "segment_id",
            "source_segment_digest",
        }
        if not isinstance(value, Mapping):
            raise ValueError("memory change source has an invalid shape")
        fields = frozenset(value)
        if fields not in {
            frozenset(expected),
            frozenset({*expected, "editor_segment_id", "editor_segment_digest"}),
        }:
            raise ValueError("memory change source has an invalid shape")
        started_on = value["started_on"]
        if not isinstance(started_on, str):
            raise ValueError("memory change source started_on must be text")
        return cls(
            memory_sequence=value["memory_sequence"],
            transaction_id=value["transaction_id"],
            conversation_id=value["conversation_id"],
            started_on=date.fromisoformat(started_on),
            segment_id=value["segment_id"],
            source_segment_digest=value["source_segment_digest"],
            editor_segment_id=value.get("editor_segment_id"),
            editor_segment_digest=value.get("editor_segment_digest"),
        )

    @staticmethod
    def _hex(value: str, length: int) -> bool:
        if len(value) != length or value != value.lower():
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class MemoryIdentityChange:
    """MERGE 或 DELETE 的最终身份处置和受控依据。"""

    action: MemoryNodeDisposition
    source_uri: MemoryURI
    target_uri: MemoryURI | None
    basis: MemoryIdentityProposalBasis

    def __post_init__(self) -> None:
        action = MemoryNodeDisposition(self.action)
        basis = MemoryIdentityProposalBasis(self.basis)
        if action not in {MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE}:
            raise ValueError("identity receipt changes support only merge or delete")
        if not isinstance(self.source_uri, MemoryURI):
            raise TypeError("identity change source_uri must be MemoryURI")
        self.source_uri.to_address()
        if action is MemoryNodeDisposition.MERGE:
            if not isinstance(self.target_uri, MemoryURI):
                raise ValueError("merge identity change requires target_uri")
            self.target_uri.to_address()
            if basis is not MemoryIdentityProposalBasis.DUPLICATE_IDENTITY:
                raise ValueError("merge identity change requires duplicate_identity")
        elif self.target_uri is not None:
            raise ValueError("delete identity change cannot contain target_uri")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "basis", basis)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "source_uri": str(self.source_uri),
            "target_uri": str(self.target_uri) if self.target_uri is not None else None,
            "basis": self.basis.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryIdentityChange:
        expected = {"action", "source_uri", "target_uri", "basis"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("memory identity change has an invalid shape")
        source_uri = value["source_uri"]
        target_uri = value["target_uri"]
        if not isinstance(source_uri, str) or not isinstance(target_uri, str | type(None)):
            raise ValueError("memory identity change URIs are invalid")
        return cls(
            action=MemoryNodeDisposition(value["action"]),
            source_uri=MemoryURI.parse(source_uri),
            target_uri=MemoryURI.parse(target_uri) if target_uri is not None else None,
            basis=MemoryIdentityProposalBasis(value["basis"]),
        )


@dataclass(frozen=True)
class MemoryPreparedNodeChange:
    """准备态绑定的旧文档摘要与无时间最终内容摘要。"""

    action: MemoryNodeChangeAction
    uri: MemoryURI
    before_digest: str | None
    expected_after_digest: str | None
    confirms_intention: bool = False

    def __post_init__(self) -> None:
        action = MemoryNodeChangeAction(self.action)
        if not isinstance(self.uri, MemoryURI):
            raise TypeError("prepared node change uri must be MemoryURI")
        address = self.uri.to_address()
        for name in ("before_digest", "expected_after_digest"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        if not isinstance(self.confirms_intention, bool):
            raise TypeError("confirms_intention must be boolean")
        if self.confirms_intention and address.kind is not MemoryKind.INTENTION:
            raise ValueError("only an Intention prepared change can refresh confirmation time")
        if action is MemoryNodeChangeAction.CREATE:
            if self.before_digest is not None:
                raise ValueError("prepared create cannot contain before state")
            if self.expected_after_digest is None:
                raise ValueError("prepared create requires expected after content")
            if address.kind is MemoryKind.INTENTION and not self.confirms_intention:
                raise ValueError("prepared Intention create must establish confirmation time")
        elif action is MemoryNodeChangeAction.UPDATE:
            if self.before_digest is None or self.expected_after_digest is None:
                raise ValueError("prepared update requires before and expected after content")
        else:
            if self.before_digest is None:
                raise ValueError("prepared delete requires before state")
            if self.expected_after_digest is not None:
                raise ValueError("prepared delete cannot contain expected after content")
            if self.confirms_intention:
                raise ValueError("prepared delete cannot refresh Intention confirmation")
        object.__setattr__(self, "action", action)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "uri": str(self.uri),
            "before_digest": self.before_digest,
            "expected_after_digest": self.expected_after_digest,
            "confirms_intention": self.confirms_intention,
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryPreparedNodeChange:
        expected = {
            "action",
            "uri",
            "before_digest",
            "expected_after_digest",
            "confirms_intention",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared memory node change has an invalid shape")
        uri = value["uri"]
        if not isinstance(uri, str):
            raise ValueError("prepared memory node change URI must be text")
        return cls(
            action=MemoryNodeChangeAction(value["action"]),
            uri=MemoryURI.parse(uri),
            before_digest=value["before_digest"],
            expected_after_digest=value["expected_after_digest"],
            confirms_intention=value["confirms_intention"],
        )


@dataclass(frozen=True)
class MemoryNodeChange:
    """一个实际提交节点的前后 revision 和物理摘要。"""

    action: MemoryNodeChangeAction
    uri: MemoryURI
    before_revision: int | None
    after_revision: int | None
    before_digest: str | None
    after_digest: str | None

    def __post_init__(self) -> None:
        action = MemoryNodeChangeAction(self.action)
        if not isinstance(self.uri, MemoryURI):
            raise TypeError("node change uri must be MemoryURI")
        self.uri.to_address()
        for name in ("before_revision", "after_revision"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be null or a positive integer")
        for name in ("before_digest", "after_digest"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        if action is MemoryNodeChangeAction.CREATE:
            if self.before_revision is not None or self.before_digest is not None:
                raise ValueError("create node change cannot contain before state")
            if self.after_revision is None or self.after_digest is None:
                raise ValueError("create node change requires after state")
        elif action is MemoryNodeChangeAction.UPDATE:
            if None in {
                self.before_revision,
                self.after_revision,
                self.before_digest,
                self.after_digest,
            }:
                raise ValueError("update node change requires before and after states")
        else:
            if self.before_revision is None or self.before_digest is None:
                raise ValueError("delete node change requires before state")
            if self.after_revision is not None or self.after_digest is not None:
                raise ValueError("delete node change cannot contain after state")
        object.__setattr__(self, "action", action)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "uri": str(self.uri),
            "before_revision": self.before_revision,
            "after_revision": self.after_revision,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryNodeChange:
        expected = {
            "action",
            "uri",
            "before_revision",
            "after_revision",
            "before_digest",
            "after_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("memory node change has an invalid shape")
        uri = value["uri"]
        if not isinstance(uri, str):
            raise ValueError("memory node change URI must be text")
        return cls(
            action=MemoryNodeChangeAction(value["action"]),
            uri=MemoryURI.parse(uri),
            before_revision=value["before_revision"],
            after_revision=value["after_revision"],
            before_digest=value["before_digest"],
            after_digest=value["after_digest"],
        )


@dataclass(frozen=True)
class MemoryChangeReceipt:
    """与一个 MemoryJob 一一对应的准备态或最终态变更回执。"""

    source: MemoryChangeSource
    state: MemoryChangeReceiptState
    prepared_at: datetime
    committed_at: datetime | None
    expected_created_uris: tuple[MemoryURI, ...]
    expected_updated_uris: tuple[MemoryURI, ...]
    expected_deleted_uris: tuple[MemoryURI, ...]
    unchanged_uris: tuple[MemoryURI, ...]
    prepared_node_changes: tuple[MemoryPreparedNodeChange, ...]
    identity_changes: tuple[MemoryIdentityChange, ...]
    added_relations: tuple[MemoryStoredLink, ...]
    removed_relations: tuple[MemoryStoredLink, ...]
    node_changes: tuple[MemoryNodeChange, ...] = ()

    SCHEMA_VERSION = "memory_change_receipt_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.source, MemoryChangeSource):
            raise TypeError("receipt source must be MemoryChangeSource")
        state = MemoryChangeReceiptState(self.state)
        prepared_at = self._timestamp(self.prepared_at, "prepared_at")
        committed_at = self._timestamp(self.committed_at, "committed_at") if self.committed_at is not None else None
        if state is MemoryChangeReceiptState.PREPARED:
            if committed_at is not None or self.node_changes:
                raise ValueError("prepared receipt cannot contain committed output")
        elif committed_at is None or committed_at < prepared_at:
            raise ValueError("committed receipt requires a valid committed_at")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "prepared_at", prepared_at)
        object.__setattr__(self, "committed_at", committed_at)
        for name in (
            "expected_created_uris",
            "expected_updated_uris",
            "expected_deleted_uris",
            "unchanged_uris",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(uri, MemoryURI) for uri in values):
                raise TypeError(f"{name} must contain MemoryURI values")
            identities = tuple(str(uri) for uri in values)
            if identities != tuple(sorted(set(identities))):
                raise ValueError(f"{name} must be unique and sorted")
        expected_sets = [
            {str(uri) for uri in getattr(self, name)}
            for name in (
                "expected_created_uris",
                "expected_updated_uris",
                "expected_deleted_uris",
            )
        ]
        if any(left & right for index, left in enumerate(expected_sets) for right in expected_sets[index + 1 :]):
            raise ValueError("receipt expected change URI sets must be disjoint")
        for name, item_type, key in (
            ("prepared_node_changes", MemoryPreparedNodeChange, lambda item: str(item.uri)),
            ("identity_changes", MemoryIdentityChange, lambda item: str(item.source_uri)),
            ("added_relations", MemoryStoredLink, lambda item: item.identity),
            ("removed_relations", MemoryStoredLink, lambda item: item.identity),
            ("node_changes", MemoryNodeChange, lambda item: str(item.uri)),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, item_type) for item in values):
                raise TypeError(f"{name} contains invalid values")
            keys = tuple(key(item) for item in values)
            if keys != tuple(sorted(set(keys))):
                raise ValueError(f"{name} must be unique and sorted")
        if {link.identity for link in self.added_relations} & {link.identity for link in self.removed_relations}:
            raise ValueError("the same relation cannot be added and removed")
        prepared_actual = {
            action: {str(change.uri) for change in self.prepared_node_changes if change.action is action}
            for action in MemoryNodeChangeAction
        }
        expected = {
            MemoryNodeChangeAction.CREATE: {str(uri) for uri in self.expected_created_uris},
            MemoryNodeChangeAction.UPDATE: {str(uri) for uri in self.expected_updated_uris},
            MemoryNodeChangeAction.DELETE: {str(uri) for uri in self.expected_deleted_uris},
        }
        if prepared_actual != expected:
            raise ValueError("prepared node changes do not match the expected change sets")
        if state is MemoryChangeReceiptState.COMMITTED:
            actual = {
                action: {str(change.uri) for change in self.node_changes if change.action is action}
                for action in MemoryNodeChangeAction
            }
            if actual != expected:
                raise ValueError("committed node changes do not match the prepared change sets")

    @property
    def changed_uris(self) -> tuple[MemoryURI, ...]:
        return tuple(change.uri for change in self.node_changes)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source": self.source.to_dict(),
            "state": self.state.value,
            "prepared_at": self._format_time(self.prepared_at),
            "committed_at": self._format_time(self.committed_at) if self.committed_at else None,
            "expected_created_uris": [str(uri) for uri in self.expected_created_uris],
            "expected_updated_uris": [str(uri) for uri in self.expected_updated_uris],
            "expected_deleted_uris": [str(uri) for uri in self.expected_deleted_uris],
            "unchanged_uris": [str(uri) for uri in self.unchanged_uris],
            "prepared_node_changes": [change.to_dict() for change in self.prepared_node_changes],
            "identity_changes": [change.to_dict() for change in self.identity_changes],
            "added_relations": [relation.to_dict() for relation in self.added_relations],
            "removed_relations": [relation.to_dict() for relation in self.removed_relations],
            "node_changes": [change.to_dict() for change in self.node_changes],
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryChangeReceipt:
        expected = {
            "schema_version",
            "source",
            "state",
            "prepared_at",
            "committed_at",
            "expected_created_uris",
            "expected_updated_uris",
            "expected_deleted_uris",
            "unchanged_uris",
            "prepared_node_changes",
            "identity_changes",
            "added_relations",
            "removed_relations",
            "node_changes",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("memory change receipt has an invalid shape")
        if value["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("memory change receipt has an unsupported schema")
        for name in (
            "expected_created_uris",
            "expected_updated_uris",
            "expected_deleted_uris",
            "unchanged_uris",
            "prepared_node_changes",
            "identity_changes",
            "added_relations",
            "removed_relations",
            "node_changes",
        ):
            if not isinstance(value[name], list):
                raise ValueError(f"memory change receipt {name} must be an array")
        committed_at = value["committed_at"]
        if committed_at is not None and not isinstance(committed_at, str):
            raise ValueError("memory change receipt committed_at must be text or null")
        return cls(
            source=MemoryChangeSource.from_dict(value["source"]),
            state=MemoryChangeReceiptState(value["state"]),
            prepared_at=cls._parse_time(value["prepared_at"], "prepared_at"),
            committed_at=cls._parse_time(committed_at, "committed_at") if committed_at else None,
            expected_created_uris=cls._uris(value["expected_created_uris"]),
            expected_updated_uris=cls._uris(value["expected_updated_uris"]),
            expected_deleted_uris=cls._uris(value["expected_deleted_uris"]),
            unchanged_uris=cls._uris(value["unchanged_uris"]),
            prepared_node_changes=tuple(
                MemoryPreparedNodeChange.from_dict(item) for item in value["prepared_node_changes"]
            ),
            identity_changes=tuple(MemoryIdentityChange.from_dict(item) for item in value["identity_changes"]),
            added_relations=tuple(MemoryStoredLink.from_dict(item) for item in value["added_relations"]),
            removed_relations=tuple(MemoryStoredLink.from_dict(item) for item in value["removed_relations"]),
            node_changes=tuple(MemoryNodeChange.from_dict(item) for item in value["node_changes"]),
        )

    @staticmethod
    def _uris(values: object) -> tuple[MemoryURI, ...]:
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("receipt URI list is invalid")
        return tuple(MemoryURI.parse(value) for value in values)

    @staticmethod
    def _timestamp(value: datetime, label: str) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{label} must be timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _format_time(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_time(value: object, label: str) -> datetime:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be text")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{label} must include timezone")
        return parsed.astimezone(UTC)


__all__ = [
    "MemoryChangeReceipt",
    "MemoryChangeReceiptError",
    "MemoryChangeReceiptState",
    "MemoryChangeSource",
    "MemoryIdentityChange",
    "MemoryNodeChange",
    "MemoryNodeChangeAction",
    "MemoryPreparedNodeChange",
]
