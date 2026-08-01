"""统一提交节点内容、删除和双向关系的可恢复事务。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from foundation.integrity import canonical_digest
from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from infrastructure.store.contracts import PathLock
from memory.document import MemoryDocument, MemoryDocumentMetadata, MemoryStoredLink
from memory.editor.identity import (
    MemoryFinalIdentityMap,
    MemoryNodeDisposition,
)
from memory.editor.link_plan import MemoryRelationPlan
from memory.editor.lock_key import MemoryDocumentLockKeyspace
from memory.editor.mutation.model import (
    MemoryMutationAction,
    MemoryMutationPlan,
)
from memory.editor.transaction_log import (
    MemoryTransactionJournal,
    MemoryTransactionJournalEntry,
    MemoryTransactionJournalError,
    MemoryTransactionJournalRecord,
    MemoryTransactionJournalState,
)
from memory.model import MemoryKind
from memory.snapshot import MemorySnapshotBatch, MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI


class MemoryCommitError(RuntimeError):
    """统一记忆事务未能完整提交并验证。"""


class MemoryCommitConflictError(MemoryCommitError):
    """语义计划依赖的任一完整快照已经变化。"""


class MemoryCommitRollbackError(MemoryCommitError):
    """发布失败后无法恢复全部旧文档。"""


class MemoryCommitRecoveryError(MemoryCommitError):
    """崩溃恢复遇到事务外写入或无法重建旧状态。"""


class MemoryCommitStatus(str, Enum):
    """统一记忆事务的结果状态。"""

    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class MemoryCommitConfig:
    """统一事务的显式锁租约。"""

    lock_ttl_seconds: int = 30

    def __post_init__(self) -> None:
        if (
            isinstance(self.lock_ttl_seconds, bool)
            or not isinstance(self.lock_ttl_seconds, int)
            or not 1 <= self.lock_ttl_seconds <= 3_600
        ):
            raise ValueError("lock_ttl_seconds must be between 1 and 3600")


def memory_logical_content_digest(
    *,
    uri: MemoryURI,
    kind: MemoryKind,
    fields: Mapping[str, Any],
    links: tuple[MemoryStoredLink, ...],
    backlinks: tuple[MemoryStoredLink, ...],
    confirms_intention: bool,
) -> str:
    """计算不受提交时间和 revision 影响的最终节点内容摘要。"""

    if not isinstance(uri, MemoryURI):
        raise TypeError("logical content uri must be a MemoryURI")
    uri.to_address()
    resolved_kind = MemoryKind(kind)
    if not isinstance(fields, Mapping) or any(not isinstance(name, str) for name in fields):
        raise TypeError("logical content fields must be a mapping with string keys")
    for label, values in {"links": links, "backlinks": backlinks}.items():
        if not isinstance(values, tuple) or any(not isinstance(value, MemoryStoredLink) for value in values):
            raise TypeError(f"logical content {label} must contain MemoryStoredLink values")
    if not isinstance(confirms_intention, bool):
        raise TypeError("confirms_intention must be boolean")
    if confirms_intention and resolved_kind is not MemoryKind.INTENTION:
        raise ValueError("only Intention logical content can refresh confirmation time")
    return canonical_digest(
        {
            "uri": str(uri),
            "memory_type": resolved_kind.value,
            "fields": fields,
            "links": [link.to_dict() for link in links],
            "backlinks": [link.to_dict() for link in backlinks],
            "confirms_intention": confirms_intention,
        }
    )


@dataclass(frozen=True)
class MemoryCommitLogicalWrite:
    """提交前确定的最终业务内容，不包含系统生成的时间与 revision。"""

    before: VersionedSnapshot[MemoryDocument]
    uri: MemoryURI
    kind: MemoryKind
    fields: Mapping[str, Any]
    links: tuple[MemoryStoredLink, ...]
    backlinks: tuple[MemoryStoredLink, ...]
    confirms_intention: bool

    def __post_init__(self) -> None:
        if not isinstance(self.before, VersionedSnapshot):
            raise TypeError("logical write before must be a VersionedSnapshot")
        if not isinstance(self.uri, MemoryURI):
            raise TypeError("logical write uri must be a MemoryURI")
        if self.before.identity != str(self.uri):
            raise ValueError("logical write snapshot identity does not match its URI")
        kind = MemoryKind(self.kind)
        if not isinstance(self.fields, Mapping) or any(not isinstance(name, str) for name in self.fields):
            raise TypeError("logical write fields must be a mapping with string keys")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        for label, values in {"links": self.links, "backlinks": self.backlinks}.items():
            if not isinstance(values, tuple) or any(not isinstance(value, MemoryStoredLink) for value in values):
                raise TypeError(f"logical write {label} must contain MemoryStoredLink values")
            identities = tuple(value.identity for value in values)
            if identities != tuple(sorted(set(identities))):
                raise ValueError(f"logical write {label} must be unique and sorted")
        if any(link.from_uri != self.uri for link in self.links):
            raise ValueError("logical write forward link has the wrong source URI")
        if any(link.to_uri != self.uri for link in self.backlinks):
            raise ValueError("logical write backlink has the wrong target URI")
        if not isinstance(self.confirms_intention, bool):
            raise TypeError("confirms_intention must be boolean")
        if self.confirms_intention and kind is not MemoryKind.INTENTION:
            raise ValueError("only an Intention logical write can refresh confirmation time")
        if self.before.exists:
            if not isinstance(self.before.value, MemoryDocument):
                raise TypeError("existing logical write snapshot must contain a MemoryDocument")
            if self.before.value.kind is not kind or MemoryURI.from_address(self.before.value.address) != self.uri:
                raise ValueError("logical write cannot change an existing node identity or kind")

    @property
    def content_digest(self) -> str:
        return memory_logical_content_digest(
            uri=self.uri,
            kind=self.kind,
            fields=self.fields,
            links=self.links,
            backlinks=self.backlinks,
            confirms_intention=self.confirms_intention,
        )


@dataclass(frozen=True)
class MemoryCommitPlan:
    """内容、最终身份与关系计划在发布前形成的统一纯计划。"""

    mutation_plan: MemoryMutationPlan
    identities: MemoryFinalIdentityMap
    relation_plan: MemoryRelationPlan
    read_set: MemorySnapshotBatch

    def __post_init__(self) -> None:
        if not isinstance(self.mutation_plan, MemoryMutationPlan):
            raise TypeError("mutation_plan must be a MemoryMutationPlan")
        if not isinstance(self.identities, MemoryFinalIdentityMap):
            raise TypeError("identities must be a MemoryFinalIdentityMap")
        if not isinstance(self.relation_plan, MemoryRelationPlan):
            raise TypeError("relation_plan must be a MemoryRelationPlan")
        if not isinstance(self.read_set, SnapshotBatch):
            raise TypeError("read_set must be a MemorySnapshotBatch")
        self._validate_mutation_identities()
        self._validate_snapshot_coverage()

    @classmethod
    def build(
        cls,
        mutation_plan: MemoryMutationPlan,
        identities: MemoryFinalIdentityMap,
        relation_plan: MemoryRelationPlan,
    ) -> MemoryCommitPlan:
        """合并所有语义读取、精确目标和一跳关系快照。"""

        batches = (
            mutation_plan.read_set.old_memories,
            mutation_plan.read_set.target_memories,
            relation_plan.read_set.snapshots,
        )
        snapshots: dict[str, VersionedSnapshot[MemoryDocument]] = {}
        for batch in batches:
            for snapshot in batch.snapshots:
                previous = snapshots.get(snapshot.identity)
                if previous is not None and previous != snapshot:
                    raise MemoryCommitConflictError(f"commit plan contains conflicting snapshots: {snapshot.identity}")
                snapshots[snapshot.identity] = snapshot
        ordered = tuple(snapshots[key] for key in sorted(snapshots))
        return cls(
            mutation_plan=mutation_plan,
            identities=identities,
            relation_plan=relation_plan,
            read_set=SnapshotBatch(
                snapshots=ordered,
                total_bytes=sum(snapshot.size_bytes for snapshot in ordered),
            ),
        )

    @property
    def retired_uris(self) -> tuple[MemoryURI, ...]:
        return self.identities.retired_uris

    @property
    def changed_uris(self) -> tuple[MemoryURI, ...]:
        mutation_uris = {str(mutation.uri) for mutation in self.mutation_plan.changed_mutations}
        relation_uris = {str(update.uri) for update in self.relation_plan.updates}
        retired = {str(uri) for uri in self.retired_uris}
        return tuple(MemoryURI.parse(identity) for identity in sorted(mutation_uris | relation_uris | retired))

    def logical_writes(self) -> tuple[MemoryCommitLogicalWrite, ...]:
        """从统一计划生成 Receipt 与事务共同使用的最终业务内容。"""

        mutations = {str(mutation.uri): mutation for mutation in self.mutation_plan.mutations}
        relation_updates = {str(update.uri): update for update in self.relation_plan.updates}
        targets = {str(mutation.uri) for mutation in self.mutation_plan.changed_mutations} | set(
            relation_updates
        )
        retired = {str(uri) for uri in self.retired_uris}
        if targets & retired:
            raise MemoryCommitError("one URI cannot be written and retired in the same commit")

        writes: list[MemoryCommitLogicalWrite] = []
        for identity in sorted(targets):
            before = self.read_set.get(identity)
            if before is None:
                raise MemoryCommitError("write target is missing from the unified read set")
            mutation = mutations.get(identity)
            relation_update = relation_updates.get(identity)
            if mutation is not None:
                kind = mutation.match.candidate.kind
                fields = mutation.fields
            else:
                if not before.exists or not isinstance(before.value, MemoryDocument):
                    raise MemoryCommitError("relation-only write requires an existing memory document")
                kind = before.value.kind
                fields = before.value.fields

            if relation_update is not None:
                links = relation_update.links
                backlinks = relation_update.backlinks
            elif before.exists:
                assert isinstance(before.value, MemoryDocument)
                links = before.value.links
                backlinks = before.value.backlinks
            else:
                links = ()
                backlinks = ()

            if not before.exists and (mutation is None or mutation.action is not MemoryMutationAction.CREATE):
                raise MemoryCommitError("missing write target requires a CREATE mutation")
            confirms_intention = kind is MemoryKind.INTENTION and (
                not before.exists or (mutation is not None and mutation.confirms_intention)
            )
            writes.append(
                MemoryCommitLogicalWrite(
                    before=before,
                    uri=MemoryURI.parse(identity),
                    kind=kind,
                    fields=fields,
                    links=links,
                    backlinks=backlinks,
                    confirms_intention=confirms_intention,
                )
            )
        return tuple(writes)

    def _validate_mutation_identities(self) -> None:
        expected_disposition = {
            MemoryMutationAction.CREATE: MemoryNodeDisposition.CREATE,
            MemoryMutationAction.UPDATE: MemoryNodeDisposition.UPDATE,
            MemoryMutationAction.NOOP: MemoryNodeDisposition.NOOP,
        }
        retired = {str(uri) for uri in self.identities.retired_uris}
        mutation_targets = {str(mutation.uri) for mutation in self.mutation_plan.mutations}
        old_memories = self.mutation_plan.read_set.old_memories
        final_uris = {str(entry.final_uri) for entry in self.identities.entries if entry.final_uri is not None}
        if retired & final_uris:
            raise ValueError("merge/delete sources must be flattened before final identity publication")
        for entry in self.identities.entries:
            if entry.source_uri is not None:
                source = old_memories.get(str(entry.source_uri))
                if source is None or not source.exists:
                    raise ValueError("final identity source requires a complete extracted old snapshot")
            if (
                entry.disposition is MemoryNodeDisposition.MERGE
                and entry.final_uri is not None
                and str(entry.final_uri) not in mutation_targets
            ):
                raise ValueError("merge target must be covered by deterministic field planning")
        for mutation in self.mutation_plan.mutations:
            entry = self.identities.entry(mutation.match.candidate.page_id)
            if entry.disposition is not expected_disposition[mutation.action]:
                raise ValueError("mutation action does not match its final identity disposition")
            if entry.final_uri != mutation.uri:
                raise ValueError("mutation URI does not match its final identity")
            if str(mutation.uri) in retired:
                raise ValueError("retired node cannot also receive a content mutation")

    def _validate_snapshot_coverage(self) -> None:
        for uri in self.changed_uris:
            snapshot = self.read_set.get(str(uri))
            if snapshot is None:
                raise ValueError(f"commit read set does not contain changed URI: {uri}")
        creates = {
            str(mutation.uri)
            for mutation in self.mutation_plan.mutations
            if mutation.action is MemoryMutationAction.CREATE
        }
        for update in self.relation_plan.updates:
            current = self.read_set.get(str(update.uri))
            if current != update.before:
                raise ValueError("relation update snapshot does not match unified commit read set")
            if not update.before.exists and str(update.uri) not in creates:
                raise ValueError("relation update cannot create a node without a content CREATE mutation")
        for uri in self.retired_uris:
            snapshot = self.read_set.get(str(uri))
            if snapshot is None or not snapshot.exists:
                raise ValueError("retired node requires a complete old snapshot")


@dataclass(frozen=True)
class MemoryCommitWrite:
    """统一事务对一个 URI 的唯一最终文档写入。"""

    before: VersionedSnapshot[MemoryDocument]
    after: MemoryDocument

    def __post_init__(self) -> None:
        if not isinstance(self.before, VersionedSnapshot):
            raise TypeError("before must be a VersionedSnapshot")
        if not isinstance(self.after, MemoryDocument):
            raise TypeError("after must be a MemoryDocument")
        if self.before.identity != str(MemoryURI.from_address(self.after.address)):
            raise ValueError("commit write before/after identities differ")
        if self.before.exists:
            assert isinstance(self.before.value, MemoryDocument)
            if self.after.metadata.revision != self.before.value.metadata.revision + 1:
                raise ValueError("existing commit write must advance one revision")
            if self.after.metadata.created_at != self.before.value.metadata.created_at:
                raise ValueError("existing commit write cannot change created_at")
            before_confirmation = self.before.value.metadata.last_confirmed_at
            after_confirmation = self.after.metadata.last_confirmed_at
            if after_confirmation != before_confirmation and after_confirmation != self.after.metadata.updated_at:
                raise ValueError("existing Intention confirmation must refresh to the commit timestamp")
        else:
            if self.after.metadata.revision != 1:
                raise ValueError("new commit write must start at revision 1")
            if self.after.metadata.created_at != self.after.metadata.updated_at:
                raise ValueError("new commit write timestamps must start equal")
            if (
                self.after.kind is MemoryKind.INTENTION
                and self.after.metadata.last_confirmed_at != self.after.metadata.created_at
            ):
                raise ValueError("new Intention confirmation must start at its creation timestamp")

    @property
    def uri(self) -> MemoryURI:
        return MemoryURI.from_address(self.after.address)


@dataclass(frozen=True)
class MemoryCommitResult:
    """统一事务最终发布的节点与关系变化。"""

    status: MemoryCommitStatus
    transaction_id: str | None
    created_uris: tuple[MemoryURI, ...]
    updated_uris: tuple[MemoryURI, ...]
    deleted_uris: tuple[MemoryURI, ...]
    unchanged_uris: tuple[MemoryURI, ...]
    added_relations: tuple[MemoryStoredLink, ...]
    removed_relations: tuple[MemoryStoredLink, ...]
    journal_cleaned: bool

    def __post_init__(self) -> None:
        status = MemoryCommitStatus(self.status)
        object.__setattr__(self, "status", status)
        for name in (
            "created_uris",
            "updated_uris",
            "deleted_uris",
            "unchanged_uris",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(uri, MemoryURI) for uri in values):
                raise TypeError(f"{name} must contain MemoryURI values")
            identities = tuple(str(uri) for uri in values)
            if identities != tuple(sorted(set(identities))):
                raise ValueError(f"{name} must be unique and sorted")
        for name in ("added_relations", "removed_relations"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(relation, MemoryStoredLink) for relation in values):
                raise TypeError(f"{name} must contain MemoryStoredLink values")
        if not isinstance(self.journal_cleaned, bool):
            raise TypeError("journal_cleaned must be boolean")


class MemoryCommitTransaction:
    """用一个锁集、一个 CAS 和一个恢复日志发布全部记忆变化。"""

    def __init__(
        self,
        tree: MemoryTree,
        snapshot_reader: MemorySnapshotReader,
        path_lock: PathLock,
        journal: MemoryTransactionJournal,
        *,
        config: MemoryCommitConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        transaction_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be a MemoryTree")
        if not isinstance(snapshot_reader, MemorySnapshotReader):
            raise TypeError("snapshot_reader must be a MemorySnapshotReader")
        if snapshot_reader.tree.root != tree.root:
            raise ValueError("snapshot_reader and tree must use the same memory root")
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be a PathLock")
        if not isinstance(journal, MemoryTransactionJournal):
            raise TypeError("journal must be a MemoryTransactionJournal")
        if journal.codec is not tree.document_codec:
            raise ValueError("journal and memory tree must share one document codec")
        canonical_journal_root = (
            tree.root.parent / "workflow" / "transactions"
        ).resolve(strict=False)
        if journal.root != canonical_journal_root:
            raise ValueError(
                "transaction journal must use the canonical sibling workflow/transactions root"
            )
        try:
            journal.root.relative_to(tree.root)
        except ValueError:
            pass
        else:
            raise ValueError("transaction journal must be stored outside the L2 memory tree")
        if config is not None and not isinstance(config, MemoryCommitConfig):
            raise TypeError("config must be a MemoryCommitConfig")
        self.tree = tree
        self.snapshot_reader = snapshot_reader
        self.path_lock = path_lock
        self.journal = journal
        self.config = config or MemoryCommitConfig()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.transaction_id_factory = transaction_id_factory or (lambda: uuid4().hex)
        self.lock_keys = MemoryDocumentLockKeyspace(tree.root)
        self.tree.bind_visibility_journal(journal)
        self.snapshot_reader.bind_visibility_journal(journal)

    def commit(
        self,
        plan: MemoryCommitPlan,
        *,
        transaction_id: str | None = None,
        retain_journal: bool = False,
    ) -> MemoryCommitResult:
        """统一 CAS、记录 PREPARED、发布、回读并标记 COMMITTED。"""

        if not isinstance(plan, MemoryCommitPlan):
            raise TypeError("plan must be a MemoryCommitPlan")
        if not isinstance(retain_journal, bool):
            raise TypeError("retain_journal must be boolean")
        if not plan.changed_uris and not (retain_journal and transaction_id is not None):
            return self._result(plan, transaction_id=None, journal_cleaned=True)

        timestamp = self._timestamp()
        writes = self._build_writes(plan, timestamp=timestamp)
        resolved_transaction_id = transaction_id or self.transaction_id_factory()
        identities = tuple(snapshot.identity for snapshot in plan.read_set.snapshots)
        publication_attempted = False
        documents_verified = False
        with ExitStack() as stack:
            guards = tuple(
                stack.enter_context(
                    self.path_lock.acquire(
                        key,
                        ttl_seconds=self.config.lock_ttl_seconds,
                    )
                )
                for key in (
                    self.lock_keys.transaction_key(),
                    *(self.lock_keys.key(identity) for identity in identities),
                )
            )
            with self.path_lock.fenced(guards):
                if self.journal.pending():
                    raise MemoryCommitRecoveryError(
                        "pending memory transaction must be recovered before a new commit"
                    )
                self._assert_current(plan.read_set)
                record = self._journal_record(
                    plan,
                    writes,
                    transaction_id=resolved_transaction_id,
                    timestamp=timestamp,
                )
                self.journal.prepare(record)
                try:
                    for write in writes:
                        publication_attempted = True
                        self.tree.write(write.after)
                    for uri in plan.retired_uris:
                        publication_attempted = True
                        deleted = self.tree.delete(uri.to_address())
                        if not deleted:
                            raise MemoryCommitError(f"retired memory disappeared during publication: {uri}")
                    self._verify_committed(plan, writes)
                    documents_verified = True
                    try:
                        self.journal.advance_visibility_generation()
                        self.journal.mark(
                            resolved_transaction_id,
                            MemoryTransactionJournalState.COMMITTED,
                            timestamp=self._timestamp(),
                        )
                    except Exception as exc:
                        raise MemoryCommitRecoveryError(
                            "memory documents are complete but COMMITTED journal state is indeterminate"
                        ) from exc
                except Exception as exc:
                    if documents_verified:
                        raise
                    if publication_attempted:
                        self._rollback(plan)
                        try:
                            self.journal.advance_visibility_generation()
                            self.journal.mark(
                                resolved_transaction_id,
                                MemoryTransactionJournalState.ROLLED_BACK,
                                timestamp=self._timestamp(),
                            )
                        except Exception:
                            pass
                    raise MemoryCommitError("unified memory commit failed and was rolled back") from exc

        journal_cleaned = False if retain_journal else self._discard_terminal(resolved_transaction_id)
        return self._result(
            plan,
            transaction_id=resolved_transaction_id,
            journal_cleaned=journal_cleaned,
        )

    def recover_pending(self, *, discard_terminal: bool = True) -> tuple[str, ...]:
        """在处理新 MemoryJob 前恢复所有 PREPARED 事务。"""

        if not isinstance(discard_terminal, bool):
            raise TypeError("discard_terminal must be boolean")
        recovered: list[str] = []
        with self.path_lock.acquire(
            self.lock_keys.transaction_key(),
            ttl_seconds=self.config.lock_ttl_seconds,
        ) as transaction_guard:
            for record in self.journal.pending():
                with ExitStack() as stack:
                    guards = tuple(
                        stack.enter_context(
                            self.path_lock.acquire(
                                self.lock_keys.key(identity),
                                ttl_seconds=self.config.lock_ttl_seconds,
                            )
                        )
                        for identity in record.lock_identities
                    )
                    with self.path_lock.fenced((transaction_guard, *guards)):
                        states = [self._entry_state(entry) for entry in record.entries]
                        if all(state == "after" for state in states):
                            terminal = MemoryTransactionJournalState.COMMITTED
                        elif all(state in {"before", "after"} for state in states):
                            self._restore_entries(record.entries)
                            if any(self._entry_state(entry) != "before" for entry in record.entries):
                                raise MemoryCommitRecoveryError("recovered transaction failed rollback verification")
                            terminal = MemoryTransactionJournalState.ROLLED_BACK
                        else:
                            raise MemoryCommitRecoveryError(
                                "prepared transaction overlaps an unknown later document state"
                            )
                        self.journal.advance_visibility_generation()
                        self.journal.mark(
                            record.transaction_id,
                            terminal,
                            timestamp=self._timestamp(),
                        )
                if discard_terminal:
                    self._discard_terminal(record.transaction_id)
                recovered.append(record.transaction_id)
        return tuple(recovered)

    def _build_writes(
        self,
        plan: MemoryCommitPlan,
        *,
        timestamp: datetime,
    ) -> tuple[MemoryCommitWrite, ...]:
        writes: list[MemoryCommitWrite] = []
        for logical in plan.logical_writes():
            if logical.before.exists:
                assert isinstance(logical.before.value, MemoryDocument)
                metadata = logical.before.value.metadata.next_revision(
                    timestamp,
                    refresh_confirmation=logical.confirms_intention,
                )
            else:
                metadata = MemoryDocumentMetadata.initial(
                    timestamp,
                    confirmed=logical.kind is MemoryKind.INTENTION,
                )
            after = self.tree.document_codec.build(
                logical.kind,
                logical.fields,
                metadata=metadata,
                links=logical.links,
                backlinks=logical.backlinks,
            )
            writes.append(MemoryCommitWrite(before=logical.before, after=after))
        return tuple(writes)

    def _journal_record(
        self,
        plan: MemoryCommitPlan,
        writes: tuple[MemoryCommitWrite, ...],
        *,
        transaction_id: str,
        timestamp: datetime,
    ) -> MemoryTransactionJournalRecord:
        after_by_uri = {str(write.uri): write.after for write in writes}
        changed = sorted(str(uri) for uri in plan.changed_uris)
        entries: list[MemoryTransactionJournalEntry] = []
        for identity in changed:
            snapshot = plan.read_set.get(identity)
            if snapshot is None:
                raise MemoryCommitError("journal target is missing from the read set")
            before = snapshot.value if snapshot.exists else None
            if before is not None and not isinstance(before, MemoryDocument):
                raise MemoryCommitError("journal snapshot contains an invalid document")
            entries.append(
                MemoryTransactionJournalEntry(
                    uri=MemoryURI.parse(identity),
                    before=before,
                    after=after_by_uri.get(identity),
                )
            )
        return MemoryTransactionJournalRecord(
            transaction_id=transaction_id,
            state=MemoryTransactionJournalState.PREPARED,
            created_at=timestamp,
            updated_at=timestamp,
            lock_identities=tuple(snapshot.identity for snapshot in plan.read_set.snapshots),
            entries=tuple(entries),
        )

    def _assert_current(self, expected: MemorySnapshotBatch) -> None:
        current = self.snapshot_reader._read_many_physical(
            snapshot.identity for snapshot in expected.snapshots
        )
        for expected_snapshot in expected.snapshots:
            current_snapshot = current.get(expected_snapshot.identity)
            if current_snapshot is None or not self._same_snapshot(
                current_snapshot,
                expected_snapshot,
            ):
                raise MemoryCommitConflictError(f"memory read set changed before commit: {expected_snapshot.identity}")

    def _verify_committed(
        self,
        plan: MemoryCommitPlan,
        writes: tuple[MemoryCommitWrite, ...],
    ) -> None:
        current = self.snapshot_reader._read_many_physical(
            snapshot.identity for snapshot in plan.read_set.snapshots
        )
        expected_after = {str(write.uri): write.after for write in writes}
        retired = {str(uri) for uri in plan.retired_uris}
        for original in plan.read_set.snapshots:
            snapshot = current.get(original.identity)
            if snapshot is None:
                raise MemoryCommitError("commit read-back omitted a locked URI")
            if original.identity in retired:
                if snapshot.exists:
                    raise MemoryCommitError("retired memory still exists after commit")
                continue
            after = expected_after.get(original.identity)
            if after is not None:
                if not snapshot.exists or snapshot.value != after:
                    raise MemoryCommitError("written memory failed full read-back validation")
                continue
            if not self._same_snapshot(snapshot, original):
                raise MemoryCommitError("unchanged memory changed during unified commit")

    def _rollback(
        self,
        plan: MemoryCommitPlan,
    ) -> None:
        """在调用方已持有的统一 fencing 临界区内恢复完整旧读集。"""

        try:
            for uri in reversed(plan.changed_uris):
                snapshot = plan.read_set.get(str(uri))
                assert snapshot is not None
                if snapshot.exists:
                    if not isinstance(snapshot.value, MemoryDocument):
                        raise MemoryCommitRollbackError("rollback snapshot contains an invalid memory document")
                    self.tree.write(snapshot.value)
                else:
                    self.tree.delete(uri.to_address())
            self._assert_current(plan.read_set)
        except Exception as exc:
            if isinstance(exc, MemoryCommitRollbackError):
                raise
            raise MemoryCommitRollbackError("unified memory rollback could not restore the original read set") from exc

    def _restore_entries(
        self,
        entries: tuple[MemoryTransactionJournalEntry, ...],
    ) -> None:
        for entry in reversed(entries):
            if entry.before is None:
                self.tree.delete(entry.uri.to_address())
            else:
                self.tree.write(entry.before)

    def _entry_state(self, entry: MemoryTransactionJournalEntry) -> str:
        snapshot = self.snapshot_reader._read_physical(entry.uri)
        if self._matches_document(snapshot, entry.after):
            return "after"
        if self._matches_document(snapshot, entry.before):
            return "before"
        return "other"

    @staticmethod
    def _matches_document(
        snapshot: VersionedSnapshot[MemoryDocument],
        document: MemoryDocument | None,
    ) -> bool:
        if document is None:
            return not snapshot.exists
        return snapshot.exists and snapshot.value == document

    @staticmethod
    def _same_snapshot(
        current: VersionedSnapshot[MemoryDocument],
        expected: VersionedSnapshot[MemoryDocument],
    ) -> bool:
        return (
            current.state is expected.state
            and current.revision == expected.revision
            and current.source_digest == expected.source_digest
        )

    def _result(
        self,
        plan: MemoryCommitPlan,
        *,
        transaction_id: str | None,
        journal_cleaned: bool,
    ) -> MemoryCommitResult:
        created = {
            str(mutation.uri): mutation.uri
            for mutation in plan.mutation_plan.mutations
            if mutation.action is MemoryMutationAction.CREATE
        }
        changed = {str(uri) for uri in plan.changed_uris}
        deleted = {str(uri) for uri in plan.retired_uris}
        updated = changed - set(created) - deleted
        unchanged = {
            str(mutation.uri): mutation.uri
            for mutation in plan.mutation_plan.mutations
            if mutation.action is MemoryMutationAction.NOOP and str(mutation.uri) not in updated
        }
        return MemoryCommitResult(
            status=(MemoryCommitStatus.UPDATED if changed else MemoryCommitStatus.UNCHANGED),
            transaction_id=transaction_id,
            created_uris=tuple(created[key] for key in sorted(created)),
            updated_uris=tuple(MemoryURI.parse(key) for key in sorted(updated)),
            deleted_uris=tuple(MemoryURI.parse(key) for key in sorted(deleted)),
            unchanged_uris=tuple(unchanged[key] for key in sorted(unchanged)),
            added_relations=plan.relation_plan.added,
            removed_relations=plan.relation_plan.removed,
            journal_cleaned=journal_cleaned,
        )

    def _discard_terminal(self, transaction_id: str) -> bool:
        try:
            self.journal.discard_terminal(transaction_id)
        except MemoryTransactionJournalError:
            return False
        return True

    def _timestamp(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("memory commit clock must return a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory commit clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


__all__ = [
    "MemoryCommitConfig",
    "MemoryCommitConflictError",
    "MemoryCommitError",
    "MemoryCommitLogicalWrite",
    "MemoryCommitPlan",
    "MemoryCommitRecoveryError",
    "MemoryCommitResult",
    "MemoryCommitRollbackError",
    "MemoryCommitStatus",
    "MemoryCommitTransaction",
    "MemoryCommitWrite",
    "memory_logical_content_digest",
]
