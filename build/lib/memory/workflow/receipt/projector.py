"""把 Editor 计划和已提交事务确定性投影为变更回执。"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from foundation.integrity import text_digest
from memory.document import MemoryDocument, MemoryDocumentCodec, MemoryStoredLink
from memory.editor.engine import MemoryEditorPlan
from memory.editor.identity import MemoryNodeDisposition
from memory.editor.mutation import MemoryMutationAction
from memory.editor.transaction import memory_logical_content_digest
from memory.editor.transaction_log import (
    MemoryTransactionJournalEntry,
    MemoryTransactionJournalRecord,
    MemoryTransactionJournalState,
)
from memory.uri import MemoryURI
from memory.workflow.receipt.model import (
    MemoryChangeReceipt,
    MemoryChangeReceiptError,
    MemoryChangeReceiptState,
    MemoryChangeSource,
    MemoryIdentityChange,
    MemoryNodeChange,
    MemoryNodeChangeAction,
    MemoryPreparedNodeChange,
)


class MemoryChangeReceiptProjector:
    """只计算回执内容，不读取或写入回执文件。"""

    def __init__(self, codec: MemoryDocumentCodec) -> None:
        if not isinstance(codec, MemoryDocumentCodec):
            raise TypeError("codec must be MemoryDocumentCodec")
        self.codec = codec

    def prepare(
        self,
        source: MemoryChangeSource,
        plan: MemoryEditorPlan,
        *,
        timestamp: datetime,
    ) -> MemoryChangeReceipt:
        if not isinstance(source, MemoryChangeSource):
            raise TypeError("source must be MemoryChangeSource")
        if not isinstance(plan, MemoryEditorPlan):
            raise TypeError("plan must be MemoryEditorPlan")
        mutations = plan.commit.mutation_plan.mutations
        created = {str(item.uri) for item in mutations if item.action is MemoryMutationAction.CREATE}
        deleted = {str(uri) for uri in plan.commit.retired_uris}
        changed = {str(uri) for uri in plan.commit.changed_uris}
        updated = changed - created - deleted
        unchanged = {
            str(item.uri)
            for item in mutations
            if item.action is MemoryMutationAction.NOOP and str(item.uri) not in updated
        }
        proposals = (
            {}
            if plan.extraction is None
            else {proposal.source_page_id: proposal for proposal in plan.extraction.candidates.identity_proposals}
        )
        identity_changes: list[MemoryIdentityChange] = []
        for entry in plan.identities.entries:
            if entry.disposition not in {MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE}:
                continue
            proposal = proposals.get(entry.page_id)
            if proposal is None or entry.source_uri is None:
                raise MemoryChangeReceiptError("final identity change is missing its validated proposal")
            identity_changes.append(
                MemoryIdentityChange(
                    action=entry.disposition,
                    source_uri=entry.source_uri,
                    target_uri=entry.final_uri,
                    basis=proposal.basis,
                )
            )
        prepared_node_changes = [
            MemoryPreparedNodeChange(
                action=(MemoryNodeChangeAction.UPDATE if logical.before.exists else MemoryNodeChangeAction.CREATE),
                uri=logical.uri,
                before_digest=(self._document_digest(logical.before.value) if logical.before.exists else None),
                expected_after_digest=logical.content_digest,
                confirms_intention=logical.confirms_intention,
            )
            for logical in plan.commit.logical_writes()
        ]
        for uri in plan.commit.retired_uris:
            snapshot = plan.commit.read_set.get(str(uri))
            if snapshot is None or not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
                raise MemoryChangeReceiptError("retired memory is missing its prepared before document")
            prepared_node_changes.append(
                MemoryPreparedNodeChange(
                    action=MemoryNodeChangeAction.DELETE,
                    uri=uri,
                    before_digest=self._document_digest(snapshot.value),
                    expected_after_digest=None,
                )
            )
        return MemoryChangeReceipt(
            source=source,
            state=MemoryChangeReceiptState.PREPARED,
            prepared_at=MemoryChangeReceipt._timestamp(timestamp, "prepared_at"),
            committed_at=None,
            expected_created_uris=self._sorted_uris(created),
            expected_updated_uris=self._sorted_uris(updated),
            expected_deleted_uris=self._sorted_uris(deleted),
            unchanged_uris=self._sorted_uris(unchanged),
            prepared_node_changes=tuple(sorted(prepared_node_changes, key=lambda item: str(item.uri))),
            identity_changes=tuple(sorted(identity_changes, key=lambda item: str(item.source_uri))),
            added_relations=plan.commit.relation_plan.added,
            removed_relations=plan.commit.relation_plan.removed,
        )

    def finalize(
        self,
        current: MemoryChangeReceipt,
        source: MemoryChangeSource,
        journal: MemoryTransactionJournalRecord,
    ) -> MemoryChangeReceipt:
        if not isinstance(current, MemoryChangeReceipt):
            raise TypeError("current must be MemoryChangeReceipt")
        if not isinstance(source, MemoryChangeSource):
            raise TypeError("source must be MemoryChangeSource")
        if not isinstance(journal, MemoryTransactionJournalRecord):
            raise TypeError("journal must be MemoryTransactionJournalRecord")
        if not current.source.matches_lookup(source) or journal.transaction_id != source.transaction_id:
            raise MemoryChangeReceiptError("transaction journal does not match the receipt source")
        if journal.state is not MemoryTransactionJournalState.COMMITTED:
            raise MemoryChangeReceiptError("only a COMMITTED transaction can finalize a change receipt")
        prepared_by_uri = {str(change.uri): change for change in current.prepared_node_changes}
        actual_prepared: list[MemoryPreparedNodeChange] = []
        node_changes: list[MemoryNodeChange] = []
        for entry in journal.entries:
            expected = prepared_by_uri.get(str(entry.uri))
            if expected is None:
                raise MemoryChangeReceiptError("transaction journal contains an unprepared node change")
            actual_prepared.append(self._prepared_change(entry, expected))
            node_changes.append(self._node_change(entry))
        if tuple(actual_prepared) != current.prepared_node_changes:
            raise MemoryChangeReceiptError("committed node content differs from the prepared receipt")
        self._verify_relations(current, journal)
        committed = MemoryChangeReceipt(
            source=current.source,
            state=MemoryChangeReceiptState.COMMITTED,
            prepared_at=current.prepared_at,
            committed_at=journal.updated_at,
            expected_created_uris=current.expected_created_uris,
            expected_updated_uris=current.expected_updated_uris,
            expected_deleted_uris=current.expected_deleted_uris,
            unchanged_uris=current.unchanged_uris,
            prepared_node_changes=current.prepared_node_changes,
            identity_changes=current.identity_changes,
            added_relations=current.added_relations,
            removed_relations=current.removed_relations,
            node_changes=tuple(node_changes),
        )
        if current.state is MemoryChangeReceiptState.COMMITTED:
            if current != committed:
                raise MemoryChangeReceiptError("committed change receipt conflicts with its transaction journal")
            return current
        return committed

    def _prepared_change(
        self,
        entry: MemoryTransactionJournalEntry,
        expected: MemoryPreparedNodeChange,
    ) -> MemoryPreparedNodeChange:
        before = entry.before
        after = entry.after
        if before is None and after is not None:
            action = MemoryNodeChangeAction.CREATE
        elif before is not None and after is not None:
            action = MemoryNodeChangeAction.UPDATE
        elif before is not None:
            action = MemoryNodeChangeAction.DELETE
        else:
            raise MemoryChangeReceiptError("transaction journal contains an empty change")
        self._verify_confirmation(before, after, expected)
        return MemoryPreparedNodeChange(
            action=action,
            uri=entry.uri,
            before_digest=self._document_digest(before),
            expected_after_digest=(
                None
                if after is None
                else memory_logical_content_digest(
                    uri=entry.uri,
                    kind=after.kind,
                    fields=after.fields,
                    links=after.links,
                    backlinks=after.backlinks,
                    confirms_intention=expected.confirms_intention,
                )
            ),
            confirms_intention=expected.confirms_intention,
        )

    @staticmethod
    def _verify_confirmation(
        before: MemoryDocument | None,
        after: MemoryDocument | None,
        expected: MemoryPreparedNodeChange,
    ) -> None:
        if after is None:
            if expected.confirms_intention:
                raise MemoryChangeReceiptError("deleted node cannot refresh Intention confirmation")
            return
        if expected.confirms_intention:
            if after.metadata.last_confirmed_at != after.metadata.updated_at:
                raise MemoryChangeReceiptError("committed Intention did not apply the prepared confirmation")
            return
        if before is None:
            if after.metadata.last_confirmed_at is not None:
                raise MemoryChangeReceiptError("unprepared Intention confirmation appeared during commit")
            return
        if after.metadata.last_confirmed_at != before.metadata.last_confirmed_at:
            raise MemoryChangeReceiptError("Intention confirmation changed outside the prepared plan")

    def _node_change(self, entry: MemoryTransactionJournalEntry) -> MemoryNodeChange:
        if not isinstance(entry, MemoryTransactionJournalEntry):
            raise TypeError("journal entry must be MemoryTransactionJournalEntry")
        before = entry.before
        after = entry.after
        if before is None and after is not None:
            action = MemoryNodeChangeAction.CREATE
        elif before is not None and after is not None:
            action = MemoryNodeChangeAction.UPDATE
        elif before is not None:
            action = MemoryNodeChangeAction.DELETE
        else:
            raise MemoryChangeReceiptError("transaction journal contains an empty change")
        return MemoryNodeChange(
            action=action,
            uri=entry.uri,
            before_revision=before.metadata.revision if before else None,
            after_revision=after.metadata.revision if after else None,
            before_digest=self._document_digest(before),
            after_digest=self._document_digest(after),
        )

    def _verify_relations(
        self,
        receipt: MemoryChangeReceipt,
        journal: MemoryTransactionJournalRecord,
    ) -> None:
        before = self._forward_links(entry.before for entry in journal.entries)
        after = self._forward_links(entry.after for entry in journal.entries)
        actual_added = set(after) - set(before)
        actual_removed = set(before) - set(after)
        if actual_added != {link.identity for link in receipt.added_relations}:
            raise MemoryChangeReceiptError("committed added relations differ from the prepared receipt")
        if actual_removed != {link.identity for link in receipt.removed_relations}:
            raise MemoryChangeReceiptError("committed removed relations differ from the prepared receipt")

    @staticmethod
    def _forward_links(
        documents: Iterable[MemoryDocument | None],
    ) -> dict[tuple[str, str, str], MemoryStoredLink]:
        result: dict[tuple[str, str, str], MemoryStoredLink] = {}
        for document in documents:
            if document is None:
                continue
            if not isinstance(document, MemoryDocument):
                raise TypeError("journal relation source must be MemoryDocument or None")
            for link in document.links:
                result[link.identity] = link
        return result

    def _document_digest(self, document: MemoryDocument | None) -> str | None:
        return text_digest(self.codec.encode(document)) if document is not None else None

    @staticmethod
    def same_change_intent(
        left: MemoryChangeReceipt,
        right: MemoryChangeReceipt,
    ) -> bool:
        """忽略状态和时间，只比较提交前由同一确定性计划产生的变更意图。"""

        return (
            left.source == right.source
            and left.prepared_node_changes == right.prepared_node_changes
            and left.expected_created_uris == right.expected_created_uris
            and left.expected_updated_uris == right.expected_updated_uris
            and left.expected_deleted_uris == right.expected_deleted_uris
            and left.unchanged_uris == right.unchanged_uris
            and left.identity_changes == right.identity_changes
            and left.added_relations == right.added_relations
            and left.removed_relations == right.removed_relations
        )

    @staticmethod
    def _sorted_uris(identities: set[str]) -> tuple[MemoryURI, ...]:
        return tuple(MemoryURI.parse(identity) for identity in sorted(identities))


__all__ = ["MemoryChangeReceiptProjector"]
