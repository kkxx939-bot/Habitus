"""MemoryCandidateBatch 的来源上下文校验：绑定原始会话与完整旧记忆，逐条列出静态 JSON Schema 表达不了的违规。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import tzinfo
from typing import TYPE_CHECKING

from habitus.memory.document import MemoryStoredLink
from habitus.memory.editor.candidate import (
    MemoryCandidate,
    MemoryCandidateError,
    MemoryCandidateViolation,
    MemoryIdentityProposalType,
    MemoryRelationAction,
)
from habitus.memory.editor.page_id import EXISTING_PAGE_ID_MAX, MemoryPageIdMap
from habitus.memory.model import MemoryKind
from habitus.memory.snapshot import MemorySnapshotBatch
from habitus.memory.uri import MemoryURI
from habitus.pre.conversation.messages import (
    ConversationMessageRole,
    ConversationSegment,
    ConversationToolResultStatus,
)

if TYPE_CHECKING:
    from habitus.memory.editor.candidate import MemoryCandidateBatch

_MISSING = object()


class MemoryCandidateContextValidator:
    """对一批候选执行来源校验；违规按六类检查的固定顺序逐条产出，文字与顺序即修补轮与剔除的依据。"""

    def __init__(self, batch: MemoryCandidateBatch) -> None:
        self.batch = batch

    def violations(
        self,
        segment: ConversationSegment,
        old_memories: MemorySnapshotBatch,
        page_ids: MemoryPageIdMap,
        *,
        zone: tzinfo,
    ) -> tuple[MemoryCandidateViolation, ...]:
        """逐条列出通不过来源校验的候选、关系与身份提议；日期按主体所在地 ``zone`` 的本地日算。

        page_id 表与本批完整旧快照不一致是系统自身的错，不落到哪一条上，直接明确失败。
        """

        expected_old_uris = {snapshot.identity for snapshot in old_memories.snapshots if snapshot.exists}
        mapped_old_uris = {uri for _page_id, uri in page_ids.existing_items()}
        if mapped_old_uris != expected_old_uris:
            raise MemoryCandidateError("page_id map does not match the complete old-memory snapshot batch")
        return (
            *self._page_reference_violations(page_ids),
            *self._identity_relation_violations(),
            *self._relation_removal_violations(old_memories, page_ids),
            *self._tool_source_violations(segment, old_memories),
            *self._event_violations(segment, zone),
            *self._completed_intention_violations(old_memories),
        )

    def _page_reference_violations(self, page_ids: MemoryPageIdMap) -> Iterator[MemoryCandidateViolation]:
        available_page_ids = set(page_ids.page_ids())
        candidate_page_ids: set[int] = set()
        for candidate in self.batch.iter_candidates():
            candidate_uri = str(MemoryURI.from_address(candidate.address))
            old_page_id = page_ids.page_id_for(candidate_uri)
            resolved_uri = page_ids.resolve(candidate.page_id)
            if candidate.page_id <= EXISTING_PAGE_ID_MAX:
                if resolved_uri is None or resolved_uri != candidate_uri:
                    yield MemoryCandidateViolation(
                        candidate, "existing memory candidate must reuse the page_id assigned to its exact URI"
                    )
            elif old_page_id is not None:
                yield MemoryCandidateViolation(
                    candidate, "candidate for an existing memory URI must reuse its existing page_id"
                )
            available_page_ids.add(candidate.page_id)
            candidate_page_ids.add(candidate.page_id)

        for relation in self.batch.relations:
            if relation.from_page_id not in available_page_ids:
                yield MemoryCandidateViolation(relation, "relation candidate references an unknown from_page_id")
            elif relation.to_page_id not in available_page_ids:
                yield MemoryCandidateViolation(relation, "relation candidate references an unknown to_page_id")

        for proposal in self.batch.identity_proposals:
            if proposal.source_page_id not in page_ids.page_ids():
                yield MemoryCandidateViolation(
                    proposal, "identity proposal source is not a complete old-memory page_id"
                )
            elif proposal.source_page_id in candidate_page_ids:
                yield MemoryCandidateViolation(
                    proposal, "identity proposal source cannot also appear as a memory candidate"
                )
            elif proposal.target_page_id is not None and proposal.target_page_id not in available_page_ids:
                yield MemoryCandidateViolation(proposal, "identity proposal references an unknown target_page_id")
            elif (
                proposal.proposal_type is MemoryIdentityProposalType.SAME_MEMORY
                and proposal.target_page_id not in candidate_page_ids
            ):
                yield MemoryCandidateViolation(
                    proposal, "same_memory target must also appear as a fully planned memory candidate"
                )

    def _identity_relation_violations(self) -> Iterator[MemoryCandidateViolation]:
        """待退休来源不能同时参与显式关系变更。"""

        retired_sources = {proposal.source_page_id for proposal in self.batch.identity_proposals}
        for relation in self.batch.relations:
            if relation.from_page_id in retired_sources or relation.to_page_id in retired_sources:
                yield MemoryCandidateViolation(
                    relation, "identity proposal source cannot also be referenced by a relation candidate"
                )

    def _relation_removal_violations(
        self,
        old_memories: MemorySnapshotBatch,
        page_ids: MemoryPageIdMap,
    ) -> Iterator[MemoryCandidateViolation]:
        """REMOVE 必须逐字命中已完整读取且双向一致的旧关系。"""

        for relation in self.batch.relations:
            if relation.action is not MemoryRelationAction.REMOVE:
                continue
            if relation.from_page_id > EXISTING_PAGE_ID_MAX or relation.to_page_id > EXISTING_PAGE_ID_MAX:
                yield MemoryCandidateViolation(relation, "relation remove must reference two existing memory page_ids")
                continue
            from_uri = page_ids.resolve(relation.from_page_id)
            to_uri = page_ids.resolve(relation.to_page_id)
            if from_uri is None or to_uri is None:
                yield MemoryCandidateViolation(relation, "relation remove references an unresolved existing page_id")
                continue
            stored = MemoryStoredLink(
                from_uri=MemoryURI.parse(from_uri),
                to_uri=MemoryURI.parse(to_uri),
                link_type=relation.link_type,
            )
            source = old_memories.get(str(stored.from_uri))
            target = old_memories.get(str(stored.to_uri))
            if (
                source is None
                or target is None
                or not source.exists
                or not target.exists
                or source.value is None
                or target.value is None
            ):
                yield MemoryCandidateViolation(
                    relation, "relation remove requires both complete old endpoint snapshots"
                )
            elif stored not in source.value.links or stored not in target.value.backlinks:
                yield MemoryCandidateViolation(
                    relation, "relation remove does not match a complete existing Link/Backlink pair"
                )

    def _tool_source_violations(
        self,
        segment: ConversationSegment,
        old_memories: MemorySnapshotBatch,
    ) -> Iterator[MemoryCandidateViolation]:
        """工具候选要有本段成功调用；本段新写的 failure_recovery 要有本段"失败后成功"。

        与同一旧工具记忆逐值相同、原样带过来的 failure_recovery 已有来源，不再要求本段证据。
        """

        calls = {
            message.tool_call_id: message
            for message in segment.messages
            if message.role is ConversationMessageRole.TOOL_CALL
        }
        successful_names: set[str] = set()
        failed_at: dict[str, list[int]] = {}
        succeeded_at: dict[str, list[int]] = {}
        for message in segment.messages:
            if message.role is not ConversationMessageRole.TOOL_RESULT:
                continue
            call = calls.get(message.tool_call_id)
            if call is None or call.tool_name != message.tool_name:
                continue
            assert message.tool_name is not None
            if message.tool_status is ConversationToolResultStatus.COMPLETED:
                successful_names.add(message.tool_name)
                succeeded_at.setdefault(message.tool_name, []).append(message.sequence)
            elif message.tool_status is ConversationToolResultStatus.ERROR:
                failed_at.setdefault(message.tool_name, []).append(message.sequence)

        for candidate in self.batch.tools:
            tool_name = candidate.fields.get("tool_name")
            if not isinstance(tool_name, str) or tool_name not in successful_names:
                yield MemoryCandidateViolation(
                    candidate, "tool candidate requires a matching successful tool_call/tool_result pair"
                )
                continue
            if "failure_recovery" not in candidate.fields:
                continue
            if candidate.fields["failure_recovery"] == _old_field_value(candidate, old_memories, "failure_recovery"):
                continue
            failures = failed_at.get(tool_name, [])
            successes = succeeded_at.get(tool_name, [])
            if not any(failure < success for failure in failures for success in successes):
                yield MemoryCandidateViolation(
                    candidate, "tool failure_recovery requires a failure followed by a successful result"
                )

    def _completed_intention_violations(self, old_memories: MemorySnapshotBatch) -> Iterator[MemoryCandidateViolation]:
        for candidate in self.batch.intentions:
            if candidate.fields.get("status") != "completed":
                continue
            if candidate.confirmed is not True:
                yield MemoryCandidateViolation(
                    candidate, "completed intention requires explicit confirmation in this conversation"
                )
                continue
            uri = str(MemoryURI.from_address(candidate.address))
            snapshot = old_memories.get(uri)
            if snapshot is None or not snapshot.exists or snapshot.value is None:
                yield MemoryCandidateViolation(
                    candidate, "completed intention must target an existing intention read in this batch"
                )
            elif snapshot.value is None:
                yield MemoryCandidateViolation(candidate, "completed intention snapshot is not a memory document")
            elif snapshot.value.kind is not MemoryKind.INTENTION:
                yield MemoryCandidateViolation(candidate, "completed intention snapshot has the wrong memory type")

    def _event_violations(self, segment: ConversationSegment, zone: tzinfo) -> Iterator[MemoryCandidateViolation]:
        """只校验模型语义结果中可确定验证的 Event 日期边界：不晚于片段最后一条消息在主体所在地的那一天。"""

        latest = segment.ended_at.astimezone(zone).date()
        for candidate in self.batch.events:
            event_date = candidate.address.event_date
            # Event 地址构造时已保证 event_date 存在（``MemoryAddress.__post_init__``），这里只为类型收窄。
            assert event_date is not None
            if event_date > latest:
                yield MemoryCandidateViolation(candidate, "future event date cannot represent an occurred Event")


def _old_field_value(candidate: MemoryCandidate, old_memories: MemorySnapshotBatch, field_name: str) -> object:
    """返回候选同一 URI 的完整旧记忆里该字段的值；没有旧节点或旧字段时返回缺失标记。"""

    snapshot = old_memories.get(str(MemoryURI.from_address(candidate.address)))
    if snapshot is None or not snapshot.exists or snapshot.value is None:
        return _MISSING
    return snapshot.value.fields.get(field_name, _MISSING)


__all__ = ["MemoryCandidateContextValidator"]
