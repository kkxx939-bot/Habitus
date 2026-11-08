"""由记忆 YAML 派生的严格结构化候选契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import tzinfo
from enum import Enum
from typing import Any, ClassVar

from habitus.foundation.integrity import canonicalize, immutable_snapshot
from habitus.memory.document import MemoryLinkType
from habitus.memory.editor.page_id import (
    EXISTING_PAGE_ID_MAX,
    MemoryPageIdError,
    MemoryPageIdMap,
    validate_page_id,
    validate_unique_page_ids,
)
from habitus.memory.model import MemoryAddress, MemoryKind
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.snapshot import MemorySnapshotBatch
from habitus.pre.conversation.messages import ConversationSegment


class MemoryCandidateError(ValueError):
    """模型候选不满足结构或来源上下文约束。"""


_KIND_TO_OUTPUT_FIELD = {
    MemoryKind.PROFILE: "profile",
    MemoryKind.PREFERENCE: "preferences",
    MemoryKind.ENTITY: "entities",
    MemoryKind.TOOL: "tools",
    MemoryKind.EVENT: "events",
    MemoryKind.INTENTION: "intentions",
}
_OUTPUT_FIELD_TO_KIND = {value: key for key, value in _KIND_TO_OUTPUT_FIELD.items()}
_MEMORY_OUTPUT_FIELDS = tuple(_OUTPUT_FIELD_TO_KIND)
_IDENTITY_OUTPUT_FIELD = "identity_proposals"
_RELATION_OUTPUT_FIELD = "relations"
_OUTPUT_FIELDS = (*_MEMORY_OUTPUT_FIELDS, _IDENTITY_OUTPUT_FIELD, _RELATION_OUTPUT_FIELD)


class MemoryIdentityProposalType(str, Enum):
    """模型只可提出同一记忆或整节点移除两类临时身份判断。"""

    SAME_MEMORY = "same_memory"
    REMOVE_MEMORY = "remove_memory"


class MemoryIdentityProposalBasis(str, Enum):
    """限制身份提议只能表达已经确认的语义依据。"""

    DUPLICATE_IDENTITY = "duplicate_identity"
    EXPLICIT_FORGET = "explicit_forget"
    FULLY_INVALIDATED = "fully_invalidated"


@dataclass(frozen=True)
class MemoryIdentityProposal:
    """使用临时 page_id 表达、等待确定性裁决的身份提议。"""

    proposal_type: MemoryIdentityProposalType
    source_page_id: int
    target_page_id: int | None
    basis: MemoryIdentityProposalBasis

    def __post_init__(self) -> None:
        try:
            proposal_type = MemoryIdentityProposalType(self.proposal_type)
            basis = MemoryIdentityProposalBasis(self.basis)
            source_page_id = validate_page_id(self.source_page_id)
        except (MemoryPageIdError, TypeError, ValueError) as exc:
            raise MemoryCandidateError("identity proposal contains an invalid type, basis or source_page_id") from exc
        if source_page_id > EXISTING_PAGE_ID_MAX:
            raise MemoryCandidateError("identity proposal source must be a fully read existing page_id")

        target_page_id = self.target_page_id
        if proposal_type is MemoryIdentityProposalType.SAME_MEMORY:
            if basis is not MemoryIdentityProposalBasis.DUPLICATE_IDENTITY:
                raise MemoryCandidateError("same_memory proposal requires duplicate_identity basis")
            try:
                target_page_id = validate_page_id(target_page_id)
            except MemoryPageIdError as exc:
                raise MemoryCandidateError("same_memory proposal requires a valid target_page_id") from exc
            if target_page_id == source_page_id:
                raise MemoryCandidateError("same_memory proposal source and target must differ")
        else:
            if basis not in {
                MemoryIdentityProposalBasis.EXPLICIT_FORGET,
                MemoryIdentityProposalBasis.FULLY_INVALIDATED,
            }:
                raise MemoryCandidateError("remove_memory proposal requires explicit_forget or fully_invalidated basis")
            if target_page_id is not None:
                raise MemoryCandidateError("remove_memory proposal cannot contain a target_page_id")

        object.__setattr__(self, "proposal_type", proposal_type)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "source_page_id", source_page_id)
        object.__setattr__(self, "target_page_id", target_page_id)

    def to_dict(self) -> dict[str, object]:
        """返回与结构化输出 Schema 一致的临时身份对象。"""

        return {
            "proposal_type": self.proposal_type.value,
            "source_page_id": self.source_page_id,
            "target_page_id": self.target_page_id,
            "basis": self.basis.value,
        }


class MemoryRelationAction(str, Enum):
    """Conversation 只可提出新增或移除一条独立关系。"""

    ADD = "add"
    REMOVE = "remove"


@dataclass(frozen=True)
class MemoryRelationCandidate:
    """使用临时页面编号表达、尚未落盘的关系变更候选。"""

    action: MemoryRelationAction
    from_page_id: int
    to_page_id: int
    link_type: MemoryLinkType

    def __post_init__(self) -> None:
        try:
            from_page_id = validate_page_id(self.from_page_id)
            to_page_id = validate_page_id(self.to_page_id)
        except MemoryPageIdError as exc:
            raise MemoryCandidateError(f"invalid relation candidate: {exc}") from exc
        if from_page_id == to_page_id:
            raise MemoryCandidateError("relation candidate cannot reference the same page twice")
        try:
            action = MemoryRelationAction(self.action)
        except ValueError as exc:
            raise MemoryCandidateError("relation candidate contains an unsupported action") from exc
        try:
            link_type = MemoryLinkType(self.link_type)
        except ValueError as exc:
            raise MemoryCandidateError("relation candidate contains an unsupported link_type") from exc
        if link_type.is_symmetric and to_page_id < from_page_id:
            from_page_id, to_page_id = to_page_id, from_page_id
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "from_page_id", from_page_id)
        object.__setattr__(self, "to_page_id", to_page_id)
        object.__setattr__(self, "link_type", link_type)

    @property
    def relation_identity(self) -> tuple[int, int, str]:
        """返回不包含动作的临时关系身份。"""

        return (self.from_page_id, self.to_page_id, self.link_type.value)

    def to_dict(self) -> dict[str, object]:
        """返回与结构化输出 Schema 一致的关系对象。"""

        return {
            "action": self.action.value,
            "from_page_id": self.from_page_id,
            "to_page_id": self.to_page_id,
            "link_type": self.link_type.value,
        }


@dataclass(frozen=True)
class MemoryCandidate:
    """一条携带临时编号、尚未决定存储操作的记忆候选。"""

    page_id: int
    kind: MemoryKind
    fields: Mapping[str, Any]
    confirmed: bool | None = None

    def __post_init__(self) -> None:
        try:
            page_id = validate_page_id(self.page_id)
        except MemoryPageIdError as exc:
            raise MemoryCandidateError(f"invalid memory candidate: {exc}") from exc
        kind = MemoryKind(self.kind)
        object.__setattr__(self, "page_id", page_id)
        object.__setattr__(self, "kind", kind)
        try:
            fields = MemorySchemaRegistry.load_default().validate(kind, self.fields)
        except (TypeError, ValueError) as exc:
            raise MemoryCandidateError(f"invalid {kind.value} candidate: {exc}") from exc
        if kind is MemoryKind.INTENTION:
            if self.confirmed is None:
                raise MemoryCandidateError("Intention candidate requires a boolean confirmed field")
        elif self.confirmed is not None:
            raise MemoryCandidateError("only Intention candidate accepts the confirmed control field")
        _reject_empty_present_strings(fields, _KIND_TO_OUTPUT_FIELD[kind])
        object.__setattr__(self, "fields", immutable_snapshot(fields))

    @property
    def address(self) -> MemoryAddress:
        """按已校验业务字段确定性生成目标地址。"""

        return MemorySchemaRegistry.load_default().address_for(self.kind, self.fields)

    def to_dict(self) -> dict[str, Any]:
        """返回临时编号和严格业务字段。"""

        result = {"page_id": self.page_id, **canonicalize(self.fields)}
        if self.kind is MemoryKind.INTENTION:
            result["confirmed"] = self.confirmed
        return result


@dataclass(frozen=True)
class MemoryCandidateViolation:
    """一条记忆候选、关系候选或身份提议没通过来源校验：是哪一条、违反了哪条规则。"""

    item: MemoryCandidate | MemoryRelationCandidate | MemoryIdentityProposal
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("violation reason must be non-empty text")

    def describe(self) -> str:
        """一行说清是哪一条、违反了什么，供修补轮请求与 Job 审计使用。"""

        item = self.item
        if isinstance(item, MemoryCandidate):
            label = f"{_KIND_TO_OUTPUT_FIELD[item.kind]} page_id={item.page_id}"
        elif isinstance(item, MemoryRelationCandidate):
            label = f"relations {item.action.value} {item.from_page_id}->{item.to_page_id} {item.link_type.value}"
        else:
            label = f"identity_proposals {item.proposal_type.value} source_page_id={item.source_page_id}"
        return f"{label}: {self.reason}"


def _violation_key(item: object) -> object:
    """同一批里认出同一条：记忆候选按批内唯一的 page_id，关系与身份提议按自身的值。"""

    if isinstance(item, MemoryCandidate):
        return ("candidate", item.page_id)
    return item


@dataclass(frozen=True)
class MemoryCandidateBatch:
    """一次模型解析返回的记忆、身份与关系候选。"""

    profile: tuple[MemoryCandidate, ...] = ()
    preferences: tuple[MemoryCandidate, ...] = ()
    entities: tuple[MemoryCandidate, ...] = ()
    tools: tuple[MemoryCandidate, ...] = ()
    events: tuple[MemoryCandidate, ...] = ()
    intentions: tuple[MemoryCandidate, ...] = ()
    identity_proposals: tuple[MemoryIdentityProposal, ...] = ()
    relations: tuple[MemoryRelationCandidate, ...] = ()

    _MEMORY_FIELD_NAMES: ClassVar[tuple[str, ...]] = _MEMORY_OUTPUT_FIELDS
    _FIELD_NAMES: ClassVar[tuple[str, ...]] = _OUTPUT_FIELDS

    def __post_init__(self) -> None:
        seen: set[MemoryAddress] = set()
        candidates: list[MemoryCandidate] = []
        for field_name in self._MEMORY_FIELD_NAMES:
            values = getattr(self, field_name)
            expected_kind = _OUTPUT_FIELD_TO_KIND[field_name]
            for candidate in values:
                if candidate.kind is not expected_kind:
                    raise MemoryCandidateError(f"{field_name} contains a candidate of the wrong type")
                if candidate.address in seen:
                    raise MemoryCandidateError("memory candidate batch contains a duplicate address")
                seen.add(candidate.address)
                candidates.append(candidate)
        try:
            validate_unique_page_ids(candidate.page_id for candidate in candidates)
        except MemoryPageIdError as exc:
            raise MemoryCandidateError(str(exc)) from exc
        if len(self.identity_proposals) != len(set(self.identity_proposals)):
            raise MemoryCandidateError("memory candidate batch contains a duplicate identity proposal")
        proposal_sources = tuple(item.source_page_id for item in self.identity_proposals)
        if len(proposal_sources) != len(set(proposal_sources)):
            raise MemoryCandidateError("one source page_id cannot have multiple identity proposals")
        proposal_targets = {
            item.target_page_id
            for item in self.identity_proposals
            if item.proposal_type is MemoryIdentityProposalType.SAME_MEMORY
        }
        if set(proposal_sources) & proposal_targets:
            raise MemoryCandidateError("identity proposals cannot form chains or cycles")
        if len(self.relations) != len(set(self.relations)):
            raise MemoryCandidateError("memory candidate batch contains a duplicate relation")
        actions_by_identity: dict[tuple[int, int, str], MemoryRelationAction] = {}
        for relation in self.relations:
            previous = actions_by_identity.setdefault(
                relation.relation_identity,
                relation.action,
            )
            if previous is not relation.action:
                raise MemoryCandidateError("memory candidate batch cannot add and remove the same relation")
        if len(self.profile) > 1:
            raise MemoryCandidateError("profile candidates must contain at most one item")

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        """从当前六类 YAML 生成供 StructuredChatClient 使用的 JSON Schema。"""

        # 编解码与来源校验模块要用本模块的值对象，放到调用时导入以免循环导入。
        from habitus.memory.editor.candidate_codec import candidate_batch_json_schema

        return candidate_batch_json_schema(cls)

    @classmethod
    def model_validate(cls, value: object) -> MemoryCandidateBatch:
        """严格解析模型返回；不忽略字段，也不进行宽松类型转换。"""

        # 编解码与来源校验模块要用本模块的值对象，放到调用时导入以免循环导入。
        from habitus.memory.editor.candidate_codec import parse_candidate_batch

        return parse_candidate_batch(cls, value)

    def validate_context(
        self,
        segment: ConversationSegment,
        old_memories: MemorySnapshotBatch,
        page_ids: MemoryPageIdMap,
        *,
        zone: tzinfo,
    ) -> MemoryCandidateBatch:
        """绑定原始会话和旧记忆，执行无法由静态 JSON Schema 表达的约束；有一条不过就明确失败。"""

        violations = self.context_violations(segment, old_memories, page_ids, zone=zone)
        if violations:
            raise MemoryCandidateError(violations[0].reason)
        return self

    def context_violations(
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

        # 编解码与来源校验模块要用本模块的值对象，放到调用时导入以免循环导入。
        from habitus.memory.editor.candidate_context_validator import MemoryCandidateContextValidator

        return MemoryCandidateContextValidator(self).violations(segment, old_memories, page_ids, zone=zone)

    def without_context_violations(
        self,
        segment: ConversationSegment,
        old_memories: MemorySnapshotBatch,
        page_ids: MemoryPageIdMap,
        *,
        zone: tzinfo,
    ) -> tuple[MemoryCandidateBatch, tuple[MemoryCandidateViolation, ...]]:
        """剔除通不过来源校验的条目，连同引用被剔候选的关系与身份提议；返回剩下的批次与剔掉的每一条及原因。"""

        batch = self
        dropped: list[MemoryCandidateViolation] = []
        while True:
            violations = batch.context_violations(segment, old_memories, page_ids, zone=zone)
            if not violations:
                return batch, tuple(dropped)
            removed: dict[object, MemoryCandidateViolation] = {}
            for violation in violations:
                removed.setdefault(_violation_key(violation.item), violation)
            dropped_page_ids = {
                violation.item.page_id for violation in removed.values() if isinstance(violation.item, MemoryCandidate)
            }
            for relation in batch.relations:
                if dropped_page_ids & {relation.from_page_id, relation.to_page_id}:
                    removed.setdefault(
                        relation, MemoryCandidateViolation(relation, "relation references a dropped memory candidate")
                    )
            for proposal in batch.identity_proposals:
                if proposal.target_page_id in dropped_page_ids:
                    removed.setdefault(
                        proposal,
                        MemoryCandidateViolation(proposal, "identity proposal targets a dropped memory candidate"),
                    )
            dropped.extend(removed.values())

            def keep(item: object, removed: dict[object, MemoryCandidateViolation] = removed) -> bool:
                return _violation_key(item) not in removed

            batch = replace(
                batch,
                **{
                    field_name: tuple(item for item in getattr(batch, field_name) if keep(item))
                    for field_name in self._FIELD_NAMES
                },
            )

    def to_dict(self) -> dict[str, object]:
        """输出与模型 JSON Schema 完全一致的候选批次。"""

        result: dict[str, object] = {
            field_name: [candidate.to_dict() for candidate in getattr(self, field_name)]
            for field_name in self._MEMORY_FIELD_NAMES
        }
        result[_IDENTITY_OUTPUT_FIELD] = [proposal.to_dict() for proposal in self.identity_proposals]
        result[_RELATION_OUTPUT_FIELD] = [relation.to_dict() for relation in self.relations]
        return result

    def iter_candidates(self) -> tuple[MemoryCandidate, ...]:
        """按 Schema 字段顺序返回本批次的全部记忆候选。"""

        return tuple(candidate for field_name in self._MEMORY_FIELD_NAMES for candidate in getattr(self, field_name))


def _reject_empty_present_strings(fields: Mapping[str, Any], field_name: str) -> None:
    empty = sorted(name for name, value in fields.items() if isinstance(value, str) and not value.strip())
    if empty:
        raise MemoryCandidateError(f"{field_name} candidate contains empty string fields: {empty}")


__all__ = [
    "MemoryCandidate",
    "MemoryCandidateBatch",
    "MemoryCandidateError",
    "MemoryCandidateViolation",
    "MemoryIdentityProposal",
    "MemoryIdentityProposalBasis",
    "MemoryIdentityProposalType",
    "MemoryRelationAction",
    "MemoryRelationCandidate",
    "MemoryLinkType",
]
