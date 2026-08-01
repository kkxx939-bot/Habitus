"""由记忆 YAML 派生的严格结构化候选契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import Enum
from itertools import combinations
from typing import Any, ClassVar

from foundation.integrity import canonicalize, immutable_snapshot
from infrastructure.editor.snapshot import SnapshotBatch
from memory.document import MemoryDocument, MemoryLinkType, MemoryStoredLink
from memory.editor.page_id import (
    EXISTING_PAGE_ID_MAX,
    MemoryPageIdError,
    MemoryPageIdMap,
    validate_page_id,
    validate_unique_page_ids,
)
from memory.model import MemoryAddress, MemoryKind
from memory.schema import (
    MemoryFieldSchema,
    MemoryFieldType,
    MemoryMergeStrategy,
    MemorySchemaRegistry,
    MemoryTypeSchema,
)
from memory.snapshot import MemorySnapshotBatch
from memory.uri import MemoryURI
from pre.conversation.messages import (
    ConversationMessageRole,
    ConversationSegment,
    ConversationToolResultStatus,
)


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
        if not isinstance(self.fields, Mapping):
            raise TypeError("memory candidate fields must be an object")
        try:
            fields = MemorySchemaRegistry.load_default().validate(kind, self.fields)
        except (TypeError, ValueError) as exc:
            raise MemoryCandidateError(f"invalid {kind.value} candidate: {exc}") from exc
        if kind is MemoryKind.INTENTION:
            if not isinstance(self.confirmed, bool):
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
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} candidates must be a tuple")
            expected_kind = _OUTPUT_FIELD_TO_KIND[field_name]
            for candidate in values:
                if not isinstance(candidate, MemoryCandidate):
                    raise TypeError(f"{field_name} contains a non-candidate value")
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
        if not isinstance(self.identity_proposals, tuple):
            raise TypeError("identity_proposals must be a tuple")
        if any(not isinstance(item, MemoryIdentityProposal) for item in self.identity_proposals):
            raise TypeError("identity_proposals contains a non-identity-proposal value")
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
        if not isinstance(self.relations, tuple):
            raise TypeError("relations must be a tuple")
        if any(not isinstance(relation, MemoryRelationCandidate) for relation in self.relations):
            raise TypeError("relations contains a non-relation-candidate value")
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

        registry = MemorySchemaRegistry.load_default()
        properties: dict[str, object] = {}
        for schema in registry.all():
            field_name = _KIND_TO_OUTPUT_FIELD[schema.kind]
            array_schema: dict[str, object] = {
                "type": "array",
                "description": schema.description,
                "items": _candidate_item_schema(schema),
            }
            if schema.kind is MemoryKind.PROFILE:
                array_schema["maxItems"] = 1
            properties[field_name] = array_schema
        properties[_RELATION_OUTPUT_FIELD] = {
            "type": "array",
            "description": (
                "只输出能由完整对话确认的长期关系变更。add 表示建立一条仍会长期成立的关系；"
                "remove 表示两个旧节点仍保留、但其既有关系已被本次完整对话明确否定。"
                "没有再次提到、同时出现、临时操作顺序或模型推测都不能产生 add 或 remove。"
                "两端只能引用本次上下文中提供或输出的 page_id，"
                "不得输出 URI 或反向链接。"
            ),
            "items": _relation_candidate_schema(),
        }
        properties[_IDENTITY_OUTPUT_FIELD] = {
            "type": "array",
            "description": (
                "只输出等待确定性身份规划的节点身份提议。same_memory 表示一个完整读取的旧节点与另一个"
                "存活节点是同一条记忆；remove_memory 只表示用户明确要求遗忘，或整个旧节点已被完整"
                "否定。提议不是耐久 MERGE/DELETE，不能输出 URI、revision 或置信度。"
            ),
            "items": _identity_proposal_schema(),
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "MemoryCandidateBatch",
            "description": (
                "从完整 ConversationSegment 中提取的六类记忆候选、临时身份提议和关系候选。"
                "记忆候选必须包含临时 page_id；只返回 Schema 声明的业务字段，"
                "不得返回 URI、存储路径、编辑动作或耐久系统元数据。"
            ),
            "type": "object",
            "additionalProperties": False,
            "required": list(cls._FIELD_NAMES),
            "properties": properties,
        }

    @classmethod
    def model_validate(cls, value: object) -> MemoryCandidateBatch:
        """严格解析模型返回；不忽略字段，也不进行宽松类型转换。"""

        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise MemoryCandidateError("memory candidate output must be an object")
        keys = set(value)
        expected = set(cls._FIELD_NAMES)
        unknown = keys - expected
        missing = expected - keys
        if unknown:
            raise MemoryCandidateError(f"memory candidate output contains unknown fields: {sorted(unknown)}")
        if missing:
            raise MemoryCandidateError(f"memory candidate output is missing fields: {sorted(missing)}")

        registry = MemorySchemaRegistry.load_default()
        parsed: dict[str, tuple[MemoryCandidate, ...]] = {}
        for field_name in cls._MEMORY_FIELD_NAMES:
            raw_items = value[field_name]
            if not isinstance(raw_items, list | tuple):
                raise MemoryCandidateError(f"{field_name} candidates must be an array")
            if field_name == "profile" and len(raw_items) > 1:
                raise MemoryCandidateError("profile candidates must contain at most one item")
            kind = _OUTPUT_FIELD_TO_KIND[field_name]
            candidates: list[MemoryCandidate] = []
            for raw_item in raw_items:
                if not isinstance(raw_item, Mapping):
                    raise MemoryCandidateError(f"{field_name} contains a non-object item")
                page_id = _required_page_id(raw_item, field_name)
                confirmed: bool | None = None
                control_fields = {"page_id"}
                if kind is MemoryKind.INTENTION:
                    if "confirmed" not in raw_item or not isinstance(raw_item["confirmed"], bool):
                        raise MemoryCandidateError("intention candidate requires boolean confirmed")
                    confirmed = raw_item["confirmed"]
                    control_fields.add("confirmed")
                business_fields = {key: item_value for key, item_value in raw_item.items() if key not in control_fields}
                try:
                    fields = registry.validate(kind, business_fields)
                except (TypeError, ValueError) as exc:
                    raise MemoryCandidateError(f"invalid {field_name} candidate: {exc}") from exc
                _reject_empty_present_strings(fields, field_name)
                candidates.append(
                    MemoryCandidate(
                        page_id=page_id,
                        kind=kind,
                        fields=fields,
                        confirmed=confirmed,
                    )
                )
            parsed[field_name] = tuple(candidates)
        raw_relations = value[_RELATION_OUTPUT_FIELD]
        if not isinstance(raw_relations, list | tuple):
            raise MemoryCandidateError("relations must be an array")
        relations = tuple(_parse_relation_candidate(raw_relation) for raw_relation in raw_relations)
        raw_identity_proposals = value[_IDENTITY_OUTPUT_FIELD]
        if not isinstance(raw_identity_proposals, list | tuple):
            raise MemoryCandidateError("identity_proposals must be an array")
        identity_proposals = tuple(_parse_identity_proposal(raw_proposal) for raw_proposal in raw_identity_proposals)
        return cls(
            profile=parsed["profile"],
            preferences=parsed["preferences"],
            entities=parsed["entities"],
            tools=parsed["tools"],
            events=parsed["events"],
            intentions=parsed["intentions"],
            identity_proposals=identity_proposals,
            relations=relations,
        )

    def validate_context(
        self,
        segment: ConversationSegment,
        old_memories: MemorySnapshotBatch,
        page_ids: MemoryPageIdMap,
    ) -> MemoryCandidateBatch:
        """绑定原始会话和旧记忆，执行无法由静态 JSON Schema 表达的约束。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        if not isinstance(old_memories, SnapshotBatch):
            raise TypeError("old_memories must be a MemorySnapshotBatch")
        if not isinstance(page_ids, MemoryPageIdMap):
            raise TypeError("page_ids must be a MemoryPageIdMap")
        self._validate_page_references(old_memories, page_ids)
        self._validate_identity_relation_separation()
        self._validate_relation_removals(old_memories, page_ids)
        self._validate_tool_sources(segment)
        self._validate_event_invariants(segment)
        self._validate_completed_intentions(old_memories)
        return self

    def _validate_page_references(
        self,
        old_memories: MemorySnapshotBatch,
        page_ids: MemoryPageIdMap,
    ) -> None:
        expected_old_uris = {snapshot.identity for snapshot in old_memories.snapshots if snapshot.exists}
        mapped_old_uris = {uri for _page_id, uri in page_ids.existing_items()}
        if mapped_old_uris != expected_old_uris:
            raise MemoryCandidateError("page_id map does not match the complete old-memory snapshot batch")

        available_page_ids = set(page_ids.page_ids())
        candidate_page_ids: set[int] = set()
        for candidate in self.iter_candidates():
            candidate_uri = str(MemoryURI.from_address(candidate.address))
            old_page_id = page_ids.page_id_for(candidate_uri)
            resolved_uri = page_ids.resolve(candidate.page_id)
            if candidate.page_id <= EXISTING_PAGE_ID_MAX:
                if resolved_uri is None or resolved_uri != candidate_uri:
                    raise MemoryCandidateError(
                        "existing memory candidate must reuse the page_id assigned to its exact URI"
                    )
            elif old_page_id is not None:
                raise MemoryCandidateError("candidate for an existing memory URI must reuse its existing page_id")
            available_page_ids.add(candidate.page_id)
            candidate_page_ids.add(candidate.page_id)

        for relation in self.relations:
            if relation.from_page_id not in available_page_ids:
                raise MemoryCandidateError("relation candidate references an unknown from_page_id")
            if relation.to_page_id not in available_page_ids:
                raise MemoryCandidateError("relation candidate references an unknown to_page_id")

        for proposal in self.identity_proposals:
            if proposal.source_page_id not in page_ids.page_ids():
                raise MemoryCandidateError("identity proposal source is not a complete old-memory page_id")
            if proposal.source_page_id in candidate_page_ids:
                raise MemoryCandidateError("identity proposal source cannot also appear as a memory candidate")
            if proposal.target_page_id is not None and proposal.target_page_id not in available_page_ids:
                raise MemoryCandidateError("identity proposal references an unknown target_page_id")
            if (
                proposal.proposal_type is MemoryIdentityProposalType.SAME_MEMORY
                and proposal.target_page_id not in candidate_page_ids
            ):
                raise MemoryCandidateError("same_memory target must also appear as a fully planned memory candidate")

    def _validate_identity_relation_separation(self) -> None:
        """待退休来源不能同时参与显式关系变更。"""

        retired_sources = {proposal.source_page_id for proposal in self.identity_proposals}
        for relation in self.relations:
            if relation.from_page_id in retired_sources or relation.to_page_id in retired_sources:
                raise MemoryCandidateError("identity proposal source cannot also be referenced by a relation candidate")

    def _validate_relation_removals(
        self,
        old_memories: MemorySnapshotBatch,
        page_ids: MemoryPageIdMap,
    ) -> None:
        """REMOVE 必须逐字命中已完整读取且双向一致的旧关系。"""

        for relation in self.relations:
            if relation.action is not MemoryRelationAction.REMOVE:
                continue
            if relation.from_page_id > EXISTING_PAGE_ID_MAX or relation.to_page_id > EXISTING_PAGE_ID_MAX:
                raise MemoryCandidateError("relation remove must reference two existing memory page_ids")
            from_uri = page_ids.resolve(relation.from_page_id)
            to_uri = page_ids.resolve(relation.to_page_id)
            if from_uri is None or to_uri is None:
                raise MemoryCandidateError("relation remove references an unresolved existing page_id")
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
                or not isinstance(source.value, MemoryDocument)
                or not isinstance(target.value, MemoryDocument)
            ):
                raise MemoryCandidateError("relation remove requires both complete old endpoint snapshots")
            if stored not in source.value.links or stored not in target.value.backlinks:
                raise MemoryCandidateError("relation remove does not match a complete existing Link/Backlink pair")

    def _validate_tool_sources(self, segment: ConversationSegment) -> None:
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

        for candidate in self.tools:
            tool_name = candidate.fields.get("tool_name")
            if not isinstance(tool_name, str) or tool_name not in successful_names:
                raise MemoryCandidateError("tool candidate requires a matching successful tool_call/tool_result pair")
            if "failure_recovery" not in candidate.fields:
                continue
            failures = failed_at.get(tool_name, [])
            successes = succeeded_at.get(tool_name, [])
            if not any(failure < success for failure in failures for success in successes):
                raise MemoryCandidateError("tool failure_recovery requires a failure followed by a successful result")

    def _validate_completed_intentions(self, old_memories: MemorySnapshotBatch) -> None:
        for candidate in self.intentions:
            if candidate.fields.get("status") != "completed":
                continue
            if candidate.confirmed is not True:
                raise MemoryCandidateError("completed intention requires explicit confirmation in this conversation")
            uri = str(MemoryURI.from_address(candidate.address))
            snapshot = old_memories.get(uri)
            if snapshot is None or not snapshot.exists or snapshot.value is None:
                raise MemoryCandidateError("completed intention must target an existing intention read in this batch")
            if not isinstance(snapshot.value, MemoryDocument):
                raise MemoryCandidateError("completed intention snapshot is not a memory document")
            if snapshot.value.kind is not MemoryKind.INTENTION:
                raise MemoryCandidateError("completed intention snapshot has the wrong memory type")

    def _validate_event_invariants(self, segment: ConversationSegment) -> None:
        """只校验模型语义结果中可确定验证的 Event 日期边界。"""

        for candidate in self.events:
            event_date = candidate.address.event_date
            if not isinstance(event_date, date):  # pragma: no cover - Event 地址由 Schema 严格生成。
                raise MemoryCandidateError("event candidate requires a calendar date")
            if event_date > segment.ended_at.date():
                raise MemoryCandidateError("future event date cannot represent an occurred Event")

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


def _candidate_item_schema(schema: MemoryTypeSchema) -> dict[str, object]:
    properties = {
        "page_id": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "本次解析周期的临时节点编号。更新旧节点时复用上下文给出的 1..99；"
                "新节点使用本批次内唯一的 100 以上编号。"
            ),
        },
        **{field.name: _candidate_field_schema(field) for field in schema.fields},
    }
    required = ["page_id", *(field.name for field in schema.fields if field.required)]
    if schema.kind is MemoryKind.INTENTION:
        properties["confirmed"] = {
            "type": "boolean",
            "description": (
                "仅表示当前完整 ConversationSegment 是否明确创建、更新或重新确认了此事项。"
                "只是为 same_memory 保留目标、但当前对话没有确认时必须为 false；"
                "系统据此维护 last_confirmed_at，模型不得输出时间戳。"
            ),
        }
        required.append("confirmed")
    item: dict[str, object] = {
        "type": "object",
        "description": schema.description,
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }
    minimum = schema.min_non_empty_content_fields
    if minimum:
        content_names = [field.name for field in schema.content_fields]
        item["anyOf"] = [{"required": list(names)} for names in combinations(content_names, minimum)]
    return item


def _relation_candidate_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": [action.value for action in MemoryRelationAction],
                "description": ("add 建立关系；remove 只撤销已读取旧关系，不删除两端节点。"),
            },
            "from_page_id": {
                "type": "integer",
                "minimum": 1,
                "description": "关系来源节点的临时 page_id。",
            },
            "to_page_id": {
                "type": "integer",
                "minimum": 1,
                "description": "关系目标节点的临时 page_id。",
            },
            "link_type": {
                "type": "string",
                "enum": [link_type.value for link_type in MemoryLinkType],
                "description": (
                    "来源节点指向目标节点的受控关系类型。\n"
                    + "\n".join(f"{link_type.value}: {link_type.description}" for link_type in MemoryLinkType)
                ),
            },
        },
        "required": ["action", "from_page_id", "to_page_id", "link_type"],
    }


def _identity_proposal_schema() -> dict[str, object]:
    source_page_id = {
        "type": "integer",
        "minimum": 1,
        "maximum": EXISTING_PAGE_ID_MAX,
        "description": "必须是本次完整读取的旧节点 page_id。",
    }
    return {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "proposal_type": {"const": MemoryIdentityProposalType.SAME_MEMORY.value},
                    "source_page_id": source_page_id,
                    "target_page_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "经过初步字段计划覆盖的存活目标 page_id。",
                    },
                    "basis": {"const": MemoryIdentityProposalBasis.DUPLICATE_IDENTITY.value},
                },
                "required": ["proposal_type", "source_page_id", "target_page_id", "basis"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "proposal_type": {"const": MemoryIdentityProposalType.REMOVE_MEMORY.value},
                    "source_page_id": source_page_id,
                    "target_page_id": {"type": "null"},
                    "basis": {
                        "type": "string",
                        "enum": [
                            MemoryIdentityProposalBasis.EXPLICIT_FORGET.value,
                            MemoryIdentityProposalBasis.FULLY_INVALIDATED.value,
                        ],
                    },
                },
                "required": ["proposal_type", "source_page_id", "target_page_id", "basis"],
            },
        ]
    }


def _required_page_id(item: Mapping[str, Any], field_name: str) -> int:
    if "page_id" not in item:
        raise MemoryCandidateError(f"{field_name} candidate is missing page_id")
    try:
        return validate_page_id(item["page_id"])
    except MemoryPageIdError as exc:
        raise MemoryCandidateError(f"invalid {field_name} candidate: {exc}") from exc


def _parse_relation_candidate(value: object) -> MemoryRelationCandidate:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise MemoryCandidateError("relations contains a non-object item")
    expected = {"action", "from_page_id", "to_page_id", "link_type"}
    keys = set(value)
    unknown = keys - expected
    missing = expected - keys
    if unknown:
        raise MemoryCandidateError(f"relation candidate contains unknown fields: {sorted(unknown)}")
    if missing:
        raise MemoryCandidateError(f"relation candidate is missing fields: {sorted(missing)}")
    return MemoryRelationCandidate(
        action=value["action"],
        from_page_id=value["from_page_id"],
        to_page_id=value["to_page_id"],
        link_type=value["link_type"],
    )


def _parse_identity_proposal(value: object) -> MemoryIdentityProposal:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise MemoryCandidateError("identity_proposals contains a non-object item")
    expected = {"proposal_type", "source_page_id", "target_page_id", "basis"}
    keys = set(value)
    unknown = keys - expected
    missing = expected - keys
    if unknown:
        raise MemoryCandidateError(f"identity proposal contains unknown fields: {sorted(unknown)}")
    if missing:
        raise MemoryCandidateError(f"identity proposal is missing fields: {sorted(missing)}")
    return MemoryIdentityProposal(
        proposal_type=value["proposal_type"],
        source_page_id=value["source_page_id"],
        target_page_id=value["target_page_id"],
        basis=value["basis"],
    )


def _candidate_field_schema(field: MemoryFieldSchema) -> dict[str, object]:
    if field.field_type is MemoryFieldType.INTEGER:
        field_type = "integer"
    elif field.field_type is MemoryFieldType.NUMBER:
        field_type = "number"
    elif field.field_type is MemoryFieldType.BOOLEAN:
        field_type = "boolean"
    else:
        field_type = "string"
    result: dict[str, object] = {
        "type": field_type,
        "description": f"{field.description}\n{_merge_output_description(field)}",
    }
    if field_type == "string":
        result["minLength"] = 1
        result["pattern"] = r"\S"
    if field.field_type is MemoryFieldType.DATE:
        result["format"] = "date"
        result["pattern"] = r"^\d{4}-\d{2}-\d{2}$"
    if field.allowed_values:
        result["enum"] = list(field.allowed_values)
    return result


def _merge_output_description(field: MemoryFieldSchema) -> str:
    if field.merge_strategy is MemoryMergeStrategy.IMMUTABLE:
        return "更新已有节点时必须逐值保留旧字段，不得改名、改址或重写。"
    if field.merge_strategy is MemoryMergeStrategy.PATCH:
        return (
            "更新已有节点且输出此字段时，必须返回合并旧记忆后的完整最终值，"
            "不得输出 SEARCH/REPLACE 或局部增量；可选字段省略表示保留旧值。"
        )
    return "更新已有节点时，此字段表示最新完整状态；提供时必须返回完整最终值，可选字段省略表示从最终记忆中移除。"


def _reject_empty_present_strings(fields: Mapping[str, Any], field_name: str) -> None:
    empty = sorted(name for name, value in fields.items() if isinstance(value, str) and not value.strip())
    if empty:
        raise MemoryCandidateError(f"{field_name} candidate contains empty string fields: {empty}")


__all__ = [
    "MemoryCandidate",
    "MemoryCandidateBatch",
    "MemoryCandidateError",
    "MemoryIdentityProposal",
    "MemoryIdentityProposalBasis",
    "MemoryIdentityProposalType",
    "MemoryRelationAction",
    "MemoryRelationCandidate",
    "MemoryLinkType",
]
