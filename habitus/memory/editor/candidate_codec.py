"""MemoryCandidateBatch 的模型契约编解码：由六类 YAML 生成结构化输出 JSON Schema，并严格解析模型返回。"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import TYPE_CHECKING, Any

from habitus.foundation.validation import is_int
from habitus.memory.document import MemoryLinkType
from habitus.memory.editor.candidate import (
    _IDENTITY_OUTPUT_FIELD,
    _KIND_TO_OUTPUT_FIELD,
    _OUTPUT_FIELD_TO_KIND,
    _RELATION_OUTPUT_FIELD,
    MemoryCandidate,
    MemoryCandidateError,
    MemoryIdentityProposal,
    MemoryIdentityProposalBasis,
    MemoryIdentityProposalType,
    MemoryRelationAction,
    MemoryRelationCandidate,
    _reject_empty_present_strings,
)
from habitus.memory.editor.page_id import EXISTING_PAGE_ID_MAX, MemoryPageIdError, validate_page_id
from habitus.memory.model import MemoryKind
from habitus.memory.schema import (
    MemoryFieldSchema,
    MemoryFieldType,
    MemoryMergeStrategy,
    MemorySchemaRegistry,
    MemoryTypeSchema,
)

if TYPE_CHECKING:
    from habitus.memory.editor.candidate import MemoryCandidateBatch


def candidate_batch_json_schema(batch_type: type[MemoryCandidateBatch]) -> dict[str, object]:
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
        "required": list(batch_type._FIELD_NAMES),
        "properties": properties,
    }


def parse_candidate_batch(batch_type: type[MemoryCandidateBatch], value: object) -> MemoryCandidateBatch:
    """严格解析模型返回；不忽略字段，也不进行宽松类型转换。"""

    not_object = "memory candidate output must be an object"
    value = _model_object(value, set(batch_type._FIELD_NAMES), "memory candidate output", not_object)
    registry = MemorySchemaRegistry.load_default()
    parsed: dict[str, tuple[MemoryCandidate, ...]] = {}
    for field_name in batch_type._MEMORY_FIELD_NAMES:
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
                fields = registry.validate_new(kind, business_fields)
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
    return batch_type(
        profile=parsed["profile"],
        preferences=parsed["preferences"],
        entities=parsed["entities"],
        tools=parsed["tools"],
        events=parsed["events"],
        intentions=parsed["intentions"],
        identity_proposals=identity_proposals,
        relations=relations,
    )


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


# 模型输出里的严格对象；下面两个解码器在构造前查类型，报错文字照抄对应值对象的 ``__post_init__``
def _model_object(value: object, expected: set[str], label: str, not_object: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise MemoryCandidateError(not_object)
    if unknown := set(value) - expected:
        raise MemoryCandidateError(f"{label} contains unknown fields: {sorted(unknown)}")
    if missing := expected - set(value):
        raise MemoryCandidateError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _parse_relation_candidate(value: object) -> MemoryRelationCandidate:
    expected = {"action", "from_page_id", "to_page_id", "link_type"}
    item = _model_object(value, expected, "relation candidate", "relations contains a non-object item")
    try:
        from_page_id, to_page_id = validate_page_id(item["from_page_id"]), validate_page_id(item["to_page_id"])
    except MemoryPageIdError as exc:
        raise MemoryCandidateError(f"invalid relation candidate: {exc}") from exc
    try:
        action = MemoryRelationAction(item["action"])
    except ValueError as exc:
        raise MemoryCandidateError("relation candidate contains an unsupported action") from exc
    try:
        link_type = MemoryLinkType(item["link_type"])
    except ValueError as exc:
        raise MemoryCandidateError("relation candidate contains an unsupported link_type") from exc
    return MemoryRelationCandidate(action=action, from_page_id=from_page_id, to_page_id=to_page_id, link_type=link_type)


def _parse_identity_proposal(value: object) -> MemoryIdentityProposal:
    expected = {"proposal_type", "source_page_id", "target_page_id", "basis"}
    item = _model_object(value, expected, "identity proposal", "identity_proposals contains a non-object item")
    try:
        proposal_type = MemoryIdentityProposalType(item["proposal_type"])
        basis = MemoryIdentityProposalBasis(item["basis"])
        source_page_id = validate_page_id(item["source_page_id"])
    except (MemoryPageIdError, TypeError, ValueError) as exc:
        raise MemoryCandidateError("identity proposal contains an invalid type, basis or source_page_id") from exc
    target_page_id = item["target_page_id"]
    if target_page_id is not None and not is_int(target_page_id):
        if proposal_type is MemoryIdentityProposalType.SAME_MEMORY:
            raise MemoryCandidateError("same_memory proposal requires a valid target_page_id")
        raise MemoryCandidateError("remove_memory proposal cannot contain a target_page_id")
    return MemoryIdentityProposal(proposal_type, source_page_id, target_page_id, basis)


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


__all__ = ["candidate_batch_json_schema", "parse_candidate_batch"]
