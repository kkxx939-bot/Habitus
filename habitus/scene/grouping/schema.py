"""归组的结构化输出 Schema。

穷尽性由输出形状承担：``assignments`` 每条输入行为恰好一行、第 n 行的 no 等于 n（长度由调用方
按行为数钉死）。契约尽量写进字段描述——模型填某个字段时读的就是它（融合层实测：贴在字段上的
约束比正文段落有效得多）。
"""

from __future__ import annotations

from typing import Any

from habitus.foundation.integrity import canonical_digest
from habitus.scene.model import SceneLinkType, SceneRole

_RELATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "occurrence_no", "scene_no", "reference_no", "pending_no"],
    "properties": {
        "kind": {
            "type": "string",
            "enum": [kind.value for kind in SceneLinkType],
            "description": "needs 这件事的前提由目标建立；results_from 这件事因目标而起。",
        },
        "occurrence_no": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "目标是本日一条行为时填它的编号；那条行为必须早于本情景的第一个成员；否则 null。",
        },
        "scene_no": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "目标是本日另一个情景时填它的编号（不能填自己，必须早开始）；否则 null。",
        },
        "reference_no": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "目标是【先前的事】里的 C 编号时填；否则 null。",
        },
        "pending_no": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "目标是【待用前提】里的 P 编号时填（前提被这件事用掉）；否则 null。四个目标字段恰好填一个。",
        },
    },
}

_PENDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "occurrence_no"],
    "properties": {
        "text": {"type": "string", "description": "一句话说清留下了什么前提，单行。"},
        "occurrence_no": {"type": "integer", "minimum": 1, "description": "建立这个前提的那条行为编号，必须是本情景的成员。"},
    },
}

_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_no", "label", "effects", "pending_effects", "relations"],
    "properties": {
        "scene_no": {"type": "integer", "minimum": 1, "description": "本次输出内的临时情景编号。"},
        "label": {"type": "string", "description": "这件事的目标，用主体自己会说的话；说不出目标就不要成立这个情景。"},
        "effects": {"type": "array", "items": {"type": "string"}, "description": "这件事留下的改变，每项单行；记录支持不了就填 []。"},
        "pending_effects": {
            "type": "array",
            "items": _PENDING_SCHEMA,
            "description": "专门为将来某件事铺好的前提，按建立那一刻判、不管当天后来有没有用掉；普通的完成不算；没有填 []。",
        },
        "relations": {"type": "array", "items": _RELATION_SCHEMA, "description": "与更早对象的关系；看不出依赖就填 []。"},
    },
}

_ASSIGNMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["no", "scene_no", "role"],
    "properties": {
        "no": {"type": "integer", "minimum": 1, "description": "行为编号；必须与本行位置一致。"},
        "scene_no": {"type": ["integer", "null"], "minimum": 1, "description": "属于哪个情景；不属于任何情景填 null。"},
        "role": {
            "type": ["string", "null"],
            "enum": [*(role.value for role in SceneRole), None],
            "description": "在该情景里的角色；scene_no 为 null 时也为 null。",
        },
    },
}

SCENE_GROUPING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scenes", "assignments"],
    "properties": {
        # 允许空：整天没有一件归得成的事时一个情景都不出（每行填 null），不逼模型发明。
        "scenes": {"type": "array", "items": _SCENE_SCHEMA},
        "assignments": {
            "type": "array",
            "items": _ASSIGNMENT_SCHEMA,
            "minItems": 1,
            "description": "每条输入行为恰好一行，顺序与输入一致，一行都不能少。",
        },
    },
}

SCHEMA_FINGERPRINT = canonical_digest(SCENE_GROUPING_JSON_SCHEMA)[:12]


def grouping_json_schema(occurrence_count: int) -> dict[str, Any]:
    """按本日行为数钉死 ``assignments`` 的长度。"""

    if isinstance(occurrence_count, bool) or not isinstance(occurrence_count, int) or occurrence_count <= 0:
        raise ValueError("occurrence_count must be a positive integer")
    schema = dict(SCENE_GROUPING_JSON_SCHEMA)
    properties = dict(schema["properties"])
    assignments = dict(properties["assignments"])
    assignments["minItems"] = occurrence_count
    assignments["maxItems"] = occurrence_count
    properties["assignments"] = assignments
    schema["properties"] = properties
    return schema


__all__ = ["SCENE_GROUPING_JSON_SCHEMA", "SCHEMA_FINGERPRINT", "grouping_json_schema"]
