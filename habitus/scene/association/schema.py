"""关联输出的 JSON Schema。

**约束贴在它约束的字段上**（与归组同一条纪律）：每个编号字段的描述里写清它能引用哪一组编号，
而不是把规则堆在系统提示词的正文里。装配层仍然逐条核对——schema 只能保证形状，保证不了那个
编号在输入里真的存在。
"""

from __future__ import annotations

from typing import Any

from habitus.foundation.integrity import canonical_digest

_LEFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "consumed_by"],
    "properties": {
        "text": {"type": "string", "description": "一句话说清这次留下了什么前提，单行。"},
        "consumed_by": {
            "type": "string",
            "description": "这个前提在等哪一类行为把它用掉（挂号等就诊、买菜等做饭、约好等打球）。"
            "填那类行为的名字；说不出在等什么，就不要列这条前提。",
        },
    },
}

_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "occurrence_no",
        "context",
        "cites",
        "causes",
        "situation_no",
        "new_situation",
        "consumed",
        "left",
    ],
    "properties": {
        "occurrence_no": {
            "type": "integer",
            "minimum": 1,
            "description": "这一条说的是哪次发生；必须是【这次发生】里列出的编号，一次一行。",
        },
        "context": {
            "type": "string",
            "description": "这次是在什么情况下发生的，一句话、单行。只能由输入里的四样材料拼出来："
            "情境事实、当天的流、更早的行为、未兑现的前提。记录里没写出来的状态不在这四样里。",
        },
        "cites": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "integer", "minimum": 1},
            "description": "上面那句话依据的是哪几行的 # 编号（当天的流，或【更早的行为】）。至少一个。"
            "情境事实不用引用。当天前后确实没有相关的事时，只引用这次发生自己那一行即可。",
        },
        "causes": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": "其中哪几行是这次发生的前因（因为有它才有了这次）。必须是 cites 的子集，必须比这次早，"
            "不能是这次自己。只是排在前面、看不出导致关系的不要列。看不出就填 []。",
        },
        "situation_no": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "这次属于【已有的情境】里的哪一种，填 S 编号的数字（S1 写 1）；都不像就填 null。",
        },
        "new_situation": {
            "type": ["string", "null"],
            "description": "situation_no 为 null 时，用一句话说这是一种什么新情境；否则填 null。"
            "只有当这次确实与已有的每一种都不同才算新——同一种事换个说法不是新情境。"
            "确实哪一种都不像、也说不出是什么新情境时，两个都填 null 是可以的。",
        },
        "consumed": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": "这次用掉了【未兑现的前提】里的哪几条，填 P 编号的数字（P1 写 1）；"
            "同一条前提只能被一次发生用掉；没有就填 []。",
        },
        "left": {
            "type": "array",
            "items": _LEFT_SCHEMA,
            "description": "这次为将来某件事铺好的前提；只是把一件事做完了不算；没有就填 []。",
        },
    },
}

ASSOCIATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entries"],
    "properties": {
        "entries": {
            "type": "array",
            "minItems": 1,
            "items": _ENTRY_SCHEMA,
            "description": "【这次发生】里每个编号恰好一行，一行都不能少。",
        }
    },
}

SCHEMA_FINGERPRINT = canonical_digest(ASSOCIATION_JSON_SCHEMA)[:12]


def association_json_schema(target_count: int) -> dict[str, Any]:
    """把条数钉死成目标数：少一行是漏了一次发生，多一行是无中生有。"""

    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count <= 0:
        raise ValueError("target_count must be a positive integer")
    schema = dict(ASSOCIATION_JSON_SCHEMA)
    properties = dict(schema["properties"])
    entries = dict(properties["entries"])
    entries["minItems"] = target_count
    entries["maxItems"] = target_count
    properties["entries"] = entries
    schema["properties"] = properties
    return schema


__all__ = ["ASSOCIATION_JSON_SCHEMA", "SCHEMA_FINGERPRINT", "association_json_schema"]
