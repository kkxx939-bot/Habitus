"""判断输出的 JSON Schema。

约束贴在它约束的字段上（与关联层同一条纪律）：每个字段的描述里写清它能引用什么，而不是把规则堆在
系统提示词正文里。装配层仍然逐条核对——schema 只能保证形状，保证不了"那张卡真的在包里"。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from habitus.foresight.judge.model import DAY_STATES, VERDICTS
from habitus.foundation.integrity import canonical_digest

_WINDOW_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "required": ["from_slot", "to_slot"],
    "properties": {
        "from_slot": {"type": "integer", "minimum": 0, "description": "最早从今天第几槽开始（含）。"},
        "to_slot": {"type": "integer", "minimum": 0, "description": "最晚到今天第几槽（含）。"},
    },
    "description": "判为「会」时，它会在今天哪一段槽号里开始；槽宽与一天的槽数在包头。"
    "不能早于此刻所在的槽。判为「不会」填 null；判为「会」或「说不准」但说不出时候也可以填 null。",
}

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind_token", "verdict", "window", "next", "basis", "note"],
    "properties": {
        "kind_token": {"type": "string", "description": "说的是哪个候选；照抄包里那一节的标题，一个候选一条。"},
        "verdict": {
            "type": "string",
            "enum": list(VERDICTS),
            "description": "会：此刻像它历史上的那几次，接下来会做；不会：此刻已经走到了别的路上；"
            "说不准：此刻场景与历次都对不上，或者材料不够。",
        },
        "window": _WINDOW_SCHEMA,
        "next": {
            "type": "array",
            "items": {"type": "string"},
            "description": "它之后紧跟着会做什么，填行为名字，原样照抄你引用的那几张卡「之后」那段里出现过的名字；"
            "没有就填 []。",
        },
        "basis": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": "你是拿此刻和哪几张卡比的，填卡的 # 编号（编号在这个候选自己的【历史】里）。"
            "判为「会」至少一张；「不会」与「说不准」也尽量指出是哪几张让你这么判。",
        },
        "note": {"type": "string", "description": "一句话说清为什么这么判，单行；只能引用包里写着的东西。"},
    },
}

JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts", "day_state", "day_note"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": _VERDICT_SCHEMA,
            "description": "包里每个摊开的候选（有【历史】那一节的）恰好一条，一条都不能少；只列名的候选不写。",
        },
        "day_state": {
            "type": "string",
            "enum": list(DAY_STATES),
            "description": "把此刻场景与各候选的卡合起来看，今天到此刻为止像不像往常的这个时候。",
        },
        "day_note": {
            "type": ["string", "null"],
            "description": "反常时一句话说是哪几件往常这时候已经有的事没出现、或出现了什么往常没有的；正常填 null。",
        },
    },
}

SCHEMA_FINGERPRINT = canonical_digest(JUDGE_JSON_SCHEMA)[:12]


def judge_json_schema(kind_tokens: Sequence[str]) -> dict[str, Any]:
    """把条数钉死成摊开的候选数、名字钉死成那几个：少一条是漏答，多一条是无中生有。"""

    names = list(kind_tokens)
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("kind_tokens must be unique non-empty strings")
    schema = dict(JUDGE_JSON_SCHEMA)
    properties = dict(schema["properties"])
    verdicts = dict(properties["verdicts"])
    verdicts["minItems"] = len(names)
    verdicts["maxItems"] = len(names)
    if names:
        item = dict(verdicts["items"])
        item_properties = dict(item["properties"])
        item_properties["kind_token"] = {**item_properties["kind_token"], "enum": names}
        item["properties"] = item_properties
        verdicts["items"] = item
    properties["verdicts"] = verdicts
    schema["properties"] = properties
    return schema


__all__ = ["JUDGE_JSON_SCHEMA", "SCHEMA_FINGERPRINT", "judge_json_schema"]
