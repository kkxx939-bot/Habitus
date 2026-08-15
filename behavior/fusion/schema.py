"""行为判断的结构化输出 Schema。

## 逐帧归属表承担穷尽性

``frames`` 每条输入片段恰好一行，第 n 行的 ``no`` 必须等于 n，长度由调用方按本段片段数钉死。
穷尽性因此是**输出的形状**，不是事后数一遍——实测证明模型做不了"把 N 个编号无重无漏地分组"这
类组合记账，但逐行回答"这一帧属于谁"没有问题。

一行可以有多个归属（并行时边界帧本来就同时属于两边），也**必须**至少有一个：读不懂的帧归给一条
``behavior`` 为空的判断，这样"不能融合的也要留下"不需要另一套结构。

## 归属直接落到 basis 一级

``assignments`` 同时给出判断编号与 ``basis`` 条目编号，于是覆盖范围与折叠结构由同一份数据推出，
模型不必把帧号写两遍。``basis_no`` 为 null 表示这条判断没有分解（不是目标行为事件）。

全部对象 ``additionalProperties: false`` 且全字段 ``required``：不适用的字段显式填 null，
省略会让"模型忘了写"与"模型认为没有"分不开。
"""

from __future__ import annotations

from typing import Any

_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["basis_no", "semantics"],
    "properties": {
        "basis_no": {"type": "integer", "minimum": 1, "description": "本条判断内的临时编号。"},
        "semantics": {
            "type": "string",
            "description": "一条构成该事件的行为事实，如「打肥皂搓手」；是折叠后的说法，不是原始帧的复述。",
        },
    },
}

_RELATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "target", "context_target"],
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["continues", "supersedes", "concurrent_with", "results_from"],
            "description": (
                "continues 延续那条判断；supersedes 修正那条判断；"
                "concurrent_with 与那条判断同时进行；"
                "results_from 这件事是那件事造成的结果。"
            ),
        },
        "target": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "本次输出里另一条判断的 judgement_no；指向【先前的判断】时填 null。",
        },
        "context_target": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "【先前的判断】里的编号（C1 就填 1）；指向本次输出时填 null。",
        },
    },
}

_JUDGEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "judgement_no",
        "subjects",
        "behavior",
        "goal",
        "summary",
        "basis",
        "status",
        "status_basis",
        "relations",
    ],
    "properties": {
        "judgement_no": {"type": "integer", "minimum": 1, "description": "本次输出内的临时判断编号。"},
        "subjects": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "谁在做这件事；必须逐字取自它覆盖的片段里出现过的称呼。两个人一起做就都写上，"
                "不要拼成一个字符串。behavior 为 null 时填 []。"
            ),
        },
        "behavior": {
            "type": ["string", "null"],
            "description": "这是什么行为，如「洗手」。判不出来就填 null，表示这段观测没读懂。",
        },
        "goal": {
            "type": ["string", "null"],
            "description": (
                "主体自己会说的那个目标，如「清洁双手」。判不出目标就填 null"
                "——那表示这只是一个动作，不是一件有目标的事。"
            ),
        },
        "summary": {
            "type": ["string", "null"],
            "description": "脱离观测也读得懂的一句话。behavior 为 null 时填 null。",
        },
        "basis": {
            "type": "array",
            "items": _FACT_SCHEMA,
            "description": "构成这件事的行为事实。**只有 goal 非 null 时才填**，其余情况填 []。",
        },
        "status": {
            "type": ["string", "null"],
            "enum": ["ongoing", "completed", "interrupted", "abandoned", None],
            "description": (
                "ongoing 还在做；completed 做完了；interrupted 做到一半被打断；"
                "abandoned 主动放弃。behavior 为 null 时填 null。"
            ),
        },
        "status_basis": {
            "type": ["string", "null"],
            "enum": ["observed", "inferred", "observation_lost", None],
            "description": (
                "observed 看到了结束；inferred 没看到但从后续推出；"
                "observation_lost 观测在这里断了、之后不知道。behavior 为 null 时填 null。"
            ),
        },
        "relations": {
            "type": "array",
            "items": _RELATION_SCHEMA,
            "description": (
                "与本次其它判断的关系；独立就填 []。一条判断可以同时有多条关系。"
                "并行只需要在一边声明，系统会自动补上另一边。"
            ),
        },
    },
}

_ASSIGNMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["judgement_no", "basis_no"],
    "properties": {
        "judgement_no": {"type": "integer", "minimum": 1},
        "basis_no": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "该判断内的 basis 条目编号；判断没有 basis 时填 null。",
        },
    },
}

_FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["no", "assignments"],
    "properties": {
        "no": {"type": "integer", "minimum": 1, "description": "片段编号；必须与本行的位置一致。"},
        "assignments": {
            "type": "array",
            "items": _ASSIGNMENT_SCHEMA,
            "minItems": 1,
            "description": "这一帧属于哪些判断。并行时可以有多个；一帧都不能空着。",
        },
    },
}

JUDGEMENT_FUSION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["judgements", "frames"],
    "properties": {
        "judgements": {"type": "array", "items": _JUDGEMENT_SCHEMA, "minItems": 1},
        "frames": {
            "type": "array",
            "items": _FRAME_SCHEMA,
            "minItems": 1,
            "description": "每条输入片段恰好一行，顺序与输入一致，一行都不能少。",
        },
    },
}


def fusion_json_schema(fragment_count: int) -> dict[str, Any]:
    """按本段片段数钉死 ``frames`` 的长度。

    把"一帧都不能少"写进 Schema 而不是留给事后校验：支持 ``json_schema`` 的供应商会直接拒绝长度
    不符的输出，不支持的也能从 Schema 文本里读到这条硬要求。
    """

    if isinstance(fragment_count, bool) or not isinstance(fragment_count, int) or fragment_count <= 0:
        raise ValueError("fragment_count must be a positive integer")
    schema = dict(JUDGEMENT_FUSION_JSON_SCHEMA)
    properties = dict(schema["properties"])
    frames = dict(properties["frames"])
    frames["minItems"] = fragment_count
    frames["maxItems"] = fragment_count
    properties["frames"] = frames
    schema["properties"] = properties
    return schema


__all__ = ["JUDGEMENT_FUSION_JSON_SCHEMA", "fusion_json_schema"]
