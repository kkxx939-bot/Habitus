"""每晚新增的提示词与 schema：把同 lane 的待定池整体交给模型，同一件事聚成一组，并为每组写好新类的条目。

词表从空开始（裁定 29），类全由这里长出来，所以粒度规则与白天归类、定期拆改同一段文本（``kinds/rules.py``），
只写规则、不放场景例子（裁定 30）。v5（2026-10-12）：删掉 v4 的"去掉具体对象后相同聚一组"与"类名去掉对象"——它在生活数据
上把订机票 + 订酒店聚成「预订」（`17-词表的问题.md` 第 12 条）；改用替代检验。"一组够不够成类"仍由 ``grower._promote``
按复现数定，模型不判。v4 去掉的"拿不准就分开"不恢复（第十批前 3 天按主题拆成一条一组，攒不够复现）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from habitus.behavior.kinds.model import BehaviorClass
from habitus.behavior.kinds.pending import PendingEntry
from habitus.behavior.kinds.rules import ENTRY_FORMAT, RULES_VERSION, SAMENESS_RULE

NIGHTLY_PROMPT_VERSION = f"behavior_kind_nightly_v5+{RULES_VERSION}"

NIGHTLY_SYSTEM_PROMPT = f"""你在整理一份"待定池"：这些行为记录白天没能归进词表里任何一类。现在把它们整体看一遍，
同一件事的记录聚成一组；每条记录恰好属于一组，一条自成一组也可以。

{SAMENESS_RULE}

白天提议的名字只是参考，可能写得太粗或太细，按上面的检验自己判。
每组按下面写出这一类的条目；excludes 里能指出该去词表里哪一类时，goes_to 填那一类的编号。

{ENTRY_FORMAT}

词表里已有的类只给你参考，不要改它们。如果某一组按替代检验就是词表里已有的某一类，把 existing 填那一类的编号
（这一组先不建新类），否则 existing 填 null。
"""


def render_user_message(classes: Mapping[str, BehaviorClass], entries: Mapping[str, PendingEntry]) -> str:
    lines = ["## 词表里已有的类（只作参考）"]
    for key, item in classes.items():
        lines.append(f"{key}  {item.name}：{item.criterion}")
    if not classes:
        lines.append("（无）")
    lines += ["", "## 待定池"]
    for key, entry in entries.items():
        lines.append(f"{key}  [{entry.day.isoformat()}] 白天提议：{entry.proposed}　｜{entry.content}")
    return "\n".join(lines)


def nightly_schema(entry_ids: Sequence[str], class_ids: Sequence[str], *, max_examples: int) -> dict[str, Any]:
    nullable_class = {"anyOf": [{"enum": list(class_ids)}, {"type": "null"}]} if class_ids else {"type": "null"}
    text = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "members": {"type": "array", "items": {"enum": list(entry_ids)}, "minItems": 1},
                        "name": text,
                        "criterion": text,
                        "reminder": text,
                        "excludes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"text": text, "goes_to": nullable_class},
                                "required": ["text", "goes_to"],
                                "additionalProperties": False,
                            },
                        },
                        "examples": {"type": "array", "items": text, "maxItems": max_examples},
                        "existing": nullable_class,
                    },
                    "required": ["members", "name", "criterion", "reminder", "excludes", "examples", "existing"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["groups"],
        "additionalProperties": False,
    }


__all__ = ["NIGHTLY_PROMPT_VERSION", "NIGHTLY_SYSTEM_PROMPT", "nightly_schema", "render_user_message"]
