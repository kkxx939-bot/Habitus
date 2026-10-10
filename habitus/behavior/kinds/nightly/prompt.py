"""每晚新增的提示词与 schema：把同 lane 的待定池整体交给模型，同一件事聚成一组，并为每组写好新类的条目。

词表从空开始（裁定 29），类全由这里长出来，所以粒度规则与白天归类同一套，只写规则、不放场景例子（裁定 30）：
去掉具体对象 / 主题之后提醒句相同就是同一件事（v4；v3 只写"对象不影响"，模型照样按主题分组）；按在完成什么分、不看谁动手。
v4 同时去掉了"拿不准就分开"：第十批前 3 天它把同一类事按主题拆成一条一组，攒不够复现，类长不出来。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from habitus.behavior.kinds.model import BehaviorClass
from habitus.behavior.kinds.pending import PendingEntry

NIGHTLY_PROMPT_VERSION = "behavior_kind_nightly_v4"

NIGHTLY_SYSTEM_PROMPT = """你在整理一份"待定池"：这些行为记录白天没能归进词表里任何一类。现在把它们整体看一遍，
同一件事的记录聚成一组；每条记录恰好属于一组，一条自成一组也可以。

判断"同一件事"用提醒句：如果要提醒这个人做这件事，会说"该……了"。
比之前先把提醒句里的具体对象、主题、内容、项目名、人名、地点、时间都去掉，只留"在做哪一类事"；去掉之后相同，就是同一件事，聚成一组。
只有去掉某个对象之后就说不清在做什么了，这个对象才算这件事的一部分、要分开。
按在完成什么分，不看谁动手：让别人或工具去完成一件事，就是那件事本身；做一件事的过程中顺带问的、查的、确认的，也还是那件事。
动手之前专门商量这件事该怎么做、与手上的事无关的另一个问题、只是安排或叫停别人而不针对任何具体的事，才各是另一件事。
白天提议的名字可能带着具体主题，不要被它带着分组，按上面去掉对象之后的样子判断。

每组写出这一类的条目：
- name：类名，写去掉具体对象之后的那件事本身，不带具体对象、主题、项目名、人名；
- criterion：一句判据，说清什么算这一类；
- reminder：提醒句，"该……了"；
- excludes：容易混进来但不算这一类的事（可以为空）；能指出它该去词表里哪一类时，goes_to 填那一类的编号；
- examples：从组里挑最有代表性的几条原话（条数上限见 schema）。

词表里已有的类只给你参考，不要改它们。如果某一组其实就是词表里已有的某一类，把 existing 填那一类的编号
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
