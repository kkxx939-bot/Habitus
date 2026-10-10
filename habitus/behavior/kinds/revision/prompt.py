"""定期拆改的提示词与 schema：模型看各类占比、跨天复现、抽样原话，只提差分（合并 / 拆分 / 改判据）。

"同一件事"与白天归类、每晚新增同一条规则，只写规则、不放场景例子（裁定 30）：去掉具体对象 / 主题之后提醒句相同就是同一件事。
v2 去掉了"个人习惯可以从通用类里切出来"——没有预置的通用类了（裁定 29），类本来就是这个人自己的。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from habitus.behavior.kinds.model import BehaviorClass
from habitus.behavior.kinds.revision.facts import Member

REVISION_PROMPT_VERSION = "behavior_kind_revision_v2"
NEW = "NEW"

REVISION_SYSTEM_PROMPT = """你在检查一份行为类词表还合不合现状。给你同一条 lane 的全部在用类：判据、提醒句、
不含、这段时间归进来的条数与占比、出现在多少个不同的日子，以及抽样的原话。

判断"同一件事"用提醒句（"该……了"）：先把里面的具体对象、主题、内容、项目名、人名、地点、时间都去掉，只留"在做哪一类事"；
去掉之后相同就是同一件事。只有去掉某个对象之后就说不清在做什么了，这个对象才算这件事的一部分。

只提必要的改动，没有就返回空列表。可以提三种：
- merge：两类其实是同一件事（按上面的办法比，提醒句相同）。class 填要并掉的那一类，target 填并入的那一类。
- split：一类里混着提醒句不同的两件事（去掉对象之后仍不同），而且两边都能跨天反复出现。class 填被拆的那一类，
  draft 写拆完后它自己的条目（收窄的判据），new_class 写拆出来的新类条目。
- revise：只是措辞不清楚、或缺一条「不含」导致和别的类混。class 填那一类，draft 写改后的条目。

条目的写法：name 写去掉具体对象之后的那件事本身，不带具体对象、主题、项目名、人名；criterion 一句判据；reminder 是"该……了"；
excludes 写容易混进来但不算本类的事，能指出该去哪一类时 goes_to 填那一类的编号（拆分时指向新类填 NEW）；
examples 从抽样里挑最有代表性的几条原话（条数上限见 schema）。

每条改动先在 reason 里写理由（引用抽样里的具体原话）。拿不准就不提——错误的合并比暂时多一类更难挽回。
"""


def render_user_message(
    classes: Mapping[str, BehaviorClass],
    counts: Mapping[str, tuple[int, int]],
    samples: Mapping[str, Sequence[Member]],
    *,
    max_steps: int,
) -> str:
    """``counts``：类编号（C…）→（条数，不同日子数）；``samples``：类编号 → 抽样成员。"""

    total = sum(count for count, _ in counts.values()) or 1
    names = {item.id: key for key, item in classes.items()}
    lines = ["## 在用类"]
    for key, item in classes.items():
        count, days = counts.get(key, (0, 0))
        lines.append(f"{key}  {item.name}　（{count} 条，占 {count / total:.0%}，跨 {days} 天）")
        lines.append(f"    判据：{item.criterion}")
        lines.append(f"    提醒句：{item.reminder}")
        for exclusion in item.excludes:
            target = names.get(exclusion.goes_to) if exclusion.goes_to is not None else None
            lines.append(f"    不含：{exclusion.text}" + (f"（→{target}）" if target else ""))
        for member in samples.get(key, ()):
            lines.append(f"    · [{member.day.isoformat()}] {member.content.render(max_steps=max_steps)}")
    return "\n".join(lines)


def _entry_schema(class_ids: Sequence[str], *, max_examples: int, allow_new: bool) -> dict[str, Any]:
    text = {"type": "string", "minLength": 1}
    targets = [*class_ids, NEW] if allow_new else list(class_ids)
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": text,
                    "criterion": text,
                    "reminder": text,
                    "excludes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"text": text, "goes_to": {"anyOf": [{"enum": targets}, {"type": "null"}]}},
                            "required": ["text", "goes_to"],
                            "additionalProperties": False,
                        },
                    },
                    "examples": {"type": "array", "items": text, "maxItems": max_examples},
                },
                "required": ["name", "criterion", "reminder", "excludes", "examples"],
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }


def revision_schema(class_ids: Sequence[str], *, max_examples: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "minLength": 1},
                        "kind": {"enum": ["merge", "split", "revise"]},
                        "class": {"enum": list(class_ids)},
                        "target": {"anyOf": [{"enum": list(class_ids)}, {"type": "null"}]},
                        "draft": _entry_schema(class_ids, max_examples=max_examples, allow_new=True),
                        "new_class": _entry_schema(class_ids, max_examples=max_examples, allow_new=False),
                    },
                    "required": ["reason", "kind", "class", "target", "draft", "new_class"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["proposals"],
        "additionalProperties": False,
    }


__all__ = ["NEW", "REVISION_PROMPT_VERSION", "REVISION_SYSTEM_PROMPT", "render_user_message", "revision_schema"]
