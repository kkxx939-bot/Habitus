"""定期拆改的提案提示词与 schema：模型看各类占比、跨天复现、抽样原话，只提合并与改判据。

"同一件事"与白天归类、每晚新增同一段文本（``kinds/rules.py``），只写规则、不放场景例子（裁定 30）。
v3（2026-10-12）：**拆分不再由这里提**——v2 让模型按"去掉对象之后仍不同"判该不该拆，35 晚 4 次提案 0 次拆分
（`17-词表的问题.md` 第 12 条）；拆分改为 ``grouping`` 对每个类的近期成员分组、``reviser`` 按复现定（`18` 第四节）。
v2 去掉了"个人习惯可以从通用类里切出来"——没有预置的通用类了（裁定 29），类本来就是这个人自己的。
``NEW`` 仍保留：收窄的原类的「不含」可以指向本次拆出来的新类（由 ``reviser`` 机械写入，模型不再填）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from habitus.behavior.kinds.model import BehaviorClass
from habitus.behavior.kinds.revision.facts import Member
from habitus.behavior.kinds.rules import ENTRY_FORMAT, RULES_VERSION, SAMENESS_RULE

REVISION_PROMPT_VERSION = f"behavior_kind_revision_v3+{RULES_VERSION}"
NEW = "NEW"

REVISION_SYSTEM_PROMPT = f"""你在检查一份行为类词表还合不合现状。给你同一条 lane 的全部在用类：判据、提醒句、
不含、这段时间归进来的条数与占比、出现在多少个不同的日子，以及抽样的原话。

{SAMENESS_RULE}

只提必要的改动，没有就返回空列表。可以提两种：
- merge：两类按替代检验其实是同一件事。class 填要并掉的那一类，target 填并入的那一类。
- revise：只是措辞不清楚、或缺一条「不含」导致和别的类混。class 填那一类，draft 按下面写改后的条目。
拆分不在这里提：一个类里混着几件事，由另一步按记录分组、按复现决定。

{ENTRY_FORMAT}
excludes 的 goes_to 填词表里那一类的编号，指不出就填 null。

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
                        "kind": {"enum": ["merge", "revise"]},
                        "class": {"enum": list(class_ids)},
                        "target": {"anyOf": [{"enum": list(class_ids)}, {"type": "null"}]},
                        "draft": _entry_schema(class_ids, max_examples=max_examples, allow_new=False),
                    },
                    "required": ["reason", "kind", "class", "target", "draft"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["proposals"],
        "additionalProperties": False,
    }


__all__ = ["NEW", "REVISION_PROMPT_VERSION", "REVISION_SYSTEM_PROMPT", "render_user_message", "revision_schema"]
