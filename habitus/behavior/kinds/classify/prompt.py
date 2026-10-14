"""白天归类的提示词、输入渲染与输出 schema（改措辞只动这一个文件；"什么算同一件事"的文本在 ``kinds/rules.py``）。

每类给判据、提醒句、不含、代表例子（设计 二）；先写理由再选（理由在 schema 里排在选择前面）。词表从空开始、类只从用户自己的
数据里长出来（裁定 29），"什么算一件事"全靠规则交代，而且**只写规则、不放任何场景的例子**（裁定 30：上线后每个用户的场景
都不同）。v6（2026-10-12）：删掉 v5 的"去掉具体对象再比"——它把订机票 + 订酒店、看日程 + 查快递、刷牙 + 洗手都并成一类
（`17-词表的问题.md` 第 7、12 条）；改用 ``rules.SAMENESS_RULE`` 的替代检验，归类只问"记录的提醒句与类的提醒句能否互相替代"。
v2 是撤回的"看目的"那一版，不复用编号。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from habitus.behavior.kinds.classify.request import ClassifyRequest
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import BehaviorClass
from habitus.behavior.kinds.rules import RULES_VERSION, SAMENESS_RULE

CLASSIFY_PROMPT_VERSION = f"behavior_kind_classify_v7+{RULES_VERSION}"
OUTSIDE = "都不是"
NOT_EVENT = "不是一件事"

_LANE_TITLES = {Lane.SESSION: "人与编码助手 / 对话助手的会话", Lane.PHYSICAL: "摄像头、手表等看到的现实活动"}

CLASSIFY_SYSTEM_PROMPT = f"""你在给行为记录归类。词表里每一类是一个"行为类"，带编号、判据、提醒句，
有的还写了「不含」（这些不算本类、应该去哪一类）和几个代表例子。对每条记录，判断它属于哪一类。

规则：
1. 判断用提醒句：先写出提醒这个人做这条记录里的事会说的那句话（"该……了"）。它和某一类的提醒句能互相替代
   （做了这条记录里的事，就算做到了那一类的提醒；反过来也算），就归那一类。
2. 怎么判"能互相替代"，按下面【什么算一件事、什么算同一件事】。
3. 看整条记录：名字、每段的概要、目标、步骤一起看，不要只看名字。
4. 先看「不含」：记录落在某一类的「不含」里，就不归那一类。
5. 清单里没有能互相替代的类，选「都不是」，并在 proposed 里给一个新类名：动作加上对着的种类（替别人办的带上角色；你只看得到这一条，
   种类先带上，粗细由每晚整理待定池时按反复不反复定）；不写具体是哪一个。
6. 记录本身不是一件事——只是正在做的某件事里的一步（继续、重试、一条命令），没有自己的目标；无意识的小动作；别人做的事；打招呼；
   只是一句感受——选「不是一件事」。一步就做完但有自己目标的事（查一下快递到哪了）是一件事。
7. 拿不准时，宁可选「都不是」，不要勉强归进一类。

{SAMENESS_RULE}

每条记录先在 reason 里写一句理由，再在 choice 里选：词表里某一类的编号、「都不是」或「不是一件事」。
"""


def render_user_message(
    lane: Lane, classes: Mapping[str, BehaviorClass], items: Mapping[str, ClassifyRequest], *, max_steps: int
) -> str:
    """``classes``：提示词里的类编号（C1…）→ 行为类；``items``：记录编号（R1…）→ 请求。"""

    names = {item.id: item.name for item in classes.values()}
    lines = [f"## 词表（{_LANE_TITLES[Lane(lane)]}）"]
    if not classes:
        lines.append("（还没有类：每条记录只判是不是一件事，是的话选「都不是」并给一个新类名）")
    for key, item in classes.items():
        lines.append(f"{key}  {item.name}")
        lines.append(f"    判据：{item.criterion}")
        lines.append(f"    提醒句：{item.reminder}")
        for exclusion in item.excludes:
            target = names.get(exclusion.goes_to) if exclusion.goes_to is not None else None
            lines.append(f"    不含：{exclusion.text}" + (f"（→{target}）" if target else ""))
        if item.examples:
            lines.append(f"    例：{'；'.join(item.examples)}")
    lines += ["", "## 待归类的记录"]
    for key, request in items.items():
        lines.append(f"{key}  {request.content.render(max_steps=max_steps)}")
    return "\n".join(lines)


def classify_schema(item_ids: Sequence[str], class_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "record": {"enum": list(item_ids)},
                        "reason": {"type": "string", "minLength": 1},
                        "choice": {"enum": [*class_ids, OUTSIDE, NOT_EVENT]},
                        "proposed": {
                            "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
                            "description": "choice 为「都不是」时必填：新类名，写动作和对着的种类、不写具体是哪一个；其余情况填 null。",
                        },
                    },
                    "required": ["record", "reason", "choice", "proposed"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


__all__ = [
    "CLASSIFY_PROMPT_VERSION",
    "CLASSIFY_SYSTEM_PROMPT",
    "NOT_EVENT",
    "OUTSIDE",
    "classify_schema",
    "render_user_message",
]
