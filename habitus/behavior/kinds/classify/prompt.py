"""白天归类的提示词、输入渲染与输出 schema（改措辞只动这一个文件）。

每类给判据、提醒句、不含、代表例子（设计 二）；先写理由再选（理由在 schema 里排在选择前面）。词表从空开始、类只从用户自己的
数据里长出来（裁定 29），"什么算一件事"全靠这里的规则交代，而且**只写规则、不放任何场景的例子**（裁定 30：上线后每个用户的场景
都不同）。规则 2 是可执行的检验——去掉提醒句里的具体对象 / 主题再比（v5；v4 只写"对象不影响"，模型照样按主题拆开）；
规则 3 是裁定 10、13 的"按在完成什么归、不看谁动手"。v2 是撤回的"看目的"那一版，不复用编号。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from habitus.behavior.kinds.classify.request import ClassifyRequest
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import BehaviorClass

CLASSIFY_PROMPT_VERSION = "behavior_kind_classify_v5"
OUTSIDE = "都不是"
NOT_EVENT = "不是一件事"

_LANE_TITLES = {Lane.SESSION: "人与编码助手 / 对话助手的会话", Lane.PHYSICAL: "摄像头、手表等看到的现实活动"}

CLASSIFY_SYSTEM_PROMPT = """你在给行为记录归类。词表里每一类是一个"行为类"，带编号、判据、提醒句，
有的还写了「不含」（这些不算本类、应该去哪一类）和几个代表例子。对每条记录，判断它属于哪一类。

规则：
1. 判断用提醒句：如果要提醒这个人做这件事，会说"该……了"。提醒句和某一类相同，就归那一类。
2. 比提醒句之前，先把里面的具体对象、主题、内容、项目名、人名、地点、时间都去掉，只留"在做哪一类事"；去掉之后相同，就是同一件事。
   只有去掉某个对象之后就说不清在做什么了，这个对象才算这件事的一部分、留在提醒句里。
3. 看在完成什么，不看谁动手：让别人或工具去完成一件事，就是那件事本身；做一件事的过程中顺带问的、查的、确认的，也还是那件事。
   动手之前专门商量这件事该怎么做、与手上的事无关的另一个问题、只是安排或叫停别人而不针对任何具体的事，才各是另一件事。
4. 看整条记录：名字、每段的概要、目标、步骤一起看，不要只看名字。
5. 先看「不含」：记录落在某一类的「不含」里，就不归那一类。
6. 清单里没有合适的类，选「都不是」，并在 proposed 里给一个新类名：按规则 2 去掉对象之后的那件事本身，不带具体对象、主题、项目名、人名。
7. 记录本身不是一件事——无意识的小动作、一个操作步骤或单条命令、别人做的事、打招呼——选「不是一件事」。
8. 拿不准时，宁可选「都不是」，不要勉强归进一类。

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
                            "description": "choice 为「都不是」时必填：新类名，写那件事本身；其余情况填 null。",
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
