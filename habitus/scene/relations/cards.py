"""关系卡片上写什么（裁定 115）：关系表里一行换成卡片字段、行没了卡片怎么办、文件名。纯函数，读写在 ``runtime.relation_cards``。

只做记录、预测与语义树的任何计算都不读卡片。写的行是关系表里跨代带着的（候选、成立、成立后没再出现、前向未复现）与退休的；
每代重算、不跨代的"检验过没过"不写。

- **行在**：状态、日期、读数从这一行来。退休行在表里没有读数与日期：保持旧卡片上的（第一次上线时已经退休、没有旧卡片的，
  读数与日期是空）。
- **行没了**：成立过的（这一段成立过，或"以前成立过"里有记录）卡片留着，状态写成"成立过、已结束"，结束日期是第一次发现它
  没了的那一代的关系表的日子；从没成立过的跟着删。
- **以前成立过**（``past``）：一条关系被清掉、或失效 / 退休之后又重新当候选，表里是新建的一行，旧的日期都丢了。卡片上这一段
  成立的日子换掉时，把旧的那段（成立日、结束日）记进 ``past``，接着写新的。
- 字段里只有日期、整数、布尔、字符串与固定四位的小数，没有元组，不放这一代的日子与写入时刻：同一张表重算得到同一组字段，
  "没变不写"才判得出来。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import date
from typing import Any

from habitus.scene.relations.conditions import Field, Template, parse_conditions
from habitus.scene.relations.engine import RelationKey
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.state import Relation, Status

#: 关系表里的行之外，卡片多一种状态：成立过、行后来被清掉了。
ENDED = "ended"
#: 卡片上的指向（地址由读写那一层按树的规矩拼：后果那个词的关系目录、前因与后果的习惯卡片）。
POINTERS = ("next", "antecedent_habit", "consequent_habit")
_READINGS = ("antecedents", "units", "treated_rate", "control_rate", "difference")
_DATES = ("discovered_on", "established_on", "ended_on", "rejected_on", "retired_on")
CARD_KEYS = frozenset(
    {
        "lane",
        "antecedent",
        "antecedent_name",
        "consequent",
        "consequent_name",
        "segment",
        "condition",
        "upward",
        "status",
        "past",
        *_DATES,
        *_READINGS,
        *POINTERS,
    }
)
#: 条件进文件名用的摘要长度（十六进制位数）。
_DIGEST_CHARS = 12

_STATUS_TEXT = {
    Status.CANDIDATE.value: "还在观察",
    Status.ESTABLISHED.value: "已经成立",
    Status.EXPIRED.value: "成立过、后来没再出现",
    Status.REJECTED.value: "观察过、没被证实",
    Status.RETIRED.value: "相关的事很久没出现，先不跟了",
    ENDED: "成立过、后来没再出现，已不再跟踪",
}
_SEGMENT_TEXT = {
    Segment.CHAIN: "紧接着",
    Segment.REST_OF_DAY: "当天之内",
    Segment.NEXT_DAY: "第二天",
    Segment.DAYS_2_7: "之后两到七天里",
    Segment.FIRST_SEEN: "第一次出现之后",
    Segment.HIGH_STATE: "做得明显比平常多的那段日子里",
    Segment.LOW_STATE: "做得明显比平常少的那段日子里",
    Segment.RETURN: "隔了很久又做之后",
    Segment.SITUATION: "",
}
_LANE_NAMES = {"session": "会话", "physical": "物理"}


def card_location(key: RelationKey) -> tuple[str, str]:
    """（前因目录名, 文件名去掉 ``.md``）：``<后果>--<段>[--<条件摘要>]``。前因、后果是概念身份（路径安全、不含 ``--``），
    段是固定的英文名；条件里有冒号、竖线与任意的地点、人名，换成条件文字 sha256 的前 12 位，完整条件写在字段里。
    同一条关系永远得到同一个名字。"""

    name = f"{key.consequent}--{key.segment.value}"
    if key.condition:
        name += "--" + hashlib.sha256(key.condition.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return key.antecedent, name


def row_fields(
    row: Relation, old: Mapping[str, Any] | None, *, names: Mapping[str, str], pointers: Mapping[str, str | None]
) -> dict[str, Any]:
    """关系表里这一行（不是"检验过没过"）对应的卡片字段。``old`` 是盘上这张卡片的字段（没有是 None）；``names`` 是词的编号 →
    名字（词表里没有的——日历情境、删了的词——沿用旧卡片上的，再没有就印身份本身）。"""

    if row.status in (Status.SPARSE, Status.TESTED):
        raise ValueError("only relations carried across generations, and retired ones, get a card")
    key = row.key
    if row.status is Status.RETIRED:
        # 退休行只有键、方向与退休那一代：读数与日期保持旧卡片上的
        kept = {name: None if old is None else old[name] for name in (*_READINGS, *_DATES)}
        dates = {**kept, "retired_on": _day(row.retired_on)}
        readings = {name: kept[name] for name in _READINGS}
    else:
        dates = {
            "discovered_on": _day(row.discovered_on),
            "established_on": _day(row.established_on),
            "ended_on": _day(row.expired_on),
            "rejected_on": _day(row.rejected_on),
            "retired_on": None,
        }
        readings = _readings(row)
    past = [] if old is None else list(old["past"])
    if old is not None and old["established_on"] is not None and old["established_on"] != dates["established_on"]:
        # 这一段成立的日子换掉了（行被清掉或失效、退休后重新当候选是新建一行）：旧的那段记进"以前成立过"
        past.append({"established_on": old["established_on"], "ended_on": old["ended_on"]})
    return {
        "lane": key.lane,
        "antecedent": key.antecedent,
        "antecedent_name": _name(key.antecedent, names, old, "antecedent_name"),
        "consequent": key.consequent,
        "consequent_name": _name(key.consequent, names, old, "consequent_name"),
        "segment": key.segment.value,
        "condition": key.condition,
        "upward": row.upward,
        "status": row.status.value,
        "past": past,
        **dates,
        **readings,
        **{name: pointers[name] for name in POINTERS},
    }


def gone_fields(old: Mapping[str, Any], night: date) -> dict[str, Any] | None:
    """卡片在、关系表（第 ``night`` 代）里已经没有这一行：成立过的留着、标"已结束"与结束日期（已经标过的不动）；从没成立过的
    是 None（删卡片）。"""

    if old["established_on"] is None and not old["past"]:
        return None
    if old["status"] == ENDED:
        return dict(old)
    ended = old["ended_on"] if old["ended_on"] is not None else night.isoformat()
    return {**old, "status": ENDED, "ended_on": ended}


def render_card(fields: Mapping[str, Any]) -> str:
    """卡片正文：给人看的几行白话，全部由字段拼出。"""

    antecedent, consequent = f"「{fields['antecedent_name']}」", f"「{fields['consequent_name']}」"
    more = "更常" if fields["upward"] else "更少"
    segment = Segment(fields["segment"])
    if segment is Segment.SITUATION:
        sentence = f"在{antecedent}这样的日子里，{more}做{consequent}。"
    elif segment.long:
        sentence = f"{antecedent}{_SEGMENT_TEXT[segment]}，{more}做{consequent}。"
    else:
        sentence = f"做完{antecedent}之后，{_SEGMENT_TEXT[segment]}{more}做{consequent}。"
    lines = [
        f"# {fields['antecedent_name']} → {fields['consequent_name']}",
        "",
        sentence,
        "",
        f"- 前因：{fields['antecedent']}；后果：{fields['consequent']}（{_LANE_NAMES.get(fields['lane'], fields['lane'])}）",
        f"- 条件：{_conditions(fields['condition'])}",
        f"- 状态：{_STATUS_TEXT[fields['status']]}",
        f"- 开始观察：{_or_unknown(fields['discovered_on'])}；成立：{_or_none(fields['established_on'])}；"
        f"结束：{_or_none(fields['ended_on'])}；没被证实：{_or_none(fields['rejected_on'])}；"
        f"不再跟踪：{_or_none(fields['retired_on'])}",
    ]
    lines.extend(
        f"- 以前成立过：{period['established_on']} 起，到 {_or_unknown(period['ended_on'])}"
        for period in fields["past"]
    )
    if fields["antecedents"] is None:
        lines.append("- 读数：没有（退休时表里不留读数）")
    else:
        lines.append(
            f"- 读数：{antecedent}一共 {fields['antecedents']} 次，算进比较的 {fields['units']} 次；"
            f"有它时{consequent}来的比例 {_percent(fields['treated_rate'])}，没有它时 {_percent(fields['control_rate'])}，"
            f"差 {_percent(fields['difference'])}"
        )
    lines.append(f"- 接下来：{fields['next']}")
    habits = [value for value in (fields["antecedent_habit"], fields["consequent_habit"]) if value is not None]
    lines.append(f"- 习惯卡片：{'；'.join(habits)}")
    return "\n".join(lines) + "\n"


def _readings(row: Relation) -> dict[str, Any]:
    test = row.tonight
    if test is None:
        return dict.fromkeys(_READINGS)
    treated, control = round(test.effect.treated_rate, 4), round(test.effect.control_rate, 4)
    return {
        "antecedents": test.antecedents,
        "units": test.effect.antecedents,
        "treated_rate": treated,
        "control_rate": control,
        "difference": round(test.effect.treated_rate - test.effect.control_rate, 4),
    }


def _name(identity: str, names: Mapping[str, str], old: Mapping[str, Any] | None, field: str) -> str:
    if identity in names:
        return names[identity]
    return str(old[field]) if old is not None else identity


def _conditions(text: str) -> str:
    if not text:
        return "无"
    parts: list[str] = []
    for item in parse_conditions(text):
        if item.template is Template.FIELD:
            parts.append(f"在「{item.value}」" if item.field is Field.PLACE else f"和「{item.value}」一起")
        elif item.template is Template.TIMELINE:
            parts.append(f"之前做过 {item.value}")
        else:
            parts.append(f"那天是「{item.value}」")
    return "、".join(parts) + f"（{text}）"


def _day(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _or_none(value: str | None) -> str:
    return "—" if value is None else value


def _or_unknown(value: str | None) -> str:
    return "不知道" if value is None else value


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


__all__ = ["CARD_KEYS", "ENDED", "POINTERS", "card_location", "gone_fields", "render_card", "row_fields"]
