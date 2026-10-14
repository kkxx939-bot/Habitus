"""定期拆改的分组（`18` 第四节，2026-10-12）：一个类的近期成员整体交给模型按"同一件事"分组，每组带一份条目。

模型只答"哪些记录是同一件事"，不答"该不该拆"——拆不拆由 ``reviser`` 按每组的复现（条数、跨天数）定。
分组用的规则文本与白天归类、每晚新增同一段（``kinds/rules.py``）。成员多于 ``batch_size`` 时分批问，
各批里提醒句相同（去掉标点与空白后逐字相同）的组合并；任何一批没答成就整类放弃这次分组——半份分组会把复现数算错。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller, KindModelOutputError
from habitus.behavior.kinds.ids import ClassId
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError
from habitus.behavior.kinds.revision.facts import Member
from habitus.behavior.kinds.revision.proposals import Draft
from habitus.behavior.kinds.rules import ENTRY_FORMAT, RULES_VERSION, SAMENESS_RULE

GROUPING_PROMPT_VERSION = f"behavior_kind_grouping_v1+{RULES_VERSION}"

GROUPING_SYSTEM_PROMPT = f"""下面这些行为记录现在都归在同一个行为类里（先给你这个类的条目，再给同一条 lane 里别的类做参考）。
请把这些记录按"同一件事"分组：每条记录恰好属于一组，一条自成一组也可以。

{SAMENESS_RULE}

如果它们按替代检验确实都是同一件事，就只分一组。
不要判断该不该拆、哪组更重要——这不由你定。
一组记录按替代检验就是参考清单里已有的某一类（它的提醒句与那一类的能互相替代），existing 填那一类的编号（C…）：
这些记录该搬到那一类去，不另起新类，条目照那一类写即可。
每组按下面写出条目；excludes 的 goes_to 填参考清单里那一类的编号，指不出就填 null。

{ENTRY_FORMAT}
"""

_NOISE = re.compile(r"[\s，。、,.!！？?「」\"'“”]+")


@dataclass(frozen=True)
class Group:
    """模型分出的一组：成员与这组的条目。"""

    members: tuple[Member, ...]
    draft: Draft
    #: 模型说这一组就是词表里已有的这一类：成员搬过去，不另起新类
    into: ClassId | None = None

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def days(self) -> int:
        return len({member.day for member in self.members})

    def recurs(self, *, min_count: int, min_days: int) -> bool:
        return self.count >= min_count and self.days >= min_days


def render_grouping_message(
    item: BehaviorClass,
    others: Mapping[str, BehaviorClass],
    members: Mapping[str, Member],
    *,
    earlier: Mapping[str, Group] = MappingProxyType({}),
    max_steps: int,
) -> str:
    """``others``：参考清单的编号（C…）→ 同 lane 别的在用类；``members``：记录编号（M…）→ 成员；
    ``earlier``：前几批已分出的组（G…）→ 组，这一批里是同一件事的记录归进去而不是另起一组。"""

    lines = ["## 这个类", f"{item.name}", f"    判据：{item.criterion}", f"    提醒句：{item.reminder}"]
    for exclusion in item.excludes:
        lines.append(f"    不含：{exclusion.text}")
    lines += ["", "## 同一条 lane 里别的类（excludes 的去向；一组记录就是其中某一类时 existing 填它的编号）"]
    for key, other in others.items():
        lines.append(f"{key}  {other.name}：{other.criterion}")
    if not others:
        lines.append("（无）")
    if earlier:
        lines += ["", "## 前几批已分出的组（这一批里按替代检验是同一件事的，existing 填它的编号、归进去）"]
        for key, group in earlier.items():
            lines.append(f"{key}  {group.draft.name}：{group.draft.reminder}")
    lines += ["", "## 记录"]
    for key, member in members.items():
        lines.append(f"{key}  [{member.day.isoformat()}] {member.content.render(max_steps=max_steps)}")
    return "\n".join(lines)


def grouping_schema(
    member_ids: Sequence[str], class_ids: Sequence[str], *, group_ids: Sequence[str] = (), max_examples: int
) -> dict[str, Any]:
    nullable_class = {"anyOf": [{"enum": list(class_ids)}, {"type": "null"}]} if class_ids else {"type": "null"}
    targets = [*group_ids, *class_ids]  # existing：前几批的组（G…）或词表里已有的类（C…）
    nullable_group = {"anyOf": [{"enum": targets}, {"type": "null"}]} if targets else {"type": "null"}
    text = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "members": {"type": "array", "items": {"enum": list(member_ids)}, "minItems": 1},
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
                        "existing": nullable_group,
                    },
                    "required": ["members", "name", "criterion", "reminder", "excludes", "examples", "existing"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["groups"],
        "additionalProperties": False,
    }


async def group_members(
    caller: KindModelCaller,
    item: BehaviorClass,
    others: Sequence[BehaviorClass],
    members: Sequence[Member],
    *,
    batch_size: int,
    max_steps: int,
    max_examples: int,
) -> tuple[tuple[Group, ...], int, tuple[str, ...]]:
    """返回（分出的组，模型调用次数，信号）。没答成时组为空、信号里说明；调用方据此跳过这个类。

    分批时后一批看得见前几批的组（编号 G…），答 ``existing`` 就归进去；没答 existing 而提醒句逐字相同的也合并。
    ``existing`` 答成词表里已有的类（C…）的组带 ``into``：各批里搬去同一类的合成一组。
    """

    ordered = sorted(members, key=lambda member: (member.day, member.occurrence))
    keyed_others = {f"C{i}": other for i, other in enumerate((o for o in others if o.id != item.id), start=1)}
    merged: list[Group] = []
    calls = 0
    for start in range(0, len(ordered), batch_size):
        batch = {f"M{i}": member for i, member in enumerate(ordered[start : start + batch_size], start=1)}
        earlier = {f"G{i}": group for i, group in enumerate(merged, start=1)}
        calls += 1
        try:
            raw = await _ask(caller, item, keyed_others, batch, earlier, max_steps=max_steps, max_examples=max_examples)
        except KindModelOutputError as exc:
            return (), calls, (f"kind_revision_grouping_rejected {item.id}: {exc}",)
        for group, existing in _groups(raw, batch, keyed_others, max_examples=max_examples):
            index = _index_of(merged, earlier, existing, group)
            if index is None:
                merged.append(group)
            else:
                merged[index] = _merge(merged[index], group, max_examples=max_examples)
    return tuple(merged), calls, ()


def _index_of(merged: Sequence[Group], earlier: Mapping[str, Group], existing: str | None, group: Group) -> int | None:
    if existing is not None:
        return merged.index(earlier[existing])
    if group.into is not None:
        return next((i for i, found in enumerate(merged) if found.into == group.into), None)
    key = sentence_key(group.draft.reminder)
    return next(
        (i for i, found in enumerate(merged) if found.into is None and sentence_key(found.draft.reminder) == key), None
    )


async def _ask(
    caller: KindModelCaller,
    item: BehaviorClass,
    others: Mapping[str, BehaviorClass],
    batch: Mapping[str, Member],
    earlier: Mapping[str, Group],
    *,
    max_steps: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    return await caller.call(
        system=GROUPING_SYSTEM_PROMPT,
        user=render_grouping_message(item, others, batch, earlier=earlier, max_steps=max_steps),
        schema=grouping_schema(tuple(batch), tuple(others), group_ids=tuple(earlier), max_examples=max_examples),
        name="behavior_kind_grouping",
        validator=lambda parsed: _validated(parsed, batch, others, earlier),
    )


def _validated(
    parsed: object, keyed: Mapping[str, Member], classes: Mapping[str, BehaviorClass], earlier: Mapping[str, Group]
) -> list[dict[str, Any]]:
    """每条记录恰好在一组；编号都认识。"""

    if not isinstance(parsed, Mapping) or set(parsed) != {"groups"} or not isinstance(parsed["groups"], list):
        raise BehaviorKindError("grouping output must be an object with a `groups` list")
    seen: list[str] = []
    for group in parsed["groups"]:
        if not isinstance(group, Mapping):
            raise BehaviorKindError("grouping group must be an object")
        members = group.get("members")
        if not isinstance(members, list) or not members or any(member not in keyed for member in members):
            raise BehaviorKindError("grouping group members must be listed records")
        seen += members
        for name in ("name", "criterion", "reminder"):
            if not isinstance(group.get(name), str) or not group[name].strip():
                raise BehaviorKindError(f"grouping group {name} must be non-empty text")
        existing = group.get("existing")
        if existing is not None and existing not in earlier and existing not in classes:
            raise BehaviorKindError("grouping existing must be an earlier group, a listed class or null")
        for exclusion in group.get("excludes") or []:
            if not isinstance(exclusion, Mapping) or (
                exclusion.get("goes_to") is not None and exclusion["goes_to"] not in classes
            ):
                raise BehaviorKindError("grouping exclusion target must be a listed class or null")
    if sorted(seen) != sorted(keyed):
        raise BehaviorKindError("every record must be in exactly one group")
    return [dict(group) for group in parsed["groups"]]


def _groups(
    raw: Sequence[Mapping[str, Any]],
    keyed: Mapping[str, Member],
    classes: Mapping[str, BehaviorClass],
    *,
    max_examples: int,
) -> list[tuple[Group, str | None]]:
    found: list[tuple[Group, str | None]] = []
    for group in raw:
        excludes: list[tuple[str, ClassId | str | None]] = [
            (str(exclusion["text"]).strip(), classes[exclusion["goes_to"]].id if exclusion["goes_to"] else None)
            for exclusion in group["excludes"]
        ]
        draft = Draft(
            name=str(group["name"]).strip(),
            criterion=str(group["criterion"]).strip(),
            reminder=str(group["reminder"]).strip(),
            excludes=tuple(excludes),
            examples=tuple(dict.fromkeys(str(text).strip() for text in group["examples"]))[:max_examples],
        )
        existing = group.get("existing")
        into = classes[existing].id if isinstance(existing, str) and existing in classes else None
        found.append((Group(tuple(keyed[key] for key in group["members"]), draft, into), None if into else existing))
    return found


def _merge(first: Group, second: Group, *, max_examples: int) -> Group:
    examples = tuple(dict.fromkeys((*first.draft.examples, *second.draft.examples)))[:max_examples]
    excludes = tuple(dict.fromkeys((*first.draft.excludes, *second.draft.excludes)))
    draft = Draft(
        name=first.draft.name,
        criterion=first.draft.criterion,
        reminder=first.draft.reminder,
        excludes=excludes,
        examples=examples,
    )
    return Group((*first.members, *second.members), draft, first.into)


def sentence_key(text: str) -> str:
    """两句提醒去掉空白与标点后逐字相同就当同一句（与 ``evidence`` 的"提醒句相同"同一口径）。"""

    return _NOISE.sub("", text).casefold()


__all__ = [
    "GROUPING_PROMPT_VERSION",
    "GROUPING_SYSTEM_PROMPT",
    "Group",
    "group_members",
    "grouping_schema",
    "sentence_key",
]
