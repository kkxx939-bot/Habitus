"""把模型的归组输出装配成可落盘的草稿：只校验我们自己产物的自洽，违反者**降级留信号**。

沿融合装配层的既定纪律：记账疏漏（指向不存在的情景、边目标晚于本情景、待用前提的产生方
不是成员、重复项）是组合式记账不是语义判断，实测提示词压不住，一律降级（丢那一项、置 null），
不整批拒——整批拒要白烧一次完整调用。**不校验**现实的形状：情景只有一个成员、没有 effects、
成员跨了午夜都照常收。

同规范身份（NFC + casefold 后的标签）且首成员同刻的两个情景机械合并（它们在地址上就是同一件
事），成员、留下的改变、待用前提与边一并并入。文本先按存储层同一口径清洗（去不可打印字符、折叠
空白），清不出内容的那一项丢弃留信号——模型输出里一个零宽字符不能让整天失败。

装配是一条小流水线：排列核对 → 收情景 → 分成员 → 丢空情景 → 同身份合并 → 逐情景整理 effects /
待用前提 / 边。每一步只读上一步的产物，信号统一收在 ``_Signals`` 里。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from habitus.foundation.ids import canonical_path_identity
from habitus.scene.grouping.model import DraftRelation, GroupingAssembly, SceneDraft, SceneGroupingInput
from habitus.scene.model import SceneLinkType, SceneRole, scene_label


class SceneGroupingAssemblyError(ValueError):
    """模型输出的形状连降级都救不回来（穷尽性被破坏）。"""


@dataclass
class _Signals:
    notes: list[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        self.notes.append(note)


@dataclass
class _Scene:
    """流水线中途的一件事：原始输出 + 清洗过的标签 + 成员。"""

    no: int
    raw: dict[str, Any]
    label: str
    members: list[tuple[int, SceneRole]] = field(default_factory=list)

    def start(self, payload: SceneGroupingInput) -> datetime:
        return min(payload.row(no).started_at.astimezone(UTC) for no, _ in self.members)

    @property
    def member_nos(self) -> set[int]:
        return {no for no, _ in self.members}


def assemble_grouping(parsed: object, payload: SceneGroupingInput) -> GroupingAssembly:
    if not isinstance(parsed, Mapping):
        raise SceneGroupingAssemblyError("grouping output must be an object")
    raw_scenes = parsed.get("scenes")
    raw_assignments = parsed.get("assignments")
    if not isinstance(raw_scenes, list) or not isinstance(raw_assignments, list):
        raise SceneGroupingAssemblyError("grouping output must carry scenes and assignments arrays")
    signals = _Signals()
    assignments = _assignments_in_order(raw_assignments, len(payload.occurrences), signals)
    scenes = _admit_scenes(raw_scenes, signals)
    unassigned = _assign_members(assignments, scenes, signals)
    scenes = _drop_empty(scenes, signals)
    scenes, merged_into = _merge_same_identity(scenes, payload, signals)
    ordered = sorted(scenes.values(), key=lambda scene: (scene.start(payload), scene.no))
    index_of = {scene.no: index for index, scene in enumerate(ordered)}
    drafts = tuple(
        SceneDraft(
            label=scene.label,
            members=tuple(sorted(scene.members, key=lambda pair: (payload.row(pair[0]).started_at.astimezone(UTC), pair[0]))),
            effects=_effects(scene, signals),
            pending_effects=_pending_effects(scene, signals),
            relations=_relations(scene, payload, index_of, merged_into, signals),
        )
        for scene in ordered
    )
    return GroupingAssembly(scenes=drafts, unassigned=tuple(unassigned), signals=tuple(signals.notes))


# ── 流水线各步 ───────────────────────────────────────────────────────────────


def _assignments_in_order(raw: list[Any], count: int, signals: _Signals) -> list[Mapping[str, Any]]:
    """穷尽性：形状已由 schema 钉死长度；编号是唯一的对应关系——是 1..N 的排列就机械排好，
    不是排列（缺号/重号）才整批不可用。"""

    nos = [item.get("no") if isinstance(item, Mapping) else None for item in raw]
    if sorted(nos, key=lambda value: (not isinstance(value, int), value)) != list(range(1, count + 1)):
        raise SceneGroupingAssemblyError(f"assignments must cover occurrences 1..{count} exactly once")
    rows = [item for item in raw if isinstance(item, Mapping)]
    if nos != list(range(1, count + 1)):
        signals.add("assignments_reordered: rows were not in input order")
        rows.sort(key=lambda item: int(item["no"]))
    return rows


def _admit_scenes(raw: list[Any], signals: _Signals) -> dict[int, _Scene]:
    scenes: dict[int, _Scene] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("scene_no"), int):
            signals.add("scene_dropped: malformed scene entry")
            continue
        no = int(item["scene_no"])
        if no in scenes:
            signals.add(f"scene_dropped: duplicate scene_no {no}")
            continue
        label = clean_line(item.get("label"))
        try:
            scene_label(label, "scene label")
        except (TypeError, ValueError) as exc:
            signals.add(f"scene_dropped: label {item.get('label')!r} unusable as an address ({exc})")
            continue
        scenes[no] = _Scene(no=no, raw=dict(item), label=label)
    return scenes


def _assign_members(assignments: list[Mapping[str, Any]], scenes: dict[int, _Scene], signals: _Signals) -> list[int]:
    unassigned: list[int] = []
    for item in assignments:
        no = int(item["no"])
        scene_no = item.get("scene_no")
        role = item.get("role")
        if scene_no is None:
            if role is not None:
                signals.add(f"role_cleared: #{no} has a role but no scene")
            unassigned.append(no)
            continue
        if scene_no not in scenes:
            signals.add(f"assignment_degraded: #{no} points at unknown scene {scene_no}")
            unassigned.append(no)
            continue
        try:
            resolved_role = SceneRole(role) if role is not None else SceneRole.OPTIONAL
        except ValueError:
            resolved_role = SceneRole.OPTIONAL
            signals.add(f"role_degraded: #{no} role {role!r} treated as optional")
        if role is None:
            signals.add(f"role_degraded: #{no} has a scene but no role, treated as optional")
        scenes[int(scene_no)].members.append((no, resolved_role))
    return unassigned


def _drop_empty(scenes: dict[int, _Scene], signals: _Signals) -> dict[int, _Scene]:
    kept: dict[int, _Scene] = {}
    for no, scene in scenes.items():
        if scene.members:
            kept[no] = scene
        else:
            signals.add(f"scene_dropped: scene {no} «{scene.raw.get('label')}» has no member")
    return kept


def _merge_same_identity(
    scenes: dict[int, _Scene], payload: SceneGroupingInput, signals: _Signals
) -> tuple[dict[int, _Scene], dict[int, int]]:
    """同规范身份同刻合并：地址就是同一件事；成员之外，effects / pending / relations 一并并入。"""

    merged_into: dict[int, int] = {}
    by_identity: dict[tuple[str, datetime], int] = {}
    for no in sorted(scenes):
        scene = scenes[no]
        key = (canonical_path_identity(scene.label, "scene label"), scene.start(payload))
        if key not in by_identity:
            by_identity[key] = no
            continue
        target = scenes[by_identity[key]]
        target.members.extend(scene.members)
        for name in ("effects", "pending_effects", "relations"):
            target.raw[name] = [*_list(target.raw.get(name)), *_list(scene.raw.get(name))]
        merged_into[no] = target.no
        signals.add(f"scene_merged: scene {no} merged into {target.no} (same label and start)")
    return {no: scene for no, scene in scenes.items() if no not in merged_into}, merged_into


def _effects(scene: _Scene, signals: _Signals) -> tuple[str, ...]:
    effects: list[str] = []
    for text in _list(scene.raw.get("effects")):
        line = clean_line(text)
        if not line:
            signals.add(f"effect_dropped: scene {scene.no} has an empty effect")
        elif line in effects:
            signals.add(f"effect_dropped: scene {scene.no} repeats «{line}»")
        else:
            effects.append(line)
    return tuple(effects)


def _pending_effects(scene: _Scene, signals: _Signals) -> tuple[tuple[str, int], ...]:
    pending: list[tuple[str, int]] = []
    for item in _list(scene.raw.get("pending_effects")):
        if not isinstance(item, Mapping) or not isinstance(item.get("occurrence_no"), int):
            signals.add(f"pending_dropped: scene {scene.no} has a malformed pending effect")
            continue
        producer = int(item["occurrence_no"])
        line = clean_line(item.get("text"))
        if not line:
            signals.add(f"pending_dropped: scene {scene.no} has an empty pending effect")
        elif producer not in scene.member_nos:
            signals.add(f"pending_dropped: scene {scene.no} pending «{line}» names non-member #{producer}")
        elif (line, producer) in pending:
            signals.add(f"pending_dropped: scene {scene.no} repeats pending «{line}»")
        else:
            pending.append((line, producer))
    return tuple(pending)


def _relations(
    scene: _Scene,
    payload: SceneGroupingInput,
    index_of: Mapping[int, int],
    merged_into: Mapping[int, int],
    signals: _Signals,
) -> tuple[DraftRelation, ...]:
    relations: list[DraftRelation] = []
    seen: set[tuple[str, str, int]] = set()
    for item in _list(scene.raw.get("relations")):
        resolved = _relation(item, scene, payload, index_of, merged_into, signals)
        if resolved is None:
            continue
        key = (resolved.kind.value, *_relation_key(resolved))
        if key in seen:
            signals.add(f"relation_dropped: scene {scene.no} repeats a {resolved.kind.value} edge")
            continue
        seen.add(key)
        relations.append(resolved)
    return tuple(relations)


def _relation(
    item: object,
    scene: _Scene,
    payload: SceneGroupingInput,
    index_of: Mapping[int, int],
    merged_into: Mapping[int, int],
    signals: _Signals,
) -> DraftRelation | None:
    """一条边：目标恰好一个，且不晚于本情景首成员（同秒放行，与 link 层 lag ≥ 0 同口径）。"""

    if not isinstance(item, Mapping):
        signals.add(f"relation_dropped: scene {scene.no} has a malformed relation")
        return None
    try:
        kind = SceneLinkType(item.get("kind"))
    except ValueError:
        signals.add(f"relation_dropped: scene {scene.no} uses unknown relation kind {item.get('kind')!r}")
        return None
    targets = {name: item.get(name) for name in ("occurrence_no", "scene_no", "reference_no", "pending_no")}
    filled = [name for name, value in targets.items() if value is not None]
    if len(filled) != 1:
        signals.add(f"relation_dropped: scene {scene.no} {kind.value} names {len(filled)} targets")
        return None
    name = filled[0]
    value = targets[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        signals.add(f"relation_dropped: scene {scene.no} {kind.value} target must be a positive integer")
        return None
    scene_start = scene.start(payload)
    prefix = f"relation_dropped: scene {scene.no} {kind.value}"
    if name == "occurrence_no":
        if value > len(payload.occurrences):
            signals.add(f"{prefix} points at unknown #{value}")
            return None
        if value in scene.member_nos or payload.row(value).started_at.astimezone(UTC) > scene_start:
            signals.add(f"{prefix} points at #{value}, not earlier than the scene")
            return None
        return DraftRelation(kind=kind, occurrence_no=value)
    if name == "scene_no":
        target = merged_into.get(value, value)
        if target == scene.no:
            signals.add(f"{prefix} points at itself")
            return None
        if target not in index_of:
            signals.add(f"{prefix} points at unknown scene {value}")
            return None
        if index_of[target] >= index_of[scene.no]:
            signals.add(f"{prefix} points at scene {value}, not earlier")
            return None
        return DraftRelation(kind=kind, scene_index=index_of[target])
    if name == "reference_no":
        if value > len(payload.scene_references):
            signals.add(f"{prefix} points at unknown reference C{value}")
            return None
        if payload.scene_references[value - 1].started_at.astimezone(UTC) > scene_start:
            signals.add(f"{prefix} points at C{value}, not earlier")
            return None
        return DraftRelation(kind=kind, reference_no=value)
    if value > len(payload.pending_references):
        signals.add(f"{prefix} points at unknown pending P{value}")
        return None
    if payload.pending_references[value - 1].producer_started_at.astimezone(UTC) > scene_start:
        signals.add(f"{prefix} points at P{value}, not earlier")
        return None
    return DraftRelation(kind=kind, pending_no=value)


# ── 小工具 ───────────────────────────────────────────────────────────────────


def clean_line(value: object) -> str:
    """与存储层 ``schema/fields.py::line`` 同口径的清洗：去不可打印字符、折叠空白、去首尾；非文本得空串。"""

    if not isinstance(value, str):
        return ""
    printable = "".join(character if character.isprintable() else " " for character in value)
    return " ".join(printable.split())


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _relation_key(relation: DraftRelation) -> tuple[str, int]:
    for name in ("occurrence_no", "scene_index", "reference_no", "pending_no"):
        value = getattr(relation, name)
        if value is not None:
            return name, int(value)
    raise AssertionError("unreachable: a draft relation always names one target")


__all__ = ["SceneGroupingAssemblyError", "assemble_grouping", "clean_line"]
