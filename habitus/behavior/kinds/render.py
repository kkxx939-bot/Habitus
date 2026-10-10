"""把一版词表渲染成给人看的清单（``kinds.md``）；只是视图，真相在变更日志。"""

from __future__ import annotations

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import BehaviorClass, Exclusion, Vocabulary

_LANE_TITLES = {Lane.SESSION: "会话 lane", Lane.PHYSICAL: "物理 lane"}


def render_catalog(vocabulary: Vocabulary, *, schema_version: str) -> str:
    lines = [f"# 行为类清单（第 {vocabulary.version} 版）", ""]
    for lane in Lane:
        classes = [item for item in vocabulary.classes.values() if item.lane is lane]
        if not classes:
            continue
        lines += [f"## {_LANE_TITLES[lane]}", ""]
        for item in classes:
            lines += _class_lines(item, vocabulary)
    lines += [f"<!-- {schema_version} version={vocabulary.version} -->", ""]
    return "\n".join(lines)


def _class_lines(item: BehaviorClass, vocabulary: Vocabulary) -> list[str]:
    head = f"- **{item.name}**〔{item.id}〕"
    if not item.active:
        successors = "、".join(str(target) for target in vocabulary.descendants(item.id))
        head += f"（停用{'，条目去了 ' + successors if successors else ''}）"
    lines = [head, f"  - 判据：{item.criterion}", f"  - 提醒句：{item.reminder}"]
    for exclusion in item.excludes:
        target = "" if exclusion.goes_to is None else f"（→{_name(vocabulary, exclusion)}）"
        lines.append(f"  - 不含：{exclusion.text}{target}")
    if item.examples:
        lines.append(f"  - 例：{'；'.join(item.examples)}")
    return lines


def _name(vocabulary: Vocabulary, exclusion: Exclusion) -> str:
    goes_to = exclusion.goes_to
    if goes_to is None or goes_to not in vocabulary.classes:
        return str(goes_to)
    return vocabulary.classes[goes_to].name


__all__ = ["render_catalog"]
