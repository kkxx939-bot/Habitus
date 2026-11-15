"""把一条 lane 的词表渲染成给人看的清单（``kinds.md``）；只是视图，真相是初始词表与词表状态。"""

from __future__ import annotations

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import Vocabulary, reminder_sentence

_LANE_TITLES = {Lane.SESSION: "会话 lane", Lane.PHYSICAL: "物理 lane"}


def render_catalog(vocabulary: Vocabulary, *, schema_version: str) -> str:
    title = _LANE_TITLES[vocabulary.lane]
    grown = sum(1 for word in vocabulary.words if word.grown)
    lines = [f"# 词表 · {title}（在用状态第 {vocabulary.revision} 版）", ""]
    lines += [
        f"在用 {len(vocabulary.active)} 个 / 共 {len(vocabulary.words)} 个（自己长的 {grown} 个）"
        f" / 已删 {len(vocabulary.deleted)} 个",
        "",
    ]
    domain = None
    for word in vocabulary.ordered():
        if word.level == 1 and word.domain != domain:
            domain = word.domain
            lines += ["", f"## {domain}", ""]
        since = vocabulary.active.get(word.id)
        notes = []
        if word.grown:
            assert word.created_at is not None
            picked = "没被选中过" if word.last_picked is None else f"最后被选中 {word.last_picked.isoformat()}"
            notes.append(f"自己长的，{word.created_at.isoformat()} 写的，{picked}")
        if since is not None:
            notes.append(f"{since.isoformat()} 起在用；{reminder_sentence(word)}")
        elif word.id in vocabulary.exited:
            notes.append(f"{vocabulary.exited[word.id].isoformat()} 退出在用")
        elif word.grown:
            notes.append("候选")
        state = f"（{'；'.join(notes)}）" if notes else ""
        lines.append(f"{'  ' * (word.level - 1)}- **{word.name}**〔{word.id}〕{word.definition}{state}")
    if vocabulary.deleted:
        lines += ["", "## 已删", ""]
        lines += [
            f"- 〔{mark.id}〕上级〔{mark.parent}〕，{mark.deleted_on.isoformat()} 删的" for mark in vocabulary.deleted
        ]
    lines += ["", f"<!-- {schema_version} revision={vocabulary.revision} -->", ""]
    return "\n".join(lines)


__all__ = ["render_catalog"]
