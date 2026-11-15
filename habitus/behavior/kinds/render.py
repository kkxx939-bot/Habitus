"""把一条 lane 的词表渲染成给人看的清单（``kinds.md``）；只是视图，真相是初始词表与在用状态。"""

from __future__ import annotations

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import Vocabulary, reminder_sentence

_LANE_TITLES = {Lane.SESSION: "会话 lane", Lane.PHYSICAL: "物理 lane"}


def render_catalog(vocabulary: Vocabulary, *, schema_version: str) -> str:
    title = _LANE_TITLES[vocabulary.lane]
    lines = [f"# 词表 · {title}（在用状态第 {vocabulary.revision} 版）", ""]
    lines += [f"在用 {len(vocabulary.active)} 个 / 共 {len(vocabulary.words)} 个", ""]
    domain = None
    for word in vocabulary.words:
        if word.level == 1 and word.domain != domain:
            domain = word.domain
            lines += ["", f"## {domain}", ""]
        since = vocabulary.active.get(word.id)
        state = "" if since is None else f"（{since.isoformat()} 起在用；{reminder_sentence(word)}）"
        lines.append(f"{'  ' * (word.level - 1)}- **{word.name}**〔{word.id}〕{word.definition}{state}")
    lines += ["", f"<!-- {schema_version} revision={vocabulary.revision} -->", ""]
    return "\n".join(lines)


__all__ = ["render_catalog"]
