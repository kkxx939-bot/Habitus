"""提取记忆时"搜内容"这个只读动作：按模型给的词在卡片字段文字里做字面匹配（照 Claude Code 的 grep、OpenViking 提取时的 search）。

不是语义搜索，不打语义分：命中的检索词个数多的在前，同数时最近更新的在前。退休的卡片不返回；冷热读不出时不剔，
和助手查记忆一侧的清单同一个口径。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from habitus.memory.document import MemoryDocument
from habitus.memory.model import MemoryAddress, MemoryDirectory
from habitus.memory.recall import MemoryRecallCandidate, MemoryRecallLifecycle, MemoryRecallTarget
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI

_SEARCH_ROOTS = (MemoryDirectory(("preferences",)), MemoryDirectory(("entities",)), MemoryDirectory(("tools",)))
_SNIPPET_BEFORE = 40
_SNIPPET_AFTER = 80


@dataclass(frozen=True)
class MemorySearchHit:
    """一张命中的卡片：地址、名字（文件名）和命中的那一小段字段文字。"""

    uri: MemoryURI
    name: str
    snippet: str

    def line(self) -> str:
        return f"- {self.uri}（{self.name}）：{self.snippet}"


def memory_content_search(
    tree: MemoryTree,
    recall_lifecycle: MemoryRecallLifecycle | None,
    *,
    keywords: tuple[str, ...],
    directory: MemoryDirectory | None,
    limit: int,
) -> tuple[MemorySearchHit, ...]:
    """在档案和三类集合目录（或只在 ``directory`` 及其子目录）的卡片字段里找这些词，返回至多 ``limit`` 条。"""

    folded = tuple(keyword.casefold() for keyword in keywords)
    scored: list[tuple[int, datetime, MemoryDocument, str]] = []
    for address in _addresses(tree, directory):
        document = tree.read(address)
        text = _field_text(document)
        hits = sum(1 for keyword in folded if keyword in text.casefold())
        if hits:
            scored.append((hits, document.metadata.updated_at, document, text))
    retired = _retired(recall_lifecycle, tuple(document for _hits, _updated, document, _text in scored))
    scored = [item for item in scored if str(MemoryURI.from_address(item[2].address)) not in retired]
    scored.sort(key=lambda item: (-item[0], -item[1].timestamp(), str(MemoryURI.from_address(item[2].address))))
    return tuple(
        MemorySearchHit(
            uri=MemoryURI.from_address(document.address),
            name=tree.path_for(document.address).name.removesuffix(".md"),
            snippet=_snippet(text, folded),
        )
        for _hits, _updated, document, text in scored[:limit]
    )


def _addresses(tree: MemoryTree, directory: MemoryDirectory | None) -> tuple[MemoryAddress, ...]:
    roots = _SEARCH_ROOTS if directory is None else (directory,)
    found: list[MemoryAddress] = []
    if directory is None and tree.exists(MemoryAddress.profile()):
        found.append(MemoryAddress.profile())
    for root in roots:
        if not tree.directory_exists(root):
            continue
        for child in (root, *tree.child_directories(root)):
            found.extend(tree.direct_addresses(child))
    return tuple(found)


def _field_text(document: MemoryDocument) -> str:
    """业务字段逐个摊平成"字段名: 值"，列表与嵌套值按出现顺序连起来。"""

    def flatten(value: object) -> str:
        if isinstance(value, dict):
            return " ".join(flatten(item) for item in value.values())
        if isinstance(value, list | tuple):
            return " ".join(flatten(item) for item in value)
        return "" if value is None else str(value)

    return "\n".join(f"{name}: {flatten(value)}" for name, value in document.fields.items())


def _snippet(text: str, folded: tuple[str, ...]) -> str:
    lowered = text.casefold()
    start = min(position for keyword in folded if (position := lowered.find(keyword)) >= 0)
    left = max(0, start - _SNIPPET_BEFORE)
    right = min(len(text), start + _SNIPPET_AFTER)
    piece = " ".join(text[left:right].split())
    return ("…" if left else "") + piece + ("…" if right < len(text) else "")


def _retired(lifecycle: MemoryRecallLifecycle | None, documents: tuple[MemoryDocument, ...]) -> frozenset[str]:
    if lifecycle is None or not documents:
        return frozenset()
    candidates = tuple(
        MemoryRecallCandidate(
            MemoryRecallTarget(
                MemoryURI.from_address(document.address), document.metadata.revision, document.metadata.created_at
            ),
            document.kind,
            document.metadata.updated_at,
        )
        for document in documents
    )
    batch = lifecycle.config.max_batch_size
    try:
        rankings = [
            ranking
            for start in range(0, len(candidates), batch)
            for ranking in lifecycle.rank(candidates[start : start + batch], now=datetime.now(UTC))
        ]
    except Exception:  # 冷热读不出时不剔（与助手查记忆一侧的清单同一口径）
        return frozenset()
    return frozenset(str(ranking.candidate.uri) for ranking in rankings if ranking.state.retired_at is not None)


__all__ = ["MemorySearchHit", "memory_content_search"]
