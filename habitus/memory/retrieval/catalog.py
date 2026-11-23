"""检索第 1 步"清单"：从各目录的模板概览读出卡片清单，配上冷热，不调模型。

最冷一档的卡片不混进普通清单，单列一组只给名字；退休的卡片不出现。清单太长或目录概览本身截断时，没列出的
卡片记下来，留给"挑"那一步给的关键词在名字和说明里匹配。每一行的 ``updated_at``、``revision`` 只给系统算冷热，
不给模型看。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime

from habitus.memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.recall import (
    MemoryRecallCandidate,
    MemoryRecallLifecycle,
    MemoryRecallTarget,
    MemoryTemperature,
)
from habitus.memory.retrieval.records import keyword_hits
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.semantic import MemoryOverview, MemoryOverviewCard
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI

_SECTION_TITLES = {
    MemoryKind.PROFILE: "档案",
    MemoryKind.PREFERENCE: "偏好（preferences/）",
    MemoryKind.ENTITY: "人和项目（entities/，按类别分两层）",
    MemoryKind.TOOL: "工具（tools/）",
}


@dataclass(frozen=True)
class CatalogCard:
    """清单里的一张卡片。``path`` 是给模型看、也让模型照抄回来的相对路径。"""

    uri: MemoryURI
    path: str
    name: str
    description: str
    kind: MemoryKind
    target: MemoryRecallTarget
    updated_at: datetime
    hotness: float | None = None
    temperature: MemoryTemperature | None = None

    def line(self) -> str:
        return f"- {self.path} — {self.description}" if self.description else f"- {self.path}"


@dataclass(frozen=True)
class MemoryCatalog:
    """一次查询的卡片清单。"""

    cards: tuple[CatalogCard, ...]
    cold: tuple[CatalogCard, ...]
    hidden: tuple[CatalogCard, ...] = ()
    overview_omitted: tuple[tuple[MemoryDirectory, int], ...] = ()
    notes: tuple[str, ...] = field(default=())

    def find(self, path: str) -> CatalogCard | None:
        """按模型照抄回来的路径找卡片（普通清单、最冷一组都算）；不在清单里的找不到。"""

        normalized = path.strip().removeprefix("memory://").strip("/")
        for card in (*self.cards, *self.cold):
            if card.path == normalized:
                return card
        return None

    @property
    def unlisted_count(self) -> int:
        return len(self.hidden) + sum(count for _directory, count in self.overview_omitted)

    def render(self) -> str:
        if not self.cards and not self.cold:
            return "（没有卡片）"
        blocks: list[str] = []
        for kind in MemoryKind:
            lines = [card.line() for card in self.cards if card.kind is kind]
            if lines:
                blocks.append(f"## {_SECTION_TITLES[kind]}\n" + "\n".join(lines))
        if self.unlisted_count:
            blocks.append(
                f"（还有 {self.unlisted_count} 张卡片没有列出；需要时在 card_keywords 里给词，系统按名字和说明匹配）"
            )
        if self.cold:
            blocks.append(
                "## 很久没用到的卡片（只给名字；有用时可以挑，每次最多一张）\n"
                + "\n".join(f"- {card.path}" for card in self.cold)
            )
        return "\n\n".join(blocks)


class MemoryCatalogReader:
    """读模板概览、算冷热、按字数上限截出给模型看的清单。"""

    def __init__(
        self,
        tree: MemoryTree,
        schema_registry: MemorySchemaRegistry,
        recall_lifecycle: MemoryRecallLifecycle,
        *,
        max_catalog_chars: int,
        overview_max_bytes: int,
        max_unlisted_scan: int,
    ) -> None:
        self.tree = tree
        self.schema_registry = schema_registry
        self.recall_lifecycle = recall_lifecycle
        self.max_catalog_chars = max_catalog_chars
        self.overview_max_bytes = overview_max_bytes
        self.max_unlisted_scan = max_unlisted_scan

    def read(
        self,
        *,
        kinds: tuple[MemoryKind, ...],
        now: datetime,
        on_recall_failure: Callable[[BaseException], None],
    ) -> MemoryCatalog:
        """读全部目录概览；会因记忆任务提交窗口抛 ``MemoryTreeConsistencyError``，由调用方短暂重试。"""

        cards: list[CatalogCard] = []
        omitted: list[tuple[MemoryDirectory, int]] = []
        notes: list[str] = []
        pending = [MemoryDirectory.root()]
        while pending:
            directory = pending.pop(0)
            if not self.tree.layer_exists(directory, MemoryLevel.OVERVIEW):
                if self.tree.direct_addresses(directory) or self.tree.child_directories(directory):
                    notes.append(f"{_label(directory)} 还没有概览，这次没有列出它的卡片")
                continue
            overview = MemoryOverview.parse(
                directory,
                self.tree.read_layer_bounded(directory, MemoryLevel.OVERVIEW, max_bytes=self.overview_max_bytes),
            )
            cards.extend(self._card(directory, row) for row in overview.cards)
            if overview.omitted_cards:
                omitted.append((directory, overview.omitted_cards))
            pending.extend(MemoryDirectory((*directory.parts, child.name)) for child in overview.children)
        if kinds:
            cards = [card for card in cards if card.kind in kinds]
            omitted = [(directory, count) for directory, count in omitted if _directory_kind(directory) in kinds]
        normal, cold = self._temperatures(tuple(cards), now=now, on_failure=on_recall_failure)
        shown, shown_cold, hidden = self._fit(normal, cold)
        return MemoryCatalog(tuple(shown), tuple(shown_cold), tuple(hidden), tuple(omitted), tuple(notes))

    def match_unlisted(
        self,
        catalog: MemoryCatalog,
        keywords: tuple[str, ...],
        limit: int,
        *,
        now: datetime,
    ) -> tuple[CatalogCard, ...]:
        """没列出的卡片里，名字或说明命中关键词的（命中词多的在前）；概览本身截掉的要读文件拿说明，读的张数有上限。
        读出来的卡片照样算冷热：退休的剔掉，最冷的带上温度，由调用方算进"每次最多一张"。"""

        scanned: list[CatalogCard] = []
        listed = {card.uri for card in (*catalog.cards, *catalog.cold, *catalog.hidden)}
        budget = self.max_unlisted_scan
        for directory, _count in catalog.overview_omitted:
            for address in self.tree.direct_addresses(directory):
                uri = MemoryURI.from_address(address)
                if uri in listed or budget <= 0:
                    continue
                budget -= 1
                document = self.tree.read(address)
                scanned.append(
                    CatalogCard(
                        uri=uri,
                        path=str(uri).removeprefix("memory://"),
                        name=self.tree.path_for(address).name.removesuffix(".md"),
                        description=self.schema_registry.get(document.kind).overview_line(document.fields),
                        kind=document.kind,
                        target=MemoryRecallTarget(uri, document.metadata.revision, document.metadata.created_at),
                        updated_at=document.metadata.updated_at,
                    )
                )
        normal, cold = self._temperatures(tuple(scanned), now=now, on_failure=lambda _error: None)
        candidates = (*catalog.hidden, *normal, *cold)
        scored = [(keyword_hits(f"{card.name}\n{card.description}", keywords), card) for card in candidates]
        matched = sorted((item for item in scored if item[0]), key=lambda item: (-item[0], item[1].path))
        return tuple(card for _hits, card in matched[:limit])

    def _card(self, directory: MemoryDirectory, row: MemoryOverviewCard) -> CatalogCard:
        address = _address(directory, row.name)
        uri = MemoryURI.from_address(address)
        return CatalogCard(
            uri=uri,
            path=str(uri).removeprefix("memory://"),
            name=row.name,
            description=row.description,
            kind=address.kind,
            target=MemoryRecallTarget(uri, row.revision, row.created_at),
            updated_at=row.updated_at,
        )

    def _temperatures(
        self,
        cards: tuple[CatalogCard, ...],
        *,
        now: datetime,
        on_failure: Callable[[BaseException], None],
    ) -> tuple[tuple[CatalogCard, ...], tuple[CatalogCard, ...]]:
        """按热度从高到低；退休的剔掉，最冷一档单列。冷热读不出时不分档，照清单顺序给。"""

        if not cards:
            return (), ()
        batch = self.recall_lifecycle.config.max_batch_size
        by_uri = {card.uri: card for card in cards}
        try:
            rankings = [
                ranking
                for start in range(0, len(cards), batch)
                for ranking in self.recall_lifecycle.rank(
                    tuple(
                        MemoryRecallCandidate(card.target, card.kind, card.updated_at)
                        for card in cards[start : start + batch]
                    ),
                    now=now,
                )
            ]
        except Exception as exc:
            on_failure(exc)
            return cards, ()
        rankings.sort(key=lambda item: (-(item.hotness or 0.0), str(item.candidate.uri)))
        normal: list[CatalogCard] = []
        cold: list[CatalogCard] = []
        for ranking in rankings:
            if ranking.state.retired_at is not None:
                continue
            card = by_uri[ranking.candidate.uri]
            ranked = replace(card, hotness=ranking.hotness, temperature=ranking.temperature)
            (cold if ranking.temperature is MemoryTemperature.COLD_2 else normal).append(ranked)
        return tuple(normal), tuple(cold)

    def _fit(
        self,
        normal: tuple[CatalogCard, ...],
        cold: tuple[CatalogCard, ...],
    ) -> tuple[list[CatalogCard], list[CatalogCard], list[CatalogCard]]:
        """按热度从高到低放进清单，直到字数上限，普通卡片先放、最冷的后放；放不下的记作没列出。"""

        used = 400
        shown: list[CatalogCard] = []
        shown_cold: list[CatalogCard] = []
        hidden: list[CatalogCard] = []
        for card in cold:
            normal_cost = sum(len(item.line()) + 1 for item in normal)
            cost = len(card.path) + 3
            if used + normal_cost + cost > self.max_catalog_chars:
                hidden.append(card)
                continue
            shown_cold.append(card)
            used += cost
        for card in normal:
            cost = len(card.line()) + 1
            if used + cost > self.max_catalog_chars:
                hidden.append(card)
                continue
            shown.append(card)
            used += cost
        order = {card.uri: index for index, card in enumerate(normal)}
        shown.sort(key=lambda card: (card.path.rsplit("/", 1)[0], order[card.uri]))
        return shown, shown_cold, hidden


def _address(directory: MemoryDirectory, name: str) -> MemoryAddress:
    parts = directory.parts
    if not parts:
        return MemoryAddress.profile()
    if parts == ("preferences",):
        return MemoryAddress.preference(name)
    if parts == ("tools",):
        return MemoryAddress.tool(name)
    if parts[0] == "entities" and len(parts) == 2:
        return MemoryAddress.entity(parts[1], name)
    raise ValueError(f"memory overview of {_label(directory)} lists a card where no card can live")


def _directory_kind(directory: MemoryDirectory) -> MemoryKind:
    parts = directory.parts
    if not parts:
        return MemoryKind.PROFILE
    return {"preferences": MemoryKind.PREFERENCE, "tools": MemoryKind.TOOL}.get(parts[0], MemoryKind.ENTITY)


def _label(directory: MemoryDirectory) -> str:
    return "/".join(directory.parts) or "记忆根"


__all__ = ["CatalogCard", "MemoryCatalog", "MemoryCatalogReader"]
