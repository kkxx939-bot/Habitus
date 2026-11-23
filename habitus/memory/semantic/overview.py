"""目录概览（L1）与目录简介（L0）的模板：确定性拼出、确定性读回，不调模型。

概览每行一项，字段之间用制表符隔开（记忆名字里不允许控制字符，说明在写入前压成一行）：

- 卡片行：``- <文件名>\\t<created_at>\\t<updated_at>\\tr<revision>\\t<说明>``
- 子目录行：``- <目录名>/\\t<项数> 项``

``created_at``、``updated_at`` 与 ``revision`` 给系统算冷热用（查询时不必读全树），检索给模型看的清单里不放它们。
一个目录超出上限时只列最新的那些卡片，末尾写明还有多少张没列出。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from habitus.foundation.timestamps import format_utc_z, parse_utc_z
from habitus.memory.model import MemoryDirectory

_HEADER_PREFIX = "# "
_OMITTED_PREFIX = "（还有 "
_OMITTED_SUFFIX = " 张卡片没有列出）"
_FORMAT_NOTE = "<!-- 每行：文件名、创建时间、更新时间、版本、说明；由模板生成，可删除重建 -->"


class MemoryOverviewFormatError(ValueError):
    """概览文件不是本模板拼出的格式（被手改或来自旧版的模型概览）。"""


@dataclass(frozen=True)
class MemoryOverviewCard:
    """目录里一张直接 L2 卡片的一行。"""

    filename: str
    description: str
    created_at: datetime
    updated_at: datetime
    revision: int

    def __post_init__(self) -> None:
        if not self.filename.endswith(".md") or any(not ch.isprintable() for ch in self.filename):
            raise ValueError("overview card filename must be a printable .md name")
        if any(not ch.isprintable() for ch in self.description):
            raise ValueError("overview card description must be one printable line")
        if self.revision <= 0:
            raise ValueError("overview card revision must be positive")

    @property
    def name(self) -> str:
        return self.filename[: -len(".md")]


@dataclass(frozen=True)
class MemoryOverviewChild:
    """一个直接子目录：名字与里面直接有多少项。"""

    name: str
    items: int

    def __post_init__(self) -> None:
        if not self.name or any(not ch.isprintable() for ch in self.name):
            raise ValueError("overview child name must be printable text")
        if self.items < 0:
            raise ValueError("overview child item count must be non-negative")


@dataclass(frozen=True)
class MemoryOverview:
    """一个目录的完整概览：卡片按 ``updated_at`` 从新到旧，子目录按名字。"""

    directory: MemoryDirectory
    cards: tuple[MemoryOverviewCard, ...]
    children: tuple[MemoryOverviewChild, ...]
    omitted_cards: int = 0

    def __post_init__(self) -> None:
        if self.omitted_cards < 0:
            raise ValueError("omitted card count must be non-negative")

    @property
    def total_items(self) -> int:
        return len(self.cards) + self.omitted_cards + len(self.children)

    def render(self) -> str:
        lines = [_HEADER_PREFIX + _directory_label(self.directory), _FORMAT_NOTE, ""]
        lines.extend(
            f"- {card.filename}\t{format_utc_z(card.created_at)}\t{format_utc_z(card.updated_at)}"
            f"\tr{card.revision}\t{card.description}"
            for card in self.cards
        )
        lines.extend(f"- {child.name}/\t{child.items} 项" for child in self.children)
        if self.omitted_cards:
            lines.append(f"{_OMITTED_PREFIX}{self.omitted_cards}{_OMITTED_SUFFIX}")
        return "\n".join(lines).rstrip() + "\n"

    def abstract(self) -> str:
        """L0：一行，说这个目录是什么、直接有多少项。"""

        return f"{_directory_label(self.directory)}：{self.total_items} 项\n"

    @classmethod
    def parse(cls, directory: MemoryDirectory, text: str) -> MemoryOverview:
        lines = text.rstrip("\n").split("\n")
        if len(lines) < 2 or lines[0] != _HEADER_PREFIX + _directory_label(directory) or lines[1] != _FORMAT_NOTE:
            raise MemoryOverviewFormatError(
                f"memory overview of {_directory_label(directory)} is not template-generated; "
                "删除这个目录下的 .overview.md 与 .abstract.md，下一次维护会按模板重建"
            )
        cards: list[MemoryOverviewCard] = []
        children: list[MemoryOverviewChild] = []
        omitted = 0
        for line in lines[2:]:
            if not line:
                continue
            try:
                if line.startswith(_OMITTED_PREFIX) and line.endswith(_OMITTED_SUFFIX):
                    omitted = int(line[len(_OMITTED_PREFIX) : -len(_OMITTED_SUFFIX)])
                    continue
                if not line.startswith("- "):
                    raise ValueError("unknown line")
                parts = line[2:].split("\t")
                if len(parts) == 2 and parts[0].endswith("/") and parts[1].endswith(" 项"):
                    children.append(MemoryOverviewChild(parts[0][:-1], int(parts[1][: -len(" 项")])))
                    continue
                if len(parts) != 5 or not parts[3].startswith("r"):
                    raise ValueError("unknown card line")
                cards.append(
                    MemoryOverviewCard(
                        filename=parts[0],
                        description=parts[4],
                        created_at=_time(parts[1]),
                        updated_at=_time(parts[2]),
                        revision=int(parts[3][1:]),
                    )
                )
            except ValueError as exc:
                raise MemoryOverviewFormatError(
                    f"memory overview of {_directory_label(directory)} has an unreadable line"
                ) from exc
        return cls(directory, tuple(cards), tuple(children), omitted)


def _time(value: str) -> datetime:
    return parse_utc_z(value, invalid="overview time must be text", naive="overview time must include a timezone")


def _directory_label(directory: MemoryDirectory) -> str:
    return "/".join(directory.parts) or "记忆根"


__all__ = [
    "MemoryOverview",
    "MemoryOverviewCard",
    "MemoryOverviewChild",
    "MemoryOverviewFormatError",
]
