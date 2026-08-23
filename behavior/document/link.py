"""行为语义 L2 文档的受控持久关系。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from behavior.uri import BehaviorURI


class BehaviorLinkType(str, Enum):
    """occurrence 之间仅有的两种前向关系；add-only 下只存前向、读侧取对称闭包。

    方向纪律沿融合层：``concurrent_with`` 由后封口的一条指向先封口的一条（语义对称，
    存储单向）；``results_from`` 由结果指回原因（结果本来就晚于原因）。
    """

    CONCURRENT_WITH = "concurrent_with"
    RESULTS_FROM = "results_from"


@dataclass(frozen=True)
class BehaviorStoredLink:
    """只保存最终 L2 URI 和受控方向语义的关系。"""

    from_uri: BehaviorURI
    to_uri: BehaviorURI
    link_type: BehaviorLinkType

    def __post_init__(self) -> None:
        for name, uri in {"from_uri": self.from_uri, "to_uri": self.to_uri}.items():
            if not isinstance(uri, BehaviorURI):
                raise TypeError(f"{name} must be a BehaviorURI")
            try:
                uri.to_address()
            except ValueError as exc:
                raise ValueError(f"{name} must identify an L2 behavior document") from exc
        try:
            link_type = BehaviorLinkType(self.link_type)
        except ValueError as exc:
            raise ValueError("behavior link contains an unsupported link_type") from exc
        if self.from_uri == self.to_uri:
            raise ValueError("behavior link cannot reference the same URI twice")
        object.__setattr__(self, "link_type", link_type)

    @property
    def identity(self) -> tuple[str, str, str]:
        return str(self.from_uri), str(self.to_uri), self.link_type.value

    def to_dict(self) -> dict[str, str]:
        return {
            "from_uri": str(self.from_uri),
            "to_uri": str(self.to_uri),
            "link_type": self.link_type.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> BehaviorStoredLink:
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ValueError("behavior link must be an object")
        expected = {"from_uri", "to_uri", "link_type"}
        if set(value) != expected:
            raise ValueError("behavior link has an invalid shape")
        from_uri = value["from_uri"]
        to_uri = value["to_uri"]
        link_type = value["link_type"]
        if not isinstance(from_uri, str) or not isinstance(to_uri, str) or not isinstance(link_type, str):
            raise ValueError("behavior link values must be strings")
        return cls(
            from_uri=BehaviorURI.parse(from_uri),
            to_uri=BehaviorURI.parse(to_uri),
            link_type=BehaviorLinkType(link_type),
        )


def normalize_stored_links(values: object, *, label: str) -> tuple[BehaviorStoredLink, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    links: list[BehaviorStoredLink] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, BehaviorStoredLink):
            raise TypeError(f"{label} must contain BehaviorStoredLink values")
        if value.identity in seen:
            raise ValueError(f"{label} contains a duplicate behavior link")
        seen.add(value.identity)
        links.append(value)
    return tuple(sorted(links, key=lambda link: link.identity))


def parse_stored_links(values: Any, *, label: str) -> tuple[BehaviorStoredLink, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be an array")
    return normalize_stored_links(
        tuple(BehaviorStoredLink.from_dict(value) for value in values),
        label=label,
    )


__all__ = [
    "BehaviorLinkType",
    "BehaviorStoredLink",
    "normalize_stored_links",
    "parse_stored_links",
]
