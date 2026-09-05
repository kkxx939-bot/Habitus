"""L2 文档中持久化的严格记忆关系模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from habitus.memory.uri import MemoryURI


class MemoryLinkType(str, Enum):
    """持久关系和模型关系候选共同使用的受控方向语义。"""

    RELATED_TO = "related_to"
    BELONGS_TO = "belongs_to"
    CAUSED_BY = "caused_by"
    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"
    EVOLVED_FROM = "evolved_from"

    @property
    def is_symmetric(self) -> bool:
        """返回关系是否不区分语义方向。"""

        return self in {MemoryLinkType.RELATED_TO, MemoryLinkType.CONTRADICTS}

    @property
    def description(self) -> str:
        """返回结构化解析时使用的明确方向定义。"""

        return {
            MemoryLinkType.RELATED_TO: "两个节点存在稳定且明确的一般关联，不区分方向；有更精确关系时不使用",
            MemoryLinkType.BELONGS_TO: "来源节点是目标节点所表达整体的成员、组成部分或下位对象",
            MemoryLinkType.CAUSED_BY: "来源节点所记录的结果或状态由目标节点直接导致",
            MemoryLinkType.DERIVED_FROM: "来源节点的信息或结论由目标节点的内容推导或沉淀而来",
            MemoryLinkType.CONTRADICTS: "两个节点的有效内容明确互不相容，不区分方向",
            MemoryLinkType.EVOLVED_FROM: "来源节点是较新状态，由目标节点所表达的较早状态演变而来",
        }[self]


@dataclass(frozen=True)
class MemoryStoredLink:
    """只保存最终 L2 URI 和关系类型的规范正向关系。"""

    from_uri: MemoryURI
    to_uri: MemoryURI
    link_type: MemoryLinkType

    def __post_init__(self) -> None:
        for name, uri in {"from_uri": self.from_uri, "to_uri": self.to_uri}.items():
            if not isinstance(uri, MemoryURI):
                raise TypeError(f"{name} must be a MemoryURI")
            try:
                uri.to_address()
            except ValueError as exc:
                raise ValueError(f"{name} must identify an L2 memory document") from exc
        try:
            link_type = MemoryLinkType(self.link_type)
        except ValueError as exc:
            raise ValueError("memory link contains an unsupported link_type") from exc
        from_uri = self.from_uri
        to_uri = self.to_uri
        if from_uri == to_uri:
            raise ValueError("memory link cannot reference the same URI twice")
        if link_type.is_symmetric and str(to_uri) < str(from_uri):
            from_uri, to_uri = to_uri, from_uri
        object.__setattr__(self, "from_uri", from_uri)
        object.__setattr__(self, "to_uri", to_uri)
        object.__setattr__(self, "link_type", link_type)

    @property
    def identity(self) -> tuple[str, str, str]:
        """返回所有关系层统一使用的确定性身份。"""

        return (str(self.from_uri), str(self.to_uri), self.link_type.value)

    def to_dict(self) -> dict[str, str]:
        """返回规范 JSON 对象。"""

        return {
            "from_uri": str(self.from_uri),
            "to_uri": str(self.to_uri),
            "link_type": self.link_type.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryStoredLink:
        """严格解析持久 JSON；不忽略字段或修复枚举。"""

        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ValueError("memory link must be an object")
        expected = {"from_uri", "to_uri", "link_type"}
        keys = set(value)
        if keys != expected:
            raise ValueError("memory link has an invalid shape")
        from_uri = value["from_uri"]
        to_uri = value["to_uri"]
        link_type = value["link_type"]
        if not isinstance(from_uri, str) or not isinstance(to_uri, str):
            raise ValueError("memory link endpoints must be URI strings")
        if not isinstance(link_type, str):
            raise ValueError("memory link type must be a string")
        return cls(
            from_uri=MemoryURI.parse(from_uri),
            to_uri=MemoryURI.parse(to_uri),
            link_type=MemoryLinkType(link_type),
        )


def normalize_stored_links(values: object, *, label: str) -> tuple[MemoryStoredLink, ...]:
    """校验关系元组、拒绝重复并按统一身份排序。"""

    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    links: list[MemoryStoredLink] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, MemoryStoredLink):
            raise TypeError(f"{label} must contain MemoryStoredLink values")
        if value.identity in seen:
            raise ValueError(f"{label} contains a duplicate memory link")
        seen.add(value.identity)
        links.append(value)
    return tuple(sorted(links, key=lambda link: link.identity))


def parse_stored_links(values: Any, *, label: str) -> tuple[MemoryStoredLink, ...]:
    """严格解析 JSON 关系数组。"""

    if not isinstance(values, list):
        raise ValueError(f"{label} must be an array")
    return normalize_stored_links(
        tuple(MemoryStoredLink.from_dict(value) for value in values),
        label=label,
    )


__all__ = [
    "MemoryLinkType",
    "MemoryStoredLink",
    "normalize_stored_links",
    "parse_stored_links",
]
