"""情景 L2 文档的受控前向边。

边只存在情景上（一次判断存一次）：``needs`` / ``results_from`` 由情景指向更早的行为树
occurrence 或情景。目标不设时间上限——一个月前的预约也可以是今天理发的前提。归属（成员）
是字段不是边，见 ``scene.model``。

``lag_seconds`` = 情景开始 − 目标开始（整秒，非负）。两端 URI 的叶名都带 ``started_at``，
所以方向（晚指早）与 lag 都是**从两端身份派生**的：构造时按 URI 计算并与传入值核对，
传入值与身份矛盾即拒绝——存一个能和它旁边的 URI 打架的数，正是校验该拦的产物矛盾。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from habitus.behavior.model import BehaviorKind
from habitus.behavior.uri import BehaviorURI, BehaviorURINodeType
from habitus.scene.model import SceneLinkType
from habitus.scene.uri import SceneURI, SceneURINodeType


def parse_link_target(value: object) -> SceneURI | BehaviorURI:
    """边的目标只能是行为树的 occurrence 文档或情景树的情景文档。"""

    if isinstance(value, SceneURI | BehaviorURI):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise TypeError("scene link target must be a URI string")
    if raw.startswith(f"{SceneURI.SCHEME}://"):
        parsed = SceneURI.parse(raw)
        if parsed.node_type is not SceneURINodeType.DOCUMENT:
            raise ValueError("scene link target must identify a scene document")
        return parsed
    if raw.startswith(f"{BehaviorURI.SCHEME}://"):
        target = BehaviorURI.parse(raw)
        if target.node_type is not BehaviorURINodeType.DOCUMENT or target.to_address().kind is not BehaviorKind.OCCURRENCE:
            raise ValueError("scene link target must identify a behavior occurrence document")
        return target
    raise ValueError("scene link target must be a scene:// or behavior:// document URI")


def link_lag_seconds(from_uri: SceneURI, to_uri: SceneURI | BehaviorURI) -> int:
    """按两端身份里的开始时刻算 lag（整秒）；目标晚于来源即方向错误。"""

    # 源可能是按天情景文档，也可能是规律级的一次关联记录——取时刻这件事不按形态分叉。
    started = from_uri.started_at().astimezone(UTC)
    target_started = (
        to_uri.started_at() if isinstance(to_uri, SceneURI) else to_uri.to_address().started_at
    ).astimezone(UTC)
    delta = (started - target_started).total_seconds()
    if delta < 0:
        raise ValueError("scene link must point at an earlier target (later points to earlier)")
    return int(delta)


@dataclass(frozen=True)
class SceneStoredLink:
    """情景指向更早对象的一条前向边。"""

    from_uri: SceneURI
    to_uri: SceneURI | BehaviorURI
    link_type: SceneLinkType
    lag_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.from_uri, SceneURI) or self.from_uri.node_type is not SceneURINodeType.DOCUMENT:
            raise TypeError("from_uri must identify a scene document")
        object.__setattr__(self, "to_uri", parse_link_target(self.to_uri))
        try:
            link_type = SceneLinkType(self.link_type)
        except ValueError as exc:
            raise ValueError("scene link contains an unsupported link_type") from exc
        object.__setattr__(self, "link_type", link_type)
        if str(self.from_uri) == str(self.to_uri):
            raise ValueError("scene link cannot reference the same URI twice")
        if isinstance(self.lag_seconds, bool) or not isinstance(self.lag_seconds, int) or self.lag_seconds < 0:
            raise ValueError("scene link lag_seconds must be a non-negative integer")
        expected = link_lag_seconds(self.from_uri, self.to_uri)
        if self.lag_seconds != expected:
            raise ValueError(f"scene link lag_seconds must equal the start-time difference of its endpoints ({expected})")

    @classmethod
    def between(cls, from_uri: SceneURI, to_uri: SceneURI | BehaviorURI | str, link_type: SceneLinkType | str) -> SceneStoredLink:
        """按两端身份派生 lag 构造一条边；调用方不必自己算。"""

        target = parse_link_target(to_uri)
        return cls(from_uri=from_uri, to_uri=target, link_type=SceneLinkType(link_type), lag_seconds=link_lag_seconds(from_uri, target))

    @property
    def identity(self) -> tuple[str, str, str]:
        return str(self.from_uri), str(self.to_uri), self.link_type.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_uri": str(self.from_uri),
            "to_uri": str(self.to_uri),
            "link_type": self.link_type.value,
            "lag_seconds": self.lag_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> SceneStoredLink:
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ValueError("scene link must be an object")
        if set(value) != {"from_uri", "to_uri", "link_type", "lag_seconds"}:
            raise ValueError("scene link has an invalid shape")
        from_uri, link_type, lag = value["from_uri"], value["link_type"], value["lag_seconds"]
        if not isinstance(from_uri, str) or not isinstance(link_type, str):
            raise ValueError("scene link values must be strings")
        if isinstance(lag, bool) or not isinstance(lag, int):
            raise ValueError("scene link lag_seconds must be an integer")
        return cls(
            from_uri=SceneURI.parse(from_uri),
            to_uri=parse_link_target(value["to_uri"]),
            link_type=SceneLinkType(link_type),
            lag_seconds=lag,
        )


def normalize_stored_links(values: object, *, label: str) -> tuple[SceneStoredLink, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    links: list[SceneStoredLink] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, SceneStoredLink):
            raise TypeError(f"{label} must contain SceneStoredLink values")
        if value.identity in seen:
            raise ValueError(f"{label} contains a duplicate scene link")
        seen.add(value.identity)
        links.append(value)
    return tuple(sorted(links, key=lambda link: link.identity))


def parse_stored_links(values: Any, *, label: str) -> tuple[SceneStoredLink, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be an array")
    return normalize_stored_links(tuple(SceneStoredLink.from_dict(value) for value in values), label=label)


__all__ = ["SceneStoredLink", "link_lag_seconds", "normalize_stored_links", "parse_link_target", "parse_stored_links"]
