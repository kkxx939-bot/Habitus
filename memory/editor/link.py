"""把临时关系候选解析为最终记忆 URI 关系操作。"""

from __future__ import annotations

from dataclasses import dataclass

from memory.document import MemoryLinkType, MemoryStoredLink
from memory.editor.candidate import (
    MemoryCandidateBatch,
    MemoryRelationAction,
    MemoryRelationCandidate,
)
from memory.editor.identity import MemoryFinalIdentityMap
from memory.uri import MemoryURI


class MemoryRelationResolutionError(ValueError):
    """最终节点身份或关系端点无法确定性解析。"""


@dataclass(frozen=True)
class MemoryResolvedRelation:
    """已经绑定最终 L2 URI、但尚未参与关系计划的操作。"""

    action: MemoryRelationAction
    from_uri: MemoryURI
    to_uri: MemoryURI
    link_type: MemoryLinkType

    def __post_init__(self) -> None:
        try:
            action = MemoryRelationAction(self.action)
        except ValueError as exc:
            raise MemoryRelationResolutionError("resolved relation contains an unsupported action") from exc
        for name, uri in {"from_uri": self.from_uri, "to_uri": self.to_uri}.items():
            if not isinstance(uri, MemoryURI):
                raise TypeError(f"{name} must be a MemoryURI")
            try:
                uri.to_address()
            except ValueError as exc:
                raise MemoryRelationResolutionError(f"{name} must identify an L2 document") from exc
        if self.from_uri == self.to_uri:
            raise MemoryRelationResolutionError("resolved relation cannot reference the same memory URI twice")
        try:
            link_type = MemoryLinkType(self.link_type)
        except ValueError as exc:
            raise MemoryRelationResolutionError("resolved relation contains an unsupported link_type") from exc
        from_uri = self.from_uri
        to_uri = self.to_uri
        if link_type.is_symmetric and str(to_uri) < str(from_uri):
            from_uri, to_uri = to_uri, from_uri
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "from_uri", from_uri)
        object.__setattr__(self, "to_uri", to_uri)
        object.__setattr__(self, "link_type", link_type)

    @property
    def identity(self) -> tuple[str, str, str]:
        """返回不包含动作的持久关系身份。"""

        return (str(self.from_uri), str(self.to_uri), self.link_type.value)

    def to_stored(self) -> MemoryStoredLink:
        """转换为 L2 文档使用的规范关系对象。"""

        return MemoryStoredLink(
            from_uri=self.from_uri,
            to_uri=self.to_uri,
            link_type=self.link_type,
        )


class MemoryRelationResolver:
    """在全部节点身份落定后解析、规范化并检查关系操作。"""

    def resolve(
        self,
        batch: MemoryCandidateBatch,
        identities: MemoryFinalIdentityMap,
    ) -> tuple[MemoryResolvedRelation, ...]:
        """把 page_id 关系转换为最终 URI 操作。"""

        if not isinstance(batch, MemoryCandidateBatch):
            raise TypeError("batch must be a MemoryCandidateBatch")
        if not isinstance(identities, MemoryFinalIdentityMap):
            raise TypeError("identities must be a MemoryFinalIdentityMap")

        for candidate in batch.iter_candidates():
            identities.entry(candidate.page_id)

        resolved: dict[tuple[str, str, str], MemoryResolvedRelation] = {}
        for relation in batch.relations:
            item = self._resolve_one(relation, identities)
            if item is None:
                continue
            previous = resolved.get(item.identity)
            if previous is not None and previous.action is not item.action:
                raise MemoryRelationResolutionError("final identity mapping makes one relation both add and remove")
            resolved.setdefault(item.identity, item)
        return tuple(resolved[key] for key in sorted(resolved))

    @staticmethod
    def _resolve_one(
        relation: MemoryRelationCandidate,
        identities: MemoryFinalIdentityMap,
    ) -> MemoryResolvedRelation | None:
        from_uri = identities.resolve(relation.from_page_id)
        to_uri = identities.resolve(relation.to_page_id)
        if from_uri is None or to_uri is None:
            raise MemoryRelationResolutionError("explicit relation operation cannot reference a deleted node")
        if from_uri == to_uri:
            return None
        return MemoryResolvedRelation(
            action=relation.action,
            from_uri=from_uri,
            to_uri=to_uri,
            link_type=relation.link_type,
        )


__all__ = [
    "MemoryRelationResolutionError",
    "MemoryRelationResolver",
    "MemoryResolvedRelation",
]
