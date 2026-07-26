"""基于最终节点身份计算 Links/Backlinks 的纯关系计划。"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from memory.document import MemoryDocument, MemoryStoredLink
from memory.editor.candidate import MemoryRelationAction
from memory.editor.identity import (
    MemoryFinalIdentityMap,
    MemoryNodeDisposition,
)
from memory.editor.link import MemoryResolvedRelation
from memory.editor.reader import MemorySnapshotBatch
from memory.uri import MemoryURI


class MemoryRelationIntegrityError(ValueError):
    """关系读集、旧双向副本或最终关系状态不完整。"""


@dataclass(frozen=True)
class MemoryRelationReadSet:
    """关系计划涉及的旧节点、新节点缺失快照和一跳邻居。"""

    structural_source_uris: tuple[MemoryURI, ...]
    snapshots: MemorySnapshotBatch

    def __post_init__(self) -> None:
        if not isinstance(self.structural_source_uris, tuple) or any(
            not isinstance(uri, MemoryURI) for uri in self.structural_source_uris
        ):
            raise TypeError("structural_source_uris must be MemoryURI values")
        source_ids = tuple(str(uri) for uri in self.structural_source_uris)
        if source_ids != tuple(sorted(set(source_ids))):
            raise MemoryRelationIntegrityError("structural source URIs must be unique and sorted")
        if not isinstance(self.snapshots, SnapshotBatch):
            raise TypeError("snapshots must be a MemorySnapshotBatch")
        for snapshot in self.snapshots.snapshots:
            MemoryURI.parse(snapshot.identity).to_address()
            if not snapshot.exists:
                continue
            if not isinstance(snapshot.value, MemoryDocument):
                raise MemoryRelationIntegrityError("relation read set contains an invalid memory document")
            if str(MemoryURI.from_address(snapshot.value.address)) != snapshot.identity:
                raise MemoryRelationIntegrityError("relation snapshot identity does not match its document")
        for identity in source_ids:
            source_snapshot = self.snapshots.get(identity)
            if source_snapshot is None or not source_snapshot.exists:
                raise MemoryRelationIntegrityError("structural relation source must be a complete old document")

    @classmethod
    def build(
        cls,
        snapshots: MemorySnapshotBatch,
        identities: MemoryFinalIdentityMap,
        operations: tuple[MemoryResolvedRelation, ...],
    ) -> MemoryRelationReadSet:
        """从上游完整快照中选择并验证关系计划所需的闭合一跳读集。"""

        if not isinstance(snapshots, SnapshotBatch):
            raise TypeError("snapshots must be a MemorySnapshotBatch")
        if not isinstance(identities, MemoryFinalIdentityMap):
            raise TypeError("identities must be a MemoryFinalIdentityMap")
        if not isinstance(operations, tuple) or any(
            not isinstance(operation, MemoryResolvedRelation) for operation in operations
        ):
            raise TypeError("operations must be MemoryResolvedRelation values")

        structural_sources = {
            str(entry.source_uri)
            for entry in identities.entries
            if entry.disposition in {MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE}
            and entry.source_uri is not None
        }
        required = set(structural_sources)
        required.update(
            str(entry.final_uri)
            for entry in identities.entries
            if entry.disposition is MemoryNodeDisposition.MERGE and entry.final_uri is not None
        )
        required.update(str(uri) for operation in operations for uri in (operation.from_uri, operation.to_uri))
        created = {
            str(entry.final_uri)
            for entry in identities.entries
            if entry.disposition is MemoryNodeDisposition.CREATE and entry.final_uri is not None
        }

        for identity in tuple(required):
            snapshot = snapshots.get(identity)
            if snapshot is None:
                raise MemoryRelationIntegrityError(f"relation read set is missing required snapshot: {identity}")
            if not snapshot.exists:
                if identity not in created:
                    raise MemoryRelationIntegrityError(f"relation endpoint does not exist: {identity}")
                continue
            assert isinstance(snapshot.value, MemoryDocument)
            if identity in structural_sources:
                required.update(str(link.to_uri) for link in snapshot.value.links)
                required.update(str(backlink.from_uri) for backlink in snapshot.value.backlinks)

        missing = sorted(identity for identity in required if snapshots.get(identity) is None)
        if missing:
            raise MemoryRelationIntegrityError(f"relation read set is missing one-hop snapshots: {missing}")
        selected = tuple(snapshot for snapshot in snapshots.snapshots if snapshot.identity in required)
        result = cls(
            structural_source_uris=tuple(MemoryURI.parse(identity) for identity in sorted(structural_sources)),
            snapshots=SnapshotBatch(
                snapshots=selected,
                total_bytes=sum(snapshot.size_bytes for snapshot in selected),
            ),
        )
        result.validate_structural_consistency()
        return result

    def snapshot(self, uri: MemoryURI | str) -> VersionedSnapshot[MemoryDocument]:
        """返回指定关系节点的完整存在或缺失快照。"""

        identity = str(MemoryURI.parse(uri))
        snapshot = self.snapshots.get(identity)
        if snapshot is None:
            raise MemoryRelationIntegrityError(f"relation read set does not contain {identity}")
        return snapshot

    def document(self, uri: MemoryURI | str) -> MemoryDocument:
        """返回指定已存在关系节点。"""

        snapshot = self.snapshot(uri)
        if not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
            raise MemoryRelationIntegrityError("relation operation requires an existing memory document")
        return snapshot.value

    def validate_structural_consistency(self) -> None:
        """只校验将被迁移或清理节点的全部一跳双向关系。"""

        for uri in self.structural_source_uris:
            document = self.document(uri)
            for link in document.links:
                target = self.document(link.to_uri)
                if link not in target.backlinks:
                    raise MemoryRelationIntegrityError("structural forward link is missing its target backlink")
            for backlink in document.backlinks:
                source = self.document(backlink.from_uri)
                if backlink not in source.links:
                    raise MemoryRelationIntegrityError("structural backlink is missing its source link")


@dataclass(frozen=True)
class MemoryRelationNodeUpdate:
    """一个 URI 在关系规划后的完整 Links/Backlinks 目标状态。"""

    before: VersionedSnapshot[MemoryDocument]
    uri: MemoryURI
    links: tuple[MemoryStoredLink, ...]
    backlinks: tuple[MemoryStoredLink, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.before, VersionedSnapshot):
            raise TypeError("before must be a VersionedSnapshot")
        if not isinstance(self.uri, MemoryURI):
            raise TypeError("uri must be a MemoryURI")
        if self.before.identity != str(self.uri):
            raise MemoryRelationIntegrityError("relation update identity does not match its snapshot")
        for name, values in {
            "links": self.links,
            "backlinks": self.backlinks,
        }.items():
            if not isinstance(values, tuple) or any(not isinstance(value, MemoryStoredLink) for value in values):
                raise TypeError(f"{name} must contain MemoryStoredLink values")
            identities = tuple(value.identity for value in values)
            if identities != tuple(sorted(set(identities))):
                raise MemoryRelationIntegrityError(f"relation update {name} must be unique and sorted")
        if self.before.exists:
            assert isinstance(self.before.value, MemoryDocument)
            if self.links == self.before.value.links and self.backlinks == self.before.value.backlinks:
                raise MemoryRelationIntegrityError("existing relation update must change at least one relation set")
        elif not self.links and not self.backlinks:
            raise MemoryRelationIntegrityError("missing relation endpoint update must create at least one relation")


@dataclass(frozen=True)
class MemoryRelationPlan:
    """不含时间、revision 和写盘副作用的最终关系计划。"""

    read_set: MemoryRelationReadSet
    operations: tuple[MemoryResolvedRelation, ...]
    updates: tuple[MemoryRelationNodeUpdate, ...]
    added: tuple[MemoryStoredLink, ...]
    removed: tuple[MemoryStoredLink, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.read_set, MemoryRelationReadSet):
            raise TypeError("read_set must be a MemoryRelationReadSet")
        if not isinstance(self.operations, tuple) or any(
            not isinstance(operation, MemoryResolvedRelation) for operation in self.operations
        ):
            raise TypeError("operations must contain MemoryResolvedRelation values")
        if not isinstance(self.updates, tuple) or any(
            not isinstance(update, MemoryRelationNodeUpdate) for update in self.updates
        ):
            raise TypeError("updates must contain MemoryRelationNodeUpdate values")
        update_ids = tuple(str(update.uri) for update in self.updates)
        if update_ids != tuple(sorted(set(update_ids))):
            raise MemoryRelationIntegrityError("relation node updates must be unique and sorted")
        for name, values in {"added": self.added, "removed": self.removed}.items():
            if not isinstance(values, tuple) or any(not isinstance(value, MemoryStoredLink) for value in values):
                raise TypeError(f"{name} must contain MemoryStoredLink values")
            keys = tuple(value.identity for value in values)
            if keys != tuple(sorted(set(keys))):
                raise MemoryRelationIntegrityError(f"{name} relations must be unique and sorted")

    def update_for(self, uri: MemoryURI | str) -> MemoryRelationNodeUpdate | None:
        """返回指定 URI 的关系目标状态。"""

        identity = str(MemoryURI.parse(uri))
        for update in self.updates:
            if str(update.uri) == identity:
                return update
        return None


class MemoryRelationPlanner:
    """按结构迁移、REMOVE、ADD 的固定顺序计算双向关系。"""

    def plan(
        self,
        identities: MemoryFinalIdentityMap,
        operations: tuple[MemoryResolvedRelation, ...],
        read_set: MemoryRelationReadSet,
    ) -> MemoryRelationPlan:
        """从一致旧图生成全部受影响节点的最终关系状态。"""

        if not isinstance(identities, MemoryFinalIdentityMap):
            raise TypeError("identities must be a MemoryFinalIdentityMap")
        if not isinstance(operations, tuple) or any(
            not isinstance(operation, MemoryResolvedRelation) for operation in operations
        ):
            raise TypeError("operations must contain MemoryResolvedRelation values")
        if not isinstance(read_set, MemoryRelationReadSet):
            raise TypeError("read_set must be a MemoryRelationReadSet")
        read_set.validate_structural_consistency()

        links_by_uri: dict[str, set[MemoryStoredLink]] = {}
        backlinks_by_uri: dict[str, set[MemoryStoredLink]] = {}
        for snapshot in read_set.snapshots.snapshots:
            if snapshot.exists:
                assert isinstance(snapshot.value, MemoryDocument)
                links_by_uri[snapshot.identity] = set(snapshot.value.links)
                backlinks_by_uri[snapshot.identity] = set(snapshot.value.backlinks)
            else:
                links_by_uri[snapshot.identity] = set()
                backlinks_by_uri[snapshot.identity] = set()
        initial_relations = {relation for relations in links_by_uri.values() for relation in relations}

        structural_old: dict[tuple[str, str, str], MemoryStoredLink] = {}
        for source_uri in read_set.structural_source_uris:
            document = read_set.document(source_uri)
            for relation in (*document.links, *document.backlinks):
                structural_old.setdefault(relation.identity, relation)

        for relation in sorted(structural_old.values(), key=lambda item: item.identity):
            self._require_pair(relation, links_by_uri, backlinks_by_uri)

        for relation in sorted(structural_old.values(), key=lambda item: item.identity):
            self._discard_pair(relation, links_by_uri, backlinks_by_uri)
            remapped_from = identities.remap_uri(relation.from_uri)
            remapped_to = identities.remap_uri(relation.to_uri)
            if remapped_from is None or remapped_to is None or remapped_from == remapped_to:
                continue
            self._add_pair(
                MemoryStoredLink(
                    from_uri=remapped_from,
                    to_uri=remapped_to,
                    link_type=relation.link_type,
                ),
                links_by_uri,
                backlinks_by_uri,
            )

        explicit_removes = tuple(
            operation for operation in operations if operation.action is MemoryRelationAction.REMOVE
        )
        explicit_adds = tuple(operation for operation in operations if operation.action is MemoryRelationAction.ADD)
        for operation in explicit_removes:
            relation = operation.to_stored()
            self._require_pair(relation, links_by_uri, backlinks_by_uri)
            self._discard_pair(relation, links_by_uri, backlinks_by_uri)
        for operation in explicit_adds:
            self._add_pair(
                operation.to_stored(),
                links_by_uri,
                backlinks_by_uri,
            )

        dead_sources = {
            str(entry.source_uri)
            for entry in identities.entries
            if entry.disposition in {MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE}
            and entry.source_uri is not None
        }
        updates: list[MemoryRelationNodeUpdate] = []
        for identity in sorted(set(links_by_uri) | set(backlinks_by_uri)):
            if identity in dead_sources:
                continue
            current_snapshot = read_set.snapshots.get(identity)
            if current_snapshot is None:
                raise MemoryRelationIntegrityError(f"final relation endpoint lacks a snapshot: {identity}")
            links = tuple(sorted(links_by_uri.get(identity, set()), key=lambda item: item.identity))
            backlinks = tuple(sorted(backlinks_by_uri.get(identity, set()), key=lambda item: item.identity))
            old_links: tuple[MemoryStoredLink, ...] = ()
            old_backlinks: tuple[MemoryStoredLink, ...] = ()
            if current_snapshot.exists:
                assert isinstance(current_snapshot.value, MemoryDocument)
                old_links = current_snapshot.value.links
                old_backlinks = current_snapshot.value.backlinks
            if links == old_links and backlinks == old_backlinks:
                continue
            updates.append(
                MemoryRelationNodeUpdate(
                    before=current_snapshot,
                    uri=MemoryURI.parse(identity),
                    links=links,
                    backlinks=backlinks,
                )
            )

        final_relations = {
            relation
            for identity, relations in links_by_uri.items()
            if identity not in dead_sources
            for relation in relations
        }
        added = tuple(sorted(final_relations - initial_relations, key=lambda item: item.identity))
        removed = tuple(sorted(initial_relations - final_relations, key=lambda item: item.identity))
        return MemoryRelationPlan(
            read_set=read_set,
            operations=operations,
            updates=tuple(updates),
            added=added,
            removed=removed,
        )

    @staticmethod
    def _require_pair(
        relation: MemoryStoredLink,
        links_by_uri: dict[str, set[MemoryStoredLink]],
        backlinks_by_uri: dict[str, set[MemoryStoredLink]],
    ) -> None:
        source = str(relation.from_uri)
        target = str(relation.to_uri)
        if (
            source not in links_by_uri
            or target not in backlinks_by_uri
            or relation not in links_by_uri[source]
            or relation not in backlinks_by_uri[target]
        ):
            raise MemoryRelationIntegrityError("relation operation does not match a complete Link/Backlink pair")

    @staticmethod
    def _discard_pair(
        relation: MemoryStoredLink,
        links_by_uri: dict[str, set[MemoryStoredLink]],
        backlinks_by_uri: dict[str, set[MemoryStoredLink]],
    ) -> None:
        links_by_uri.setdefault(str(relation.from_uri), set()).discard(relation)
        backlinks_by_uri.setdefault(str(relation.to_uri), set()).discard(relation)

    @staticmethod
    def _add_pair(
        relation: MemoryStoredLink,
        links_by_uri: dict[str, set[MemoryStoredLink]],
        backlinks_by_uri: dict[str, set[MemoryStoredLink]],
    ) -> None:
        source = str(relation.from_uri)
        target = str(relation.to_uri)
        if source not in links_by_uri or target not in backlinks_by_uri:
            raise MemoryRelationIntegrityError("final relation references an unavailable live endpoint")
        links_by_uri[source].add(relation)
        backlinks_by_uri[target].add(relation)


__all__ = [
    "MemoryRelationIntegrityError",
    "MemoryRelationNodeUpdate",
    "MemoryRelationPlan",
    "MemoryRelationPlanner",
    "MemoryRelationReadSet",
]
