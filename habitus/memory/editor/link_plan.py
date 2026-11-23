"""基于最终节点身份计算 Links/Backlinks 的纯关系计划。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.memory.document import MemoryDocument, MemoryStoredLink
from habitus.memory.editor.candidate import MemoryRelationAction
from habitus.memory.editor.identity import (
    MemoryFinalIdentityMap,
    MemoryNodeDisposition,
)
from habitus.memory.editor.link import MemoryResolvedRelation
from habitus.memory.snapshot import MemorySnapshotBatch
from habitus.memory.uri import MemoryURI

_LOGGER = logging.getLogger(__name__)


class MemoryRelationIntegrityError(ValueError):
    """关系读集、旧双向副本或最终关系状态不完整。"""


@dataclass(frozen=True)
class MemoryRelationReadSet:
    """关系计划涉及的旧节点、新节点缺失快照和一跳邻居。"""

    structural_source_uris: tuple[MemoryURI, ...]
    snapshots: MemorySnapshotBatch

    def __post_init__(self) -> None:
        source_ids = tuple(str(uri) for uri in self.structural_source_uris)
        if source_ids != tuple(sorted(set(source_ids))):
            raise MemoryRelationIntegrityError("structural source URIs must be unique and sorted")
        for snapshot in self.snapshots.snapshots:
            MemoryURI.parse(snapshot.identity).to_address()
            if not snapshot.exists:
                continue
            if snapshot.value is None:
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
        """从上游完整快照中选出关系计划所需的节点及其一跳邻居；邻居不存在不算错（悬空关联，规划时清掉）。"""

        structural_sources = _structural_sources(identities)
        required = relation_endpoints(identities, operations)
        created = {
            str(entry.final_uri)
            for entry in identities.entries
            if entry.disposition is MemoryNodeDisposition.CREATE and entry.final_uri is not None
        }

        neighbours: set[str] = set()
        for identity in sorted(required):
            snapshot = snapshots.get(identity)
            if snapshot is None:
                raise MemoryRelationIntegrityError(f"relation read set is missing required snapshot: {identity}")
            if not snapshot.exists:
                if identity not in created:
                    raise MemoryRelationIntegrityError(f"relation endpoint does not exist: {identity}")
                continue
            assert isinstance(snapshot.value, MemoryDocument)
            neighbours.update(one_hop(snapshot.value))
        required |= neighbours

        missing = sorted(identity for identity in required if snapshots.get(identity) is None)
        if missing:
            raise MemoryRelationIntegrityError(f"relation read set is missing one-hop snapshots: {missing}")
        selected = tuple(snapshot for snapshot in snapshots.snapshots if snapshot.identity in required)
        return cls(
            structural_source_uris=tuple(MemoryURI.parse(identity) for identity in sorted(structural_sources)),
            snapshots=SnapshotBatch(
                snapshots=selected,
                total_bytes=sum(snapshot.size_bytes for snapshot in selected),
            ),
        )

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
        if not snapshot.exists or snapshot.value is None:
            raise MemoryRelationIntegrityError("relation operation requires an existing memory document")
        return snapshot.value


@dataclass(frozen=True)
class MemoryRelationNodeUpdate:
    """一个 URI 在关系规划后的完整 Links/Backlinks 目标状态。"""

    before: VersionedSnapshot[MemoryDocument]
    uri: MemoryURI
    links: tuple[MemoryStoredLink, ...]
    backlinks: tuple[MemoryStoredLink, ...]

    def __post_init__(self) -> None:
        if self.before.identity != str(self.uri):
            raise MemoryRelationIntegrityError("relation update identity does not match its snapshot")
        for name, values in {
            "links": self.links,
            "backlinks": self.backlinks,
        }.items():
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
        update_ids = tuple(str(update.uri) for update in self.updates)
        if update_ids != tuple(sorted(set(update_ids))):
            raise MemoryRelationIntegrityError("relation node updates must be unique and sorted")
        for name, values in {"added": self.added, "removed": self.removed}.items():
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
    """按结构迁移、REMOVE、ADD、修复悬空与单边关联的固定顺序计算双向关系。"""

    def plan(
        self,
        identities: MemoryFinalIdentityMap,
        operations: tuple[MemoryResolvedRelation, ...],
        read_set: MemoryRelationReadSet,
    ) -> MemoryRelationPlan:
        """从旧图生成全部受影响节点的最终关系状态。

        关联两边分开写、不保证一致（裁定 111）：旧图里指向不存在文件的、只有一边的关联不再报错。
        这一批要写到的节点顺手修掉：指向不存在文件的丢掉；正向有、对方缺反向的给对方补上；
        反向有、对方没有正向的丢掉（以正向为准）。
        """

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

        dead_sources = _structural_sources(identities)
        created = {
            str(entry.final_uri)
            for entry in identities.entries
            if entry.disposition is MemoryNodeDisposition.CREATE and entry.final_uri is not None
        }
        live = {
            snapshot.identity
            for snapshot in read_set.snapshots.snapshots
            if (snapshot.exists or snapshot.identity in created) and snapshot.identity not in dead_sources
        }

        structural_old: dict[tuple[str, str, str], MemoryStoredLink] = {}
        for source_uri in read_set.structural_source_uris:
            document = read_set.document(source_uri)
            for relation in (*document.links, *document.backlinks):
                structural_old.setdefault(relation.identity, relation)

        for relation in sorted(structural_old.values(), key=lambda item: item.identity):
            self._discard_pair(relation, links_by_uri, backlinks_by_uri)
            remapped_from = identities.remap_uri(relation.from_uri)
            remapped_to = identities.remap_uri(relation.to_uri)
            if remapped_from is None or remapped_to is None or remapped_from == remapped_to:
                continue
            moved = MemoryStoredLink(from_uri=remapped_from, to_uri=remapped_to, link_type=relation.link_type)
            if str(remapped_from) not in live or str(remapped_to) not in live:
                _skip(moved, "the other end does not exist")
                continue
            self._add_pair(moved, links_by_uri, backlinks_by_uri)

        explicit_removes = tuple(
            operation for operation in operations if operation.action is MemoryRelationAction.REMOVE
        )
        explicit_adds = tuple(operation for operation in operations if operation.action is MemoryRelationAction.ADD)
        for operation in explicit_removes:
            relation = operation.to_stored()
            self._require_stored(relation, links_by_uri, backlinks_by_uri)
            self._discard_pair(relation, links_by_uri, backlinks_by_uri)
        for operation in explicit_adds:
            self._add_pair(
                operation.to_stored(),
                links_by_uri,
                backlinks_by_uri,
            )

        self._repair_written(identities, read_set, live, links_by_uri, backlinks_by_uri)

        updates: list[MemoryRelationNodeUpdate] = []
        for identity in sorted(set(links_by_uri) | set(backlinks_by_uri)):
            if identity in dead_sources:
                continue
            current_snapshot = read_set.snapshots.get(identity)
            if current_snapshot is None:
                raise MemoryRelationIntegrityError(f"final relation endpoint lacks a snapshot: {identity}")
            if not self._changed(identity, read_set, links_by_uri, backlinks_by_uri):
                continue
            if not current_snapshot.exists and identity not in created:
                continue
            updates.append(
                MemoryRelationNodeUpdate(
                    before=current_snapshot,
                    uri=MemoryURI.parse(identity),
                    links=tuple(sorted(links_by_uri.get(identity, set()), key=lambda item: item.identity)),
                    backlinks=tuple(sorted(backlinks_by_uri.get(identity, set()), key=lambda item: item.identity)),
                )
            )

        final_relations = {
            relation for identity, relations in links_by_uri.items() if identity in live for relation in relations
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
    def _changed(
        identity: str,
        read_set: MemoryRelationReadSet,
        links_by_uri: dict[str, set[MemoryStoredLink]],
        backlinks_by_uri: dict[str, set[MemoryStoredLink]],
    ) -> bool:
        snapshot = read_set.snapshots.get(identity)
        old_links: set[MemoryStoredLink] = set()
        old_backlinks: set[MemoryStoredLink] = set()
        if snapshot is not None and snapshot.exists:
            assert isinstance(snapshot.value, MemoryDocument)
            old_links = set(snapshot.value.links)
            old_backlinks = set(snapshot.value.backlinks)
        return links_by_uri.get(identity, set()) != old_links or backlinks_by_uri.get(identity, set()) != old_backlinks

    def _repair_written(
        self,
        identities: MemoryFinalIdentityMap,
        read_set: MemoryRelationReadSet,
        live: set[str],
        links_by_uri: dict[str, set[MemoryStoredLink]],
        backlinks_by_uri: dict[str, set[MemoryStoredLink]],
    ) -> None:
        """这一批要写到的节点（字段有变化的、关系已经有变化的）逐个修掉悬空与单边关联。"""

        written = {
            str(entry.final_uri)
            for entry in identities.entries
            if entry.final_uri is not None
            and entry.disposition
            in {MemoryNodeDisposition.CREATE, MemoryNodeDisposition.UPDATE, MemoryNodeDisposition.MERGE}
        }
        written |= {
            identity
            for identity in links_by_uri
            if identity in live and self._changed(identity, read_set, links_by_uri, backlinks_by_uri)
        }
        known = {snapshot.identity for snapshot in read_set.snapshots.snapshots}
        for identity in sorted(written & live):
            self._repair(identity, known, live, links_by_uri, backlinks_by_uri)

    @staticmethod
    def _repair(
        identity: str,
        known: set[str],
        live: set[str],
        links_by_uri: dict[str, set[MemoryStoredLink]],
        backlinks_by_uri: dict[str, set[MemoryStoredLink]],
    ) -> None:
        """修掉一个要写节点上的悬空与单边关联：正向为准。另一端不在读集里的（没读到，判不了）原样留着。"""

        for link in sorted(links_by_uri.get(identity, set()), key=lambda item: item.identity):
            target = str(link.to_uri)
            if target not in known:
                continue
            if target not in live:
                links_by_uri[identity].discard(link)
                _skip(link, "dropped: the target does not exist")
            elif link not in backlinks_by_uri[target]:
                backlinks_by_uri[target].add(link)
        for backlink in sorted(backlinks_by_uri.get(identity, set()), key=lambda item: item.identity):
            source = str(backlink.from_uri)
            if source not in known:
                continue
            if source not in live:
                backlinks_by_uri[identity].discard(backlink)
                _skip(backlink, "dropped: the source does not exist")
            elif backlink not in links_by_uri[source]:
                backlinks_by_uri[identity].discard(backlink)
                _skip(backlink, "dropped: the source has no forward link")

    @staticmethod
    def _require_stored(
        relation: MemoryStoredLink,
        links_by_uri: dict[str, set[MemoryStoredLink]],
        backlinks_by_uri: dict[str, set[MemoryStoredLink]],
    ) -> None:
        """显式 REMOVE 的关联至少要在一边真实存在（两边不必一致）。"""

        if relation not in links_by_uri.get(str(relation.from_uri), set()) and relation not in backlinks_by_uri.get(
            str(relation.to_uri), set()
        ):
            raise MemoryRelationIntegrityError("relation REMOVE does not match a stored Link or Backlink")

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


def _structural_sources(identities: MemoryFinalIdentityMap) -> set[str]:
    """这一批合并掉或删掉的旧节点。"""

    return {
        str(entry.source_uri)
        for entry in identities.entries
        if entry.disposition in {MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE}
        and entry.source_uri is not None
    }


def relation_endpoints(
    identities: MemoryFinalIdentityMap,
    operations: tuple[MemoryResolvedRelation, ...],
) -> set[str]:
    """关系规划必须读到的节点：合并删除的来源、合并目标、这一批要写的节点和显式关系的两端。"""

    required = _structural_sources(identities)
    required.update(
        str(entry.final_uri)
        for entry in identities.entries
        if entry.final_uri is not None
        and entry.disposition
        in {MemoryNodeDisposition.CREATE, MemoryNodeDisposition.UPDATE, MemoryNodeDisposition.MERGE}
    )
    required.update(str(uri) for operation in operations for uri in (operation.from_uri, operation.to_uri))
    return required


def one_hop(document: MemoryDocument) -> set[str]:
    """一个节点的全部一跳邻居地址（正向的目标与反向的来源）。"""

    return {str(link.to_uri) for link in document.links} | {str(backlink.from_uri) for backlink in document.backlinks}


def _skip(relation: MemoryStoredLink, reason: str) -> None:
    _LOGGER.warning(
        "memory relation skipped: %s -[%s]-> %s (%s)",
        relation.from_uri,
        relation.link_type.value,
        relation.to_uri,
        reason,
    )


__all__ = [
    "MemoryRelationIntegrityError",
    "one_hop",
    "relation_endpoints",
    "MemoryRelationNodeUpdate",
    "MemoryRelationPlan",
    "MemoryRelationPlanner",
    "MemoryRelationReadSet",
]
