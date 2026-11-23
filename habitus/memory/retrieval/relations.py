"""读到一张卡片时按它的 Links/Backlinks 有界补一跳邻居。"""

from __future__ import annotations

import logging
from dataclasses import replace

from habitus.memory.document import MemoryDocument, MemoryStoredLink
from habitus.memory.retrieval.model import (
    MemoryMatchedMemory,
    MemoryRelatedMemory,
    MemorySearchError,
    MemorySearchServiceConfig,
)
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.uri import MemoryURI

_LOGGER = logging.getLogger(__name__)


class RelationExpander:
    """按系统维护的双向关系为直接命中补充有界一跳邻居，不递归、不修改关系。"""

    def __init__(
        self,
        *,
        snapshot_reader: MemorySnapshotReader,
        config: MemorySearchServiceConfig,
    ) -> None:
        self.snapshot_reader = snapshot_reader
        self.config = config

    def expand(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
    ) -> tuple[MemoryMatchedMemory, ...]:
        if not memories or self.config.max_relation_neighbors_total == 0:
            return memories
        direct_documents = {memory.uri: memory.document for memory in memories}
        neighbor_documents: dict[MemoryURI, MemoryDocument] = {}
        selected_relations = self._select(
            memories,
            direct_documents=direct_documents,
            neighbor_documents=neighbor_documents,
        )
        return self._attach(
            memories,
            selected_relations,
            direct_documents=direct_documents,
            neighbor_documents=neighbor_documents,
        )

    def _select(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
        *,
        direct_documents: dict[MemoryURI, MemoryDocument],
        neighbor_documents: dict[MemoryURI, MemoryDocument],
    ) -> dict[MemoryURI, list[MemoryStoredLink]]:
        """按稳定关系身份顺序为每条直接命中挑选一跳关系，读到的邻居文档写入 ``neighbor_documents``。"""

        selected_relations: dict[MemoryURI, list[MemoryStoredLink]] = {memory.uri: [] for memory in memories}
        assigned_relations: set[tuple[str, str, str]] = set()
        for memory in memories:
            relations = tuple(
                sorted((*memory.document.links, *memory.document.backlinks), key=lambda link: link.identity)
            )
            for relation in relations:
                neighbor = relation.to_uri if relation.from_uri == memory.uri else relation.from_uri
                if relation.identity in assigned_relations:
                    continue
                document = direct_documents.get(neighbor) or neighbor_documents.get(neighbor)
                if document is None:
                    document = self._read_neighbor(neighbor)
                    if document is None:
                        _LOGGER.warning("memory relation points to a missing L2 document, skipped: %s", neighbor)
                        continue
                    neighbor_documents[neighbor] = document
                if not self._reciprocal(memory.document, document, relation):
                    _LOGGER.warning("one-sided memory relation between %s and %s, skipped", memory.uri, neighbor)
                    continue
                if len(selected_relations[memory.uri]) >= self.config.max_relation_neighbors_per_match:
                    break
                if len(assigned_relations) >= self.config.max_relation_neighbors_total:
                    break
                selected_relations[memory.uri].append(relation)
                assigned_relations.add(relation.identity)
            if len(assigned_relations) >= self.config.max_relation_neighbors_total:
                break
        return selected_relations

    def _read_neighbor(self, neighbor: MemoryURI) -> MemoryDocument | None:
        try:
            snapshot = self.snapshot_reader.read(neighbor)
        except Exception as exc:
            raise MemorySearchError("failed to read one-hop related memory document") from exc
        if not snapshot.exists or snapshot.value is None:
            return None
        return snapshot.value

    def _attach(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
        selected_relations: dict[MemoryURI, list[MemoryStoredLink]],
        *,
        direct_documents: dict[MemoryURI, MemoryDocument],
        neighbor_documents: dict[MemoryURI, MemoryDocument],
    ) -> tuple[MemoryMatchedMemory, ...]:
        expanded: list[MemoryMatchedMemory] = []
        for memory in memories:
            related: list[MemoryRelatedMemory] = []
            for relation in selected_relations[memory.uri]:
                neighbor_uri = relation.to_uri if relation.from_uri == memory.uri else relation.from_uri
                document = direct_documents.get(neighbor_uri)
                if document is None:
                    document = neighbor_documents[neighbor_uri]
                related.append(MemoryRelatedMemory(relation=relation, document=document))
            expanded.append(replace(memory, related=tuple(related)))
        return tuple(expanded)

    @staticmethod
    def _reciprocal(
        seed: MemoryDocument,
        neighbor: MemoryDocument,
        relation: MemoryStoredLink,
    ) -> bool:
        """另一边也记着这条关联；两边分开写、可能不一致（裁定 111），不一致的跳过。"""

        seed_uri = MemoryURI.from_address(seed.address)
        neighbor_uri = MemoryURI.from_address(neighbor.address)
        if {seed_uri, neighbor_uri} != {relation.from_uri, relation.to_uri}:
            raise MemorySearchError("memory relation endpoint does not match the loaded documents")
        if relation.from_uri == seed_uri:
            return relation in seed.links and relation in neighbor.backlinks
        return relation in seed.backlinks and relation in neighbor.links


__all__ = ["RelationExpander"]
