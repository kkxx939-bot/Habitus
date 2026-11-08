"""SearchService 主召回之后的有界 Links/Backlinks 一跳关系扩展。"""

from __future__ import annotations

from datetime import datetime

from habitus.memory.document import MemoryDocument, MemoryStoredLink
from habitus.memory.intention import (
    MemoryIntentionRecallScope,
    MemoryIntentionReview,
    MemoryIntentionReviewer,
    intention_matches_scope,
)
from habitus.memory.model import MemoryKind
from habitus.memory.retrieval.model import (
    MemoryMatchedMemory,
    MemoryRelatedMemory,
    MemorySearchError,
    MemorySearchServiceConfig,
)
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.uri import MemoryURI


def review_intention(
    reviewer: MemoryIntentionReviewer,
    document: MemoryDocument,
    *,
    now: datetime,
) -> MemoryIntentionReview | None:
    """只给 Intention 生成复核提醒，其他类型没有提醒。"""

    if document.kind is not MemoryKind.INTENTION:
        return None
    return reviewer.review(document, now=now)


class RelationExpander:
    """按系统维护的双向关系为直接命中补充有界一跳邻居，不递归、不修改关系。"""

    def __init__(
        self,
        *,
        snapshot_reader: MemorySnapshotReader,
        config: MemorySearchServiceConfig,
        intention_reviewer: MemoryIntentionReviewer,
    ) -> None:
        self.snapshot_reader = snapshot_reader
        self.config = config
        self.intention_reviewer = intention_reviewer

    def expand(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
        *,
        roots: tuple[MemoryURI, ...],
        intention_scope: MemoryIntentionRecallScope,
        review_now: datetime,
    ) -> tuple[MemoryMatchedMemory, ...]:
        if not memories or self.config.max_relation_neighbors_total == 0:
            return memories
        direct_documents = {memory.uri: memory.document for memory in memories}
        neighbor_documents: dict[MemoryURI, MemoryDocument] = {}
        selected_relations = self._select(
            memories,
            roots=roots,
            intention_scope=intention_scope,
            direct_documents=direct_documents,
            neighbor_documents=neighbor_documents,
        )
        return self._attach(
            memories,
            selected_relations,
            direct_documents=direct_documents,
            neighbor_documents=neighbor_documents,
            review_now=review_now,
        )

    def _select(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
        *,
        roots: tuple[MemoryURI, ...],
        intention_scope: MemoryIntentionRecallScope,
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
                if not any(neighbor.matches_prefix(root) for root in roots):
                    continue
                document = direct_documents.get(neighbor) or neighbor_documents.get(neighbor)
                if document is None:
                    document = self._read_neighbor(neighbor)
                    neighbor_documents[neighbor] = document
                self._require_reciprocal(memory.document, document, relation)
                if document.kind is MemoryKind.INTENTION and not intention_matches_scope(
                    document.fields.get("status"),
                    intention_scope,
                ):
                    # 不符合召回范围的关系不能先占用一跳配额。
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

    def _read_neighbor(self, neighbor: MemoryURI) -> MemoryDocument:
        try:
            snapshot = self.snapshot_reader.read(neighbor)
        except Exception as exc:
            raise MemorySearchError("failed to read one-hop related memory document") from exc
        if not snapshot.exists or snapshot.value is None:
            raise MemorySearchError("memory relation points to a missing L2 document")
        return snapshot.value

    def _attach(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
        selected_relations: dict[MemoryURI, list[MemoryStoredLink]],
        *,
        direct_documents: dict[MemoryURI, MemoryDocument],
        neighbor_documents: dict[MemoryURI, MemoryDocument],
        review_now: datetime,
    ) -> tuple[MemoryMatchedMemory, ...]:
        expanded: list[MemoryMatchedMemory] = []
        for memory in memories:
            related: list[MemoryRelatedMemory] = []
            for relation in selected_relations[memory.uri]:
                neighbor_uri = relation.to_uri if relation.from_uri == memory.uri else relation.from_uri
                document = direct_documents.get(neighbor_uri)
                if document is None:
                    document = neighbor_documents[neighbor_uri]
                related.append(
                    MemoryRelatedMemory(
                        relation=relation,
                        document=document,
                        intention_review=review_intention(self.intention_reviewer, document, now=review_now),
                    )
                )
            expanded.append(
                MemoryMatchedMemory(
                    hit=memory.hit,
                    document=memory.document,
                    matched_queries=memory.matched_queries,
                    intention_review=memory.intention_review,
                    related=tuple(related),
                )
            )
        return tuple(expanded)

    @staticmethod
    def _require_reciprocal(
        seed: MemoryDocument,
        neighbor: MemoryDocument,
        relation: MemoryStoredLink,
    ) -> None:
        seed_uri = MemoryURI.from_address(seed.address)
        neighbor_uri = MemoryURI.from_address(neighbor.address)
        if {seed_uri, neighbor_uri} != {relation.from_uri, relation.to_uri}:
            raise MemorySearchError("memory relation endpoint does not match the loaded documents")
        if relation.from_uri == seed_uri:
            valid = relation in seed.links and relation in neighbor.backlinks
        else:
            valid = relation in seed.backlinks and relation in neighbor.links
        if not valid:
            raise MemorySearchError("memory relation Links/Backlinks are inconsistent")


__all__ = ["RelationExpander", "review_intention"]
