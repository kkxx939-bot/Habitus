"""不可变预测样本的确定性发布边界。"""

from __future__ import annotations

from collections.abc import Sequence

from foundation.integrity import canonical_digest
from prediction.document import PredictionDocument
from prediction.publication_journal import (
    PredictionPublicationJournal,
    PredictionPublicationRecord,
    PredictionPublicationState,
)
from prediction.tree import PredictionTree
from prediction.uri import PredictionURI


class PredictionReadBackError(RuntimeError):
    """样本创建后无法从预测树完整回读。"""


class PredictionPublisher:
    """稳定排序并幂等发布一批不可变预测样本。"""

    def __init__(
        self,
        tree: PredictionTree,
        *,
        journal: PredictionPublicationJournal | None = None,
    ) -> None:
        if not isinstance(tree, PredictionTree):
            raise TypeError("tree must be a PredictionTree")
        self.tree = tree
        self.journal = journal or PredictionPublicationJournal(
            tree.root.parent / f".{tree.root.name}.prediction-publications"
        )

    def publish(self, documents: Sequence[PredictionDocument]) -> tuple[PredictionDocument, ...]:
        """发布一批样本；中途失败后可用同一批文档安全重放。"""

        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of PredictionDocument values")
        by_uri: dict[str, PredictionDocument] = {}
        for document in documents:
            if not isinstance(document, PredictionDocument):
                raise TypeError("documents must contain PredictionDocument values")
            uri = str(PredictionURI.from_address(document.address))
            existing = by_uri.get(uri)
            if existing is not None and existing != document:
                raise ValueError("one prediction batch cannot bind one URI to different documents")
            by_uri[uri] = document

        published: list[PredictionDocument] = []
        identities = tuple(
            sorted(
                (
                    uri,
                    canonical_digest({"encoded": self.tree.document_codec.encode(document)}),
                )
                for uri, document in by_uri.items()
            )
        )
        job_id = canonical_digest({"contract": "prediction-publication-v1", "documents": identities})
        prepared = self.journal.prepare(
            PredictionPublicationRecord(job_id, identities, PredictionPublicationState.PREPARED)
        )
        if prepared.documents != identities:
            raise PredictionReadBackError("prediction publication journal does not match the requested batch")
        ordered_documents = tuple(by_uri[uri] for uri in sorted(by_uri))
        self.tree.create_many(ordered_documents)
        for uri in sorted(by_uri):
            document = by_uri[uri]
            read_back = self.tree.read(document.address)
            if read_back != document:
                raise PredictionReadBackError(f"prediction sample read-back mismatch: {uri}")
            published.append(read_back)
        committed = self.journal.commit(job_id)
        if committed.state is not PredictionPublicationState.COMMITTED:
            raise PredictionReadBackError("prediction publication journal did not reach COMMITTED")
        return tuple(published)


__all__ = ["PredictionPublisher", "PredictionReadBackError"]
