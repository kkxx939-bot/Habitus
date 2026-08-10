"""完整 Pattern generation 的验证、发布与激活。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from foundation.integrity import text_digest
from prediction.model import PredictionPatternKind
from prediction.pattern.contracts import PredictionPatternRepository
from prediction.pattern.document import PredictionPatternDocument
from prediction.pattern.generation import (
    PredictionPatternGenerationEntry,
    PredictionPatternGenerationManifest,
    PredictionPatternGenerationRecord,
    PredictionPatternGenerationStore,
    default_generation_store_root,
)
from prediction.uri import PredictionURI


class PredictionPatternPublicationError(RuntimeError):
    """完整 generation 无法被规范物化或回读。"""


class PredictionPatternGenerationPublisher:
    """只有完整、可回读的 generation 才能成为 active。"""

    def __init__(
        self,
        tree: PredictionPatternRepository,
        *,
        store: PredictionPatternGenerationStore | None = None,
    ) -> None:
        if not isinstance(tree, PredictionPatternRepository):
            raise TypeError("tree must satisfy PredictionPatternRepository")
        self.tree = tree
        self.store = store or PredictionPatternGenerationStore(
            default_generation_store_root(tree.root)
        )

    def publish(
        self,
        documents: Sequence[PredictionPatternDocument],
        *,
        created_at: datetime | None = None,
        learning_identity: Mapping[str, Any] | None = None,
    ) -> tuple[PredictionPatternGenerationRecord, ...]:
        """按 State 分组发布；每个 State 及其 Branch 独立成代、独立激活。

        ``learning_identity`` 是学习器的匹配关键参数快照（时段参数、词表
        版本），随 manifest 落盘并进入 manifest 摘要，供预测端构造期握手，
        防止学习/预测两侧配置漂移时静默命中语义错误的状态。
        """

        ordered = self._validated_documents(documents)
        timestamp = created_at or datetime.now(timezone.utc)
        grouped: dict[str, list[PredictionPatternDocument]] = defaultdict(list)
        for document in ordered:
            grouped[document.address.state_key].append(document)
        return tuple(
            self._publish_state(
                grouped[state_key], created_at=timestamp, learning_identity=learning_identity
            )
            for state_key in sorted(grouped)
        )

    def _publish_state(
        self,
        documents: Sequence[PredictionPatternDocument],
        *,
        created_at: datetime,
        learning_identity: Mapping[str, Any] | None,
    ) -> PredictionPatternGenerationRecord:
        state = next(item for item in documents if item.kind is PredictionPatternKind.STATE)
        entries = tuple(
            sorted(
                PredictionPatternGenerationEntry(
                    str(PredictionURI.from_pattern_address(document.address)),
                    text_digest(self.tree.pattern_codec.encode(document)),
                )
                for document in documents
            )
        )
        manifest = PredictionPatternGenerationManifest(
            generation_id=state.fields["pattern_generation"],
            projection_version=state.fields["projection_version"],
            created_at=created_at,
            logical_state_key=state.fields["logical_state_key"],
            entries=entries,
            learning_identity=learning_identity,
        )
        prepared = self.store.prepare(manifest)
        manifest = prepared.manifest

        for document in documents:
            self.tree.create_pattern(document)
        self._verify_materialized(manifest)
        ready = self.store._mark_ready(manifest.logical_state_key, manifest.generation_id)
        return self.store._activate(ready.manifest.logical_state_key, ready.manifest.generation_id)

    def _validated_documents(
        self,
        documents: Sequence[PredictionPatternDocument],
    ) -> tuple[PredictionPatternDocument, ...]:
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of PredictionPatternDocument values")
        by_uri: dict[str, PredictionPatternDocument] = {}
        for document in documents:
            if not isinstance(document, PredictionPatternDocument):
                raise TypeError("documents must contain PredictionPatternDocument values")
            uri = str(PredictionURI.from_pattern_address(document.address))
            existing = by_uri.get(uri)
            if existing is not None and existing != document:
                raise ValueError("one generation cannot bind one Pattern URI to different content")
            by_uri[uri] = document
        if not by_uri:
            raise ValueError("pattern generation cannot be empty")
        ordered = tuple(
            sorted(
                by_uri.values(),
                key=lambda document: (
                    0 if document.kind is PredictionPatternKind.STATE else 1,
                    str(PredictionURI.from_pattern_address(document.address)),
                ),
            )
        )
        generations = {document.fields["pattern_generation"] for document in ordered}
        versions = {document.fields["projection_version"] for document in ordered}
        if len(generations) != 1 or len(versions) != 1:
            raise ValueError("one Pattern generation must use one generation ID and projection version")
        states = {
            document.address.state_key: document
            for document in ordered
            if document.kind is PredictionPatternKind.STATE
        }
        if not states:
            raise ValueError("pattern generation requires at least one StatePattern")
        branches_by_state: dict[str, list[PredictionPatternDocument]] = defaultdict(list)
        for document in ordered:
            if document.kind is PredictionPatternKind.STATE:
                continue
            if document.address.state_key not in states:
                raise ValueError("Pattern generation Branch requires its State in the same manifest")
            branches_by_state[document.address.state_key].append(document)
        for branches in branches_by_state.values():
            if sum(branch.fields["conditional_probability"] for branch in branches) > 1.000000001:
                raise ValueError("Pattern generation outgoing branch probabilities cannot exceed one")
        return ordered

    def _verify_materialized(self, manifest: PredictionPatternGenerationManifest) -> None:
        for entry in manifest.entries:
            address = PredictionURI.parse(entry.uri).to_pattern_address()
            document = self.tree.read_pattern(address)
            if document.fields["pattern_generation"] != manifest.generation_id:
                raise PredictionPatternPublicationError(
                    "Pattern document belongs to another generation"
                )
            if document.fields["projection_version"] != manifest.projection_version:
                raise PredictionPatternPublicationError(
                    "Pattern document uses another projection version"
                )
            if text_digest(self.tree.pattern_codec.encode(document)) != entry.digest:
                raise PredictionPatternPublicationError(
                    "Pattern document digest does not match its manifest"
                )


__all__ = ["PredictionPatternGenerationPublisher", "PredictionPatternPublicationError"]
