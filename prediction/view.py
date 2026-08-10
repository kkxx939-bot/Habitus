"""不可变事实样本之上的有效版本读取视图。"""

from __future__ import annotations

from prediction.document import PredictionDocument
from prediction.model import PredictionKind
from prediction.tree import PredictionTree

_PAGE_SIZE = 10_000


class PredictionSampleView:
    """底层保留历史；默认消费者按逻辑身份读取最新有效 Consequence。"""

    def __init__(self, tree: PredictionTree) -> None:
        if not isinstance(tree, PredictionTree):
            raise TypeError("tree must be a PredictionTree")
        self.tree = tree

    def latest_effective_consequences(
        self,
        *,
        projection_version: str | None = None,
    ) -> tuple[PredictionDocument, ...]:
        if projection_version is not None and (
            not isinstance(projection_version, str) or not projection_version.strip()
        ):
            raise ValueError("projection_version must be non-empty text or None")
        selected: dict[str, tuple[int, PredictionDocument]] = {}
        after = None
        while True:
            page = self.tree.list_addresses(
                PredictionKind.CONSEQUENCE,
                limit=_PAGE_SIZE,
                after=after,
            )
            for address in page:
                document = self.tree.read(address)
                version = document.fields["provenance"]["projection_version"]
                if projection_version is not None and version != projection_version:
                    continue
                logical_id = document.fields["logical_sample_id"]
                revision = _outcome_revision(document)
                current = selected.get(logical_id)
                if current is None or (revision, address.materialization_id) > (
                    current[0],
                    current[1].address.materialization_id,
                ):
                    selected[logical_id] = revision, document
            if len(page) < _PAGE_SIZE:
                break
            after = page[-1]
        return tuple(
            item[1]
            for item in sorted(
                selected.values(),
                key=lambda item: (item[1].fields["logical_sample_id"], item[1].address.materialization_id),
            )
        )


def _outcome_revision(document: PredictionDocument) -> int:
    revisions = tuple(
        binding["revision"]
        for binding in document.fields["provenance"]["source_bindings"]
        if binding["member_type"] == "outcome"
    )
    if len(revisions) != 1:
        raise ValueError("ConsequenceSample must bind exactly one Outcome source revision")
    return revisions[0]


__all__ = ["PredictionSampleView"]
