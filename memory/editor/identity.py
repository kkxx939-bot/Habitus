"""节点规划完成后供关系层使用的最终身份映射。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from memory.document import MemoryDocument
from memory.editor.candidate import (
    MemoryIdentityProposalType,
)
from memory.editor.extraction.model import MemoryExtractionResult
from memory.editor.mutation.model import MemoryMutationPlan
from memory.editor.page_id import (
    EXISTING_PAGE_ID_MAX,
    MemoryPageIdMap,
    validate_page_id,
)
from memory.model import MemoryKind
from memory.schema import MemorySchemaRegistry
from memory.uri import MemoryURI


class MemoryFinalIdentityError(ValueError):
    """节点处置不能形成唯一、完整的最终身份映射。"""


class MemoryIdentityPlanningError(MemoryFinalIdentityError):
    """临时身份提议不能通过确定性安全规则。"""


class MemoryNodeDisposition(str, Enum):
    """一个临时页面编号在节点规划后的最终处置。"""

    CREATE = "create"
    UPDATE = "update"
    NOOP = "noop"
    MERGE = "merge"
    DELETE = "delete"


@dataclass(frozen=True)
class MemoryFinalIdentity:
    """一个 page_id 的来源身份、最终身份与节点处置。"""

    page_id: int
    disposition: MemoryNodeDisposition
    source_uri: MemoryURI | None
    final_uri: MemoryURI | None

    def __post_init__(self) -> None:
        page_id = validate_page_id(self.page_id)
        try:
            disposition = MemoryNodeDisposition(self.disposition)
        except ValueError as exc:
            raise MemoryFinalIdentityError("final identity contains an unsupported disposition") from exc
        source_uri = self._document_uri(self.source_uri, "source_uri")
        final_uri = self._document_uri(self.final_uri, "final_uri")

        if disposition is MemoryNodeDisposition.CREATE:
            if page_id <= EXISTING_PAGE_ID_MAX:
                raise MemoryFinalIdentityError("create disposition requires a new page_id")
            if source_uri is not None or final_uri is None:
                raise MemoryFinalIdentityError("create disposition requires only a final URI")
        elif disposition in {
            MemoryNodeDisposition.UPDATE,
            MemoryNodeDisposition.NOOP,
        }:
            if source_uri is None or final_uri is None or source_uri != final_uri:
                raise MemoryFinalIdentityError("update/noop disposition must preserve one existing URI")
        elif disposition is MemoryNodeDisposition.MERGE:
            if source_uri is None or final_uri is None or source_uri == final_uri:
                raise MemoryFinalIdentityError("merge disposition requires different source and final URIs")
        elif source_uri is None or final_uri is not None:
            raise MemoryFinalIdentityError("delete disposition requires a source URI and no final URI")

        object.__setattr__(self, "page_id", page_id)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "source_uri", source_uri)
        object.__setattr__(self, "final_uri", final_uri)

    @staticmethod
    def _document_uri(value: MemoryURI | None, label: str) -> MemoryURI | None:
        if value is None:
            return None
        if not isinstance(value, MemoryURI):
            raise TypeError(f"{label} must be a MemoryURI or None")
        try:
            value.to_address()
        except ValueError as exc:
            raise MemoryFinalIdentityError(f"{label} must identify an L2 memory document") from exc
        return value


@dataclass(frozen=True)
class MemoryFinalIdentityMap:
    """覆盖相关旧节点与本批候选的不可变最终身份表。"""

    entries: tuple[MemoryFinalIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, MemoryFinalIdentity) for entry in self.entries
        ):
            raise TypeError("final identity entries must be MemoryFinalIdentity values")
        page_ids = tuple(entry.page_id for entry in self.entries)
        if page_ids != tuple(sorted(set(page_ids))):
            raise MemoryFinalIdentityError("final identity entries must have unique sorted page_ids")

        source_targets: dict[str, str | None] = {}
        live_final_uris: list[str] = []
        for entry in self.entries:
            if entry.disposition in {
                MemoryNodeDisposition.CREATE,
                MemoryNodeDisposition.UPDATE,
                MemoryNodeDisposition.NOOP,
            }:
                assert entry.final_uri is not None
                live_final_uris.append(str(entry.final_uri))
            if entry.source_uri is None:
                continue
            source = str(entry.source_uri)
            final = str(entry.final_uri) if entry.final_uri is not None else None
            previous = source_targets.setdefault(source, final)
            if previous != final:
                raise MemoryFinalIdentityError("one source URI cannot have multiple final identities")
        if len(live_final_uris) != len(set(live_final_uris)):
            raise MemoryFinalIdentityError("one live final URI cannot have multiple node identities")
        live_targets = set(live_final_uris)
        for entry in self.entries:
            if (
                entry.disposition is MemoryNodeDisposition.MERGE
                and entry.final_uri is not None
                and str(entry.final_uri) not in live_targets
            ):
                raise MemoryFinalIdentityError("merge disposition must target a separately planned live node")

    @classmethod
    def from_mutation_plan(
        cls,
        plan: object,
        page_ids: MemoryPageIdMap,
    ) -> MemoryFinalIdentityMap:
        """把当前 CREATE/UPDATE/NOOP 纯内容计划提升为完整身份表。"""

        from memory.editor.mutation.model import (
            MemoryMutationAction,
            MemoryMutationPlan,
        )

        if not isinstance(plan, MemoryMutationPlan):
            raise TypeError("plan must be a MemoryMutationPlan")
        if not isinstance(page_ids, MemoryPageIdMap):
            raise TypeError("page_ids must be a MemoryPageIdMap")

        entries: dict[int, MemoryFinalIdentity] = {
            page_id: MemoryFinalIdentity(
                page_id=page_id,
                disposition=MemoryNodeDisposition.NOOP,
                source_uri=MemoryURI.parse(uri),
                final_uri=MemoryURI.parse(uri),
            )
            for page_id, uri in page_ids.existing_items()
        }
        disposition_by_action = {
            MemoryMutationAction.CREATE: MemoryNodeDisposition.CREATE,
            MemoryMutationAction.UPDATE: MemoryNodeDisposition.UPDATE,
            MemoryMutationAction.NOOP: MemoryNodeDisposition.NOOP,
        }
        for mutation in plan.mutations:
            page_id = mutation.match.candidate.page_id
            disposition = disposition_by_action[mutation.action]
            source_uri = None if disposition is MemoryNodeDisposition.CREATE else mutation.uri
            entries[page_id] = MemoryFinalIdentity(
                page_id=page_id,
                disposition=disposition,
                source_uri=source_uri,
                final_uri=mutation.uri,
            )
        return cls(tuple(entries[key] for key in sorted(entries)))

    def entry(self, page_id: int) -> MemoryFinalIdentity:
        """返回一个 page_id 的最终处置；缺失时明确失败。"""

        normalized = validate_page_id(page_id)
        for entry in self.entries:
            if entry.page_id == normalized:
                return entry
        raise MemoryFinalIdentityError(f"final identity map does not contain page_id {normalized}")

    def resolve(self, page_id: int) -> MemoryURI | None:
        """返回最终 URI；DELETE 返回 None。"""

        return self.entry(page_id).final_uri

    def validate_context(
        self,
        page_ids: MemoryPageIdMap,
        candidate_page_ids: tuple[int, ...],
    ) -> None:
        """确认身份表没有遗漏、重定向或凭空加入解析上下文之外的编号。"""

        if not isinstance(page_ids, MemoryPageIdMap):
            raise TypeError("page_ids must be a MemoryPageIdMap")
        if not isinstance(candidate_page_ids, tuple):
            raise TypeError("candidate_page_ids must be a tuple")
        normalized_candidates = tuple(validate_page_id(page_id) for page_id in candidate_page_ids)
        if len(normalized_candidates) != len(set(normalized_candidates)):
            raise MemoryFinalIdentityError("candidate page_ids must be unique")
        expected = page_ids.page_ids() | set(normalized_candidates)
        actual = {entry.page_id for entry in self.entries}
        if actual != expected:
            raise MemoryFinalIdentityError("final identity map must exactly cover old nodes and candidates")
        for old_page_id, old_uri in page_ids.existing_items():
            entry = self.entry(old_page_id)
            if entry.source_uri is None or str(entry.source_uri) != old_uri:
                raise MemoryFinalIdentityError("existing page_id source URI cannot be omitted or redirected")

    def remap_uri(self, uri: MemoryURI | str) -> MemoryURI | None:
        """把旧关系端点映射到存活 URI；未处置邻居保持原身份。"""

        parsed = MemoryURI.parse(uri)
        parsed.to_address()
        identity = str(parsed)
        for entry in self.entries:
            if entry.source_uri is not None and str(entry.source_uri) == identity:
                return entry.final_uri
        return parsed

    @property
    def deleted_uris(self) -> tuple[MemoryURI, ...]:
        """返回需要物理删除的旧节点 URI。"""

        return tuple(
            MemoryURI.parse(identity)
            for identity in sorted(
                str(entry.source_uri)
                for entry in self.entries
                if entry.disposition is MemoryNodeDisposition.DELETE and entry.source_uri is not None
            )
        )

    @property
    def retired_uris(self) -> tuple[MemoryURI, ...]:
        """返回 DELETE 与 MERGE 后都不应继续存在的来源 URI。"""

        return tuple(
            MemoryURI.parse(identity)
            for identity in sorted(
                str(entry.source_uri)
                for entry in self.entries
                if entry.disposition in {MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE}
                and entry.source_uri is not None
            )
        )

    @property
    def created_uris(self) -> tuple[MemoryURI, ...]:
        """返回本计划中从缺失状态创建的最终 URI。"""

        return tuple(
            MemoryURI.parse(identity)
            for identity in sorted(
                str(entry.final_uri)
                for entry in self.entries
                if entry.disposition is MemoryNodeDisposition.CREATE and entry.final_uri is not None
            )
        )

    @property
    def merged_uri_map(self) -> dict[str, MemoryURI]:
        """返回旧 URI 到存活 URI 的显式 MERGE 映射。"""

        return {
            str(entry.source_uri): entry.final_uri
            for entry in self.entries
            if entry.disposition is MemoryNodeDisposition.MERGE
            and entry.source_uri is not None
            and entry.final_uri is not None
        }


class MemoryIdentityPlanner:
    """只裁决已审查身份提议，不解释自然语言或直接修改存储。"""

    _MERGE_KINDS = frozenset(
        {
            MemoryKind.PREFERENCE,
            MemoryKind.ENTITY,
            MemoryKind.INTENTION,
        }
    )

    def __init__(self, registry: MemorySchemaRegistry | None = None) -> None:
        if registry is not None and not isinstance(registry, MemorySchemaRegistry):
            raise TypeError("registry must be a MemorySchemaRegistry")
        self.registry = registry or MemorySchemaRegistry.load_default()

    def plan(
        self,
        extraction: MemoryExtractionResult,
    ) -> MemoryFinalIdentityMap:
        """把安全的临时提议提升为完整 CREATE/UPDATE/NOOP/MERGE/DELETE 身份表。"""

        if not isinstance(extraction, MemoryExtractionResult):
            raise TypeError("extraction must be a reviewed MemoryExtractionResult")
        batch = extraction.candidates
        mutations = extraction.mutations
        page_ids = extraction.page_ids

        base = MemoryFinalIdentityMap.from_mutation_plan(mutations, page_ids)
        entries = {entry.page_id: entry for entry in base.entries}
        mutation_by_page = {mutation.match.candidate.page_id: mutation for mutation in mutations.mutations}
        proposal_sources = {proposal.source_page_id for proposal in batch.identity_proposals}
        proposal_targets = {
            proposal.target_page_id
            for proposal in batch.identity_proposals
            if proposal.proposal_type is MemoryIdentityProposalType.SAME_MEMORY
        }
        if proposal_sources & proposal_targets:
            raise MemoryIdentityPlanningError("identity proposals cannot form merge chains or cycles")

        relation_page_ids = {
            page_id for relation in batch.relations for page_id in (relation.from_page_id, relation.to_page_id)
        }
        if proposal_sources & relation_page_ids:
            raise MemoryIdentityPlanningError(
                "retired identity proposal sources cannot participate in relation candidates"
            )

        for proposal in sorted(batch.identity_proposals, key=lambda item: item.source_page_id):
            source = entries.get(proposal.source_page_id)
            if source is None or source.source_uri is None:
                raise MemoryIdentityPlanningError("identity proposal source must be a complete old-memory identity")
            if proposal.source_page_id in mutation_by_page:
                raise MemoryIdentityPlanningError("identity proposal source cannot also receive a content mutation")
            source_document = self._old_document(mutations, source.source_uri)

            if proposal.proposal_type is MemoryIdentityProposalType.REMOVE_MEMORY:
                entries[proposal.source_page_id] = MemoryFinalIdentity(
                    page_id=proposal.source_page_id,
                    disposition=MemoryNodeDisposition.DELETE,
                    source_uri=source.source_uri,
                    final_uri=None,
                )
                continue

            assert proposal.target_page_id is not None
            target = entries.get(proposal.target_page_id)
            if target is None or target.final_uri is None:
                raise MemoryIdentityPlanningError("same_memory target must resolve to a live identity")
            if target.disposition not in {
                MemoryNodeDisposition.CREATE,
                MemoryNodeDisposition.UPDATE,
                MemoryNodeDisposition.NOOP,
            }:
                raise MemoryIdentityPlanningError("same_memory target must be a separately planned live node")
            if proposal.target_page_id not in mutation_by_page:
                raise MemoryIdentityPlanningError(
                    "same_memory target must be explicitly covered by preliminary field planning"
                )
            target_document, target_fields, target_kind = self._target_state(
                mutations,
                target,
                proposal.target_page_id,
            )
            if source_document.kind is not target_kind:
                raise MemoryIdentityPlanningError("same_memory can only merge nodes of the same memory type")
            if source_document.kind not in self._MERGE_KINDS:
                raise MemoryIdentityPlanningError("automatic same_memory merge is not allowed for this memory type")
            if target.disposition is MemoryNodeDisposition.NOOP:
                assert target_document is not None
                if not self._content_fields_equal(
                    source_document,
                    target_kind,
                    target_fields,
                ):
                    raise MemoryIdentityPlanningError(
                        "same_memory target without an update does not exactly contain the source content"
                    )
            entries[proposal.source_page_id] = MemoryFinalIdentity(
                page_id=proposal.source_page_id,
                disposition=MemoryNodeDisposition.MERGE,
                source_uri=source.source_uri,
                final_uri=target.final_uri,
            )

        result = MemoryFinalIdentityMap(tuple(entries[key] for key in sorted(entries)))
        result.validate_context(
            page_ids,
            tuple(candidate.page_id for candidate in batch.iter_candidates()),
        )
        return result

    @staticmethod
    def _old_document(
        mutations: MemoryMutationPlan,
        uri: MemoryURI,
    ) -> MemoryDocument:
        snapshot = mutations.read_set.old_memories.get(str(uri))
        if snapshot is None or not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
            raise MemoryIdentityPlanningError("identity proposal source requires a complete extracted old snapshot")
        return snapshot.value

    def _target_state(
        self,
        mutations: MemoryMutationPlan,
        target: MemoryFinalIdentity,
        target_page_id: int,
    ) -> tuple[MemoryDocument | None, Mapping[str, object], MemoryKind]:
        mutation = next(
            (item for item in mutations.mutations if item.match.candidate.page_id == target_page_id),
            None,
        )
        if mutation is None:
            raise MemoryIdentityPlanningError("same_memory target is missing its preliminary mutation")
        document = mutation.match.snapshot.value if mutation.match.snapshot.exists else None
        if document is not None and not isinstance(document, MemoryDocument):
            raise MemoryIdentityPlanningError("same_memory target snapshot is not a memory document")
        return document, mutation.fields, mutation.match.candidate.kind

    def _content_fields_equal(
        self,
        source: MemoryDocument,
        target_kind: MemoryKind,
        target_fields: Mapping[str, object],
    ) -> bool:
        schema = self.registry.get(target_kind)
        names = tuple(field.name for field in schema.content_fields)
        return all(
            (name in source.fields) == (name in target_fields) and source.fields.get(name) == target_fields.get(name)
            for name in names
        )


__all__ = [
    "MemoryFinalIdentity",
    "MemoryFinalIdentityError",
    "MemoryFinalIdentityMap",
    "MemoryIdentityPlanner",
    "MemoryIdentityPlanningError",
    "MemoryNodeDisposition",
]
