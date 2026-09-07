"""草稿 → 情景文档：编号换 URI、算 lag、逐篇编码；编不成的那一篇降级留信号，不让整天失败。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from habitus.scene.document import SceneDocument, SceneDocumentMetadata, SceneStoredLink
from habitus.scene.grouping.model import DraftRelation, GroupingAssembly, SceneDraft, SceneGroupingInput
from habitus.scene.model import SceneAddress
from habitus.scene.tree import SceneTree
from habitus.scene.uri import SceneURI


def materialize_documents(
    scene_tree: SceneTree,
    day: date,
    payload: SceneGroupingInput,
    assembly: GroupingAssembly,
    *,
    source_digest: str,
    scene_version: str,
    created_at: datetime,
) -> tuple[tuple[SceneDocument, ...], tuple[str, ...]]:
    codec = scene_tree.document_codec
    signals: list[str] = []
    kept: list[tuple[int, SceneAddress, dict[str, Any], SceneDraft]] = []
    seen_addresses: set[SceneAddress] = set()
    for index, draft in enumerate(assembly.scenes):
        fields = _fields(day, payload, draft, source_digest=source_digest, scene_version=scene_version)
        try:
            address = scene_tree.registry.address_for(fields)
        except Exception as exc:  # noqa: BLE001 - 单篇降级
            signals.append(f"scene_dropped: «{draft.label}» cannot be materialized ({exc})")
            continue
        if address in seen_addresses:
            signals.append(f"scene_dropped: «{draft.label}» collides with an earlier scene at the same address")
            continue
        seen_addresses.add(address)
        kept.append((index, address, fields, draft))
    address_by_index = {index: address for index, address, _, _ in kept}
    documents: list[SceneDocument] = []
    for index, address, fields, draft in kept:
        from_uri = SceneURI.from_address(address)
        links: list[SceneStoredLink] = []
        for relation in draft.relations:
            link = _link(from_uri, relation, payload, address_by_index, index)
            if link is None:
                signals.append(f"relation_dropped: «{draft.label}» {relation.kind.value} target was dropped")
                continue
            links.append(link)
        try:
            documents.append(codec.build(fields, metadata=SceneDocumentMetadata(created_at=created_at), links=tuple(links)))
        except Exception as exc:  # noqa: BLE001 - 单篇降级
            signals.append(f"scene_dropped: «{draft.label}» cannot be encoded ({exc})")
    return tuple(documents), tuple(signals)


def _fields(day: date, payload: SceneGroupingInput, draft: SceneDraft, *, source_digest: str, scene_version: str) -> dict[str, Any]:
    started_at = min((payload.row(no).started_at for no, _ in draft.members), key=lambda value: value.astimezone(UTC))
    ended_at = max((payload.row(no).last_observed_at for no, _ in draft.members), key=lambda value: value.astimezone(UTC))
    if ended_at.astimezone(UTC) < started_at.astimezone(UTC):
        ended_at = started_at
    return {
        "occurred_on": day,
        "label": draft.label,
        "started_at": started_at,
        "ended_at": ended_at,
        "members": tuple({"uri": payload.row(no).uri, "role": role.value} for no, role in draft.members),
        "effects": draft.effects,
        "pending_effects": tuple({"text": text, "producer_uri": payload.row(no).uri} for text, no in draft.pending_effects),
        "source_digest": source_digest,
        "scene_version": scene_version,
    }


def _link(
    from_uri: SceneURI,
    relation: DraftRelation,
    payload: SceneGroupingInput,
    address_by_index: Mapping[int, SceneAddress],
    index: int,
) -> SceneStoredLink | None:
    target: str
    if relation.occurrence_no is not None:
        target = payload.row(relation.occurrence_no).uri
    elif relation.scene_index is not None:
        if relation.scene_index >= index or relation.scene_index not in address_by_index:
            return None
        target = str(SceneURI.from_address(address_by_index[relation.scene_index]))
    elif relation.reference_no is not None:
        target = payload.scene_references[relation.reference_no - 1].uri
    else:
        assert relation.pending_no is not None
        target = payload.pending_references[relation.pending_no - 1].producer_uri
    # 装配层已按开始瞬时挡过"目标晚于本情景"，lag 必然 ≥ 0；违反即我们自己的产物不自洽，让它抛。
    return SceneStoredLink.between(from_uri, target, relation.kind)


__all__ = ["materialize_documents"]
