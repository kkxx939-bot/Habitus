"""把某一天的行为树、前 D 天的情景树与待用前提清单装成一次归组的输入。

只读：行为树经 ``BehaviorTree.read_day``（已知的撞车消歧重复机械跳过，与预测夜批同口径）；
情景树只读每天当前生效的一代。参照的边界：先前的事与"上一次"以 ``lookback_days`` 为界，
待用前提以 ``pending_expiry_days`` 为界——它们是引用范围，不是"跨日关系只许这么远"。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.scene.grouping.model import (
    GapRow,
    LastOccurrenceReference,
    OccurrenceRow,
    PendingReference,
    SceneGroupingInput,
    SceneReference,
)
from habitus.scene.ledger import pending_before
from habitus.scene.tree import SceneTree
from habitus.scene.uri import SceneURI


def build_grouping_input(
    day: date,
    *,
    subject: str,
    behavior_tree: BehaviorTree,
    scene_tree: SceneTree,
    lookback_days: int,
    pending_expiry_days: int,
) -> SceneGroupingInput | None:
    """当天没有可归组的行为时返回 None（零情景日仍由调用方发布空的一代）。"""

    documents = [
        document
        for document in behavior_tree.read_day(BehaviorKind.OCCURRENCE, day)
        if document.fields.get("original_name") is None
    ]
    if not documents:
        return None
    documents.sort(key=lambda document: (document.address.started_at.astimezone(UTC), document.address.identity_name))
    index_by_uri = {str(BehaviorURI.from_address(document.address)): no for no, document in enumerate(documents, start=1)}
    rows = tuple(_row(no, document, index_by_uri) for no, document in enumerate(documents, start=1))
    gaps = tuple(
        GapRow(started_at=gap.address.started_at, ended_at=datetime.fromisoformat(str(gap.fields["ended_at"])))
        for gap in behavior_tree.read_day(BehaviorKind.GAP, day)
    )
    earliest = day - timedelta(days=lookback_days)
    scene_days = [covered for covered in scene_tree.list_days() if earliest <= covered < day]
    scene_references: list[SceneReference] = []
    label_by_member: dict[str, str] = {}
    for covered in scene_days:
        for document in scene_tree.read_day(covered):
            label = document.address.label
            scene_references.append(
                SceneReference(
                    no=len(scene_references) + 1,
                    uri=str(SceneURI.from_address(document.address)),
                    day=covered,
                    label=label,
                    started_at=document.address.started_at,
                    effects=tuple(str(text) for text in document.fields["effects"]),
                    pending=tuple(str(item["text"]) for item in document.fields["pending_effects"]),
                )
            )
            for member in document.fields["members"]:
                label_by_member[str(member["uri"])] = label
    pending_references = tuple(
        PendingReference(
            no=no,
            text=item.text,
            producer_uri=item.producer_uri,
            producer_started_at=item.producer_started_at,
            scene_uri=item.scene_uri,
            created_on=item.created_on,
        )
        for no, item in enumerate(pending_before(scene_tree, day, expiry_days=pending_expiry_days), start=1)
    )
    return SceneGroupingInput(
        day=day,
        subject=subject,
        occurrences=rows,
        gaps=gaps,
        scene_references=tuple(scene_references),
        pending_references=pending_references,
        last_occurrences=_last_occurrences(day, rows, behavior_tree, earliest, label_by_member),
    )


def _row(no: int, document: BehaviorDocument, index_by_uri: dict[str, int]) -> OccurrenceRow:
    fields = document.fields
    links: list[tuple[str, int]] = []
    for link in document.links:
        target = index_by_uri.get(str(link.to_uri))
        if target is not None:
            links.append((link.link_type.value, target))
    return OccurrenceRow(
        no=no,
        uri=str(BehaviorURI.from_address(document.address)),
        name=str(fields["name"]),
        kind_token=str(fields["kind_token"]),
        started_at=document.address.started_at,
        last_observed_at=datetime.fromisoformat(str(fields["last_observed_at"])),
        status=str(fields["status"]),
        status_basis=str(fields["status_basis"]),
        goal=None if fields.get("goal") is None else str(fields["goal"]),
        summary=str(fields["summary"]),
        subjects=tuple(str(subject) for subject in fields["subjects"]),
        basis=tuple(str(step["semantics"]) for step in fields["basis"]),
        links=tuple(links),
    )


def _last_occurrences(
    day: date,
    rows: tuple[OccurrenceRow, ...],
    behavior_tree: BehaviorTree,
    earliest: date,
    label_by_member: dict[str, str],
) -> tuple[LastOccurrenceReference, ...]:
    """今日各 kind 在回看窗内上一次出现的情形；窗内没出现过的不列（不说"没有"，只是少说）。"""

    by_kind: dict[str, list[int]] = {}
    for row in rows:
        by_kind.setdefault(row.kind_token, []).append(row.no)
    wanted = sorted(by_kind)
    found: dict[str, LastOccurrenceReference] = {}
    probe = day - timedelta(days=1)
    while probe >= earliest and len(found) < len(wanted):
        documents = sorted(
            behavior_tree.read_day(BehaviorKind.OCCURRENCE, probe),
            key=lambda document: document.address.started_at.astimezone(UTC),
            reverse=True,
        )
        for document in documents:  # 当天最晚的一条才是"上一次"
            token = str(document.fields["kind_token"])
            if token in found or token not in wanted or document.fields.get("original_name") is not None:
                continue
            found[token] = LastOccurrenceReference(
                kind_token=token,
                days_ago=(day - probe).days,
                scene_label=label_by_member.get(str(BehaviorURI.from_address(document.address))),
                today_nos=tuple(by_kind[token]),
            )
        probe -= timedelta(days=1)
    return tuple(found[token] for token in wanted if token in found)


__all__ = ["build_grouping_input"]
