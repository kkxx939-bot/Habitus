"""从情景树读出某一天开始之前仍未兑现的待用前提。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from habitus.behavior.uri import BehaviorURI
from habitus.scene.model import SceneLinkType
from habitus.scene.tree import SceneTree
from habitus.scene.uri import SceneURI


@dataclass(frozen=True)
class PendingItem:
    """一条未兑现的待用前提：建立它的行为 URI 就是将来 needs 边的目标。"""

    text: str
    producer_uri: str
    producer_started_at: datetime
    scene_uri: str
    scene_label: str
    created_on: date


def pending_before(scene_tree: SceneTree, day: date, *, expiry_days: int) -> tuple[PendingItem, ...]:
    """``day`` 之前 ``expiry_days`` 天内建立、且到 ``day`` 前一天为止未被任何 needs 边兑现的项，按建立顺序。

    兑现的判据只看我们自己的产物：某篇情景文档带 ``needs`` 边、目标等于该项的产生行为或
    所在情景。同一天内后建立后兑现的也算兑现（做饭 needs 当天下午的买菜）。
    """

    if not isinstance(scene_tree, SceneTree):
        raise TypeError("scene_tree must be a SceneTree")
    if isinstance(day, datetime) or not isinstance(day, date):
        raise TypeError("day must be a date")
    if isinstance(expiry_days, bool) or not isinstance(expiry_days, int) or expiry_days <= 0:
        raise ValueError("expiry_days must be a positive integer")
    earliest = day - timedelta(days=expiry_days)
    produced: list[PendingItem] = []
    consumed: set[str] = set()
    for covered in scene_tree.list_days():
        if covered < earliest or covered >= day:
            continue
        for document in scene_tree.read_day(covered):
            scene_uri = str(SceneURI.from_address(document.address))
            member_starts = {member["uri"]: member for member in document.fields["members"]}
            for item in document.fields["pending_effects"]:
                producer = str(item["producer_uri"])
                if producer not in member_starts:
                    # Schema 已保证产生方是成员；这里只是防御性跳过，不硬失败。
                    continue
                produced.append(
                    PendingItem(
                        text=str(item["text"]),
                        producer_uri=producer,
                        producer_started_at=_member_started_at(producer),
                        scene_uri=scene_uri,
                        scene_label=document.address.label,
                        created_on=covered,
                    )
                )
            for link in document.links:
                if link.link_type is SceneLinkType.NEEDS:
                    consumed.add(str(link.to_uri))
    return tuple(
        item for item in produced if item.producer_uri not in consumed and item.scene_uri not in consumed
    )


def _member_started_at(uri: str) -> datetime:
    return BehaviorURI.parse(uri).to_address().started_at


__all__ = ["PendingItem", "pending_before"]
