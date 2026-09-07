"""情景文档的跨字段不变量；只查我们自己产物的自洽，不替现实立法。

会失败的都是产物内部矛盾：日期与开始时刻不符、开始时刻不等于首成员的开始（``started_at``
的**定义**就是首成员开始时刻，成员 URI 的叶名里带着它，零 IO 可核）、结束早于开始、待用前提
的产生方不是成员。**不会**因为"情景只有一个成员""没有 effects""成员跨了午夜""成员时间跨了别的
事"而失败——那些是现实的形状。
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from habitus.behavior.uri import BehaviorURI
from habitus.scene.model import scene_label
from habitus.scene.schema.model import SceneSchemaError


def validate_payload(payload: dict[str, Any]) -> None:
    scene_label(payload["label"], "scene label")
    if payload["occurred_on"] != payload["started_at"].date():
        raise SceneSchemaError("occurred_on must match the local started_at date")
    if payload["ended_at"] < payload["started_at"]:
        raise SceneSchemaError("ended_at cannot precede started_at")
    if not payload["members"]:
        raise SceneSchemaError("a scene must have at least one member")
    first_member = min(
        BehaviorURI.parse(member["uri"]).to_address().started_at.astimezone(UTC) for member in payload["members"]
    )
    if payload["started_at"].astimezone(UTC) != first_member:
        raise SceneSchemaError("started_at must equal the start of the earliest member")
    member_uris = {member["uri"] for member in payload["members"]}
    for pending in payload["pending_effects"]:
        if pending["producer_uri"] not in member_uris:
            raise SceneSchemaError("a pending effect must be produced by a member of the scene")
    if not payload["scene_version"]:
        raise SceneSchemaError("a scene must record the grouping version that produced it")


__all__ = ["validate_payload"]
