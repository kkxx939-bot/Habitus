"""关系表的落盘形状：每条 lane 每晚一份规范 JSON（全部关系当晚的读数与状态），状态迁移一行一条。

规范化：解码时把读到的对象按规范形式重新序列化比对，一个字节不同就是损坏，不做"尽力解析"。
样本不够的只存各段的条数，已检的只存几项读数（每晚从数据重算，不跨夜带历史）；候选、成立、失效、前向未复现存全。
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from habitus.foundation.integrity import CanonicalSerializationError, canonical_json
from habitus.scene.relations.engine import RelationKey, RelationTest, Subset, Verdict
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.state import LaneState, Relation, Status, Transition
from habitus.scene.relations.stats import Effect

SCHEMA = "habitus.scene.relations.night/v2"
TRANSITION_SCHEMA = "habitus.scene.relations.transition/v1"


class RelationRecordError(ValueError):
    """关系表的编码与我们自己的产物矛盾。"""


def encode_night(state: LaneState) -> str:
    return _canonical(
        {
            "schema": SCHEMA,
            "lane": state.lane,
            "night": state.night,
            "family": state.family,
            "relations": [_relation(item) for item in state.relations],
            "sparse": state.sparse,
        }
    )


def decode_night(text: str, *, lane: str, night: date) -> LaneState:
    payload = _parse(text)
    try:
        if payload["schema"] != SCHEMA or payload["lane"] != lane or payload["night"] != night.isoformat():
            raise RelationRecordError("relation night does not match its file")
        state = LaneState(
            lane=lane,
            night=night,
            family=int(payload["family"]),
            relations=tuple(_relation_from(item, lane) for item in payload["relations"]),
            sparse={str(name): int(count) for name, count in payload["sparse"].items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RelationRecordError(f"relation night is not valid: {exc}") from exc
    if encode_night(state) != text:
        raise RelationRecordError("relation night is not in canonical form")
    return state


def encode_transition(item: Transition) -> str:
    return _canonical(
        {
            "schema": TRANSITION_SCHEMA,
            "night": item.night,
            "key": _key(item.key),
            "before": item.before,
            "after": item.after,
            "reason": item.reason,
        }
    )


def decode_transition(text: str, *, lane: str) -> Transition:
    payload = _parse(text)
    try:
        if payload["schema"] != TRANSITION_SCHEMA:
            raise RelationRecordError("not a relation transition")
        item = Transition(
            night=date.fromisoformat(payload["night"]),
            key=_key_from(payload["key"], lane),
            before=None if payload["before"] is None else Status(payload["before"]),
            after=Status(payload["after"]),
            reason=str(payload["reason"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RelationRecordError(f"relation transition is not valid: {exc}") from exc
    if encode_transition(item) != text:
        raise RelationRecordError("relation transition is not in canonical form")
    return item


# ── 编 ──────────────────────────────────────────────────────────────────────


def _relation(item: Relation) -> dict[str, Any]:
    body: dict[str, Any] = {"key": _key(item.key), "status": item.status, "upward": item.upward}
    if item.tonight is not None:
        body["tonight"] = _test(item.tonight, full=item.status.carried)
    if item.status.carried or item.forward is not None:
        body.update(
            {
                "discovered_on": item.discovered_on,
                "forward_passed_on": item.forward_passed_on,
                "established_on": item.established_on,
                "expired_on": item.expired_on,
                "rejected_on": item.rejected_on,
                "prior_only": item.prior_only,
                "forward": None if item.forward is None else _subset(item.forward),
                "maintenance": None if item.maintenance is None else _subset(item.maintenance),
                "forward_needed": item.forward_needed,
                "maintained_through": item.maintained_through,
                "misses": item.misses,
            }
        )
    return body


def _key(key: RelationKey) -> dict[str, Any]:
    return {
        "antecedent": key.antecedent,
        "consequent": key.consequent,
        "segment": key.segment,
        "condition": key.condition,
    }


def _test(test: RelationTest, *, full: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "verdict": test.verdict,
        "antecedents": test.antecedents,
        "blocks": test.blocks,
        "block_days": test.block_days,
        "effect_units": test.effect.antecedents,
        "treated_rate": test.effect.treated_rate,
        "control_rate": test.effect.control_rate,
        "p_value": test.p_value,
        "minimum_p": test.minimum_p,
        "upward": test.upward,
        "weight": test.weight,
        "more_needed": test.more_needed,
    }
    if full or test.verdict is Verdict.SIGNIFICANT:
        body.update(
            {
                "heterogeneity_p": test.heterogeneity_p,
                "leave_one_out": test.leave_one_out,
                "stable": test.stable,
                "interval": test.interval,
                "relative_interval": test.relative_interval,
                "large_enough": test.large_enough,
                "excess_share": test.excess_share,
                "same_event_share": test.same_event_share,
            }
        )
    return body


def _subset(item: Subset) -> dict[str, Any]:
    return {
        "antecedents": item.antecedents,
        "blocks": item.blocks,
        "treated_rate": item.effect.treated_rate,
        "control_rate": item.effect.control_rate,
        "p_value": item.p_value,
        "interval": item.interval,
        "relative_interval": item.relative_interval,
    }


# ── 解 ──────────────────────────────────────────────────────────────────────


def _relation_from(body: dict[str, Any], lane: str) -> Relation:
    key = _key_from(body["key"], lane)
    status = Status(body["status"])
    tonight = None if "tonight" not in body else _test_from(body["tonight"], key)
    return Relation(
        key=key,
        status=status,
        upward=bool(body["upward"]),
        tonight=tonight,
        discovered_on=_day(body.get("discovered_on")),
        forward_passed_on=_day(body.get("forward_passed_on")),
        established_on=_day(body.get("established_on")),
        expired_on=_day(body.get("expired_on")),
        rejected_on=_day(body.get("rejected_on")),
        prior_only=bool(body.get("prior_only", False)),
        forward=None if body.get("forward") is None else _subset_from(body["forward"]),
        maintenance=None if body.get("maintenance") is None else _subset_from(body["maintenance"]),
        forward_needed=None if body.get("forward_needed") is None else int(body["forward_needed"]),
        maintained_through=_day(body.get("maintained_through")),
        misses=int(body.get("misses", 0)),
    )


def _key_from(body: dict[str, Any], lane: str) -> RelationKey:
    return RelationKey(
        lane=lane,
        antecedent=str(body["antecedent"]),
        consequent=str(body["consequent"]),
        segment=Segment(body["segment"]),
        condition=str(body["condition"]),
    )


def _test_from(body: dict[str, Any], key: RelationKey) -> RelationTest:
    antecedents = int(body["antecedents"])
    return RelationTest(
        key=key,
        verdict=Verdict(body["verdict"]),
        antecedents=antecedents,
        blocks=int(body["blocks"]),
        block_days=float(body["block_days"]),
        effect=Effect(
            antecedents=int(body["effect_units"]),
            treated_rate=float(body["treated_rate"]),
            control_rate=float(body["control_rate"]),
        ),
        p_value=float(body["p_value"]),
        minimum_p=float(body["minimum_p"]),
        upward=bool(body["upward"]),
        weight=float(body["weight"]),
        more_needed=body["more_needed"],
        heterogeneity_p=body.get("heterogeneity_p"),
        leave_one_out=body.get("leave_one_out"),
        stable=body.get("stable"),
        interval=_pair(body.get("interval")),
        relative_interval=_pair(body.get("relative_interval")),
        large_enough=body.get("large_enough"),
        excess_share=body.get("excess_share"),
        same_event_share=body.get("same_event_share"),
    )


def _subset_from(body: dict[str, Any]) -> Subset:
    antecedents = int(body["antecedents"])
    return Subset(
        antecedents=antecedents,
        blocks=int(body["blocks"]),
        effect=Effect(
            antecedents=antecedents,
            treated_rate=float(body["treated_rate"]),
            control_rate=float(body["control_rate"]),
        ),
        p_value=float(body["p_value"]),
        interval=_pair(body.get("interval")),
        relative_interval=_pair(body.get("relative_interval")),
    )


def _pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    low, high = value
    return (float(low), float(high))


def _day(value: Any) -> date | None:
    return None if value is None else date.fromisoformat(value)


def _canonical(payload: dict[str, Any]) -> str:
    try:
        return canonical_json(payload)
    except CanonicalSerializationError as exc:
        raise RelationRecordError("relation record is not canonically serializable") from exc


def _parse(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise RelationRecordError("relation record is not JSON") from exc
    if not isinstance(payload, dict):
        raise RelationRecordError("relation record must be a JSON object")
    return payload


__all__ = [
    "SCHEMA",
    "TRANSITION_SCHEMA",
    "RelationRecordError",
    "decode_night",
    "decode_transition",
    "encode_night",
    "encode_transition",
]
