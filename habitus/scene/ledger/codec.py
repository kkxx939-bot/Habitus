"""承诺、结算、提醒的 JSON 形状。规范 JSON 落盘；时间按各自的偏移存 ISO 文本（不折成 UTC——锚是本地时刻）。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from habitus.foundation.integrity import CanonicalSerializationError, canonical_json
from habitus.scene.hypotheses.model import Aspect
from habitus.scene.ledger.model import (
    Claim,
    ClaimRef,
    Intervention,
    InterventionResponse,
    LedgerError,
    Opportunity,
    OpportunityPass,
    OpportunitySnapshot,
    Outcome,
    Response,
    Settlement,
    WindowSpan,
)
from habitus.scene.occurrences.model import ConceptHit

LEDGER_SCHEMA_VERSION = "scene_ledger_v4"


class LedgerSchemaError(ValueError):
    """这条记录是**旧口径**写的（``schema_version`` 不是当前那个）。

    它不是"文件损坏"：账本没有版本迁移（定义变了就整个重算），所以拒绝读是对的——但**报出来的原因
    必须分得开**。一条旧文件会让这条假设的 `claims_for` / `open_claims` / `settle_due_all` 全部抛错、
    夜批停在第一条，而运维看到"corrupt"只会去查磁盘。重算路径要能机械认出"这是旧口径，删了重算"。
    """


class LedgerCodecError(ValueError):
    """账本记录的 JSON 与我们自己的产物矛盾。"""


def encode_claim(claim: Claim) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_type": "claim",
        "hypothesis_identity": claim.hypothesis_identity,
        "hypothesis_fingerprint": claim.hypothesis_fingerprint,
        "aspect": claim.aspect.value,
        "trigger_uri": claim.trigger_uri,
        "antecedent_hits": [{"concept": hit.concept, "grade": hit.grade} for hit in claim.antecedent_hits],
        "antecedent_uris": list(claim.antecedent_uris),
        "situation_snapshot": list(claim.situation_snapshot),
        "situations_checked": list(claim.situations_checked),
        "control": None
        if claim.control is None
        else {
            "generation": claim.control.generation,
            "opportunities": [
                {"at": _stamp(item.at), "start": _stamp(item.span.start), "end": _stamp(item.span.end), "probability": item.probability}
                for item in claim.control.opportunities
            ],
        },
        "created_at": _stamp(claim.created_at),
    }
    return _dump(payload)


def decode_claim(raw: bytes, *, expected: ClaimRef | None = None) -> Claim:
    payload = _load(raw, "claim")
    try:
        control = payload.get("control")
        claim = Claim(
            hypothesis_identity=_text(payload, "hypothesis_identity"),
            hypothesis_fingerprint=_text(payload, "hypothesis_fingerprint"),
            aspect=Aspect(payload.get("aspect")),
            trigger_uri=_text(payload, "trigger_uri"),
            antecedent_hits=tuple(_hit(item) for item in _list(payload, "antecedent_hits")),
            antecedent_uris=tuple(str(item) for item in _list(payload, "antecedent_uris")),
            situation_snapshot=tuple(str(item) for item in _list(payload, "situation_snapshot")),
            situations_checked=tuple(str(item) for item in _list(payload, "situations_checked")),
            control=None if control is None else _snapshot(control),
            created_at=_moment(payload, "created_at"),
        )
    except (LedgerError, TypeError, ValueError) as exc:
        raise LedgerCodecError(f"claim record is not valid: {exc}") from exc
    if expected is not None and claim.ref != expected:
        raise LedgerCodecError("claim record does not carry the identity of its address")
    if encode_claim(claim) != raw:
        raise LedgerCodecError("claim record is not in canonical form")
    return claim


def encode_settlement(settlement: Settlement) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_type": "settlement",
        "ref": _ref(settlement.ref),
        "outcome": settlement.outcome.value,
        "observed_at": None if settlement.observed_at is None else _stamp(settlement.observed_at),
        "fulfilling_uri": settlement.fulfilling_uri,
        "latency_hours": settlement.latency_hours,
        "opportunity_index": settlement.opportunity_index,
        "passes": [{"at": _stamp(item.at), "observed": item.observed} for item in settlement.passes],
        "count": settlement.count,
        "reason": settlement.reason,
        "releasing_uri": settlement.releasing_uri,
        "settled_at": _stamp(settlement.settled_at),
    }
    return _dump(payload)


def decode_settlement(raw: bytes, *, expected: ClaimRef | None = None) -> Settlement:
    payload = _load(raw, "settlement")
    try:
        observed = payload.get("observed_at")
        settlement = Settlement(
            ref=_parse_ref(payload.get("ref")),
            outcome=Outcome(payload.get("outcome")),
            settled_at=_moment(payload, "settled_at"),
            observed_at=None if observed is None else datetime.fromisoformat(str(observed)),
            fulfilling_uri=payload.get("fulfilling_uri"),
            latency_hours=payload.get("latency_hours"),
            opportunity_index=payload.get("opportunity_index"),
            passes=tuple(_pass(item) for item in _list(payload, "passes")),
            count=payload.get("count"),
            reason=payload.get("reason"),
            releasing_uri=payload.get("releasing_uri"),
        )
    except (LedgerError, TypeError, ValueError) as exc:
        raise LedgerCodecError(f"settlement record is not valid: {exc}") from exc
    if expected is not None and settlement.ref != expected:
        raise LedgerCodecError("settlement record does not carry the identity of its address")
    if encode_settlement(settlement) != raw:
        raise LedgerCodecError("settlement record is not in canonical form")
    return settlement


def encode_intervention(intervention: Intervention) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_type": "intervention",
        "ref": _ref(intervention.ref),
        "reminded_at": _stamp(intervention.reminded_at),
        "note": intervention.note,
        "recorded_at": _stamp(intervention.recorded_at),
    }
    return _dump(payload)


def decode_intervention(raw: bytes, *, expected: ClaimRef | None = None) -> Intervention:
    payload = _load(raw, "intervention")
    try:
        intervention = Intervention(
            ref=_parse_ref(payload.get("ref")),
            reminded_at=_moment(payload, "reminded_at"),
            recorded_at=_moment(payload, "recorded_at"),
            note=payload.get("note"),
        )
    except (LedgerError, TypeError, ValueError) as exc:
        raise LedgerCodecError(f"intervention record is not valid: {exc}") from exc
    if expected is not None and intervention.ref != expected:
        raise LedgerCodecError("intervention record does not carry the identity of its address")
    if encode_intervention(intervention) != raw:
        raise LedgerCodecError("intervention record is not in canonical form")
    return intervention


def encode_response(response: InterventionResponse) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_type": "response",
        "ref": _ref(response.ref),
        "reminded_at": _stamp(response.reminded_at),
        "responded_at": _stamp(response.responded_at),
        "response": response.response.value,
        "note": response.note,
    }
    return _dump(payload)


def decode_response(raw: bytes, *, expected: ClaimRef | None = None) -> InterventionResponse:
    payload = _load(raw, "response")
    try:
        response = InterventionResponse(
            ref=_parse_ref(payload.get("ref")),
            reminded_at=_moment(payload, "reminded_at"),
            responded_at=_moment(payload, "responded_at"),
            response=Response(payload.get("response")),
            note=payload.get("note"),
        )
    except (LedgerError, TypeError, ValueError) as exc:
        raise LedgerCodecError(f"response record is not valid: {exc}") from exc
    if expected is not None and response.ref != expected:
        raise LedgerCodecError("response record does not carry the identity of its address")
    if encode_response(response) != raw:
        raise LedgerCodecError("response record is not in canonical form")
    return response


# ── 内部 ────────────────────────────────────────────────────────────────────────


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="microseconds")


def _dump(payload: Mapping[str, Any]) -> bytes:
    try:
        return canonical_json(payload).encode("utf-8")
    except CanonicalSerializationError as exc:
        raise LedgerCodecError("ledger payload is not canonically serializable") from exc


def _load(raw: bytes, record_type: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerCodecError(f"{record_type} record is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LedgerCodecError(f"{record_type} record is not an object")
    found = payload.get("schema_version")
    if found != LEDGER_SCHEMA_VERSION:
        raise LedgerSchemaError(f"{record_type} record was written as {found!r}, not {LEDGER_SCHEMA_VERSION!r}; this account has to be recomputed")
    if payload.get("record_type") != record_type:
        raise LedgerCodecError(f"record claims to be {payload.get('record_type')!r}, not {record_type!r}")
    return payload


def _text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise LedgerCodecError(f"field {key!r} must be text")
    return value


def _list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise LedgerCodecError(f"field {key!r} must be a list")
    return value


def _moment(payload: Mapping[str, Any], key: str) -> datetime:
    return datetime.fromisoformat(_text(payload, key))


def _hit(item: object) -> ConceptHit:
    if not isinstance(item, dict) or set(item) != {"concept", "grade"}:
        raise LedgerCodecError("an antecedent hit must be an object with concept and grade")
    return ConceptHit(concept=item["concept"], grade=item["grade"])


def _snapshot(item: object) -> OpportunitySnapshot:
    if not isinstance(item, dict) or set(item) != {"generation", "opportunities"} or not isinstance(item["opportunities"], list):
        raise LedgerCodecError("control must be an object with generation and opportunities")
    opportunities = []
    for entry in item["opportunities"]:
        if not isinstance(entry, dict) or set(entry) != {"at", "start", "end", "probability"}:
            raise LedgerCodecError("an opportunity must be an object with at, start, end and probability")
        opportunities.append(
            Opportunity(
                at=datetime.fromisoformat(str(entry["at"])),
                span=WindowSpan(datetime.fromisoformat(str(entry["start"])), datetime.fromisoformat(str(entry["end"]))),
                probability=entry["probability"],
            )
        )
    return OpportunitySnapshot(generation=str(item["generation"]), opportunities=tuple(opportunities))


def _pass(item: object) -> OpportunityPass:
    if not isinstance(item, dict) or set(item) != {"at", "observed"}:
        raise LedgerCodecError("a pass must be an object with at and observed")
    return OpportunityPass(at=datetime.fromisoformat(str(item["at"])), observed=item["observed"])


def _ref(ref: ClaimRef) -> dict[str, str]:
    return {"hypothesis_identity": ref.hypothesis_identity, "day": ref.day.isoformat(), "leaf": ref.leaf}


def _parse_ref(item: object) -> ClaimRef:
    if not isinstance(item, dict) or set(item) != {"hypothesis_identity", "day", "leaf"}:
        raise LedgerCodecError("ref must be an object with hypothesis_identity, day and leaf")
    return ClaimRef(str(item["hypothesis_identity"]), date.fromisoformat(str(item["day"])), str(item["leaf"]))


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "LedgerSchemaError",
    "LedgerCodecError",
    "decode_claim",
    "decode_intervention",
    "decode_response",
    "decode_settlement",
    "encode_claim",
    "encode_intervention",
    "encode_response",
    "encode_settlement",
]
