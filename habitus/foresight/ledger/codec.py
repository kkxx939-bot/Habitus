"""账本记录的 JSON 形状（按预期的六种、按账的四种）。落盘与读回由组合根做，这里只管编解码——两边用同一份，账本才能重放。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import (
    AccountFact,
    AccountLevel,
    Antecedent,
    Delivery,
    Expectation,
    Omission,
    Response,
    Ruling,
    Settlement,
    UnitFeedback,
    UnitLevel,
    UnitSetting,
)

#: v9（裁定 64）：递送多了「待取」「宿主没取」两种原因、「说了」带 ``via`` / ``turn``（谁说的、在哪一轮）、「待取」可带话。
#: v8 及更早一律按不支持拒（项目没上线，不留读回的双轨）。
LEDGER_SCHEMA_VERSION = "foresight_ledger_v9"


def encode_expectation(item: Expectation) -> dict[str, Any]:
    start, end = item.window
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "expectation",
        "expectation_id": item.expectation_id,
        "kind_token": item.kind_token,
        "day": item.day.isoformat(),
        "formed_at": item.formed_at.isoformat(),
        "source": item.source,
        "trigger": item.trigger,
        "trigger_instance": item.trigger_instance,
        "window": [start.isoformat(), end.isoformat()],
        "deadline": item.deadline.isoformat(),
        "observed_from": item.observed_from.isoformat(),
        "basis": list(item.basis),
        "opportunities": item.opportunities,
        "hits": item.hits,
        "antecedents": [
            {"relation": a.relation, "present": a.present, "upward": a.upward, "source": a.source}
            for a in item.antecedents
        ],
        "level": item.level,
        "generation": item.generation,
        "night": item.night.isoformat() if item.night is not None else None,
        "version": item.version,
        "implementation": item.implementation,
    }


def decode_expectation(raw: Mapping[str, Any]) -> Expectation:
    _check(raw, "expectation")
    try:
        window = raw["window"]
        if not isinstance(window, list) or len(window) != 2:
            raise ForesightError("expectation window must be a two-item list")
        antecedents = raw.get("antecedents", [])
        if not isinstance(antecedents, list):
            raise ForesightError("expectation antecedents must be a list")
        return Expectation(
            expectation_id=_text(raw, "expectation_id"),
            kind_token=_text(raw, "kind_token"),
            day=date.fromisoformat(_text(raw, "day")),
            formed_at=_datetime(raw, "formed_at"),
            source=_text(raw, "source"),
            trigger=_text(raw, "trigger"),
            trigger_instance=_text(raw, "trigger_instance"),
            window=(datetime.fromisoformat(str(window[0])), datetime.fromisoformat(str(window[1]))),
            deadline=_datetime(raw, "deadline"),
            observed_from=_datetime(raw, "observed_from"),
            basis=_strings(raw, "basis"),
            opportunities=int(raw["opportunities"]),
            hits=int(raw["hits"]),
            antecedents=tuple(
                Antecedent(
                    relation=str(entry["relation"]),
                    present=bool(entry["present"]),
                    upward=bool(entry["upward"]),
                    source=str(entry.get("source", "")),
                )
                for entry in antecedents
            ),
            level=int(raw["level"]),
            generation=str(raw["generation"]),
            night=date.fromisoformat(str(raw["night"])) if raw["night"] is not None else None,
            version=str(raw["version"]),
            implementation=str(raw["implementation"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"expectation record is malformed: {exc}") from exc


def encode_delivery(item: Delivery) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "delivery",
        "expectation_id": item.expectation_id,
        "decided_at": item.decided_at.isoformat(),
        "level": item.level,
        "spoken": item.spoken,
        "reason": item.reason,
        "p_speak": item.p_speak,
        "text": item.text,
        "evidence": list(item.evidence),
        "via": item.via,
        "turn": item.turn,
    }


def decode_delivery(raw: Mapping[str, Any]) -> Delivery:
    _check(raw, "delivery")
    try:
        return Delivery(
            expectation_id=_text(raw, "expectation_id"),
            decided_at=_datetime(raw, "decided_at"),
            level=int(raw["level"]),
            spoken=bool(raw["spoken"]),
            reason=_text(raw, "reason"),
            p_speak=float(raw["p_speak"]),
            text=None if raw.get("text") is None else str(raw["text"]),
            evidence=tuple(str(item) for item in raw.get("evidence", [])),
            via=None if raw.get("via") is None else str(raw["via"]),
            turn=None if raw.get("turn") is None else str(raw["turn"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"delivery record is malformed: {exc}") from exc


def encode_response(item: Response) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "response",
        "expectation_id": item.expectation_id,
        "responded_at": item.responded_at.isoformat(),
        "polarity": item.polarity,
        "turn": item.turn,
    }


def decode_response(raw: Mapping[str, Any]) -> Response:
    _check(raw, "response")
    try:
        return Response(
            expectation_id=_text(raw, "expectation_id"),
            responded_at=_datetime(raw, "responded_at"),
            polarity=_text(raw, "polarity"),
            turn=_text(raw, "turn"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"response record is malformed: {exc}") from exc


def encode_settlement(item: Settlement) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "settlement",
        "expectation_id": item.expectation_id,
        "kind_token": item.kind_token,
        "day": item.day.isoformat(),
        "settled_at": item.settled_at.isoformat(),
        "outcome": item.outcome,
        "occurrence_uri": item.occurrence_uri,
        "offset_minutes": item.offset_minutes,
    }


def decode_settlement(raw: Mapping[str, Any]) -> Settlement:
    _check(raw, "settlement")
    try:
        return Settlement(
            expectation_id=_text(raw, "expectation_id"),
            kind_token=_text(raw, "kind_token"),
            day=date.fromisoformat(_text(raw, "day")),
            settled_at=_datetime(raw, "settled_at"),
            outcome=_text(raw, "outcome"),
            occurrence_uri=None if raw.get("occurrence_uri") is None else str(raw["occurrence_uri"]),
            offset_minutes=None if raw.get("offset_minutes") is None else int(raw["offset_minutes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"settlement record is malformed: {exc}") from exc


def encode_ruling(item: Ruling) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "ruling",
        "expectation_id": item.expectation_id,
        "ruled_at": item.ruled_at.isoformat(),
        "ruling": item.ruling,
        "occurrence_uri": item.occurrence_uri,
        "delay_minutes": item.delay_minutes,
    }


def decode_ruling(raw: Mapping[str, Any]) -> Ruling:
    _check(raw, "ruling")
    try:
        return Ruling(
            expectation_id=_text(raw, "expectation_id"),
            ruled_at=_datetime(raw, "ruled_at"),
            ruling=_text(raw, "ruling"),
            occurrence_uri=None if raw.get("occurrence_uri") is None else str(raw["occurrence_uri"]),
            delay_minutes=None if raw.get("delay_minutes") is None else int(raw["delay_minutes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"ruling record is malformed: {exc}") from exc


def encode_omission(item: Omission) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "omission",
        "occurrence_uri": item.occurrence_uri,
        "kind_token": item.kind_token,
        "day": item.day.isoformat(),
        "noticed_at": item.noticed_at.isoformat(),
        "had_expectation": item.had_expectation,
    }


def decode_omission(raw: Mapping[str, Any]) -> Omission:
    _check(raw, "omission")
    try:
        return Omission(
            occurrence_uri=_text(raw, "occurrence_uri"),
            kind_token=_text(raw, "kind_token"),
            day=date.fromisoformat(_text(raw, "day")),
            noticed_at=_datetime(raw, "noticed_at"),
            had_expectation=bool(raw["had_expectation"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"omission record is malformed: {exc}") from exc


def encode_fact(item: AccountFact) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "fact",
        "account": item.account,
        "at": item.at.isoformat(),
        "good": item.good,
        "strength": item.strength,
        "kind": item.kind,
        "expectation_id": item.expectation_id,
        "offset_minutes": item.offset_minutes,
        "written_at": None if item.written_at is None else item.written_at.isoformat(),
    }


def decode_fact(raw: Mapping[str, Any]) -> AccountFact:
    _check(raw, "fact")
    try:
        return AccountFact(
            account=_text(raw, "account"),
            at=_datetime(raw, "at"),
            good=bool(raw["good"]),
            strength=float(raw["strength"]),
            kind=_text(raw, "kind"),
            expectation_id=_text(raw, "expectation_id"),
            offset_minutes=None if raw.get("offset_minutes") is None else int(raw["offset_minutes"]),
            written_at=None if raw.get("written_at") is None else _datetime(raw, "written_at"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"fact record is malformed: {exc}") from exc


def encode_setting(item: UnitSetting) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "unit_setting",
        "kind_token": item.kind_token,
        "at": item.at.isoformat(),
        "setting": item.setting,
    }


def decode_setting(raw: Mapping[str, Any]) -> UnitSetting:
    _check(raw, "unit_setting")
    try:
        return UnitSetting(kind_token=_text(raw, "kind_token"), at=_datetime(raw, "at"), setting=_text(raw, "setting"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"unit setting record is malformed: {exc}") from exc


def encode_feedback(item: UnitFeedback) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "unit_feedback",
        "kind_token": item.kind_token,
        "at": item.at.isoformat(),
        "good": item.good,
        "expectation_id": item.expectation_id,
        "polarity": item.polarity,
    }


def decode_feedback(raw: Mapping[str, Any]) -> UnitFeedback:
    _check(raw, "unit_feedback")
    try:
        return UnitFeedback(
            kind_token=_text(raw, "kind_token"),
            at=_datetime(raw, "at"),
            good=bool(raw["good"]),
            expectation_id=_text(raw, "expectation_id"),
            polarity=None if raw.get("polarity") is None else _text(raw, "polarity"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"unit feedback record is malformed: {exc}") from exc


def encode_level(item: AccountLevel) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "level",
        "account": item.account,
        "at": item.at.isoformat(),
        "level": item.level,
    }


def decode_level(raw: Mapping[str, Any]) -> AccountLevel:
    _check(raw, "level")
    try:
        return AccountLevel(account=_text(raw, "account"), at=_datetime(raw, "at"), level=int(raw["level"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"level record is malformed: {exc}") from exc


def encode_unit_level(item: UnitLevel) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record": "unit_level",
        "kind_token": item.kind_token,
        "at": item.at.isoformat(),
        "level": item.level,
    }


def decode_unit_level(raw: Mapping[str, Any]) -> UnitLevel:
    _check(raw, "unit_level")
    try:
        return UnitLevel(kind_token=_text(raw, "kind_token"), at=_datetime(raw, "at"), level=int(raw["level"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"unit level record is malformed: {exc}") from exc


def _check(raw: Mapping[str, Any], record: str) -> None:
    if raw.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ForesightError(
            f"ledger record has schema {raw.get('schema_version')!r}, expected {LEDGER_SCHEMA_VERSION}"
        )
    if raw.get("record") != record:
        raise ForesightError(f"ledger record is a {raw.get('record')!r}, expected {record}")


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise ForesightError(f"{key} must be text")
    return value


def _datetime(raw: Mapping[str, Any], key: str) -> datetime:
    return datetime.fromisoformat(_text(raw, key))


def _strings(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw[key]
    if not isinstance(value, list):
        raise ForesightError(f"{key} must be a list")
    return tuple(str(item) for item in value)


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "decode_delivery",
    "decode_expectation",
    "decode_fact",
    "decode_feedback",
    "decode_level",
    "decode_omission",
    "decode_response",
    "decode_ruling",
    "decode_setting",
    "decode_settlement",
    "decode_unit_level",
    "encode_delivery",
    "encode_expectation",
    "encode_fact",
    "encode_feedback",
    "encode_level",
    "encode_omission",
    "encode_response",
    "encode_ruling",
    "encode_setting",
    "encode_settlement",
    "encode_unit_level",
]
