"""承诺与结算的 JSON 形状。落盘与读回由组合根做，这里只管编解码——两边用同一份，账本才能重放。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import Claim, Settlement
from habitus.foresight.model import CandidateNumbers, RecurrenceNumbers

LEDGER_SCHEMA_VERSION = "foresight_ledger_v2"


def encode_claim(claim: Claim) -> dict[str, Any]:
    numbers = claim.numbers
    recurrence = numbers.recurrence
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "claim_id": claim.claim_id,
        "kind_token": claim.kind_token,
        "day": claim.day.isoformat(),
        "slot": claim.slot,
        "slot_minutes": claim.slot_minutes,
        "window": list(claim.window),
        "judged_at": claim.judged_at.isoformat(),
        "generation": claim.generation,
        "judge_version": claim.judge_version,
        "basis": list(claim.basis),
        "situations": list(claim.situations),
        "numbers": {
            "marginal": numbers.marginal,
            "hazard": numbers.hazard,
            "cumulative": numbers.cumulative,
            "lift_all_day": numbers.lift_all_day,
            "lift_weekday": numbers.lift_weekday,
            "count": numbers.count,
            "n_eff": numbers.n_eff,
            "trend": numbers.trend,
            "trend_n_eff": numbers.trend_n_eff,
            "done_today": numbers.done_today,
            "recurrence": None
            if recurrence is None
            else {
                "p10": recurrence.p10,
                "p50": recurrence.p50,
                "p90": recurrence.p90,
                "sample_count": recurrence.sample_count,
                "overdue": recurrence.overdue,
            },
        },
        "conditions": [list(pair) for pair in claim.conditions],
        "condition_keys": list(claim.condition_keys),
        "facts_version": claim.facts_version,
    }


def decode_claim(raw: Mapping[str, Any]) -> Claim:
    try:
        if raw["schema_version"] != LEDGER_SCHEMA_VERSION:
            raise ForesightError(f"unknown ledger schema: {raw.get('schema_version')!r}")
        numbers = raw["numbers"]
        recurrence = numbers["recurrence"]
        return Claim(
            claim_id=str(raw["claim_id"]),
            kind_token=str(raw["kind_token"]),
            day=date.fromisoformat(raw["day"]),
            slot=int(raw["slot"]),
            slot_minutes=int(raw["slot_minutes"]),
            window=(int(raw["window"][0]), int(raw["window"][1])),
            judged_at=datetime.fromisoformat(raw["judged_at"]),
            generation=str(raw["generation"]),
            judge_version=str(raw["judge_version"]),
            basis=tuple(str(item) for item in raw["basis"]),
            situations=tuple(str(item) for item in raw["situations"]),
            numbers=CandidateNumbers(
                marginal=float(numbers["marginal"]),
                hazard=float(numbers["hazard"]),
                cumulative=float(numbers["cumulative"]),
                lift_all_day=float(numbers["lift_all_day"]),
                lift_weekday=float(numbers["lift_weekday"]),
                count=float(numbers["count"]),
                n_eff=float(numbers["n_eff"]),
                trend=None if numbers["trend"] is None else float(numbers["trend"]),
                trend_n_eff=float(numbers["trend_n_eff"]),
                recurrence=None
                if recurrence is None
                else RecurrenceNumbers(
                    p10=float(recurrence["p10"]),
                    p50=float(recurrence["p50"]),
                    p90=float(recurrence["p90"]),
                    sample_count=float(recurrence["sample_count"]),
                    overdue=None if recurrence["overdue"] is None else float(recurrence["overdue"]),
                ),
                done_today=int(numbers["done_today"]),
            ),
            conditions=tuple((str(k), str(v)) for k, v in raw["conditions"]),
            condition_keys=tuple(str(item) for item in raw["condition_keys"]),
            facts_version=str(raw["facts_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"claim record is malformed: {exc}") from exc


def encode_settlement(item: Settlement) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "claim_id": item.claim_id,
        "kind_token": item.kind_token,
        "day": item.day.isoformat(),
        "situations": list(item.situations),
        "outcome": item.outcome,
        "occurrence_uri": item.occurrence_uri,
        "slot_offset": item.slot_offset,
        "settled_at": item.settled_at.isoformat(),
        "reminded": item.reminded,
        "response": item.response,
    }


def decode_settlement(raw: Mapping[str, Any]) -> Settlement:
    try:
        if raw["schema_version"] != LEDGER_SCHEMA_VERSION:
            raise ForesightError(f"unknown ledger schema: {raw.get('schema_version')!r}")
        return Settlement(
            claim_id=str(raw["claim_id"]),
            kind_token=str(raw["kind_token"]),
            day=date.fromisoformat(raw["day"]),
            situations=tuple(str(item) for item in raw["situations"]),
            outcome=str(raw["outcome"]),
            occurrence_uri=None if raw["occurrence_uri"] is None else str(raw["occurrence_uri"]),
            slot_offset=None if raw["slot_offset"] is None else int(raw["slot_offset"]),
            settled_at=datetime.fromisoformat(raw["settled_at"]),
            reminded=bool(raw["reminded"]),
            response=None if raw["response"] is None else str(raw["response"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"settlement record is malformed: {exc}") from exc


__all__ = ["LEDGER_SCHEMA_VERSION", "decode_claim", "decode_settlement", "encode_claim", "encode_settlement"]
