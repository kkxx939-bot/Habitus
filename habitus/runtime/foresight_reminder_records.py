"""提醒记录挂在账和出口外面的那一层（裁定 113）：不改账本、出口的类，只包一层。

- **账**：全部写账都经过账本的四个方法（记递送、记回话、记结算、记裁决），每支的账本全程是同一个实例。``RecordingLedgerStore``
  在它们写完之后，对这一条预期：这次写的是「待取」、或者它已经有记录，就从账整份重算记录里来自账的那几段（``reminder_fields``）。
  结算会对每条预期都调：先看记录文件在不在，没有就什么都不读。
- **出口**：取走、放回只在出口的取走记录里。``RecordingOutletStore`` 在出口写取走记录时把这条请求此刻的整段取走状态（历次取走的
  时刻、宿主、后来怎样了，放回几次）传给记录；出口删掉取走记录时不动记录，那一段留在记录里。
- **写记录失败不让写账失败**：账是预测的依据，记录只是记录。失败写日志、计数（``ReminderRecords.failures``）。
- **拿锁**：出口是持着自己的锁写账、写取走记录的，于是这里的更新都在出口的锁里面；记录的锁是最里面的一把，更新时只读账、
  不读出口（取走那一段由出口直接把值传进来）。

改判（「已做」改中性、作废）写在按账存的事实里，事实不带预期所在的日子、压实时会被删，从它重算会丢东西：记录里的结算是
首次结算的结果加后补裁决，不反映改判。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from habitus.config.foresight import ForesightStoreConfig
from habitus.foresight.ledger import AWAITING_HOST, Delivery, Response, Ruling, Settlement
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.reminder_records import ReminderRecords

logger = logging.getLogger("habitus.reminders")


def reminder_fields(
    ledger: ForesightLedgerStore,
    day: date,
    expectation_id: str,
    *,
    lane: str,
    kind_name: Callable[[str], str | None],
    zone: ZoneInfo,
) -> dict[str, Any] | None:
    """一条预期在账上的全部往来拼成记录里来自账的那几段；账上没有这条预期、或它从没有过「待取」是 None。
    时刻统一写成配置时区的本地偏移（账里「说了」是 UTC 写法，别的递送带本地偏移）。"""

    expectation = ledger.expectation(day, expectation_id)
    if expectation is None:
        return None
    mine = sorted(
        (item for item in ledger.deliveries_on(day) if item.expectation_id == expectation_id),
        key=lambda item: item.decided_at,
    )
    awaiting = [item for item in mine if item.reason == AWAITING_HOST]
    if not awaiting:
        return None
    first = awaiting[0]
    # 决定要说之后的递送：说出口的那一条，或者没说成的终局（已发生 / 宿主没取 / 依据失效 / 已停用）
    later = [item for item in mine if item.decided_at >= first.decided_at and item.reason != AWAITING_HOST]
    spoken = next((item for item in later if item.spoken), None)
    unspoken = next((item for item in later if not item.spoken), None)
    responses = sorted(
        (item for item in ledger.responses_on(day) if item.expectation_id == expectation_id),
        key=lambda item: (item.responded_at, item.turn),
    )
    settlement = next((item for item in ledger.settlements_on(day) if item.expectation_id == expectation_id), None)
    rulings = sorted(
        (item for item in ledger.rulings_on(day) if item.expectation_id == expectation_id),
        key=lambda item: item.ruled_at,
    )

    def at(value: datetime) -> str:
        return value.astimezone(zone).isoformat()

    return {
        "expectation_id": expectation_id,
        "lane": lane,
        "kind_token": expectation.kind_token,
        "kind_name": kind_name(expectation.kind_token) or expectation.kind_token,
        "day": day.isoformat(),
        "source": expectation.source,
        "window_start": at(expectation.window[0]),
        "window_end": at(expectation.window[1]),
        "deadline": at(expectation.deadline),
        "decided_at": at(first.decided_at),
        "text": first.text,
        "spoken": _spoken(spoken, at),
        "unspoken": None if unspoken is None else {"at": at(unspoken.decided_at), "reason": unspoken.reason},
        "responses": [_response(item, at) for item in responses],
        "settlement": _settlement(settlement, at),
        "rulings": [_ruling(item, at) for item in rulings],
    }


def _spoken(item: Delivery | None, at: Callable[[datetime], str]) -> dict[str, Any] | None:
    if item is None:
        return None
    return {"at": at(item.decided_at), "text": item.text, "via": item.via, "turn": item.turn}


def _response(item: Response, at: Callable[[datetime], str]) -> dict[str, Any]:
    # 只有方向、时刻与哪一轮：回应的原话不在账里，要看就按"哪一轮"回到那一轮的会话原文
    return {"at": at(item.responded_at), "polarity": item.polarity, "turn": item.turn}


def _settlement(item: Settlement | None, at: Callable[[datetime], str]) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "at": at(item.settled_at),
        "outcome": item.outcome,
        "offset_minutes": item.offset_minutes,
        "occurrence_uri": item.occurrence_uri,
    }


def _ruling(item: Ruling, at: Callable[[datetime], str]) -> dict[str, Any]:
    return {
        "at": at(item.ruled_at),
        "ruling": item.ruling,
        "delay_minutes": item.delay_minutes,
        "occurrence_uri": item.occurrence_uri,
    }


class RecordingLedgerStore(ForesightLedgerStore):
    """一支的账本，四个写账的方法写完之后顺手更新这一条的提醒记录。"""

    def __init__(
        self,
        root: str | Path,
        *,
        records: ReminderRecords,
        kind_name: Callable[[str], str | None],
        zone: ZoneInfo,
        limits: ForesightStoreConfig | None = None,
    ) -> None:
        super().__init__(root, limits=limits)
        self.records = records
        self.kind_name = kind_name
        self.zone = zone

    def record_delivery(self, day: date, item: Delivery) -> bool:
        written = super().record_delivery(day, item)
        self.follow(day, item.expectation_id, start=item.reason == AWAITING_HOST)
        return written

    def record_response(self, day: date, item: Response) -> bool:
        written = super().record_response(day, item)
        self.follow(day, item.expectation_id)
        return written

    def record_settlement(self, item: Settlement) -> bool:
        written = super().record_settlement(item)
        self.follow(item.day, item.expectation_id)
        return written

    def record_ruling(self, day: date, item: Ruling) -> bool:
        written = super().record_ruling(day, item)
        self.follow(day, item.expectation_id)
        return written

    def follow(
        self, day: date, expectation_id: str, *, start: bool = False, claims: Mapping[str, Any] | None = None
    ) -> None:
        """更新这一条的记录：``start`` 为假时没有记录就什么都不做（只看文件在不在）。失败只报出、计数，不往上抛。"""

        try:
            if not start and not self.records.exists(day, expectation_id):
                return
            self.records.update(
                day,
                expectation_id,
                lambda: reminder_fields(
                    self, day, expectation_id, lane=self.records.lane, kind_name=self.kind_name, zone=self.zone
                ),
                claims=claims,
            )
        except Exception:  # noqa: BLE001 - 记录只是记录：写不成不让写账失败
            self.records.failures += 1
            logger.exception("reminder record %s (%s) was not updated", expectation_id, self.records.lane)


class RecordingOutletStore(OutletStore):
    """一支出口的传输状态，写取走记录之后把这条请求此刻的整段取走状态交给提醒记录。"""

    def __init__(
        self, root: str | Path, *, ledger: RecordingLedgerStore, limits: ForesightStoreConfig | None = None
    ) -> None:
        super().__init__(root, limits=limits)
        self.ledger = ledger

    def write_claim(self, request_id: str, payload: Mapping[str, Any]) -> None:
        super().write_claim(request_id, payload)
        try:
            day, claims = _claim_section(payload, self.ledger.zone)
        except (KeyError, TypeError, ValueError):
            self.ledger.records.failures += 1
            logger.exception("claim %s could not be read for its reminder record", request_id)
            return
        # 取走记录只为「待取」的请求而有：记录还没有（当初没写成）就照账补建
        self.ledger.follow(day, request_id, start=True, claims=claims)

    def remove_claim(self, request_id: str) -> bool:
        """出口直接删掉取走记录（过了最晚时限、时限后才问、问之前已经做了）时不经过 ``write_claim``：删之前把这一刻还没结束的
        那一次取走在记录里收尾（结局记"出口清掉了"，为什么看账上之后的递送），不让记录永远停在"还在手里"。"""

        try:
            # 读不出来的那份记到自己的清单里，不进出口共享的"读不出来"计数（出口清扫时自己数一遍）
            raw = self._files.read(self._claims / f"{request_id}.json", dict, sink=[])
            closing = (
                None
                if raw is None or raw.get("holder") is None
                else {
                    **raw,
                    "holder": None,
                    "history": [*raw["history"], {**raw["holder"], "ended": raw["holder"].get("ended") or "removed"}],
                }
            )
            section = None if closing is None else _claim_section(closing, self.ledger.zone)
        except (KeyError, TypeError, ValueError, AttributeError):
            self.ledger.records.failures += 1
            logger.exception("claim %s could not be read before it was removed", request_id)
            section = None
        removed = super().remove_claim(request_id)
        if section is not None:
            self.ledger.follow(section[0], request_id, claims=section[1])
        return removed


def _claim_section(payload: Mapping[str, Any], zone: ZoneInfo) -> tuple[date, dict[str, Any]]:
    """出口取走记录的形状（``runtime.foresight_outlet._encode_claim``）→ 预期所在的那一天、记录里的取走那一段。"""

    holders = [*payload["history"], *([payload["holder"]] if payload["holder"] is not None else [])]
    moments = sorted(
        (datetime.fromisoformat(str(item["claimed_at"])), str(item["host"]), item["ended"]) for item in holders
    )
    handoffs = [
        {
            "claimed_at": moment.astimezone(zone).isoformat(),
            "host": host,
            "ended": None if ended is None else str(ended),
        }
        for moment, host, ended in moments
    ]
    return date.fromisoformat(str(payload["request"]["day"])), {
        "attempts": int(payload["attempts"]),
        "handoffs": handoffs,
    }


__all__ = ["RecordingLedgerStore", "RecordingOutletStore", "reminder_fields"]
