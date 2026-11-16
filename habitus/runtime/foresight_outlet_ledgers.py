"""提醒出口里的两支（丙二 D4，裁定 42）：出口只有宿主里的会话一个，人只有一个；两支共用这一个出口，共用的是出口，不是数据。

- 出口的传输状态按支存：``<foresight_root>/<lane>/outlet``（``OutletStores``）。宿主来取时看两支的可取请求合在一起、按最晚时限给；
  每条请求、取走记录、问过的提问都带是哪一支，写进那一支的目录；宿主来取过（活性）两支各记一份。递送那一拍每支只换自己那一支的
  可取请求（``BranchOutlet``）。
- 「说了」与回话写回请求所属那一支的账本，查"这条预期账上有过什么"也只查那一支（``OutletLedgers``）；出口里没有这一支（请求是
  坏的）按读不了处理。类名按那一支的词表取（``ReminderOutlet.kind_labels``）。
- 两支之间不仲裁：各支的"同一刻最多一条请求"照各支自己的单元算（``foresight.delivery``）——跨支再挡就是两条 lane 之间的关联。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger import Delivery
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet_store import OutletStore

if TYPE_CHECKING:
    from habitus.runtime.foresight_outlet import OpenRequest, ReminderOutlet


#: 一支出口的传输状态在 ``<foresight_root>/<lane>/outlet``。
OUTLET_DIRECTORY = "outlet"


class OutletStores:
    """两支的出口传输状态合成出口看到的那一份（``OutletStore`` 的读写面）：读合在一起，写按记录带的支进那一支的目录。"""

    def __init__(self, stores: Mapping[str, OutletStore]) -> None:
        if not stores:
            raise ValueError("the reminder outlet keeps at least one branch's transport state")
        self.stores = dict(stores)

    def of(self, lane: str) -> OutletStore:
        store = self.stores.get(lane)
        if store is None:
            raise ForesightError(f"the reminder outlet has no {lane} branch")
        return store

    def read_open(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for store in self.stores.values() for row in store.read_open())

    def drop_open(self, request_id: str) -> None:
        for store in self.stores.values():
            rows = store.read_open()
            kept = [row for row in rows if row.get("request_id") != request_id]
            if len(kept) != len(rows):
                store.write_open(kept)

    def claims(self) -> dict[str, dict[str, Any]]:
        return {key: raw for store in self.stores.values() for key, raw in store.claims().items()}

    def read_claim(self, request_id: str) -> dict[str, Any] | None:
        return next((found for store in self.stores.values() if (found := store.read_claim(request_id))), None)

    def claimed(self, request_id: str) -> bool:
        return any(store.claimed(request_id) for store in self.stores.values())

    def write_claim(self, request_id: str, payload: Mapping[str, Any]) -> None:
        request = payload.get("request")
        self.of(str(request.get("lane") if isinstance(request, Mapping) else None)).write_claim(request_id, payload)

    def remove_claim(self, request_id: str) -> bool:
        return any([store.remove_claim(request_id) for store in self.stores.values()])

    def broken_claims(self) -> tuple[str, ...]:
        return tuple(key for store in self.stores.values() for key in store.broken_claims())

    def asked(self) -> dict[str, dict[str, Any]]:
        return {key: raw for store in self.stores.values() for key, raw in store.asked().items()}

    def read_asked(self, key: str) -> dict[str, Any] | None:
        return next((found for store in self.stores.values() if (found := store.read_asked(key))), None)

    def write_asked(self, key: str, payload: Mapping[str, Any]) -> None:
        self.of(str(payload.get("lane"))).write_asked(key, payload)

    def remove_asked(self, key: str) -> bool:
        return any([store.remove_asked(key) for store in self.stores.values()])

    def broken_asked(self) -> tuple[str, ...]:
        return tuple(key for store in self.stores.values() for key in store.broken_asked())

    def take_unreadable(self) -> tuple[str, ...]:
        return tuple(path for store in self.stores.values() for path in store.take_unreadable())

    def touch_host(self, host: str, at: datetime) -> None:
        """宿主来取过：两支各记一份（活性是宿主的，不是哪一支的；各支的目录里都留着，一支的文件不依赖另一支）。"""

        for store in self.stores.values():
            store.touch_host(host, at)

    def hosts(self) -> dict[str, datetime]:
        found: dict[str, datetime] = {}
        for store in self.stores.values():
            for host, seen in store.hosts().items():
                found[host] = max(seen, found.get(host, seen))
        return found


class OutletLedgers:
    def __init__(self, ledgers: Mapping[str, ForesightLedgerStore]) -> None:
        if not ledgers:
            raise ValueError("the reminder outlet writes at least one branch's ledger")
        self._ledgers = dict(ledgers)

    def get(self, lane: str) -> ForesightLedgerStore | None:
        return self._ledgers.get(lane)

    def __contains__(self, lane: object) -> bool:
        return lane in self._ledgers

    def of(self, lane: str) -> ForesightLedgerStore:
        """那一支的账本；出口里没有这一支抛 ``ForesightError``（与读不了同一类）。"""

        ledger = self._ledgers.get(lane)
        if ledger is None:
            raise ForesightError(f"the reminder outlet has no {lane} branch")
        return ledger

    def deliveries(self, lane: str, day: date, expectation_id: str) -> list[Delivery]:
        """这条预期在它那一支账上那一天的全部递送。读不了往上抛。"""

        return [item for item in self.of(lane).deliveries_on(day) if item.expectation_id == expectation_id]

    def latest_reason(self, lane: str, day: date, expectation_id: str) -> str | None:
        """这条预期账上最晚一条递送的原因（按决定时刻）；读不到当没有。"""

        try:
            mine = self.deliveries(lane, day, expectation_id)
        except (ForesightError, OSError):
            return None
        return max(mine, key=lambda item: item.decided_at).reason if mine else None

    def has_spoken(self, lane: str, day: date, expectation_id: str) -> bool:
        """这条预期账上已经有一条「说了」；读不到当没有。"""

        try:
            return any(item.spoken for item in self.deliveries(lane, day, expectation_id))
        except (ForesightError, OSError):
            return False


class BranchOutlet:
    """一支的递送那一拍看到的出口（``OutletPort``）：活性看整个出口（宿主只有一个），发布只换这一支的请求。"""

    def __init__(self, outlet: ReminderOutlet, lane: str) -> None:
        if lane not in outlet.ledgers:
            raise ValueError(f"the reminder outlet has no {lane} branch")
        self.outlet = outlet
        self.lane = lane

    def alive(self, now: datetime) -> bool:
        return self.outlet.alive(now)

    def publish(self, rows: Sequence[OpenRequest], *, now: datetime) -> int:
        return self.outlet.publish(self.lane, rows, now=now)


__all__ = ["OUTLET_DIRECTORY", "BranchOutlet", "OutletLedgers", "OutletStores"]
