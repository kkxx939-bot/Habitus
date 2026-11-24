"""递送那一拍的摆法（``DeliveryStage.run_once``，每 15 分钟一拍）：上级 P、下级 C 两条预期、账到提醒档、宿主活着；
断言读账上每条预期的递送原因与时刻、可取清单。几份递送测试共用。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from habitus.foresight import Seen
from habitus.foresight.delivery import done_since
from habitus.foresight.hierarchy import KindHierarchy
from habitus.foresight.ledger import Delivery, Expectation
from habitus.foresight.opportunities import OpportunityTable
from habitus.runtime.foresight_delivery import JudgeMaterials
from tests.unit.runtime.test_foresight_delivery import DAY0, Outlet, earn, stage

PARENT, CHILD = "s-k0109", "s-k0280"  # 做报表（上级）、做渠道日报（下级）
OTHER = "s-k0200"
KINDS = KindHierarchy({PARENT: frozenset({CHILD})})
DAY = DAY0.date()
MIDNIGHT = DAY0.replace(hour=0) + timedelta(days=1)


def clock(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return DAY0.replace(hour=hour, minute=minute, second=second)


def made(
    identity: str,
    kind: str,
    window: tuple[datetime, datetime],
    *,
    deadline: datetime = MIDNIGHT,
    observed: datetime | None = None,
    trigger: str = "rel",
    source: str = "日子",
    instance: str | None = None,
) -> Expectation:
    formed = observed if observed is not None else clock(0)
    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=DAY,
        formed_at=formed,
        source=source,
        trigger=trigger,
        trigger_instance=instance if instance is not None else f"occ:{identity}.md",
        window=window,
        deadline=deadline,
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def parent_item(**overrides: Any) -> Expectation:
    return made("p", PARENT, (clock(9, 10), clock(9, 40)), **overrides)


def child_item(**overrides: Any) -> Expectation:
    window = overrides.pop("window", (clock(11), clock(11, 15)))
    return made("c", CHILD, window, **overrides)


def site(
    tmp_path: Path, *items: Expectation, unearned: Sequence[str] = (), **overrides: Any
) -> tuple[Any, Any, Outlet]:
    """一份账本：``items`` 记上，它们的账都到提醒档（``unearned`` 里的类除外）。"""

    outlet = Outlet()
    run, ledger = stage(tmp_path, outlet=outlet, tick_seconds=900.0, **overrides)
    earned: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.kind_token, item.trigger, item.source)
        if item.kind_token not in unearned and key not in earned:
            earn(ledger, kind=item.kind_token, trigger=item.trigger, source=item.source)
            earned.add(key)
        ledger.record_expectation(item)
    return run, ledger, outlet


def materials(
    now: datetime,
    *,
    kinds: KindHierarchy = KINDS,
    table: OpportunityTable | None = None,
    rows: Sequence[Seen] = (),
) -> JudgeMaterials:
    seen = tuple(row for row in rows if row.started_at <= now)
    return JudgeMaterials(
        table,
        lambda uris: {},
        happened=lambda item: done_since(item, seen, now=now, kinds=kinds),
        seen=seen,
        kinds=kinds,
    )


def walk(run: Any, start: datetime, end: datetime, **options: Any) -> None:
    """从 ``start`` 到 ``end``（含）每 15 分钟一拍。"""

    now = start
    while now <= end:
        asyncio.run(run.run_once(now, materials(now, **options)))
        now += timedelta(minutes=15)


def book(ledger: Any, *identities: str) -> dict[str, list[tuple[str, str]]]:
    """账上每条预期的（递送原因, 决定时刻），按决定先后。"""

    found: dict[str, list[tuple[str, str]]] = {identity: [] for identity in identities}
    for item in sorted(ledger.deliveries_on(DAY), key=lambda delivery: delivery.decided_at):
        if item.expectation_id in found:
            found[item.expectation_id].append((item.reason, item.decided_at.strftime("%H:%M")))
    return found


def listed(outlet: Outlet) -> list[str]:
    return [row.request_id for row in outlet.published[-1]]


def said(identity: str, moment: datetime) -> Delivery:
    return Delivery(identity, moment, 1, True, "说了", 1.0, "题目", via="codex", turn=f"turn-{identity}")
