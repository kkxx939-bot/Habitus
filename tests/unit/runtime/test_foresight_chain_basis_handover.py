"""裁定 92 第 2 条②的回归（由 ``37-材料/readme/repro-92/test_repro92_chain_basis.py`` 改成，断言已翻转）：事件链（查天气 →
吃维生素，成立、效应够大）当了依据，它的最晚时限正落在时窗末后第一拍（待办 Q7，不在本次范围），「待取」写下就过了时限、宿主没取；
被它「单元已选」吸收的日子那条（时限到 24:00）接手——同一拍进「待取」，他之后开口就取得到。依据说成了的照旧吸收（见
``test_foresight_delivery_handover``）。

真实链路：真实行为树、``build_table``（带一条成立的 chain 关系）、``ForesightTick`` + ``DeliveryStage``、真实 ``ReminderOutlet``
（宿主来取 ``claim``）。替身：两本账由 ``Seeded.earn`` 种在提醒档；语义树读数手填（成立、效应够大）。配置取默认：槽 15 分钟、
转移窗 3 槽。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from habitus.foresight import NoUnsealed
from habitus.foresight.opportunities import TreeReading, build_table
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_outlet import ReminderOutlet
from habitus.runtime.foresight_outlet_store import OutletStore
from habitus.runtime.foresight_tick import ForesightTick
from habitus.scene.concepts import ConceptSet
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import Relation, RelationKey, Segment, Status, build_timelines
from habitus.series.reader import read_series
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_daily_miss_and_silent_day import (
    CONFIG,
    LOSS,
    RULERS,
    TICK,
    ZONE,
    day_expectations,
    read_side,
)
from tests.unit.runtime.test_foresight_delivery import EVIDENCE, Seeded
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import Site, at, publish

WEATHER, VITAMIN = "查天气", "吃维生素"
START, TODAY = date(2026, 10, 2), date(2026, 10, 14)


class W:
    pass


def build(tmp: Path, *, chain: bool, reading: TreeReading | None) -> W:
    w = W()
    w.site = Site(tmp, now=at(START, 3, 0))
    root = tmp / "foresight" / "session"
    w.ledger, w.opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    w.concepts = ConceptSet([base(WEATHER), base(VITAMIN)])
    w.vit = kind_id(VITAMIN)
    # 历史：每天 07:45 查天气（10 分钟），吃维生素 08:22 / 08:38 交替（同一条链段：链末槽 07:45 之后 ≤ 3 槽）
    day = START
    flip = False
    while day < TODAY:
        publish(w.site.behavior_tree, day, WEATHER, 7, 45)
        publish(w.site.behavior_tree, day, VITAMIN, 8, 38 if flip else 22)
        flip = not flip
        day += timedelta(days=1)
    series = read_series(w.site.behavior_tree, cutoff=TODAY)
    line = build_timelines(series, w.concepts, {}, CONFIG)["session"]
    relations = (
        [
            Relation(
                key=RelationKey("session", kind_id(WEATHER), w.vit, Segment.CHAIN),
                status=Status.ESTABLISHED,
                upward=True,
            )
        ]
        if chain
        else []
    )
    table = build_table(
        line,
        relations,
        w.concepts,
        night=TODAY,
        relation_night=TODAY,
        slot_minutes=15,
        window_slots=3,
        recurrences=recurrences_of(series, RULERS),
    )
    if chain:
        (relation,) = table.relations
        table = replace(table, relations=(replace(relation, reading=reading),))
        w.relation_key = relation.key
    w.opportunities.write(table)
    seeded = Seeded(root, w.ledger)
    seeded.earn(kind=w.vit, trigger="interval", hits=8)
    if chain:
        seeded.earn(kind=w.vit, trigger=w.relation_key, hits=8, source="事件")
    w.outlet = ReminderOutlet(
        OutletStore(root / "outlet"),
        w.ledger,
        liveness=timedelta(hours=48),
        lease=timedelta(minutes=30),
        max_attempts=3,
        claim_limit=3,
        similarity=0.6,
    )
    w.host = w.outlet.tools[0].host
    w.delivery = DeliveryStage(
        w.ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=10,
        tick_seconds=TICK.total_seconds(),
        outlet=w.outlet,
        history=seeded.store,
    )
    return w


def run_morning(w: W, *, ask_at: time) -> dict[str, object]:
    """夜里那条线 01:15 建好今天的日子预期；今天 07:45 查了天气、没吃维生素。07:00 宿主来过一次（活着）；ask_at 他开口、宿主来取。"""

    day_expectations(w, TODAY)  # type: ignore[arg-type]
    publish(w.site.behavior_tree, TODAY, WEATHER, 7, 45)
    w.outlet.claim(w.host, "s-early", datetime.combine(TODAY, time(7, 0), tzinfo=ZONE), 1)
    claimed: list[str] = []
    published: list[tuple[str, str]] = []
    now = datetime.combine(TODAY, time(7, 0), tzinfo=ZONE)
    asked = datetime.combine(TODAY, ask_at, tzinfo=ZONE)
    while now <= datetime.combine(TODAY, time(10, 0), tzinfo=ZONE):
        if now - TICK < asked <= now:
            claimed = [item.request_id for item in w.outlet.claim(w.host, "s-talk", asked, 3)]
        asyncio.run(
            ForesightTick(
                w.ledger,
                w.opportunities,
                read_side(w),
                NoUnsealed(),
                clock=lambda c=now: c,
                delivery=w.delivery,
                tick_seconds=TICK.total_seconds(),
                recent=35,
                tolerance=LOSS,
            ).run_once()
        )
        for row in w.outlet.store.read_open():
            published.append((now.strftime("%H:%M"), str(row["deadline"])[11:16]))
        now += TICK
    rows = []
    decisions: dict[str, list[tuple[str, str]]] = {}
    for item in w.ledger.deliveries_on(TODAY):
        decisions.setdefault(item.expectation_id, []).append(
            (item.decided_at.astimezone(ZONE).strftime("%H:%M"), item.reason)
        )
    for item in sorted(w.ledger.expectations_on(TODAY), key=lambda e: e.formed_at):
        if item.kind_token != w.vit:
            continue
        rows.append(
            {
                "id": item.expectation_id,
                "source": item.source,
                "formed": item.formed_at.astimezone(ZONE).strftime("%H:%M"),
                "window": tuple(x.astimezone(ZONE).strftime("%H:%M") for x in item.window),
                "deadline": item.deadline.astimezone(ZONE).strftime("%m-%d %H:%M"),
                "decisions": sorted(decisions.get(item.expectation_id, [])),
            }
        )
    return {"rows": rows, "claimed": claimed, "published": sorted(set(published))}


def show(result: dict[str, object]) -> None:
    for row in result["rows"]:  # type: ignore[union-attr]
        print(row)
    print("claimed:", result["claimed"])
    print("open list (tick, deadline):", result["published"])


def test_the_day_reminder_takes_over_once_the_chain_basis_was_never_claimed(tmp_path: Path) -> None:
    w = build(tmp_path, chain=True, reading=TreeReading(0.95, 0.30, True))
    result = run_morning(w, ask_at=time(9, 10))
    show(result)
    rows = {row["source"]: row for row in result["rows"]}  # type: ignore[union-attr]
    event, day = rows["事件"], rows["日子"]
    assert event["window"] == ("08:22", "08:38") and event["deadline"].endswith("08:45")
    assert event["decisions"] == [("08:30", "等时窗末"), ("08:45", "待取"), ("09:00", "宿主没取")]  # Q7：依据说不成
    assert day["decisions"] == [("08:30", "单元已选"), ("09:00", "待取")]  # 日子那条同一拍接手
    assert result["claimed"] == [day["id"]]  # 09:10 他开口，取走的是日子那条（时限 24:00）
    assert ("09:00", "00:00") in result["published"]  # type: ignore[operator]


def test_before_the_basis_fails_nothing_extra_is_claimable(tmp_path: Path) -> None:
    """08:53 开口：依据那条 08:45 的「待取」时限已过、日子那条要到 09:00 那拍才接手——这一刻仍取不到（Q7 留待办）。"""

    w = build(tmp_path, chain=True, reading=TreeReading(0.95, 0.30, True))
    result = run_morning(w, ask_at=time(8, 53))
    assert result["claimed"] == []


def test_control_day_account_alone_is_claimable(tmp_path: Path) -> None:
    w = build(tmp_path, chain=False, reading=None)
    result = run_morning(w, ask_at=time(8, 53))
    (day,) = result["rows"]  # type: ignore[misc]
    assert day["decisions"] == [("08:30", "等时窗末"), ("08:45", "待取")]
    assert day["deadline"].endswith("10-15 00:00") and result["claimed"] == [day["id"]]
