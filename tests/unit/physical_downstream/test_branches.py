"""第二步丙二：两支分开（场景 C1–C5、C7；共享说明 3.1、3.2）。

两支各一整套：换代的条件、锁、退避、完成标记、卡住的日子各支各算各记；一支失败不拖另一支。改动前物理一支没有换代，
这几条在"物理一支做了 / 报了"那一处失败。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from habitus.model_client.errors import ModelTransportError
from habitus.runtime.health import RuntimeHealthStatus
from tests.unit.physical_downstream.site import Named, World, at, delivery, texts_under

D13 = date(2026, 4, 13)
DAILY = [("吃药", 8, 0, 10), ("跑步", 18, 0, 30)]
SAID = "帮我写本周周报"


def _session(world: World) -> None:
    world.router.pick_default = Named("写周报")


def _day(world: World, day: date, *, physical: bool = True, session: bool = True) -> None:
    """``day`` 这一天：会话 09:00 一轮、物理一份整天的交付（次日 00:05 送到）。不过夜。"""

    if session:
        world.set_clock(at(day, 9, 2))
        world.turn("conv", at(day, 9), SAID)
    if physical:
        from tests.unit.physical_downstream.site import day_delivery

        world.deliver(day_delivery(day, DAILY), now=at(day + timedelta(days=1), 0, 5))


def _first_slot_with(world: World, lane: str, day: str, start: Any, end: Any) -> int | None:
    """从 ``start`` 到 ``end`` 逐槽跑，``lane`` 的 ``day`` 那张完成标记第一次出现在第几槽（没出现是 None）。"""

    found = None
    moment, index = start, 0
    while moment <= end:
        world.slot(moment)
        if found is None and day in world.marks(lane):
            found = index
        moment += timedelta(minutes=15)
        index += 1
    return found


def test_c1_both_branches_each_generate_in_the_same_slot_and_write_their_own_mark(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _session(world)
        for offset in range(2):
            day = D13 + timedelta(days=offset)
            _day(world, day)
            world.night(day + timedelta(days=1))
        day = D13 + timedelta(days=2)
        _day(world, day)
        night = day + timedelta(days=1)
        slots: dict[str, int | None] = {}
        moment, index = at(night, 0), 0
        while moment <= at(night, 1, 30):
            world.slot(moment)
            for lane in ("session", "physical"):
                if lane not in slots and night.isoformat() in world.marks(lane):
                    slots[lane] = index
            moment += timedelta(minutes=15)
            index += 1
        assert set(slots) == {"session", "physical"}, slots
        assert slots["session"] == slots["physical"]
        assert world.marks("session") == world.marks("physical") == ["2026-04-14", "2026-04-15", "2026-04-16"]
    finally:
        world.close()


def test_c2_a_must_step_failure_in_the_physical_generation_does_not_stop_the_session_one(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _session(world)
        # 物理一支的机会表写不进去：机会表是必须成的一步。只让"写"失败（阶段 C 改的造法）：原来把
        # ``<foresight_root>/physical/opportunities`` 占成文件，预测拍每拍读机会表也跟着抛，预测拍水位一直不推，
        # 换代的水位条件（AGENTS.md 换代那一条）永远不满足，物理一支根本没开始换代，测不到"必须成的一步失败"。
        pipeline = world.runtime.components.pipeline("physical")
        assert pipeline is not None and pipeline.generation.generation.foresight is not None
        store = pipeline.generation.generation.foresight.store

        def refuse(*_: object, **__: object) -> None:
            raise OSError("opportunities: disk refused the physical table")

        store.write = refuse
        for offset in range(2):
            day = D13 + timedelta(days=offset)
            _day(world, day)
            world.night(day + timedelta(days=1))
        assert world.marks("session") == ["2026-04-14", "2026-04-15"]
        assert world.marks("physical") == []
        reports = texts_under(world.scene("physical") / "generation" / "reports")
        assert "opportunities" in reports, "the physical branch reported no failed generation"
        health = asyncio.run(world.runtime.health())
        physical = [
            check
            for check in health.checks
            if check.status is not RuntimeHealthStatus.HEALTHY and "physical" in f"{check.name} {check.detail}"
        ]
        assert any("retry" in check.detail for check in physical), [(c.name, c.detail) for c in health.checks]
        session = [
            check
            for check in health.checks
            if check.name.startswith("generation")
            and check.status is not RuntimeHealthStatus.HEALTHY
            and "physical" not in f"{check.name} {check.detail}"
        ]
        assert session == []
    finally:
        world.close()


def _lock_keys(world: World, lane: str) -> list[str]:
    """语义树写锁的键（由语义树根派生，``runtime.scene.scene_write_lock_key``）：改动后这一支的根是 ``<scene_root>/<lane>``，
    改动前只有 ``<scene_root>`` 一个根（会话一支用）。两个都占上。"""

    roots = [world.scene_root / lane] + ([world.scene_root] if lane == "session" else [])
    return [
        "scene:generation:" + hashlib.sha256(str(root.expanduser().absolute()).encode("utf-8")).hexdigest()[:24]
        for root in roots
    ]


def _short_leases(raw: dict[str, Any]) -> None:
    raw["scene"] = {"enabled": True, "generation_lease_ttl_seconds": 5, "generation_heartbeat_seconds": 2}


def test_c3_the_session_scene_lock_held_elsewhere_stops_only_the_session_slot(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0), edit=_short_leases)
    try:
        _session(world)
        _day(world, D13)
        lock_store = world.runtime.components.infrastructure.path_lock.lock_store
        tokens = [lock_store.acquire(key, ttl_seconds=3600) for key in _lock_keys(world, "session")]
        try:
            world.night(D13 + timedelta(days=1))
        finally:
            for token in tokens:
                lock_store.release(token)
        assert world.marks("session") == []
        assert world.marks("physical") == ["2026-04-14"]
    finally:
        world.close()


def _fail_fast(raw: dict[str, Any]) -> None:
    raw["scene"] = {"enabled": True}
    raw["models"]["call_failures"] = {"breaker_failure_threshold": 1}


def test_c4_a_model_outage_pauses_both_branch_queues_through_the_one_chat_breaker(tmp_path: Path) -> None:
    """阶段 C 的造法（物理那条没选词的判断放在今天这一代之前，``unsealed`` 才认欠账）。裁定 108 起"整体退避"并进了对话熔断：
    两支共用一个对话模型、一个熔断——物理一支的选词把模型调不通，熔断打开，两支的队列都停在试探之前、不再问模型；健康检查
    只有一项 ``model_chat`` 写着原因，模型队列那几项不报退避。"""

    world = World(tmp_path, start=at(D13, 0), edit=_fail_fast)
    try:
        _session(world)
        _day(world, D13)
        world.night(D13 + timedelta(days=1))
        yesterday = D13 + timedelta(days=1)
        today = yesterday + timedelta(days=1)
        _day(world, yesterday)
        world.slots_between(at(today, 0), at(today, 0, 45))
        assert "2026-04-14" in world.marks("physical"), "the physical branch never generated"
        world.set_clock(at(today, 1))
        world.deliver(
            delivery("late", [("吃药", at(today, 0, 40), at(today, 0, 50))],
                     [(at(today, 0), at(today, 1), "observing")], sent_at=at(today, 1)),
        )  # fmt: skip
        world.reduce()
        world.router.physical.extend([ModelTransportError("down")] * 20)
        before = len(world.router.physical_calls())
        reports = world.drain(at(today, 1))
        asked = len(world.router.physical_calls())
        assert asked > before, "the physical queue never asked for the judgement"
        assert reports["physical"].backing_off_until is not None, reports
        health = asyncio.run(world.runtime.health())
        chat = next(check for check in health.checks if check.name == "model_chat")
        assert chat.status is RuntimeHealthStatus.DEGRADED and "network_error" in chat.detail, chat
        assert not any("backoff" in check.name for check in health.checks)
        # 熔断开着、没到试探的时候：两支都不再问模型
        reports = world.drain(at(today, 1) + timedelta(seconds=30))
        assert len(world.router.physical_calls()) == asked
        assert all(report.backing_off_until is not None for report in reports.values()), reports
    finally:
        world.close()


def test_c5_a_physical_word_and_a_session_word_on_the_same_days_never_meet(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _session(world)
        for offset in range(3):
            day = D13 + timedelta(days=offset)
            _day(world, day)
            world.night(day + timedelta(days=1))
        physical, session = world.scene("physical"), world.session_scene()
        assert (physical / "concepts").is_dir(), "the physical branch has no concepts"
        assert any(path.name.startswith("p-k") for path in (physical / "concepts").iterdir())
        assert "s-k" not in texts_under(physical / "concepts") + texts_under(physical / "relations")
        assert "p-k" not in texts_under(session / "concepts") + texts_under(session / "relations")
        assert "p-k" not in texts_under(world.foresight("session") / "opportunities")
        assert "s-k" not in texts_under(world.foresight("physical") / "opportunities")
    finally:
        world.close()


def test_c7_a_gap_in_the_physical_closed_days_stalls_only_the_physical_branch_and_health_says_so(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _session(world)
        for offset in range(3):
            day = D13 + timedelta(days=offset)
            _day(world, day)
            world.night(day + timedelta(days=1))
        # 物理一支的定稿日账上去掉 04-14：之后归约只往最后定稿日之后推，它一直断着
        ledger = Path(world.physical.tree.root).parent / "reduction" / "day_closure.json"
        body = json.loads(ledger.read_text(encoding="utf-8"))
        body["closed"].pop("2026-04-14")
        ledger.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        day = D13 + timedelta(days=3)
        _day(world, day)
        world.night(day + timedelta(days=1), until=(2, 30))
        assert "2026-04-17" in world.marks("session")
        assert "2026-04-17" not in world.marks("physical")
        assert "2026-04-16" in world.marks("physical"), "the physical branch never generated before the gap"
        health = asyncio.run(world.runtime.health())
        stuck = [check for check in health.checks if "stuck" in check.name]
        assert any("physical" in f"{check.name} {check.detail}" and "2026-04-14" in check.detail for check in stuck), [
            (c.name, c.detail) for c in health.checks
        ]
        assert not any("physical" not in f"{check.name} {check.detail}" for check in stuck)
    finally:
        world.close()
