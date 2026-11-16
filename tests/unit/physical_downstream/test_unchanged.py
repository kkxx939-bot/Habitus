"""第二步丙二：应当不变的（预期表第 1 节 S2、S4、S5；S1 是重放比对、S3 是现有三组测试不动就过，见报告）。

改动前后都应当通过。会话一支的语义树目录用 ``World.session_scene``（改动前 ``<scene_root>/``、改动后 ``<scene_root>/session/``）。
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.behavior.fusion.lanes.physical.model import PhysicalDeliveryError
from habitus.runtime.health import RuntimeHealthStatus
from tests.unit.physical_downstream.site import (
    MEDICINE,
    RUN,
    WEEKLY,
    Named,
    World,
    at,
    day_delivery,
    delivery,
)

D13 = date(2026, 4, 13)


def test_s2_no_physical_record_the_session_branch_generates_and_the_physical_one_costs_nothing(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.router.pick_default = Named("写周报")
        for offset in range(3):
            day = D13 + timedelta(days=offset)
            world.set_clock(at(day, 9, 2))
            world.turn("conv", at(day, 9), "帮我写本周周报")
            calls = len(world.router.calls)
            world.night(day + timedelta(days=1))
            # 夜里那几槽：两支的选词 0 次（会话一支的判断带着词；物理一支一条判断都没有）
            assert [call.kind for call in world.router.calls[calls:] if call.kind == "pick"] == []
        assert world.marks("session") == ["2026-04-14", "2026-04-15", "2026-04-16"]
        assert world.router.physical_calls() == []
        assert not [report for report in world.slots if isinstance(report, BaseException)]
        assert world.tokens("session") and all(token == WEEKLY for token in world.tokens("session").values())
        health = asyncio.run(world.runtime.health())
        flagged = [
            check
            for check in health.checks
            if check.status is not RuntimeHealthStatus.HEALTHY and "generation" in check.name
        ]
        assert flagged == []
    finally:
        world.close()


def test_s4_the_physical_entry_fusion_and_reduction_land_records_and_pick_at_the_reduction(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        payload = delivery(
            "d-1",
            [("吃药", at(D13, 8), at(D13, 8, 10)), ("跑步", at(D13, 18), at(D13, 18, 30))],
            [(at(D13, 0), at(D13, 20), "observing"), (at(D13, 20), at(D13 + timedelta(days=1), 0), "not_worn")],
            sent_at=at(D13 + timedelta(days=1), 0),
        )
        outcomes = world.deliver(payload, now=at(D13 + timedelta(days=1), 0, 5))
        assert len(outcomes) == 3  # 两段行为 + 一段没在看
        assert world.router.physical_calls() == []  # 物理的判断不带选词结果
        again = world.deliver(payload)  # 同一份再送一遍不重复写
        assert [item.disposition.value for item in again] != [item.disposition.value for item in outcomes]
        world.set_clock(at(D13 + timedelta(days=1), 1, 30))
        world.reduce()
        assert len(world.router.physical_calls()) == 2  # 归约那一步一条一次
        assert sorted(world.tokens("physical").values()) == sorted([MEDICINE, RUN])
        assert len(world.gaps("physical")) == 1
        with pytest.raises(PhysicalDeliveryError):  # 早于补发跨度的整份拒收
            world.deliver(day_delivery(D13 - timedelta(days=30), [("吃药", 8, 0, 10)]))
    finally:
        world.close()


def test_s5_one_behaviour_at_one_moment_puts_one_request_at_the_outlet(tmp_path: Path) -> None:
    """现有口径（``runtime.foresight_delivery`` 模块文档：同一刻只接手一条）：同一个行为同一刻的两条预期，出口里只有一条请求。"""

    day = date(2026, 4, 14)
    world = World(tmp_path, start=at(day, 0))
    try:
        world.use("session", WEEKLY)
        world.earn("session", WEEKLY, at(day, 0))
        window = (at(day, 9), at(day, 9, 30))
        world.expect("session", "weekly-day", WEEKLY, window=window, deadline=at(day, 23, 59))
        world.expect("session", "weekly-ctx", WEEKLY, window=window, deadline=at(day, 23, 59), source="情境")
        assert world.claim(at(day, 8), session="probe") == ()  # 宿主来过（出口活着）
        world.slots_between(at(day, 9), at(day, 10), reduce=False)
        claimed = world.claim(at(day, 10, 5))
        assert len(claimed) == 1
        reasons = world.reasons("session", day)
        assert sum(1 for made in reasons.values() if "待取" in made) == 1
    finally:
        world.close()
