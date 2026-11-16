"""第二步丙二：盘上布局与读数（预期表 D1、D2、D4、D5，场景 C6）。

改动前应当失败：语义树直接写在 ``<scene_root>/`` 下、物理一支没有语义树与预测层、读数不带支、TODO 还在。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from tests.unit.physical_downstream.site import REPOSITORY_ROOT, Named, World, at, files_under, texts_under

D13 = date(2026, 4, 13)
OLD_LAYOUT = ("concepts", "occurrences", "relations", "advice", "generation", "model_queue")
DAILY = [("吃药", 8, 0, 10), ("跑步", 18, 0, 30)]


def _run(world: World, days: int = 3) -> None:
    world.router.pick_default = Named("写周报")
    world.physical_days(D13, [DAILY] * days, session="帮我写本周周报")


def test_d1_each_branch_has_its_own_scene_and_foresight_directory(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _run(world)
        for lane in ("session", "physical"):
            assert (world.scene(lane) / "generation").is_dir(), f"{lane}: no {world.scene(lane)}/generation"
            assert world.foresight(lane).is_dir(), f"{lane}: no {world.foresight(lane)}"
        left = sorted(name for name in OLD_LAYOUT if (world.scene_root / name).exists())
        assert left == [], f"the scene tree still writes directly under <scene_root>/: {left}"
    finally:
        world.close()


@pytest.mark.parametrize("folder", OLD_LAYOUT)
def test_d1_an_old_layout_directory_under_the_scene_root_is_refused_at_start_up_naming_it(
    tmp_path: Path, folder: str
) -> None:
    from tests.unit.word_pick.site import config, runtime_of

    def options(raw: dict[str, object]) -> None:
        raw["foresight"] = {"enabled": True}

    habitus_config = config(tmp_path, edit=options)
    (habitus_config.scene_root / folder).mkdir(parents=True)
    with pytest.raises(Exception) as caught:  # noqa: PT011 - 报什么错由实现定，只要明确报错、点出目录
        runtime, _ = runtime_of(habitus_config)
        import asyncio

        asyncio.run(runtime.close())
    assert folder in str(caught.value)


def test_d2_the_physical_branch_has_its_own_generation_concepts_relations_table_ledger_and_outlet(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _run(world, days=4)
        scene, foresight = world.scene("physical"), world.foresight("physical")
        assert world.marks("physical") == ["2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17"]
        assert (scene / "concepts").is_dir()
        assert (scene / "relations" / "physical").is_dir()
        found = files_under(foresight)
        assert any(name.startswith("opportunities/physical/") for name in found), found
        assert any(name.startswith("unit_history/physical/") for name in found), found
        assert "tick_watermark.json" in found
        assert any(name.startswith("outlet/") for name in found), found
        # 模型队列流水：今天这一代之后物理一支还没封口的一条判断，队列选一次词、记进物理一支的流水
        today = D13 + timedelta(days=4)
        world.deliver(
            {
                "source": "physical/s12",
                "device": "watch-01",
                "delivery_id": "late",
                "sent_at": at(today, 10, 15).isoformat(),
                "segments": [
                    {"id": "late-0", "behavior": "吃药", "started_at": at(today, 10).isoformat(),
                     "ended_at": at(today, 10, 10).isoformat()},
                ],
                "coverage": [
                    {"started_at": at(today, 0).isoformat(), "ended_at": at(today, 10, 15).isoformat(),
                     "state": "observing"}
                ],
            },
            now=at(today, 10, 15),
        )  # fmt: skip
        world.drain(at(today, 10, 20))
        assert files_under(scene / "model_queue"), "the physical model queue journal is empty"
        assert not files_under(world.scene("session") / "model_queue") or "p-k" not in texts_under(
            world.scene("session") / "model_queue"
        )
    finally:
        world.close()


def test_d4_the_generation_report_and_the_metrics_say_which_branch(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _run(world)
        reports = world.scene("physical") / "generation" / "reports"
        assert files_under(reports), "the physical branch wrote no generation report"
        assert "physical" in texts_under(reports)
        assert "session" in texts_under(world.scene("session") / "generation" / "reports")
        metrics = world.runtime.prometheus_metrics()
        runs = [line for line in metrics.splitlines() if line.startswith("habitus_generation_runs_total{")]
        assert any('lane="physical"' in line or 'branch="physical"' in line for line in runs), runs
        assert any('lane="session"' in line or 'branch="session"' in line for line in runs), runs
    finally:
        world.close()


def test_d5_the_todo_that_wires_only_the_session_branch_is_gone() -> None:
    runtime = REPOSITORY_ROOT / "habitus" / "runtime"
    found = [
        f"{path.name}:{number}"
        for path in sorted(runtime.glob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "TODO(BHV-SESSION-LANE-001)：预测层" in line or "只接会话那一支" in line
    ]
    assert found == []


def test_c6_removing_the_physical_scene_directory_leaves_the_session_branch_untouched(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _run(world)
        physical = world.scene("physical")
        assert physical.is_dir(), "no physical scene directory to remove"
        import shutil

        before = {name: (world.scene("session") / name).read_bytes() for name in files_under(world.scene("session"))}
        shutil.rmtree(physical)
        after = {name: (world.scene("session") / name).read_bytes() for name in files_under(world.scene("session"))}
        assert after == before
        day = D13 + timedelta(days=3)
        world.set_clock(at(day, 9, 2))
        world.turn("conv", at(day, 9), "帮我写本周周报")
        world.night(day + timedelta(days=1))
        assert "2026-04-17" in world.marks("session")
        assert not [report for report in world.slots[-7:] if isinstance(report, BaseException)]
    finally:
        world.close()
