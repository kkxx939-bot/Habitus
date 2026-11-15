"""词表状态的跨进程写（第二步乙修补）：两个进程同时写同一棵树的 ``kinds.state.json``，谁的修改都不丢。

两个写者是会话 lane 选词时建节点、换代里数词表；它们可能在两个进程里（同一个根上开着两个 Runtime）。写入整段持行为根锁库上的
租约，所以盘上的写入序号正好等于两边写入的次数之和。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration import kinds_mp_worker as worker_module

pytestmark = pytest.mark.multiprocess
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(worker_module.__file__)
N = 25


def _run(root: Path, *workers: tuple[str, str]) -> dict[str, object]:
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    procs = [
        subprocess.Popen([sys.executable, str(SCRIPT), str(root), mode, worker_id, str(N)], env=env)
        for mode, worker_id in workers
    ]
    (root / "go").write_text("go", encoding="utf-8")
    assert all(proc.wait(timeout=120) == 0 for proc in procs)
    assert all((root / f"{worker_id}.done").exists() for _mode, worker_id in workers)
    state: dict[str, object] = json.loads((root / "tree" / "kinds.state.json").read_text(encoding="utf-8"))
    return state


@pytest.mark.parametrize("round_", range(3))
def test_two_processes_growing_nodes_at_once_lose_none(tmp_path: Path, round_: int) -> None:
    state = _run(tmp_path, ("grow", "甲"), ("grow", "乙"))
    grown = state["grown"]
    assert isinstance(grown, list)
    names = sorted(item["name"] for item in grown)
    assert names == sorted([f"甲{index:02d}号" for index in range(N)] + [f"乙{index:02d}号" for index in range(N)])
    assert len({item["id"] for item in grown}) == 2 * N
    assert state["sequence"] == 2 * N and state["deleted"] == []


@pytest.mark.parametrize("round_", range(3))
def test_a_generation_counting_and_a_turn_growing_in_two_processes_both_land(tmp_path: Path, round_: int) -> None:
    state = _run(tmp_path, ("grow", "甲"), ("count", "换"))
    grown, active = state["grown"], state["active"]
    assert isinstance(grown, list) and isinstance(active, list)
    assert sorted(item["name"] for item in grown) == sorted(f"甲{index:02d}号" for index in range(N))
    assert len(active) == N  # 每一次数词表都多转一个在用的词
    assert state["revision"] == N and state["sequence"] == 2 * N
