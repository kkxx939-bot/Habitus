"""换代与预测拍的多进程（E10；实时化方案第 9 节）：真实 OS 进程挂同一个根、同一份 SQLite 锁库——

- 两个进程的预测 worker 同时到了同一槽：只换代一次（完成标记一份、语义树那一步只开始一次）——阶段 2 第二小步起没有换代租约，
  守的是"预测根租约 + 完成标记"（A2-5）；
- 一个做到一半（语义树那一步里）被 ``kill -9``：预测根租约过期之前另一个拿不到，过期后接管、把这一代从头做完（A2-10）；
- 两个进程同时跑预测拍：预测根租约让写账本串行（没有交叠的写），同一身份只写一份。

取代夜线删掉的 ``test_nightly_multiprocess.py``；``multiprocess`` 标记，CI 的 ``pytest -q`` 照常跑到。Windows 没有 ``SIGKILL``，跳过。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.integration import generation_mp_worker as worker_module
from tests.unit.runtime.test_foresight_tick import TODAY, site_with_three_mornings, stores
from tests.unit.scene.fixtures import publish

pytestmark = [
    pytest.mark.multiprocess,
    pytest.mark.skipif(sys.platform == "win32", reason="需要 POSIX 的 SIGKILL"),
]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(worker_module.__file__)
TTL = worker_module.TTL_SECONDS


def _spawn(
    root: Path, mode: str, worker_id: str, looks: int = 1, scene_seconds: float = 0.0
) -> subprocess.Popen[bytes]:
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), str(root), mode, worker_id, str(looks), str(scene_seconds)], env=env
    )


def _events(root: Path) -> list[dict[str, object]]:
    path = root / "events.log"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _wait_for(root: Path, predicate, timeout: float = 60.0) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _events(root)
        if predicate(events):
            return events
        time.sleep(0.05)
    raise AssertionError(f"timed out; events: {_events(root)[-10:]}")


def _go(root: Path, procs: list[subprocess.Popen[bytes]]) -> None:
    """等每个进程都装好、停在起跑线（各写一次 ready 不方便：装配要读树，给足时间），再一起放行。"""

    time.sleep(1.0)
    (root / "go").write_text("go", encoding="utf-8")
    _wait_for(root, lambda events: sum(1 for item in events if item["ev"] == "go") == len(procs))


def _marks(root: Path) -> list[str]:
    return sorted(path.name for path in (root / "generation").glob("*.json"))


def test_two_processes_on_one_root_generate_the_day_once(tmp_path: Path) -> None:
    procs = [
        _spawn(tmp_path, "gen", "a", looks=3, scene_seconds=1.0),
        _spawn(tmp_path, "gen", "b", looks=3, scene_seconds=1.0),
    ]
    _go(tmp_path, procs)
    assert all(proc.wait(timeout=120) == 0 for proc in procs)
    events = _events(tmp_path)
    generated = [item for item in events if item["ev"] == "look" and item["got"] is not None]
    assert len(generated) == 1 and generated[0]["got"]["complete"] is True  # type: ignore[index]
    assert len([item for item in events if item["ev"] == "scene_start"]) == 1
    assert len(_marks(tmp_path)) == 1


def test_a_holder_killed_midway_is_taken_over_after_its_lease_expires(tmp_path: Path) -> None:
    first = _spawn(tmp_path, "gen", "a", looks=1, scene_seconds=30.0)
    _go(tmp_path, [first])
    _wait_for(tmp_path, lambda events: any(item["ev"] == "scene_start" and item["wid"] == "a" for item in events))
    os.remove(tmp_path / "go")
    second = _spawn(tmp_path, "gen", "b", looks=40, scene_seconds=0.0)
    (tmp_path / "go").write_text("go", encoding="utf-8")
    _wait_for(tmp_path, lambda events: any(item["ev"] == "look" and item["wid"] == "b" for item in events))
    events = _events(tmp_path)
    assert [item["got"] for item in events if item["ev"] == "look" and item["wid"] == "b"][0] is None  # a 还拿着
    first.send_signal(signal.SIGKILL)
    killed_at = time.time()
    first.wait(timeout=10)
    assert second.wait(timeout=120) == 0
    events = _events(tmp_path)
    took_over = [item for item in events if item["ev"] == "look" and item["wid"] == "b" and item["got"] is not None]
    assert len(took_over) == 1 and took_over[0]["got"]["complete"] is True  # type: ignore[index]
    # 接管在 a 死后、租约过期之后（最晚一个租期再加这一代本身的用时）
    assert killed_at < float(took_over[0]["t"]) < killed_at + TTL + 30.0  # type: ignore[arg-type]
    assert [item["wid"] for item in events if item["ev"] == "scene_start"] == ["a", "b"]
    assert len(_marks(tmp_path)) == 1


def _seed_ticks(root: Path) -> None:
    site, _codes = site_with_three_mornings(root)
    stores(root, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)  # 这一拍要建一条「事件」预期


def _overlapping_writes(events: list[dict[str, object]]) -> int:
    writing: set[object] = set()
    overlaps = 0
    for item in sorted(events, key=lambda row: float(row["t"])):  # type: ignore[arg-type]
        if item["ev"] == "record_start":
            if writing - {item["wid"]}:
                overlaps += 1
            writing.add(item["wid"])
        elif item["ev"] == "record_end":
            writing.discard(item["wid"])
    return overlaps


def test_two_processes_ticking_one_ledger_write_it_one_at_a_time(tmp_path: Path) -> None:
    """不持预测根租约直接写账本时，两个进程会同时写同一身份（这里会变红）。"""

    _seed_ticks(tmp_path)
    procs = [_spawn(tmp_path, "tick", "a"), _spawn(tmp_path, "tick", "b")]
    _go(tmp_path, procs)
    assert all(proc.wait(timeout=120) == 0 for proc in procs)
    events = _events(tmp_path)
    ticks = [item for item in events if item["ev"] == "tick"]
    assert len(ticks) == 2 and all(item["skipped"] is None for item in ticks)
    assert _overlapping_writes(events) == 0
    written = [item["id"] for item in events if item["ev"] == "record_end" and item["wrote"]]
    assert sum(int(item["recorded"]) for item in ticks) == len(written) == len(set(written)) == 1  # type: ignore[arg-type]
