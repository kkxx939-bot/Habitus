"""夜线多进程（共享说明 #5、#6；验证轮的对抗测试收进仓库）：两个真实 OS 进程挂同一个 root、同一份 SQLite 锁库——

- 同时跑同一晚：每一段只跑一次；
- 一个在结构段里被 ``kill -9``：租约过期之前另一个拿不到，过期后递增 generation 接管、把这一晚跑完；
- 一个在结构段里被 ``SIGSTOP``、另一个接管跑完，再 ``SIGCONT``：恢复的陈旧进程续不上租约、写不进账，账逐字节不变；
- 陈旧认领（全局租约已被接管）的续约与结算全部被拒。

真实进程加真实租约过期，这几条一共约 20 秒（``multiprocess`` 标记）；CI 的 ``pytest -q`` 照常跑到。Windows 没有 ``SIGSTOP`` /
``SIGKILL`` 的语义，跳过。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts.lock import LockLostError
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.runtime.nightly import DONE, STEPS, StepResult
from habitus.runtime.nightly_book import require
from habitus.runtime.nightly_jobs import NightClaimLostError, NightJobs, NightLedger, claim, claimed, renewed, settled
from habitus.runtime.nightly_settings import NightlySettings
from tests.integration import nightly_mp_worker as worker_module

pytestmark = [
    pytest.mark.multiprocess,
    pytest.mark.skipif(sys.platform == "win32", reason="需要 POSIX 的 SIGKILL / SIGSTOP"),
]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(worker_module.__file__)
TTL = worker_module.TTL_SECONDS
NIGHT = worker_module.NIGHT


def _spawn(root: Path, worker_id: str, looks: int, *slow: str) -> subprocess.Popen[bytes]:
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)}
    return subprocess.Popen([sys.executable, str(SCRIPT), str(root), worker_id, str(looks), *slow], env=env)


def _events(root: Path) -> list[dict[str, object]]:
    path = root / "events.log"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _wait_for(root: Path, predicate, timeout: float = 20.0) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _events(root)
        if predicate(events):
            return events
        time.sleep(0.05)
    raise AssertionError(f"timed out; events: {_events(root)[-10:]}")


def _starts(events: list[dict[str, object]], step: str) -> list[dict[str, object]]:
    return [item for item in events if item["ev"] == "start" and item["step"] == step]


def _ledger(root: Path) -> NightLedger:
    ledger = NightJobs(root / "jobs", artifact_root=root, kept_nights=366).read(NIGHT)
    assert ledger is not None
    return ledger


def test_two_processes_on_one_night_run_every_step_once(tmp_path: Path) -> None:
    first, second = _spawn(tmp_path, "a", 40), _spawn(tmp_path, "b", 40)
    assert first.wait(timeout=60) == 0 and second.wait(timeout=60) == 0
    events = _events(tmp_path)
    assert all(len(_starts(events, step)) == 1 for step in STEPS)
    ledger = _ledger(tmp_path)
    assert ledger.complete and {job.claim_generation for job in ledger.jobs} == {1}


def test_a_killed_process_is_taken_over_only_after_its_lease_expires_with_a_new_generation(tmp_path: Path) -> None:
    victim = _spawn(tmp_path, "p1", 5, "structure=30")
    _wait_for(tmp_path, lambda events: bool(_starts(events, "structure")))
    killed_at = time.time()
    victim.send_signal(signal.SIGKILL)
    victim.wait(timeout=10)
    heir = _spawn(tmp_path, "p2", 60)
    assert heir.wait(timeout=60) == 0
    events = _events(tmp_path)
    taken = [item for item in _starts(events, "structure") if item["wid"] == "p2"]
    assert len(taken) == 1 and float(taken[0]["t"]) - killed_at >= TTL - 0.6  # 租约过期之前接不走
    assert [item["wid"] for item in _starts(events, "days")] == ["p2"]
    ledger = _ledger(tmp_path)
    assert ledger.complete and ledger.job("structure").claim_generation == 2 and ledger.job("structure").attempts == 2


def test_a_stopped_process_that_wakes_up_after_a_takeover_cannot_touch_the_ledger(tmp_path: Path) -> None:
    sleeper = _spawn(tmp_path, "p1", 5, "structure=4")
    _wait_for(tmp_path, lambda events: bool(_starts(events, "structure")))
    sleeper.send_signal(signal.SIGSTOP)
    try:
        heir = _spawn(tmp_path, "p2", 60)
        assert heir.wait(timeout=60) == 0
        finished = (tmp_path / "jobs" / f"{NIGHT.isoformat()}.json").read_bytes()
        assert _ledger(tmp_path).complete
    finally:
        sleeper.send_signal(signal.SIGCONT)
    sleeper.wait(timeout=30)
    events = _events(tmp_path)
    errors = [str(item["error"]) for item in events if item["wid"] == "p1" and item["ev"] == "error"]
    assert errors and any(
        name in error for error in errors for name in ("NightlyLeaseLostError", "LockLostError", "NightClaimLostError")
    )
    assert not [item for item in _starts(events, "days") if item["wid"] == "p1"]
    assert (tmp_path / "jobs" / f"{NIGHT.isoformat()}.json").read_bytes() == finished  # 陈旧进程没改账


def test_a_stale_claim_cannot_renew_or_settle_after_a_takeover(tmp_path: Path) -> None:
    jobs = NightJobs(tmp_path / "jobs", artifact_root=tmp_path, kept_nights=366)
    stale_store, fresh_store = SQLiteLockStore(tmp_path / "locks.sqlite3"), SQLiteLockStore(tmp_path / "locks.sqlite3")
    stale_lock = stale_store.acquire("scene:nightly:mp", 1)

    def stale_fence():  # type: ignore[no-untyped-def]
        return stale_store.fenced([stale_lock], 1)

    now = datetime.now(UTC)
    jobs.update(NIGHT, lambda ledger: ledger or NightLedger.fresh(NIGHT, now), fence=stale_fence)
    stale = claim(require(jobs.read(NIGHT)), "prepare", worker_id="A", now=now)
    jobs.update(NIGHT, lambda ledger: claimed(require(ledger), stale, now=now, ttl_seconds=1), fence=stale_fence)
    time.sleep(2.2)  # A 的全局租约与认领都过期
    fresh_lock = fresh_store.acquire("scene:nightly:mp", 30)

    def fresh_fence():  # type: ignore[no-untyped-def]
        return fresh_store.fenced([fresh_lock], 30)

    now = datetime.now(UTC)
    fresh = claim(require(jobs.read(NIGHT)), "prepare", worker_id="B", now=now)
    jobs.update(NIGHT, lambda ledger: claimed(require(ledger), fresh, now=now, ttl_seconds=30), fence=fresh_fence)
    assert fresh.claim_generation == stale.claim_generation + 1 and fresh_lock.fence == stale_lock.fence + 1
    settings = NightlySettings()

    def settle_with(token):  # type: ignore[no-untyped-def]
        return lambda ledger: settled(
            require(ledger),
            token,
            StepResult("prepare", DONE),
            now=now,
            duration_seconds=1.0,
            backoff=settings.backoff,
            max_attempts=3,
        )

    attempts = (
        (lambda ledger: renewed(require(ledger), stale, now=now, ttl_seconds=30), stale_fence, LockLostError),
        (settle_with(stale), stale_fence, LockLostError),
        (lambda ledger: renewed(require(ledger), stale, now=now, ttl_seconds=30), fresh_fence, NightClaimLostError),
        (settle_with(stale), fresh_fence, NightClaimLostError),
    )
    for change, fence, error in attempts:
        before = jobs.path(NIGHT).read_bytes()
        with pytest.raises(error):
            jobs.update(NIGHT, change, fence=fence)
        assert jobs.path(NIGHT).read_bytes() == before
    with pytest.raises(LockLostError):
        stale_store.renew(stale_lock, 30)
    jobs.update(NIGHT, settle_with(fresh), fence=fresh_fence)
    assert require(jobs.read(NIGHT)).job("prepare").state == "done"
