"""换代与预测拍的两把租约（``runtime.generation_lease.Lease``；实时化方案第 9 节、用户 11-10「两把锁」）。

- ``Lease`` 本身：进程内两段互斥；另一个持有者（另一份锁库连接，等同另一个进程）拿着时拿不到；持有者崩了（不放、不续）
  要等租约过期才能接管，接管拿到的是更大的 fencing 序号；续不上、租约要过期时取消持有它的那一段并抛 ``LeaseLostError``，
  一次续不上而租约还没到期不取消。
- 预测根租约的持有范围：换代里写账本的那几步（结算、机会表、日子预期、完成标记）持有它，调模型的语义树那一步不持有；
  预测拍整拍持有它——别人拿着时这一拍不写账本、不推进水位（报 ``LEDGER_BUSY``），放开后下一拍照常建。

用真实的 ``SQLiteLockStore``（两个实例连同一个文件，等同两个进程）。
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.foresight import NoUnsealed
from habitus.infrastructure.store.contracts.lock import LockLostError, LockToken
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_tick import LEDGER_BUSY, ForesightTick
from habitus.runtime.generation_lease import Lease, LeaseBusyError, LeaseLostError
from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore
from tests.unit.runtime.test_foresight_tick import TODAY as TICK_DAY
from tests.unit.runtime.test_foresight_tick import TOLERANCE, site_with_three_mornings, stores
from tests.unit.runtime.test_generation import generate, wired
from tests.unit.scene.fixtures import SUBJECT, at, publish

KEY = "foresight:root:test"


def locks(tmp_path: Path) -> SQLiteLockStore:
    return SQLiteLockStore(tmp_path / "locks.sqlite3")


def lease(store: SQLiteLockStore, *, ttl: float = 5.0, heartbeat: float = 1.0) -> Lease:
    return Lease(store, KEY, ttl_seconds=ttl, heartbeat_seconds=heartbeat, poll_seconds=0.05)


def held_elsewhere(store: SQLiteLockStore) -> bool:
    """此刻这把租约是不是被人拿着：用另一份锁库连接试拿一下（拿到就马上放掉）。"""

    try:
        token = store.acquire(KEY, 1)
    except TimeoutError:
        return True
    store.release(token)
    return False


# ── Lease 本身 ──────────────────────────────────────────────────────────


def test_two_holders_in_one_process_take_turns(tmp_path) -> None:
    shared = lease(locks(tmp_path))

    async def scenario() -> None:
        async with shared.held(wait_seconds=0.0):
            with pytest.raises(LeaseBusyError):
                async with shared.held(wait_seconds=0.0):
                    pass  # pragma: no cover - 拿不到
        async with shared.held(wait_seconds=0.0):
            pass

    asyncio.run(scenario())


def test_another_holder_keeps_the_lease_until_it_lets_go(tmp_path) -> None:
    mine, theirs = lease(locks(tmp_path)), lease(locks(tmp_path))

    async def scenario() -> None:
        async with theirs.held(wait_seconds=0.0):
            with pytest.raises(LeaseBusyError):
                async with mine.held(wait_seconds=0.2):
                    pass  # pragma: no cover - 拿不到
        async with mine.held(wait_seconds=0.0):
            pass

    asyncio.run(scenario())


def test_a_crashed_holder_is_taken_over_only_after_its_lease_expires(tmp_path) -> None:
    """持有者崩了（拿了不续也不放）：租期之内别人拿不到；过期后接管，拿到更大的 fencing 序号。"""

    crashed = locks(tmp_path).acquire(KEY, 1)
    survivor = lease(locks(tmp_path), ttl=2.0, heartbeat=0.5)
    got: list[LockToken] = []

    async def scenario() -> float:
        with pytest.raises(LeaseBusyError):
            async with survivor.held(wait_seconds=0.0):
                pass  # pragma: no cover - 还没过期
        started = time.monotonic()
        async with survivor.held(wait_seconds=5.0) as token:
            got.append(token)
        return time.monotonic() - started

    waited = asyncio.run(scenario())
    assert got and got[0].fence > crashed.fence
    assert waited < 4.0


class FlakyRenewals:
    """包一层真实锁库：续租按给定的剧本失败（``True`` = 这一次续不上）。"""

    def __init__(self, inner: SQLiteLockStore, script: list[bool]) -> None:
        self.inner = inner
        self.script = script
        self.renewals = 0

    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken:
        return self.inner.acquire(lock_key, ttl_seconds)

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken:
        fails = self.script[self.renewals] if self.renewals < len(self.script) else self.script[-1]
        self.renewals += 1
        if fails:
            raise LockLostError("renewal refused")
        return self.inner.renew(token, ttl_seconds)

    def release(self, token: LockToken) -> None:
        self.inner.release(token)


def test_a_lease_that_cannot_be_renewed_cancels_the_holder_before_it_expires(tmp_path) -> None:
    flaky = FlakyRenewals(locks(tmp_path), [True])
    held = Lease(flaky, KEY, ttl_seconds=2.0, heartbeat_seconds=0.5, poll_seconds=0.05)  # type: ignore[arg-type]
    reached: list[str] = []

    async def scenario() -> None:
        async with held.held(wait_seconds=0.0):
            await asyncio.sleep(10.0)
            reached.append("end")  # pragma: no cover - 应该在租约到期前被取消

    started = time.monotonic()
    with pytest.raises(LeaseLostError):
        asyncio.run(scenario())
    assert reached == [] and time.monotonic() - started < 3.0
    assert flaky.renewals >= 1


def test_one_missed_renewal_before_expiry_does_not_cancel(tmp_path) -> None:
    flaky = FlakyRenewals(locks(tmp_path), [True, False])
    held = Lease(flaky, KEY, ttl_seconds=2.0, heartbeat_seconds=0.5, poll_seconds=0.05)  # type: ignore[arg-type]

    async def scenario() -> str:
        async with held.held(wait_seconds=0.0):
            await asyncio.sleep(2.5)  # 比一个租期还长：靠第二次续租活下来
        return "done"

    assert asyncio.run(scenario()) == "done"
    assert flaky.renewals >= 2


# ── 预测根租约的持有范围 ────────────────────────────────────────────────


def test_the_generation_holds_the_ledger_lease_only_around_its_ledger_writes(tmp_path) -> None:
    """换代租约之外，写账本的那几步（结算、机会表、日子预期、完成标记）持预测根租约；探模型、续做迁移、词表、调模型的语义树
    都不持有它（换代调模型的几个小时不挡预测拍）。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    probe = locks(tmp_path)
    run.ledger_lease = lease(locks(tmp_path))
    seen: dict[str, bool] = {}

    settle = run.settlement.run_once  # type: ignore[union-attr]

    def settle_probe():  # type: ignore[no-untyped-def]
        seen["结算"] = held_elsewhere(probe)
        return settle()

    run.settlement.run_once = settle_probe  # type: ignore[union-attr,method-assign]
    table = run.foresight.run  # type: ignore[union-attr]

    def table_probe(**kwargs):  # type: ignore[no-untyped-def]
        seen["机会表"] = held_elsewhere(probe)
        return table(**kwargs)

    run.foresight.run = table_probe  # type: ignore[union-attr,method-assign]
    day_run = run.days.run_once  # type: ignore[union-attr]

    async def days_probe():  # type: ignore[no-untyped-def]
        seen["日子"] = await asyncio.to_thread(held_elsewhere, probe)
        return await day_run()

    run.days.run_once = days_probe  # type: ignore[union-attr,method-assign]
    write = run.marks.write

    def mark_probe(mark):  # type: ignore[no-untyped-def]
        seen["完成标记"] = held_elsewhere(probe)
        return write(mark)

    run.marks.write = mark_probe  # type: ignore[method-assign]
    scene = run.scene.run

    async def scene_probe(**kwargs):  # type: ignore[no-untyped-def]
        seen["语义树"] = await asyncio.to_thread(held_elsewhere, probe)
        return await scene(**kwargs)

    run.scene.run = scene_probe  # type: ignore[method-assign]

    async def vocabulary_probe(_night) -> None:  # type: ignore[no-untyped-def]
        seen["词表"] = await asyncio.to_thread(held_elsewhere, probe)

    async def migrate_probe() -> None:
        seen["迁移"] = await asyncio.to_thread(held_elsewhere, probe)

    run.vocabulary_jobs = (vocabulary_probe,)
    run.migration_jobs = (migrate_probe,)
    result = generate(run)
    assert result.complete
    assert seen == {
        "迁移": False,
        "结算": True,
        "词表": False,
        "语义树": False,
        "机会表": True,
        "日子": True,
        "完成标记": True,
    }
    assert held_elsewhere(probe) is False  # 做完放开


def _tick(tmp_path: Path, held: Lease) -> tuple[ForesightTick, object]:
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TICK_DAY, "讨论方案", 10, 0)  # 刚讨论完：这一拍建一条「事件」预期
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    tick = ForesightTick(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        clock=lambda: at(TICK_DAY, 10, 15),
        recent=35,
        tolerance=TOLERANCE,
        tick_seconds=900.0,
        track=TickTrack(TickWatermarkStore(ledger.root), slot=timedelta(minutes=15), tolerance_slots=2),
        lease=held,
    )
    return tick, ledger


def test_a_tick_writes_the_ledger_only_while_holding_the_root_lease(tmp_path) -> None:
    probe = locks(tmp_path)
    tick, ledger = _tick(tmp_path, lease(locks(tmp_path)))
    seen: list[bool] = []
    record = ledger.record_expectation  # type: ignore[attr-defined]

    def record_probe(item):  # type: ignore[no-untyped-def]
        seen.append(held_elsewhere(probe))
        return record(item)

    ledger.record_expectation = record_probe  # type: ignore[attr-defined]
    report = asyncio.run(tick.run_once())
    assert report.recorded == 1 and seen and all(seen)
    assert held_elsewhere(probe) is False


def test_a_tick_while_another_holder_writes_the_ledger_writes_nothing_and_comes_back_next_slot(tmp_path) -> None:
    """另一个进程（换代写账本的那几步、另一个预测拍）拿着预测根租约：这一拍等满一个租期也拿不到，就不建、不写水位，报
    ``LEDGER_BUSY``；对方放开后下一拍照常建。不持租约直接写账本时这里会变红。"""

    tick, ledger = _tick(tmp_path, lease(locks(tmp_path), ttl=2.0, heartbeat=0.5))
    other = lease(locks(tmp_path), ttl=5.0, heartbeat=1.0)
    watermark = TickWatermarkStore(ledger.root)  # type: ignore[attr-defined]

    async def while_held():  # type: ignore[no-untyped-def]
        async with other.held(wait_seconds=0.0):
            return await tick.run_once()

    report = asyncio.run(while_held())
    assert report.skipped == LEDGER_BUSY and report.recorded == 0
    assert ledger.expectations_on(TICK_DAY) == ()  # type: ignore[attr-defined]
    assert watermark.read().ticked_at is None
    again = asyncio.run(tick.run_once())
    assert again.skipped is None and again.recorded == 1
