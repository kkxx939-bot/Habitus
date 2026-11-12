"""换代并进预测 worker、三把锁（实时化阶段 2 第二小步；``立项-A2.md`` 第 4 节里单测守得住的那几行）。

- A2-4：换代那一槽的预测拍在换代做完之后、同一槽里做，从当天零点起补建，``lateness_seconds`` 是换代用时；换代占用的那一段
  不写成「未观测」；
- A2-6：换代的语义树一步持着语义树写锁时，模型队列的落盘等它做完；锁内没有模型调用；
- A2-7：换代在做时队列不取下一项，做完后继续；
- A2-8：队列一项的模型调用卡住时，语义树写锁是放开的（换代不等这次调用）；
- A2-9：换代必须成的那一步失败、或那一眼抛了：这一拍照做、不建预期，按退避在之后的槽再试，健康面报出；
- 只开语义树、不开预测层：同一个 worker 每槽持预测根租约只看换代。

A2-5、A2-10（两个进程、中途被杀）在 ``tests/integration/test_generation_multiprocess.py``；A2-11 在配置测试；A2-12 在
``tests/integration/test_runtime_assembly.py``。预测根租约的持有范围在 ``test_generation_lease.py``。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.conditions import LocalCalendar
from habitus.foresight import NoUnsealed
from habitus.model_client import ChatClient
from habitus.model_client.structured import StructuredChatClient
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_tick import LEDGER_BUSY, ForesightTick
from habitus.runtime.foresight_worker import NO_TICK, ForesightWorker
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_trace import GENERATION_SPAN, TraceFiles
from habitus.runtime.generation_watermark import TickTrack, TickWatermark, TickWatermarkStore
from habitus.runtime.health import _generation_state_checks
from habitus.runtime.model_queue import ModelQueueState
from habitus.runtime.scene_lock import SceneWriteLock, unlocked
from habitus.scene.occurrences.mapper import ConceptMapper, MapperConfig
from tests.lock_store import ProcessLocalLockStore
from tests.unit.foresight.scripted_model import model_config
from tests.unit.runtime.test_foresight_tick import TODAY as TICK_DAY
from tests.unit.runtime.test_foresight_tick import TOLERANCE, site_with_three_mornings, stores
from tests.unit.runtime.test_generation import wired
from tests.unit.runtime.test_generation_runner import worker_for
from tests.unit.runtime.test_model_queue import queue_for, scene_files, site_and_scene
from tests.unit.runtime.test_scene_night import NOW
from tests.unit.scene.fixtures import SUBJECT, at, publish

SHANGHAI = ZoneInfo("Asia/Shanghai")


def lease(store=None, key: str = "foresight:root:test", ttl: float = 5.0) -> Lease:  # type: ignore[no-untyped-def]
    return Lease(store or ProcessLocalLockStore(), key, ttl_seconds=ttl, heartbeat_seconds=ttl / 4, poll_seconds=0.02)


def scene_lock() -> SceneWriteLock:
    return SceneWriteLock(lease(key="scene:generation:test"))


def generation(tmp_path, order: list[str], **options):  # type: ignore[no-untyped-def]
    (tmp_path / "gen").mkdir(exist_ok=True)
    return wired(tmp_path / "gen", order, **options)


def tick_for(tmp_path, clock: list[datetime], **options):  # type: ignore[no-untyped-def]
    """一拍：前三天每天 10:00 讨论方案之后修改代码，今天 10:00 刚讨论完——10:15 那一拍建一条「事件」预期。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TICK_DAY, "讨论方案", 10, 0)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), SHANGHAI)
    values = dict(
        clock=lambda: clock[0],
        recent=35,
        tolerance=TOLERANCE,
        tick_seconds=900.0,
        track=TickTrack(TickWatermarkStore(ledger.root), slot=timedelta(minutes=15), tolerance_slots=2),
        lease=lease(),
    )
    values.update(options)
    return ForesightTick(ledger, opportunities, read_side, NoUnsealed(), **values), ledger  # type: ignore[arg-type]


def slot_worker(tick, runner, clock: list[datetime], **options) -> ForesightWorker:  # type: ignore[no-untyped-def]
    return ForesightWorker(
        tick,
        generation=runner,
        interval_seconds=900.0,
        shutdown_timeout_seconds=1.0,
        zone=SHANGHAI,
        clock=lambda: clock[0],
        **options,
    )


def taking(runner, clock: list[datetime], spent: timedelta) -> None:  # type: ignore[no-untyped-def]
    """让换代的那一眼在模拟时钟上用掉 ``spent``（真实的换代照做）。"""

    look = runner.slot

    async def slow() -> None:
        await look()
        clock[0] += spent

    runner.slot = slow


# ── A2-4：换代那一槽的预测拍 ────────────────────────────────────────────


def test_a2_4_the_generations_slot_ticks_after_it_from_midnight_and_is_late_by_the_generation(tmp_path) -> None:
    """10:10 的槽边界到了要换代：先做这一代（用了 5 分钟），同一槽里接着做预测拍——从当天零点起补建，span 上的
    ``lateness_seconds`` 就是换代用时；trace 上有这一代的 ``generation.run``。换代放回独立的循环、或换代那一槽跳过预测拍，
    这里都会变红。"""

    order: list[str] = []
    run = generation(tmp_path, order)
    traces = TraceFiles(tmp_path / "traces", artifact_root=tmp_path)
    runner = worker_for(run, tmp_path)
    runner.traces = traces
    clock = [at(TICK_DAY, 10, 10)]
    tick, _ledger = tick_for(tmp_path, clock, trace=traces.sink)
    ticking = tick._run

    async def ticked(now, entered):  # type: ignore[no-untyped-def]
        order.append("预测拍")
        return await ticking(now, entered)

    tick._run = ticked  # type: ignore[method-assign]
    taking(runner, clock, timedelta(minutes=5))
    report = asyncio.run(slot_worker(tick, runner, clock).slot(at(TICK_DAY, 10, 10)))

    assert runner.last_result is not None and runner.last_result.complete
    assert order[-2:] == ["日子", "预测拍"]  # 换代的最后一步之后才是这一拍
    assert report.skipped is None and report.recorded == 1 and report.catch_up  # 同一槽做了一拍，从零点起补建
    ticks = [item for item in traces.read(TICK_DAY) if item["name"] == "foresight.tick"]
    assert len(ticks) == 1 and ticks[0]["attributes"]["lateness_seconds"] == 300.0  # type: ignore[index]
    assert [item for item in traces.read(run.today()) if item["name"] == GENERATION_SPAN]


def test_the_time_a_generation_takes_is_not_written_as_unwatched(tmp_path) -> None:
    """上一拍 09:55，10:10 进这一槽、换代用了 40 分钟（超过两槽的容差）：这一段系统在看（在换代），不写「未观测」。
    真的没在看的那一段（上一拍 08:00）照写，但只写到进这一槽的那一刻，不把换代用时算进去。空当量到换代做完的那一刻时
    这里会变红。"""

    for previous, expected in ((at(TICK_DAY, 9, 55), None), (at(TICK_DAY, 8, 0), at(TICK_DAY, 10, 10))):
        root = tmp_path / previous.strftime("%H%M")
        root.mkdir()
        run = generation(root, [])
        runner = worker_for(run, root)
        clock = [at(TICK_DAY, 10, 10)]
        written: list[tuple[datetime, datetime]] = []
        tick, ledger = tick_for(root, clock)
        assert tick.track is not None
        tick.track.unwatched = lambda start, end, _written=written: _written.append((start, end))
        tick.track.store.write(TickWatermark(ticked_at=previous))
        taking(runner, clock, timedelta(minutes=40))
        report = asyncio.run(slot_worker(tick, runner, clock).slot(at(TICK_DAY, 10, 10)))

        assert runner.last_result is not None and runner.last_result.complete
        mark = TickWatermarkStore(ledger.root).read()
        assert mark.ticked_at == at(TICK_DAY, 10, 50)  # 这一拍在换代做完之后
        if expected is None:
            assert report.unwatched is None and written == [] and mark.unwatched_until is None
        else:
            assert report.unwatched == (previous, expected) and written == [(previous, expected)]
            assert mark.unwatched_until == expected


# ── A2-9：换代没成不挡这一拍 ────────────────────────────────────────────


def test_a2_9_a_failed_must_step_does_not_stop_the_tick_and_is_retried_in_a_later_slot(tmp_path) -> None:
    """机会表那一步（必须成）失败：今天没有这一代，这一拍照做、不建预期（白天那道闸读完成标记）；退避期内的槽不重做，
    退避到点的那一槽再试、做成，那一槽的预测拍就建了；期间健康面报出没写成与下一次重试的时刻。"""

    order: list[str] = []
    run = generation(tmp_path, order)
    table = run.foresight.run  # type: ignore[union-attr]
    broken = [True]

    def flaky(**kwargs):  # type: ignore[no-untyped-def]
        if broken[0]:
            raise OSError("disk")
        return table(**kwargs)

    run.foresight.run = flaky  # type: ignore[union-attr,method-assign]
    runner = worker_for(run, tmp_path)
    wall = [NOW]
    runner._clock = lambda: wall[0]
    clock = [at(TICK_DAY, 10, 15)]
    tick, _ledger = tick_for(tmp_path, clock, finished=lambda _day: run.marks.finished(run.today()))
    worker = slot_worker(tick, runner, clock)

    first = asyncio.run(worker.slot())
    assert runner.last_result is not None and runner.last_result.failed == "opportunities"
    assert first.skipped is not None and first.recorded == 0 and first.unwatched is None  # 这一拍做了，只是不建
    checks = {item.name: item for item in _generation_state_checks(runner)}
    assert checks["generation"].status.value == "degraded" and "next retry at" in checks["generation"].detail

    broken[0] = False
    asyncio.run(worker.slot())  # 还在退避里：不重做
    assert order.count("语义树") == 1 and run.marks.finished(run.today()) is not True
    wall[0] = NOW + timedelta(hours=2)
    again = asyncio.run(worker.slot())  # 退避到点的那一槽
    assert order.count("语义树") == 2 and run.marks.finished(run.today()) is True
    assert again.skipped is None and again.recorded == 1
    assert "generation" not in {item.name for item in _generation_state_checks(runner)}


def test_a_look_that_raises_does_not_stop_the_tick_and_is_reported_until_a_look_succeeds(tmp_path) -> None:
    """看要不要换代的那一眼自己抛了（读盘出错）：这一槽的预测拍照做；健康面报 ``generation_look``，下一槽那一眼顺利就消掉。"""

    run = generation(tmp_path, [])
    ready = run.ready
    broken = [True]

    def flaky():  # type: ignore[no-untyped-def]
        if broken[0]:
            raise OSError("disk")
        return ready()

    run.ready = flaky  # type: ignore[method-assign]
    runner = worker_for(run, tmp_path)
    clock = [at(TICK_DAY, 10, 15)]
    tick, _ledger = tick_for(tmp_path, clock)
    worker = slot_worker(tick, runner, clock)

    report = asyncio.run(worker.slot())
    assert report.skipped is None and report.recorded == 1
    assert isinstance(runner.look_error, OSError)
    checks = {item.name: item for item in _generation_state_checks(runner)}
    assert checks["generation_look"].detail == "last_error:OSError" and not checks["generation_look"].critical

    broken[0] = False
    asyncio.run(worker.slot())
    assert runner.look_error is None and runner.last_error is None
    assert runner.last_result is not None and runner.last_result.complete
    assert "generation_look" not in {item.name for item in _generation_state_checks(runner)}


# ── A2-7：换代在做时队列不取下一项 ──────────────────────────────────────


def test_the_generation_raises_the_flag_while_it_runs_and_lowers_it_even_when_it_fails(tmp_path) -> None:
    order: list[str] = []
    run = generation(tmp_path, order)
    state = ModelQueueState()
    run.generating = state.set_generating
    seen: list[bool] = []
    table = run.foresight.run  # type: ignore[union-attr]
    fail = [False]

    def probe(**kwargs):  # type: ignore[no-untyped-def]
        seen.append(state.generating)
        if fail[0]:
            raise OSError("disk")
        return table(**kwargs)

    run.foresight.run = probe  # type: ignore[union-attr,method-assign]
    fail[0] = True
    assert asyncio.run(run.run(run.today())).failed == "opportunities" and state.generating is False
    fail[0] = False
    assert asyncio.run(run.run(run.today())).complete and state.generating is False
    assert seen == [True, True]


def test_a2_7_the_queue_takes_no_item_while_a_generation_runs_and_goes_on_afterwards(tmp_path) -> None:
    """同一进程的换代在做（标志立着）：队列认了欠账也不取第一项，一次模型都不调；标志撤掉后接着做完。队列不看标志时
    这里会变红。"""

    site, scene, provider = site_and_scene(tmp_path)
    queue = queue_for(tmp_path, scene, site)
    queue.state.set_generating(True)
    before = scene_files(tmp_path)

    async def main():  # type: ignore[no-untyped-def]
        draining = asyncio.create_task(queue.drain())
        await asyncio.sleep(0.6)
        assert not draining.done() and provider.calls == 0 and scene_files(tmp_path) == before
        queue.state.set_generating(False)
        return await draining

    report = asyncio.run(main())
    assert report.clear and report.done and provider.calls > 0


def test_a_queue_told_to_stop_while_waiting_for_the_generation_returns(tmp_path) -> None:
    site, scene, provider = site_and_scene(tmp_path)
    queue = queue_for(tmp_path, scene, site)
    queue.state.set_generating(True)
    stop = [False]

    async def main():  # type: ignore[no-untyped-def]
        draining = asyncio.create_task(queue.drain(stop=lambda: stop[0]))
        await asyncio.sleep(0.3)
        stop[0] = True
        return await asyncio.wait_for(draining, timeout=5.0)

    report = asyncio.run(main())
    assert report.done == () and provider.calls == 0 and not report.clear


# ── A2-6、A2-8：语义树写锁 ──────────────────────────────────────────────


def locked_queue(tmp_path, *, gate: asyncio.Event | None = None):  # type: ignore[no-untyped-def]
    """带语义树写锁的队列；映射器走 ``unlocked`` 包过的客户端。``asked`` 记每次模型调用那一刻队列是不是持着锁，
    ``wrote`` 记每次写命中 / 盖完成章那一刻是不是持着锁。"""

    site, scene, provider = site_and_scene(tmp_path)
    lock = scene_lock()
    asked: list[bool] = []
    wrote: list[bool] = []

    class Probed(StructuredChatClient):
        async def complete_json_async(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            asked.append(lock.writing_now)
            if gate is not None:
                await gate.wait()
            return await super().complete_json_async(*args, **kwargs)

    client = unlocked(Probed(ChatClient(model_config(), provider), validation_retries=1), lock)
    scene.mapper_for = lambda concepts: ConceptMapper(
        client, concepts, config=MapperConfig(transient_retry_delay_seconds=0.0), subject=SUBJECT, clock=lambda: NOW
    )
    for name in ("write", "complete_day"):
        inner = getattr(scene.hits, name)

        def probed(*args, _inner=inner, **kwargs):  # type: ignore[no-untyped-def]
            wrote.append(lock.writing_now)
            return _inner(*args, **kwargs)

        setattr(scene.hits, name, probed)
    return queue_for(tmp_path, scene, site, scene_lock=lock), lock, provider, asked, wrote


def test_a2_6_the_queue_writes_only_after_the_generations_scene_step_and_never_asks_the_model_inside_the_lock(
    tmp_path,
) -> None:
    """换代的语义树一步持着语义树写锁：队列要落盘的那一项等着，盘上一个字节都不动、一次模型都不调；这一步做完后队列接着做，
    每一次落盘都在锁内、每一次模型调用都在锁外。队列的落盘不拿写锁、或调模型时不放锁，这里都会变红。"""

    queue, lock, provider, asked, wrote = locked_queue(tmp_path)
    before = scene_files(tmp_path)

    async def main():  # type: ignore[no-untyped-def]
        async with lock.held():  # 换代的语义树一步
            draining = asyncio.create_task(queue.drain())
            await asyncio.sleep(0.6)
            assert not draining.done() and provider.calls == 0 and scene_files(tmp_path) == before
        return await draining

    report = asyncio.run(main())
    assert report.clear and scene_files(tmp_path) != before
    assert asked and not any(asked)  # 调模型的那一刻都没持锁
    assert wrote and all(wrote)  # 落盘的那一刻都持着锁
    assert lock.writing_now is False


def test_a2_8_a_model_call_that_hangs_leaves_the_scene_lock_free_for_the_generation(tmp_path) -> None:
    """队列一项的模型调用卡住：语义树写锁是放开的，换代的语义树一步马上拿得到（不等这次调用）；这一步做完、模型答复回来，
    队列拿回锁接着写完。"""

    # 现场在事件循环外搭好：生效时点（阶段 2 C）之后现场里先由"换代"同步概念（``site_and_scene`` 里自己 ``asyncio.run``）
    gate = asyncio.Event()
    queue, lock, _provider, asked, wrote = locked_queue(tmp_path, gate=gate)

    async def main():  # type: ignore[no-untyped-def]
        draining = asyncio.create_task(queue.drain())
        while not asked:
            await asyncio.sleep(0.01)
        async with asyncio.timeout(2.0):
            async with lock.held():  # 换代的语义树一步：不等卡住的那次调用
                written = len(wrote)
                gate.set()  # 这一步还没做完，模型答复回来了：队列得等这一步做完才落盘
                await asyncio.sleep(0.3)
                assert len(wrote) == written
        report = await draining
        return report, wrote, written, lock

    report, wrote, written, lock = asyncio.run(main())
    assert report.clear and len(wrote) > written and all(wrote) and lock.writing_now is False


def test_a_model_call_outside_a_queue_item_passes_straight_through(tmp_path) -> None:
    """不在队列的一项里的调用方（离线一次做完的那条路）经同一个客户端调模型：锁没人持，原样过、不去拿锁。"""

    lock = scene_lock()

    async def main() -> bool:
        async with lock.asking():
            async with asyncio.timeout(1.0):
                async with lock.held():
                    return lock.writing_now

    assert asyncio.run(main()) is False


# ── 只开语义树、不开预测层 ──────────────────────────────────────────────


def test_without_the_foresight_layer_the_same_worker_only_looks_at_the_generation_under_the_root_lease(
    tmp_path,
) -> None:
    order: list[str] = []
    run = generation(tmp_path, order)
    runner = worker_for(run, tmp_path)
    store = ProcessLocalLockStore()
    clock = [at(TICK_DAY, 10, 15)]
    worker = slot_worker(None, runner, clock, lease=lease(store, ttl=1.0))
    held: list[bool] = []
    scene = run.scene.generate

    async def probe(**kwargs):  # type: ignore[no-untyped-def]
        try:
            store.release(store.acquire("foresight:root:test", 1))
            held.append(False)
        except TimeoutError:
            held.append(True)
        return await scene(**kwargs)

    run.scene.generate = probe  # type: ignore[method-assign]

    async def while_another_process_holds_it():  # type: ignore[no-untyped-def]
        async with lease(store).held():
            return await worker.slot()

    busy = asyncio.run(while_another_process_holds_it())
    assert busy.skipped == LEDGER_BUSY and order == []  # 拿不到预测根租约：这一槽不换代
    report = asyncio.run(worker.slot())
    assert report.skipped == NO_TICK and runner.last_result is not None and runner.last_result.complete
    assert held == [True]  # 整代在预测根租约之内
