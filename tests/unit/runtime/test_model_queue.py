"""模型队列（实时化方案第 4 节；阶段 2 第一小步 A1）：原来换代里调模型的活搬到这里，口径不变。

- 队列做到没有欠账、再换代，盘上与"一晚一次做完"（``SceneNightlyRun.run``，A1 之前换代里的那一段）逐字节相同；
- 欠账每一轮从盘上现算：做到一半停下，换一个全新的队列对象接着做，结果相同；
- 已封口的日子的活在还不能做时不算欠着；
- 一项没做成按重试账退避，队列接着做下一项；锁被占是还没准备好；
- 模型调用失败：这一项在进程里等、不放弃，别的照做；连着几项都失败就整体退避（逐次加倍），期间不逐项试、项一个不少；
- 健康面：欠账过多、正在整体退避、有活欠着跨过了换代。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from habitus.behavior.reduction import BehaviorReductionBusyError
from habitus.behavior.retry import RetryReport
from habitus.foundation.lifecycle import BackoffPolicy
from habitus.infrastructure.lifecycle import RetryLedger
from habitus.model_client import ModelQuotaError, ModelTransportError, ModelUnreachable
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_marks import GenerationMark, GenerationMarks
from habitus.runtime.model_queue import (
    CATEGORIES,
    DONE,
    FAILED,
    MAPPING,
    RETRY,
    SYNC,
    UNFINISHED,
    UNREACHABLE,
    VOCABULARY,
    WAITING,
    ModelQueue,
    ModelQueueState,
    QueueJournal,
    QueueSettings,
    VocabularyBranch,
)
from habitus.runtime.model_queue_health import queue_problems
from habitus.series.reader import SeriesReader, read_series
from tests.lock_store import ProcessLocalLockStore
from tests.unit.runtime.test_scene_night import (
    NOW,
    SLEEP,
    TONIGHT,
    FakeCatalog,
    nightly,
    publish_tonight,
    site_with_three_bedtimes,
)
from tests.unit.scene.fixtures import CST

TODAY = NOW.date()


def queue_for(tmp_path, scene, site, *, clock=lambda: NOW, sealed=lambda: TODAY, **options) -> ModelQueue:  # type: ignore[no-untyped-def]
    values = dict(
        scene=scene,
        series=SeriesReader(site.behavior_tree, name="model_queue"),
        today=lambda: clock().astimezone(CST).date(),
        sealed_day=sealed,
        vocabulary=lambda: (1, False),
        retries=RetryLedger(
            tmp_path / "scene",
            owner="model-queue",
            policy=BackoffPolicy(base_seconds=60.0, cap_seconds=3600.0),
            max_attempts=3,
            clock=clock,
        ),
        lease=Lease(ProcessLocalLockStore(), "scene:model-queue:test", ttl_seconds=60.0, heartbeat_seconds=20.0),
        journal=QueueJournal(tmp_path / "scene" / "model_queue", artifact_root=tmp_path / "scene", zone=CST),
        clock=clock,
    )
    values.update(options)
    return ModelQueue(**values)  # type: ignore[arg-type]


def site_and_scene(tmp_path, **options):  # type: ignore[no-untyped-def]
    """三天就寝的历史 + 今晚封口的两条记录。生效时点（阶段 2 C）之后概念只由换代的同步词表写：这里先让"换代"把概念同步好；
    词表另记着一条迁移（今晚那条就寝改过编号），所以队列的同步词表一项还欠着（重判牵到的日子、记下同步到的版本）——
    下面各条拿它当"排在映射前面的一项"用。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    if "catalog" not in options:
        options["catalog"] = FakeCatalog()
    scene, provider = nightly(tmp_path, hits, site, **options)
    bed, _meal = publish_tonight(site)
    options["catalog"].move(bed, "待定", SLEEP)
    asyncio.run(scene.syncing.run(read_series(site.behavior_tree, cutoff=TODAY), NOW, [], remap=False))
    return site, scene, provider


def scene_files(root: Path) -> dict[str, bytes]:
    """语义树盘面（概念、命中、关系表）；不含重试账与模型队列自己的流水。"""

    found = {}
    for path in sorted((root / "scene").rglob("*")):
        relative = path.relative_to(root / "scene").as_posix()
        if path.is_file() and not relative.startswith(("model_queue/", "lifecycle/")) and "retry" not in relative:
            found[relative] = path.read_bytes()
    return found


def settled_files(root: Path) -> dict[str, bytes]:
    """``scene_files``，命中记录的 ``mapped_at`` 抹平：映射按槽之后一条记录什么时候映射不影响内容，只有这个时刻不同。"""

    from dataclasses import replace

    from habitus.scene.occurrences.document import decode, encode

    found = scene_files(root)
    for name, payload in found.items():
        if name.startswith("occurrences/") and name.endswith(".md"):
            found[name] = encode(replace(decode(payload.decode("utf-8")), mapped_at=NOW)).encode("utf-8")
    return found


def outcomes(report) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    return [(item.category, item.outcome) for item in report.done]


def test_a_queue_stopped_halfway_is_picked_up_from_disk_by_a_fresh_queue(tmp_path) -> None:
    """欠账不记"做到哪了"的账：做完第一项就停，换一个全新的队列对象（没有任何内存状态）接着做，盘面与不中断的相同。"""

    site, scene, _provider = site_and_scene(tmp_path / "straight")
    asyncio.run(queue_for(tmp_path / "straight", scene, site).drain())

    site, scene, _provider = site_and_scene(tmp_path / "killed")
    first = queue_for(tmp_path / "killed", scene, site)
    seen: list[int] = []

    def stop() -> bool:
        seen.append(1)
        return len(seen) > 1  # 做完第一项之后停

    stopped = asyncio.run(first.drain(stop=stop))
    assert outcomes(stopped) == [(SYNC, DONE)] and TONIGHT not in scene.hits.days_done()
    resumed = asyncio.run(queue_for(tmp_path / "killed", scene, site).drain())
    assert resumed.clear and (SYNC, DONE) not in outcomes(resumed) and (MAPPING, DONE) in outcomes(resumed)
    assert scene_files(tmp_path / "killed") == scene_files(tmp_path / "straight")


def test_sealed_work_is_not_owed_until_the_day_is_sealed(tmp_path) -> None:
    """已封口的日子的活（同步词表、写概念、候选条件与先验）只在换代的触发条件（除今天的完成标记外）都满足时才算欠着：
    早于这一刻不做。映射按槽之后映射不在其列：记录所在的槽过后留够余量就映射，不等那一天封口——但日完成章不由队列盖。
    （原来这条断言封口之前队列什么都不做；映射那一半按阶段 2 第三小步改了。）"""

    site, scene, _provider = site_and_scene(tmp_path)
    sealed: list[object] = [None]
    queue = queue_for(tmp_path, scene, site, sealed=lambda: sealed[0])
    report = asyncio.run(queue.drain())
    assert {category for category, _outcome in outcomes(report)} == {MAPPING} and report.clear
    assert report.owed == dict.fromkeys(CATEGORIES, 0)
    assert len(scene.hits.read_day(TONIGHT)) == len(read_series(site.behavior_tree, cutoff=TODAY).on(TONIGHT))
    assert TONIGHT not in scene.hits.days_done()  # 映射了，没盖章：章由换代封口时盖
    sealed[0] = TODAY
    assert (SYNC, DONE) in outcomes(asyncio.run(queue.drain()))


def test_a_failed_item_backs_off_on_the_retry_ledger_and_the_queue_moves_on(tmp_path) -> None:
    """一项没做成：按重试账退避，后面的类照做；退避期内不重排，过了再做。不丢项。"""

    clock = [NOW]
    site, scene, _provider = site_and_scene(tmp_path)
    queue = queue_for(tmp_path, scene, site, clock=lambda: clock[0])
    real = scene.syncing.run
    broken = [True]

    async def sync(*args, **kwargs):  # type: ignore[no-untyped-def]
        if broken[0]:
            raise OSError("disk")
        return await real(*args, **kwargs)

    scene.syncing.run = sync  # type: ignore[method-assign]
    report = asyncio.run(queue.drain())
    assert outcomes(report)[0] == (SYNC, FAILED) and (MAPPING, DONE) in outcomes(report)
    assert report.owed[SYNC] == 1 and not report.clear
    broken[0] = False
    assert (SYNC, DONE) not in outcomes(asyncio.run(queue.drain()))  # 还在退避：这一趟不重排
    clock[0] = NOW + timedelta(seconds=61)
    assert outcomes(asyncio.run(queue.drain()))[0] == (SYNC, DONE)


class _Entered:
    item_id = "0" * 32
    entered_at = NOW - timedelta(hours=1)


def _failing_sync(scene, error):  # type: ignore[no-untyped-def]
    """同步词表那一项调模型时报 ``error``（``error`` 换成 None 之后照常做）；返回装着它的那个格子。"""

    real = scene.syncing.run
    holder = [error]

    async def run(*args, **kwargs):  # type: ignore[no-untyped-def]
        if holder[0] is not None:
            raise holder[0]
        return await real(*args, **kwargs)

    scene.syncing.run = run  # type: ignore[method-assign]
    return holder


def _unreachable_from(cause: Exception) -> ModelUnreachable:
    """``model_client.asking.ask`` 抛的那一种：``ModelUnreachable``，起因挂在 ``__cause__`` 上。"""

    error = ModelUnreachable(f"model: {type(cause).__name__}")
    error.__cause__ = cause
    return error


def test_one_item_that_keeps_failing_on_the_model_waits_alone_and_is_never_given_up(tmp_path) -> None:
    """模型调用失败的那一条规则，前一半：单独一项一直失败——只有它在等（进程内的退避，逐次加倍），不记落盘的重试账、失败
    多少次都不放弃，退避期内不重问；别的项与后面的类照做；模型好了就做完。队列不进整体退避。"""

    site, scene, _provider = site_and_scene(tmp_path)
    clock = [NOW]
    key = f"sync:{TODAY.isoformat()}"
    failing = _failing_sync(scene, _unreachable_from(ModelTransportError("read timed out")))
    queue = queue_for(tmp_path, scene, site, clock=lambda: clock[0])
    report = asyncio.run(queue.drain())
    assert outcomes(report)[0] == (SYNC, UNREACHABLE) and not queue.state.down()
    assert (MAPPING, DONE) in outcomes(report)  # 后面的类照做
    assert queue.retries.entry(key) is None and report.owed[SYNC] == 1 and not report.clear  # 不记重试账，仍算欠着
    assert asyncio.run(queue.drain()).done == ()  # 退避期内不重问
    for attempt in range(1, 8):  # 比重试账的次数上限（3）多得多：不放弃，退避逐次加倍
        clock[0] += timedelta(seconds=60 * 2**attempt)
        again = asyncio.run(queue.drain())
        assert outcomes(again) == [(SYNC, UNREACHABLE)] and not queue.state.down(), attempt
        assert queue.retries.entry(key) is None and queue.state.waits[key][0] == attempt + 1
    failing[0] = None
    clock[0] += timedelta(hours=2)
    assert asyncio.run(queue.drain()).clear and key not in queue.state.waits


@pytest.mark.parametrize(
    "error",
    [ModelTransportError("503 service unavailable"), ModelQuotaError("insufficient balance")],
    ids=["transport", "quota"],
)
def test_every_call_failing_backs_the_whole_queue_off_after_n_items_and_nothing_is_given_up(tmp_path, error) -> None:  # type: ignore[no-untyped-def]
    """后一半：所有调用都失败（不分错误种类，这里超时类与额度类各跑一遍）——连着配置的项数（不同的项）都失败、中间没有
    一项做成，整体退避（期间不逐项试）；到点接着做，头一项又失败就再退、时长加倍；有一项做成才清零、算恢复。没有一项被
    放弃，恢复后做完，盘面与没出过事的相同。"""

    recovered_at = NOW + timedelta(seconds=400)
    site, scene, _provider = site_and_scene(tmp_path / "straight")
    asyncio.run(queue_for(tmp_path / "straight", scene, site, clock=lambda: recovered_at).drain())

    site, scene, provider = site_and_scene(tmp_path / "outage")
    good = provider.complete_async

    async def refused(request):  # type: ignore[no-untyped-def]
        raise error

    provider.complete_async = refused  # type: ignore[method-assign]
    failing = _failing_sync(scene, _unreachable_from(error))
    clock = [NOW]
    queue = queue_for(
        tmp_path / "outage",
        scene,
        site,
        clock=lambda: clock[0],
        settings=QueueSettings(backoff_base_seconds=60.0, backoff_max_seconds=600.0, unreachable_after_failures=2),
    )
    report = asyncio.run(queue.drain())
    # 映射按条之后，两项失败之间夹着一条不问模型的记录做成了（连续数清过一次）：这一趟不整体退避，两项各自在进程里等
    assert outcomes(report) == [(SYNC, UNREACHABLE), (MAPPING, DONE), (MAPPING, UNREACHABLE)] and not queue.state.down()
    assert report.owed[SYNC] == 1 and report.owed[MAPPING] == 1
    assert asyncio.run(queue.drain()).done == ()  # 各自的退避期内不重问
    clock[0] = NOW + timedelta(seconds=61)
    report = asyncio.run(queue.drain())  # 到点：同步又失败——连着两项（不同的项）失败、中间没有一项做成：整体退避
    assert outcomes(report) == [(SYNC, UNREACHABLE)] and queue.state.down()
    assert report.backing_off_until == clock[0] + timedelta(seconds=60)
    assert asyncio.run(queue.drain()).done == ()  # 整体退避期内不试
    clock[0] += timedelta(seconds=61)
    report = asyncio.run(queue.drain())  # 到点接着做：头一项又失败——马上再次整体退避、加倍
    assert outcomes(report) == [(SYNC, UNREACHABLE)] and queue.state.failures == 2
    assert report.backing_off_until == clock[0] + timedelta(seconds=120)
    assert [name for name, _detail in queue_problems(queue.state, queue.settings, clock[0])] == ["model_queue_backoff"]
    failing[0] = None
    provider.complete_async = good  # type: ignore[method-assign]
    clock[0] = recovered_at
    report = asyncio.run(queue.drain())
    assert report.clear and not queue.state.down() and queue.state.streak == set() and queue.state.failures == 0
    assert queue.retries.size() == 0  # 没有一项进过落盘的重试账
    assert settled_files(tmp_path / "outage") == settled_files(tmp_path / "straight")


def test_a_retry_item_stuck_on_the_model_waits_alone_and_the_later_work_is_done(tmp_path) -> None:
    """出错重试里一项卡在"模型用不了"上（会话轮没描述成、或归约归类时模型没答，归约自己吞了只在报告里记）：按一次模型
    调用失败算——只卡住的那一项在进程里等（留在堆里、仍算欠着、退避期内不重问），队列不进整体退避，排在后面的类照做。"""

    site, scene, _provider = site_and_scene(tmp_path)
    asked: list[object] = []

    class Retry:
        def due(self, *, through):  # type: ignore[no-untyped-def]
            return (object.__new__(_Entered),)

        async def run_once(self, *, through, skip=()):  # type: ignore[no-untyped-def]
            asked.append(frozenset(skip))
            return RetryReport(taken=1, unavailable=1, stuck_items=(_Entered.item_id,))

    clock = [NOW]
    queue = queue_for(tmp_path, scene, site, retry=Retry(), clock=lambda: clock[0])
    report = asyncio.run(queue.drain())
    assert outcomes(report)[0] == (RETRY, UNREACHABLE) and not queue.state.down()
    assert (SYNC, DONE) in outcomes(report) and (MAPPING, DONE) in outcomes(report)  # 后面的类照做
    assert queue.state.waiting(f"retry:{_Entered.item_id}", clock[0]) and report.owed[RETRY] == 1  # 项还欠着
    assert asyncio.run(queue.drain()).done == () and len(asked) == 1  # 退避期内不重问
    clock[0] = NOW + timedelta(seconds=61)
    assert outcomes(asyncio.run(queue.drain())) == [(RETRY, UNREACHABLE)] and len(asked) == 2  # 退避到点再试
    assert not queue.state.down()


def test_a_queue_halted_by_the_backoff_still_counts_what_the_later_categories_owe(tmp_path) -> None:
    """复验 H2：整体退避、一轮中途返回时，排在后面的各类也从盘上认一遍"还欠几项"（只认、不做）——换代报告的 ``queue_owed``
    与健康面读到的才是全的。"""

    site, scene, provider = site_and_scene(tmp_path)
    _failing_sync(scene, ModelQuotaError("insufficient balance"))
    queue = queue_for(tmp_path, scene, site, settings=QueueSettings(unreachable_after_failures=1))
    report = asyncio.run(queue.drain())
    assert outcomes(report) == [(SYNC, UNREACHABLE)] and queue.state.down() and provider.calls == 0
    records = len(read_series(site.behavior_tree, cutoff=TODAY).on(TONIGHT))
    assert (
        report.owed[SYNC] == 1 and report.owed[MAPPING] == records
    )  # 今晚封口的那一天还没映射（按条数）：认出来了，没做
    assert scene.hits.read_day(TONIGHT) == () and queue.state.owed_now()[MAPPING] == records


def test_what_is_owed_survives_midnight_until_it_is_really_cleared(tmp_path) -> None:
    """修复轮 G4：过了零点、当天还没到能做已封口的活的时候，各类"还欠几项""从什么时候起欠着"不清零；跨过的代数接着数，
    健康面的告警不自己消掉；只有这一类真的清完才归零。"""

    site, scene, _provider = site_and_scene(tmp_path)
    marks = GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path / "scene")
    clock = [NOW]
    sealed: list[object] = [TODAY]
    queue = queue_for(tmp_path, scene, site, clock=lambda: clock[0], sealed=lambda: sealed[0], marks=marks)
    real = scene.syncing.run

    async def broken(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk")

    scene.syncing.run = broken  # type: ignore[method-assign]
    # 生效时点（阶段 2 C）：同步词表这一类在今天这一代完成之后才算欠着——先有今天的完成标记；跨过的是下一次换代
    marks.write(GenerationMark(day=TODAY, finished_at=NOW - timedelta(minutes=5)))
    asyncio.run(queue.drain())
    since = queue.state.since[SYNC]
    clock[0] = NOW + timedelta(days=1)
    sealed[0] = None  # 零点之后、今天还没封口到能做的时候
    report = asyncio.run(queue.drain())
    assert report.owed[SYNC] == 1 and queue.state.since[SYNC] == since and not report.clear
    marks.write(GenerationMark(day=TODAY + timedelta(days=1), finished_at=clock[0] + timedelta(minutes=5)))
    clock[0] += timedelta(minutes=10)
    sealed[0] = TODAY + timedelta(days=1)
    asyncio.run(queue.drain())  # 新的一天、这一天的换代也做完了：同步词表仍然没做成——欠着跨过了一次换代
    assert queue.state.since[SYNC] == since
    assert dict(queue_problems(queue.state, queue.settings, clock[0]))["model_queue_crossed"].endswith("sync×1")
    marks.write(GenerationMark(day=TODAY + timedelta(days=2), finished_at=clock[0] + timedelta(days=1)))
    clock[0] += timedelta(days=1, minutes=10)
    sealed[0] = TODAY + timedelta(days=2)
    asyncio.run(queue.drain())  # 又一天：跨过的代数接着数
    assert dict(queue_problems(queue.state, queue.settings, clock[0]))["model_queue_crossed"].endswith("sync×2")
    scene.syncing.run = real  # type: ignore[method-assign]
    clock[0] += timedelta(hours=1)
    assert asyncio.run(queue.drain()).owed[SYNC] == 0 and SYNC not in queue.state.since
    assert "model_queue_crossed" not in dict(queue_problems(queue.state, queue.settings, clock[0]))


def test_a_category_that_cannot_read_what_it_owes_does_not_stop_the_later_ones_and_is_not_silent(tmp_path) -> None:
    """修复轮 G5：一类认欠账时读盘抛了——这一类这一轮记失败（原因码进状态与健康面），后面的类照做；这一趟做完之后把这个错
    抛出来（不静默）。读得了之后自己恢复。"""

    site, scene, _provider = site_and_scene(tmp_path)
    queue = queue_for(tmp_path, scene, site)
    real = scene.syncing.plan
    broken = [True]

    async def plan(*args, **kwargs):  # type: ignore[no-untyped-def]
        if broken[0]:
            raise OSError("unreadable")
        return await real(*args, **kwargs)

    scene.syncing.plan = plan  # type: ignore[method-assign]
    try:
        asyncio.run(queue.drain())
    except OSError:
        pass
    else:
        raise AssertionError("the read error must not be silent")
    # 排在后面的映射这一趟照做了（映射按槽之后队列只映射、不盖章）
    assert len(scene.hits.read_day(TONIGHT)) == len(read_series(site.behavior_tree, cutoff=TODAY).on(TONIGHT))
    assert queue.state.errors == {SYNC: "OSError"}
    assert dict(queue_problems(queue.state, queue.settings, NOW))["model_queue_scan"].endswith("sync: OSError")
    broken[0] = False
    assert asyncio.run(queue.drain()).clear and queue.state.errors == {}
    assert queue_problems(queue.state, queue.settings, NOW) == []


def test_vocabulary_jobs_run_once_a_night_and_a_held_sweep_lock_is_not_a_failure(tmp_path) -> None:
    """词表：这一晚还欠着的支各做一次（先于语义树的各类）；做完判据还说欠着是没做成、按重试账退避；sweep 锁被另一进程拿着
    是还没准备好，不算失败、不退避。"""

    site, scene, _provider = site_and_scene(tmp_path)
    due = {"session": True, "physical": True, "stuck": True}
    ran: list[tuple[str, object]] = []

    def branch(name: str, *, finishes: bool = True, busy: bool = False) -> VocabularyBranch:
        async def run(night):  # type: ignore[no-untyped-def]
            if busy:
                raise BehaviorReductionBusyError("sweep lock held")
            ran.append((name, night))
            due[name] = not finishes

        return VocabularyBranch(name, lambda _night: due[name], run)

    queue = queue_for(
        tmp_path,
        scene,
        site,
        vocabulary_branches=(branch("session"), branch("stuck", finishes=False), branch("physical")),
    )
    report = asyncio.run(queue.drain())
    assert outcomes(report)[:3] == [(VOCABULARY, DONE), (VOCABULARY, UNFINISHED), (VOCABULARY, DONE)]
    assert ran == [("session", TODAY), ("stuck", TODAY), ("physical", TODAY)]
    # 生效时点（阶段 2 C）：词表有一支没做成，排在它后面的写概念、补问、同步词表先不做（补做的要与准时做的同一份材料）；
    # 不依赖它们的映射照做。原来这里断言词表之后接着就是同步词表。
    assert {category for category, _outcome in outcomes(report)[3:]} == {MAPPING}
    assert report.owed[VOCABULARY] == 1 and queue.retries.blocked(f"vocabulary:stuck:{TODAY.isoformat()}")

    site, scene, _provider = site_and_scene(tmp_path / "busy")
    due["session"] = True
    queue = queue_for(tmp_path / "busy", scene, site, vocabulary_branches=(branch("session", busy=True),))
    report = asyncio.run(queue.drain())
    assert outcomes(report)[0] == (VOCABULARY, WAITING)
    assert not queue.retries.blocked(f"vocabulary:session:{TODAY.isoformat()}")


def test_two_queues_on_one_root_do_not_work_at_the_same_time(tmp_path) -> None:
    """队列租约：另一个进程正拿着时这一趟什么都不做（``held_elsewhere``），不算没有欠账。"""

    site, scene, provider = site_and_scene(tmp_path)
    queue = queue_for(tmp_path, scene, site)

    async def main():  # type: ignore[no-untyped-def]
        async with queue.lease.held(wait_seconds=0.0):
            return await queue.drain()

    report = asyncio.run(main())
    assert report.held_elsewhere and not report.clear and report.done == () and provider.calls == 0


def test_health_reports_a_large_backlog_and_work_owed_across_generations_until_it_clears(tmp_path) -> None:
    """健康面：欠账到了阈值（项数、或最老一项等了多久）报出各类条数；有活欠着的时候换代写成了，报哪几类、跨了几代；清完自己消掉。"""

    state = ModelQueueState()
    settings = QueueSettings(warning_items=3, warning_wait_seconds=3_600.0)
    assert queue_problems(state, settings, NOW) == []
    state.record(MAPPING, 2, NOW)
    assert queue_problems(state, settings, NOW + timedelta(minutes=30)) == []  # 没到阈值
    state.record(SYNC, 1, NOW)
    found = dict(queue_problems(state, settings, NOW))
    assert "owed 3: sync=1, mapping=2" in found["model_queue_backlog"]
    state.record(SYNC, 0, NOW)
    assert (
        "oldest waited 3600s" in dict(queue_problems(state, settings, NOW + timedelta(hours=1)))["model_queue_backlog"]
    )

    site, scene, _provider = site_and_scene(tmp_path)
    marks = GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path / "scene")
    clock = [NOW]
    queue = queue_for(tmp_path, scene, site, clock=lambda: clock[0], marks=marks)

    async def broken(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk")

    real = scene.syncing.run
    scene.syncing.run = broken  # type: ignore[method-assign]
    # 生效时点（阶段 2 C）：这一类在今天这一代完成之后才算欠着；"欠着的时候换代写成了"说的是下一次换代
    marks.write(GenerationMark(day=TODAY, finished_at=NOW - timedelta(minutes=5)))
    asyncio.run(queue.drain())
    marks.write(GenerationMark(day=TODAY + timedelta(days=1), finished_at=NOW + timedelta(minutes=5)))
    clock[0] = NOW + timedelta(minutes=10)
    asyncio.run(queue.drain())
    assert dict(queue_problems(queue.state, queue.settings, clock[0]))["model_queue_crossed"].endswith("sync×1")
    scene.syncing.run = real  # type: ignore[method-assign]
    clock[0] = NOW + timedelta(hours=1)
    assert asyncio.run(queue.drain()).clear
    assert queue_problems(queue.state, queue.settings, clock[0]) == []
