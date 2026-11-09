"""夜线 Job 化的端到端（验证轮 #5 的对抗测试收进仓库）：``finished`` 三值经真实 ``look()`` 跑出来的账、过了那天没跑完记 FAILED
且不在新的一天补跑、账坏了明确报错（最新一晚的账坏了挡住后面的晚，人工删掉坏账文件后下一眼重新入队，裁定 94 第 9 条
2026-11-09 补记）。"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from habitus.runtime.nightly import PREPARE, STEPS, STRUCTURE
from habitus.runtime.nightly_jobs import FAILED, NIGHT_PASSED, SUCCEEDED, NightJobsError
from tests.unit.runtime.test_nightly import worker_for
from tests.unit.runtime.test_nightly_line import ALL, wired
from tests.unit.runtime.test_scene_night import NOW


def _day(clock, days):
    clock[0] = NOW + timedelta(days=days)


def test_finished_true_false_none_through_real_looks(tmp_path) -> None:
    order: list[str] = []
    clock = [NOW]
    migrating = [False]
    run = wired(tmp_path, order, migrating=lambda: migrating[0], clock=lambda: clock[0])
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    n0 = run.tonight()
    # 第 0 晚：结构卡在迁移（没跑完、还在当天）——此时还没有任何跑完的一晚：None（不是 False）
    migrating[0] = True
    asyncio.run(worker.look())
    assert worker.finished(n0) is None
    # 第 1 晚：过了那天，第 0 晚记 FAILED；第 1 晚跑完 → True；第 0 晚在第一次跑完之前 → None
    migrating[0] = False
    _day(clock, 1)
    run.closed_days = lambda: (n0,)
    ledger = asyncio.run(worker.look())
    n1 = n0 + timedelta(days=1)
    assert ledger is not None and ledger.night == n1 and ledger.complete
    assert worker.finished(n1) is True and worker.finished(n0) is None
    # 第 2 晚：模型调不通，前置处理停下 → False（已有更早跑完的一晚）
    from habitus.model_client import ModelUnreachable

    async def down() -> None:
        raise ModelUnreachable("x")

    run.probe = down
    _day(clock, 2)
    run.closed_days = lambda: (n0, n1)
    asyncio.run(worker.look())
    n2 = n0 + timedelta(days=2)
    assert worker.finished(n2) is False

    # 第 3 晚没开机（没有账），第 4 晚跑完：第 3 晚 False、第 4 晚 True、第 2 晚过了那天被记 FAILED 仍 False
    async def up() -> None:
        order.append("探模型")

    run.probe = up
    _day(clock, 4)
    n3, n4 = n0 + timedelta(days=3), n0 + timedelta(days=4)
    run.closed_days = lambda: (n0, n1, n2, n3)
    ledger = asyncio.run(worker.look())
    assert ledger is not None and ledger.night == n4 and ledger.complete
    old = worker.jobs.read(n2)
    assert old is not None and old.job(PREPARE).state == FAILED and old.job(PREPARE).reason.startswith(NIGHT_PASSED)
    assert [worker.finished(n) for n in (n0, n1, n2, n3, n4, n4 + timedelta(days=1))] == [
        None,
        True,
        False,
        False,
        True,
        False,
    ]
    # 保留期外：只留 1 晚时清掉更早的账 → 留存里第一次跑完的是 n4，之前的都 None
    worker.jobs.kept_nights = 1
    worker.jobs.prune(n4)
    assert [worker.finished(n) for n in (n1, n2, n3, n4)] == [None, None, None, True]


def test_expiry_side_effects_report_and_no_rerun_of_old_steps(tmp_path) -> None:
    """过了那天没跑完：旧晚没成的段 FAILED（含 STOPPED 等退避的），旧晚的段不在新的一天上再跑；报告同步重写；新晚从头跑。"""

    import json

    order: list[str] = []
    clock = [NOW]
    fails = [True]
    run = wired(tmp_path, order, settle_fails=lambda: fails[0], clock=lambda: clock[0])
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    n0 = run.tonight()
    asyncio.run(worker.look())
    assert worker.jobs.read(n0).job("settle").state == "stopped"
    fails[0] = False
    _day(clock, 1)
    run.closed_days = lambda: (n0,)
    order.clear()
    ledger = asyncio.run(worker.look())
    assert order == ALL  # 只跑新的一晚（旧晚的结算没在新的一天补跑）
    old = worker.jobs.read(n0)
    assert [old.job(s).state for s in STEPS] == [SUCCEEDED, FAILED, FAILED, FAILED, FAILED]
    assert old.job("settle").reason == f"{NIGHT_PASSED}（结算出错: OSError）"
    assert old.job("vocabulary").reason == NIGHT_PASSED
    report = json.loads((tmp_path / "nightly" / "reports" / f"{n0.isoformat()}.json").read_text())
    assert report["steps"]["settle"]["state"] == FAILED
    assert ledger is not None and ledger.complete


def test_expiry_happens_even_before_the_next_night_is_ready(tmp_path) -> None:
    """新的一天零点刚过、昨天（=旧晚）还没定稿时，旧晚就被记 FAILED（不等它；与旧版"那一晚放弃"同义，现在写进了账）。"""

    order: list[str] = []
    clock = [NOW]
    migrating = [True]
    run = wired(tmp_path, order, migrating=lambda: migrating[0], clock=lambda: clock[0])
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    n0 = run.tonight()
    asyncio.run(worker.look())
    _day(clock, 1)
    run.closed_days = lambda: ()  # 昨天还没定稿：新的一晚还不能跑
    got = asyncio.run(worker.look())
    assert got is not None and got.night == n0 and got.job(STRUCTURE).state == FAILED


def test_a_corrupt_ledger_of_the_latest_night_blocks_every_later_night(tmp_path) -> None:
    """最新那一晚的账坏了 → 之后每一晚的 look 都在读 latest 时抛 NightJobsError，新的一晚不入队，健康面报读不了；人工删掉坏账
    文件之后下一眼重新入队（裁定 94 第 9 条 2026-11-09 补记）。"""

    order: list[str] = []
    clock = [NOW]
    run = wired(tmp_path, order, clock=lambda: clock[0])
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    n0 = run.tonight()
    asyncio.run(worker.look())
    worker.jobs.path(n0).write_text("{broken")
    for days in (1, 2, 3):
        _day(clock, days)
        run.closed_days = lambda d=days: tuple(n0 + timedelta(days=i) for i in range(d))
        order.clear()
        with pytest.raises(NightJobsError):
            asyncio.run(worker.look())
        assert order == []
    assert worker.jobs.nights() == (n0,)
    from habitus.runtime.health import _night_line_checks

    (check,) = _night_line_checks(worker.jobs)
    assert check.detail == "unreadable:NightJobsError"
    with pytest.raises(NightJobsError):
        worker.finished(n0 + timedelta(days=3))  # 唯一的账读不了：不吞成"认不出"，明确报错（codex R2）
    # 人工处置：删掉坏账文件，下一眼重新入队、照常跑
    worker.jobs.path(n0).unlink()
    order.clear()
    ledger = asyncio.run(worker.look())
    assert ledger is not None and ledger.night == n0 + timedelta(days=3) and ledger.complete and order == ALL


def test_a_corrupt_older_ledger_does_not_block_later_nights_but_asking_about_it_raises(tmp_path) -> None:
    order: list[str] = []
    clock = [NOW]
    run = wired(tmp_path, order, clock=lambda: clock[0])
    worker = worker_for(run, tmp_path, clock=lambda: clock[0])
    n0 = run.tonight()
    for days in (0, 1, 2):
        _day(clock, days)
        run.closed_days = lambda d=days: tuple(n0 + timedelta(days=i - 1) for i in range(d + 1))
        asyncio.run(worker.look())
    n1 = n0 + timedelta(days=1)
    worker.jobs.path(n1).write_text("{broken")
    _day(clock, 3)
    run.closed_days = lambda: tuple(n0 + timedelta(days=i) for i in range(3))
    ledger = asyncio.run(worker.look())
    assert ledger is not None and ledger.complete
    assert worker.finished(n0) is True and worker.finished(n0 + timedelta(days=3)) is True
    with pytest.raises(NightJobsError):
        worker.finished(n1)  # 问到坏的那一晚：明确报错，不当它停了（codex R2）
