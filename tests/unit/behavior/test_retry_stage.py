"""每晚跑批的第一步：按天重试白天读不懂的（裁定 67、68）——补回、按天收、每项试完就删、探活只停会话轮、上一晚中途被杀
剩下的不再试只撤回、重启时就清。"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.fusion.lanes.session import TurnHandler, split_turns
from habitus.behavior.fusion.lanes.session.model import encode_turn
from habitus.behavior.retry import (
    Admission,
    Attempt,
    Reduction,
    RetryBranch,
    RetryCause,
    RetryItem,
    RetryKind,
    RetryReport,
    RetryStage,
    retry_id,
)
from habitus.model_client import ModelQuotaError
from tests.unit.behavior.session_fixtures import Stores, description, lane, prompt, reply

DAY = date(2026, 10, 20)
AT = datetime(2026, 10, 21, 3, 0, tzinfo=UTC)


def _turn(text: str, start: int):  # type: ignore[no-untyped-def]
    (turn,) = split_turns(f"c-{text}", (prompt(start, text), reply(start + 1, "好的", minutes=start + 2)))
    return turn


def _turn_item(turn, day: date = DAY) -> RetryItem:  # type: ignore[no-untyped-def]
    return RetryItem(
        retry_id(RetryKind.TURN, turn.key), RetryKind.TURN, RetryCause.GIVEN_UP, day, AT, encode_turn(turn)
    )


def _chain_item(name: str, day: date = DAY, cause: RetryCause = RetryCause.UNAVAILABLE) -> RetryItem:
    """一条链：默认是归类时模型调不通放一边的（夜里放回、再归约一次，裁定 94 第 3 条）。"""

    return RetryItem(
        retry_id(RetryKind.CHAIN, name),
        RetryKind.CHAIN,
        cause,
        day,
        AT,
        {"records": [{"name": name}]},
    )


class _Chains:
    """链处理器的替身：放回、按名字答成没成、记撤回了谁。"""

    def __init__(self, good: frozenset[str] = frozenset()) -> None:
        self.good = good
        self.abandoned: list[str] = []
        self.attempted: list[str] = []

    async def attempt(self, item: RetryItem) -> Attempt:
        self.attempted.append(item.payload["records"][0]["name"])
        return Attempt.RESTORED

    def recovered(self, item: RetryItem) -> bool:
        return item.payload["records"][0]["name"] in self.good

    def abandon(self, item: RetryItem) -> None:
        self.abandoned.append(item.payload["records"][0]["name"])


class _Reduce:
    """一支的归约替身：按 ``outcomes`` 依次答（用完了就是做成了）。"""

    def __init__(self, *outcomes: Reduction) -> None:
        self.runs = 0
        self.outcomes = list(outcomes)

    async def __call__(self) -> Reduction:
        self.runs += 1
        return self.outcomes.pop(0) if self.outcomes else Reduction.DONE


def _stage(stores: Stores, session, chains: _Chains | None = None, reduce: _Reduce | None = None):  # type: ignore[no-untyped-def]
    handlers: dict[RetryKind, Any] = {RetryKind.CHAIN: chains or _Chains()}
    if session is not None:
        handlers[RetryKind.TURN] = TurnHandler(session)
    return RetryStage([RetryBranch("session", stores.backlog, handlers, reduce or _Reduce())])


def test_a_turn_is_recovered_and_its_day_is_closed_to_new_items(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    turn = _turn("统计一下行数", 0)
    stores.backlog.put(_turn_item(turn))
    reduce = _Reduce()

    report = asyncio.run(_stage(stores, session, reduce=reduce).run_once(through=DAY))

    assert report == RetryReport(taken=1, recovered=1)
    assert provider.calls == 1 and reduce.runs == 1 and stores.backlog.count() == 0
    assert stores.backlog.through() == DAY
    # 当天重试、当天废弃：那一天跑过之后再来属于它的不收（裁定 68 第 3 条）
    assert stores.backlog.put(_turn_item(_turn("晚到的一轮", 30))) is Admission.LATE
    assert stores.backlog.refused() == {"full": 0, "late": 1}


def test_items_of_today_wait_for_todays_night(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [])
    today = _turn_item(_turn("零点之后的一轮", 0), day=DAY + timedelta(days=1))
    stores.backlog.put(today)

    report = asyncio.run(_stage(stores, session).run_once(through=DAY))

    assert report.taken == 0 and provider.calls == 0
    assert [item.item_id for item in stores.backlog.snapshot().items] == [today.item_id]


def test_the_first_unavailable_turn_stops_the_night(tmp_path: Path) -> None:
    """模型整个调不通不重试（裁定 94 第 4 条）：第一条会话轮就额度满——记一项 ``unavailable``（线据它停下这一晚），剩下的不逐条
    等、链也不再放回；堆里的留着，属于已推过的日子，下一晚按上一晚剩下的撤回。"""

    stores = Stores(tmp_path)
    session, _provider = lane(stores, [])

    async def quota(_turn, _reminders=()):  # type: ignore[no-untyped-def]
        raise ModelQuotaError("quota exhausted")

    session.describer.describe = quota  # type: ignore[method-assign]
    chains = _Chains(good=frozenset({"链"}))
    stores.backlog.put(_chain_item("链"))
    stores.backlog.put(_turn_item(_turn("统计一下行数", 0)))
    stores.backlog.put(_turn_item(_turn("再查一下", 10)))

    report = asyncio.run(_stage(stores, session, chains).run_once(through=DAY))

    assert report.unreachable and report == RetryReport(taken=3, unavailable=1)
    assert chains.attempted == [] and stores.backlog.count() == 3
    assert stores.backlog.through() == DAY


def test_a_night_killed_midway_leaves_no_item_to_be_tried_twice(tmp_path: Path) -> None:
    """被杀在第二项上：第一项写回了、还没归约（等归约之后才删），第二项没试——两项都属于已经跑过的那天，下一晚只撤回、不再试
    （第一项写回的判断也撤回，不让它之后在白天落进已经结算的日子），重启时就清（裁定 68 第 3 条：同一轮不会补两遍）。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description(), description()])
    first, second = _turn("第一轮", 0), _turn("第二轮", 10)
    stores.backlog.put(_turn_item(first))
    stores.backlog.put(_turn_item(second))
    handler = TurnHandler(session)
    calls = {"n": 0}
    real = handler.attempt

    async def killed_on_the_second(item: RetryItem) -> Attempt:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("process killed")
        return await real(item)

    handler.attempt = killed_on_the_second  # type: ignore[method-assign]
    stage = RetryStage([RetryBranch("session", stores.backlog, {RetryKind.TURN: handler}, _Reduce())])
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(stage.run_once(through=DAY))
    left = {item.item_id for item in stores.backlog.snapshot().items}
    assert left == {_turn_item(first).item_id, _turn_item(second).item_id} and provider.calls == 1

    withdrawn: list[str] = []
    handler.abandon = lambda item: withdrawn.append(item.item_id)  # type: ignore[method-assign]
    report = asyncio.run(stage.run_once(through=DAY + timedelta(days=1)))
    assert report == RetryReport(taken=2, leftover=2) and set(withdrawn) == left and provider.calls == 1

    # 重启时就清：不等下一晚
    stores.backlog.put(_turn_item(_turn("又一轮", 20), day=DAY + timedelta(days=2)))
    stores.backlog.advance(DAY + timedelta(days=2))
    assert stage.clean_leftovers() == 1 and stores.backlog.count() == 0


def test_a_restored_chain_counts_as_recovered_only_if_the_reduction_consumed_it(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    chains = _Chains(good=frozenset({"洗手"}))
    stores.backlog.put(_chain_item("洗手"))
    stores.backlog.put(_chain_item("坏名字"))
    reduce = _Reduce()

    report = asyncio.run(_stage(stores, None, chains, reduce).run_once(through=DAY))

    assert report == RetryReport(taken=2, recovered=1, deterministic=1) and reduce.runs == 1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # 答得不合格（校验用尽）：这一项内容不合格，后面的照常补回
        ("ModelStructuredOutputError", RetryReport(taken=3, recovered=2, invalid=1)),
        # 服务端答坏了（failover_eligible）、模型配置用不了：模型整个调不通，记 unavailable、不再试别的（线据此停下）
        ("ModelResponseError", RetryReport(taken=3, unavailable=1)),
        ("ModelConfigurationError", RetryReport(taken=3, unavailable=1)),
    ],
)
def test_other_model_errors_on_a_turn_are_counted_or_stop_the_night(
    tmp_path: Path, error: str, expected: RetryReport
) -> None:
    """E2：第一项让模型抛 ``ModelClientError`` 的其他子类——格式不合按"内容不合格"计数、堆清空；模型用不了 / 没配好不重试，
    记 ``unavailable``、剩下的留在堆里（裁定 94 错误分类）。"""

    import habitus.model_client as model_client

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description(), description()])
    real = session.describer.describe
    calls = {"n": 0}

    async def first_fails(turn, reminders=()):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise getattr(model_client, error)("scripted")
        return await real(turn, reminders)

    session.describer.describe = first_fails  # type: ignore[method-assign]
    for index, text in enumerate(("第一轮", "第二轮", "第三轮")):
        stores.backlog.put(_turn_item(_turn(text, index * 10)))

    report = asyncio.run(_stage(stores, session).run_once(through=DAY))

    assert report == expected
    assert stores.backlog.count() == (3 if report.unreachable else 0)


def test_a_branch_whose_backlog_is_empty_is_not_reduced(tmp_path: Path) -> None:
    """E15：计数按支——会话一支补回了，物理一支的堆是空的：只跑会话一支的归约（两支互不影响，裁定 35、42）。"""

    from habitus.behavior.retry import RetryBacklogStore

    stores = Stores(tmp_path / "session")
    session, _provider = lane(stores, [description()])
    stores.backlog.put(_turn_item(_turn("统计一下行数", 0)))
    session_reduce, physical_reduce = _Reduce(), _Reduce()
    physical_backlog = RetryBacklogStore(tmp_path / "physical", max_items=100)
    stage = RetryStage(
        [
            RetryBranch("session", stores.backlog, {RetryKind.TURN: TurnHandler(session)}, session_reduce),
            RetryBranch("physical", physical_backlog, {RetryKind.CHAIN: _Chains()}, physical_reduce),
        ]
    )

    report = asyncio.run(stage.run_once(through=DAY))

    assert report == RetryReport(taken=1, recovered=1)
    assert (session_reduce.runs, physical_reduce.runs) == (1, 0)


# ── 变异测试补的探针（段 4 E19）──────────────────────────────────────────────────────


def test_after_the_first_unavailable_turn_the_rest_never_reach_the_model(tmp_path: Path) -> None:
    """R39：第一条会话轮就"模型用不了"，这一晚停下、剩下的轮不再调模型（额度用完不白烧调用）。"""

    stores = Stores(tmp_path)
    session, _provider = lane(stores, [])
    calls = {"n": 0}

    async def quota(_turn, _reminders=()):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise ModelQuotaError("quota exhausted")

    session.describer.describe = quota  # type: ignore[method-assign]
    for index in range(3):
        stores.backlog.put(_turn_item(_turn(f"第{index}轮", 10 * index)))

    report = asyncio.run(_stage(stores, session).run_once(through=DAY))
    assert report.unreachable and calls["n"] == 1


def test_a_deterministic_failure_on_retry_is_counted_as_deterministic(tmp_path: Path) -> None:
    """R42：重试时描述器确定性失败，计成"确定性失败"，不计成"内容不合格"（健康面按类分开数，裁定 67 第 12 条）。"""

    from habitus.foundation.lifecycle import DeterministicFailure

    stores = Stores(tmp_path)
    session, _provider = lane(stores, [])

    async def broken(_turn, _reminders=()):  # type: ignore[no-untyped-def]
        raise DeterministicFailure("input exceeds the limit")

    session.describer.describe = broken  # type: ignore[method-assign]
    stores.backlog.put(_turn_item(_turn("超长的一轮", 0)))

    report = asyncio.run(_stage(stores, session).run_once(through=DAY))

    assert report == RetryReport(taken=1, deterministic=1)


def test_cleaning_last_nights_leftovers_withdraws_what_they_half_wrote_first(tmp_path: Path) -> None:
    """R43：重启时清上一晚剩下的项，先撤回它写到一半的东西再删（不留没归约的判断钉住封口）。"""

    stores = Stores(tmp_path)
    session, _provider = lane(stores, [])
    handler = TurnHandler(session)
    withdrawn: list[str] = []
    handler.abandon = lambda item: withdrawn.append(item.item_id)  # type: ignore[method-assign]
    item = _turn_item(_turn("剩下的一轮", 0))
    stores.backlog.put(item)
    stores.backlog.advance(DAY)
    stage = RetryStage([RetryBranch("session", stores.backlog, {RetryKind.TURN: handler}, _Reduce())])

    assert stage.clean_leftovers() == 1
    assert withdrawn == [item.item_id]


def test_errors_we_cannot_process_or_refused_by_content_safety_are_not_tried_again(tmp_path: Path) -> None:
    """裁定 94 错误分类：内容安全拦下的、我们自己处理不了的（链无可读名 / 物化失败、写不成）夜里不白试——直接删、按类计数
    （健康面列出），处理器一次也不调；能重试的照常试。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    chains = _Chains(good=frozenset({"归类调不通"}))
    stores.backlog.put(_chain_item("物化失败", cause=RetryCause.DETERMINISTIC))
    stores.backlog.put(_chain_item("归类调不通"))
    refused = _turn("被内容安全拦下", 0)
    stores.backlog.put(
        RetryItem(
            retry_id(RetryKind.TURN, refused.key), RetryKind.TURN, RetryCause.REFUSED, DAY, AT, encode_turn(refused)
        )
    )

    report = asyncio.run(_stage(stores, session, chains).run_once(through=DAY))

    assert report == RetryReport(taken=3, recovered=1, deterministic=1, refused=1)
    assert chains.attempted == ["归类调不通"] and provider.calls == 0 and stores.backlog.count() == 0


def test_a_bug_on_one_item_is_held_to_that_item_and_the_written_ones_are_reduced_tonight(tmp_path: Path) -> None:
    """裁定 94 第 8 条：重试堆里一项的处理器抛了（代码 bug）——只拦这一项、记成我们自己处理不了的，后面的项照试，这一支已经
    放回的照常在这一晚归约（不留到白天落进已经结算的日子）。"""

    stores = Stores(tmp_path)

    class _Buggy(_Chains):
        async def attempt(self, item: RetryItem) -> Attempt:
            if item.payload["records"][0]["name"] == "坏的":
                raise KeyError("records")
            return await super().attempt(item)

    chains = _Buggy(good=frozenset({"先放回", "后放回"}))
    for name in ("先放回", "坏的", "后放回"):
        stores.backlog.put(_chain_item(name))
    reduce = _Reduce()

    report = asyncio.run(_stage(stores, None, chains, reduce).run_once(through=DAY))

    assert report == RetryReport(taken=3, recovered=2, deterministic=1)
    assert set(chains.attempted) == {"先放回", "后放回"} and reduce.runs == 1 and stores.backlog.count() == 0


def test_a_lock_held_by_another_process_is_not_ready_and_the_next_look_tries_that_item_again(tmp_path: Path) -> None:
    """裁定 94 第 9 条：一项的锁被另一进程拿着（``TimeoutError``）不是错误、更不是模型调不通，是还没准备好——这一项不删，
    这一支记成在等（线停在前置处理、不结算）；后面的照试、写回的照常归约。下一眼同一晚接着跑：只再试那一项，计数接着往下数。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description(), description()])
    real = session.describer.describe
    calls = {"n": 0}

    async def first_busy(turn, reminders=()):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("lock busy")
        return await real(turn, reminders)

    session.describer.describe = first_busy  # type: ignore[method-assign]
    for index, text in enumerate(("第一轮", "第二轮")):
        stores.backlog.put(_turn_item(_turn(text, index * 10)))
    stage = _stage(stores, session)

    report = asyncio.run(stage.run_once(through=DAY))

    assert report == RetryReport(taken=2, recovered=1, waiting=("session",))
    assert not report.ready and not report.unreachable
    assert provider.calls == 1 and stores.backlog.count() == 1

    again = asyncio.run(stage.run_once(through=DAY))
    assert again == RetryReport(taken=2, recovered=2) and again.ready
    assert provider.calls == 2 and stores.backlog.count() == 0


def test_a_busy_reduction_keeps_the_written_items_and_resumes_without_asking_the_model_again(tmp_path: Path) -> None:
    """review-94b 中 1：重试段里这一支的归约碰到锁被另一进程拿着——是还没准备好，不是吞掉：写回的那一项不删、这一支记成在等
    （线不往下结算，写回的判断不会在白天落进已经结算的日子）；下一眼直接归约它、不再重新描述（不再问一次模型），归约成了才删。
    撤回这一处改动（归约锁被占时照删、照往下走）这里就红。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    turn = _turn("统计一下行数", 0)
    stores.backlog.put(_turn_item(turn))
    reduce = _Reduce(Reduction.BUSY)
    stage = _stage(stores, session, reduce=reduce)

    first = asyncio.run(stage.run_once(through=DAY))

    assert first == RetryReport(taken=1, waiting=("session",)) and not first.ready
    assert stores.backlog.count() == 1 and provider.calls == 1 and reduce.runs == 1
    assert session.retried(turn)  # 写回的判断还在等归约

    second = asyncio.run(stage.run_once(through=DAY))

    assert second == RetryReport(taken=1, recovered=1) and second.ready
    assert provider.calls == 1 and reduce.runs == 2 and stores.backlog.count() == 0


def test_a_reduction_that_finds_the_model_unusable_stops_the_night_and_keeps_tonights_counts(tmp_path: Path) -> None:
    """review-94b 低 7：重试段里这一支的归约碰到模型整个用不了——报告照样带着这一晚的计数（取出几项、哪几项没补回），并标
    ``unreachable`` 交线停下；不是抛出去让线丢掉整份计数。"""

    stores = Stores(tmp_path)
    chains = _Chains()
    for name in ("放回一", "放回二"):
        stores.backlog.put(_chain_item(name))

    report = asyncio.run(_stage(stores, None, chains, _Reduce(Reduction.UNREACHABLE)).run_once(through=DAY))

    assert report == RetryReport(taken=2, unavailable=2, reduction_unreachable=True) and report.unreachable
    assert stores.backlog.count() == 0


def test_a_reduction_bug_stops_the_night_and_the_written_items_are_reduced_first_next_time(tmp_path: Path) -> None:
    """裁定 94 第 9 条：重试段里归约抛了是 bug——原样抛给线（停下这一晚、报出），不吞成一句没人读的原因；写回的那一项留在堆里，
    下一次先归约它（不当上一晚剩下的撤回，也不再问一次模型）。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    turn = _turn("统计一下行数", 0)
    stores.backlog.put(_turn_item(turn))
    calls = {"n": 0}

    async def broken_once() -> Reduction:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyError("verdicts")
        return Reduction.DONE

    stage = RetryStage([RetryBranch("session", stores.backlog, {RetryKind.TURN: TurnHandler(session)}, broken_once)])
    with pytest.raises(KeyError):
        asyncio.run(stage.run_once(through=DAY))
    assert stores.backlog.count() == 1 and session.retried(turn)

    report = asyncio.run(stage.run_once(through=DAY + timedelta(days=1)))
    assert report == RetryReport(recovered=1) and provider.calls == 1 and stores.backlog.count() == 0


def test_a_write_bug_on_retry_is_counted_as_ours_not_as_a_bad_model_answer(tmp_path: Path) -> None:
    """review-94b 低 7：重试一轮时我们自己写入出错（``ValueError`` / ``BehaviorFusionError`` 这类校验不过）是 bug——按"我们自己
    处理不了"计数、只拦这一项，不再报成"格式不合"（那是模型答了但结构化校验用尽）。撤回 ``RETRY_INVALID`` 的改动这里就红。"""

    from habitus.behavior.fusion.errors import BehaviorFusionError

    for error in (ValueError("bad write"), BehaviorFusionError("bad payload")):
        stores = Stores(tmp_path / type(error).__name__)
        session, _provider = lane(stores, [description()])

        def broken(*_args, _error=error, **_kwargs):  # type: ignore[no-untyped-def]
            raise _error

        session.recorder.record = broken  # type: ignore[method-assign]
        stores.backlog.put(_turn_item(_turn("统计一下行数", 0)))

        report = asyncio.run(_stage(stores, session).run_once(through=DAY))

        assert report == RetryReport(taken=1, deterministic=1), type(error).__name__
