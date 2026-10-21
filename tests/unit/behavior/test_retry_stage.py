"""每晚跑批的第一步：按天重试白天读不懂的（裁定 67、68）——补回、按天收、每项试完就删、探活只停会话轮、上一晚中途被杀
剩下的不再试只撤回、重启时就清。"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.fusion.lanes.session import split_turns
from habitus.behavior.fusion.lanes.session.model import encode_turn
from habitus.behavior.retry import (
    Admission,
    Attempt,
    RetryBranch,
    RetryCause,
    RetryItem,
    RetryKind,
    RetryReport,
    RetryStage,
    retry_id,
)
from habitus.behavior.retry.handlers import TurnHandler
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


def _chain_item(name: str, day: date = DAY) -> RetryItem:
    return RetryItem(
        retry_id(RetryKind.CHAIN, name),
        RetryKind.CHAIN,
        RetryCause.DETERMINISTIC,
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
    def __init__(self) -> None:
        self.runs = 0

    async def __call__(self) -> str | None:
        self.runs += 1
        return None


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


def test_the_first_unavailable_turn_calls_off_the_turns_but_not_the_chains(tmp_path: Path) -> None:
    """探活：第一条会话轮就额度满——剩下的会话轮不逐条等、按丢弃删；链的放回不调模型，照常做（裁定 67 第 8 条）。"""

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

    assert report == RetryReport(taken=3, recovered=1, unavailable=2)
    assert chains.attempted == ["链"] and stores.backlog.count() == 0


def test_a_night_killed_midway_leaves_no_item_to_be_tried_twice(tmp_path: Path) -> None:
    """每项试完立刻删；被杀在第二项上：第一项已经删了，剩下的那一项属于已经跑过的那天——下一晚只撤回、不再试，
    重启时就清（裁定 68 第 3 条：同一轮不会补两遍）。"""

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
    (left,) = stores.backlog.snapshot().items
    assert left.item_id == _turn_item(second).item_id and provider.calls == 1

    withdrawn: list[str] = []
    handler.abandon = lambda item: withdrawn.append(item.item_id)  # type: ignore[method-assign]
    report = asyncio.run(stage.run_once(through=DAY + timedelta(days=1)))
    assert report == RetryReport(taken=1, leftover=1) and withdrawn == [left.item_id] and provider.calls == 1

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
