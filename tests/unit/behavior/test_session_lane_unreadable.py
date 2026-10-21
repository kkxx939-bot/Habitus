"""会话 lane 确定性地写不成的一轮：写成一段没读懂、凭据被覆盖，不再每次重做（26 号清单 2.4.4）。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from habitus.behavior.fusion.errors import BehaviorFusionLimitError, BehaviorFusionTruncatedError
from habitus.behavior.fusion.lanes.session import TurnDisposition, split_turns
from habitus.behavior.fusion.lanes.session.model import decode_turn
from habitus.behavior.retry import RetryCause, RetryKind
from tests.unit.behavior.session_fixtures import Stores, description, lane, prompt, reply

CONVERSATION = "codex-rollout-unreadable"


def _turn():
    (turn,) = split_turns(CONVERSATION, (prompt(0, "统计一下行数"), reply(1, "8,792 行。", minutes=3)))
    return turn


def test_an_oversized_receipt_turns_the_round_into_an_unreadable_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    original = stores.receipts.put
    calls = {"n": 0}

    def put(receipt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BehaviorFusionLimitError("fusion receipt exceeds its configured file bound")
        return original(receipt)

    monkeypatch.setattr(stores.receipts, "put", put)
    turn = _turn()

    (outcome,) = asyncio.run(session.consume((turn,)))

    assert outcome.disposition is TurnDisposition.UNREADABLE
    (judgement,) = stores.judgements.list()
    assert judgement["behavior"] is None  # 可读的那条拿走了，只剩没读懂
    assert judgement["judgement_id"] == outcome.judgement_id
    # 这一轮整轮进每晚重试的堆（裁定 67），撤下的那条可读判断不另存
    (item,) = stores.backlog.snapshot().items
    assert (item.kind, item.cause) == (RetryKind.TURN, RetryCause.DETERMINISTIC)
    assert decode_turn(item.payload) == turn
    recorder = stores.recorder()
    assert recorder.is_recorded(turn)  # 凭据被覆盖：封口前沿不再被它钉住
    again, _ = lane(stores, [])
    (repeat,) = asyncio.run(again.consume((turn,)))
    assert repeat.disposition is TurnDisposition.ALREADY_RECORDED
    assert provider.calls == 1


def test_a_truncated_description_is_recorded_as_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stores = Stores(tmp_path)
    session, _provider = lane(stores, [description()])

    async def truncated(_turn, _reminders=()):
        raise BehaviorFusionTruncatedError("output truncated twice")

    monkeypatch.setattr(session.describer, "describe", truncated)
    (outcome,) = asyncio.run(session.consume((_turn(),)))
    assert outcome.disposition is TurnDisposition.UNREADABLE
    (judgement,) = stores.judgements.list()
    assert judgement["behavior"] is None


def test_a_crash_after_the_unreadable_judgement_landed_is_finished_on_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没读懂的判断落了、回执没落时被杀：下一次认出它是没读懂、补写回执，不当成写了一半的可读判断去续写（L10.4；
    原先续写时构造"没有行为却带状态"的判断，每轮都抛错，这份源永远到不了终态）。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    original = stores.receipts.put
    calls = {"n": 0}

    def put(receipt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BehaviorFusionLimitError("fusion receipt exceeds its configured file bound")
        if calls["n"] == 2:
            raise RuntimeError("process killed")
        return original(receipt)

    monkeypatch.setattr(stores.receipts, "put", put)
    turn = _turn()
    with pytest.raises(RuntimeError):
        asyncio.run(session.consume((turn,)))
    (landed,) = stores.judgements.list()
    assert landed["behavior"] is None

    again, _ = lane(stores, [])
    (outcome,) = asyncio.run(again.consume((turn,)))

    assert outcome.disposition is TurnDisposition.UNREADABLE
    assert outcome.judgement_id == landed["judgement_id"]
    assert [item["judgement_id"] for item in stores.judgements.list()] == [landed["judgement_id"]]
    assert stores.recorder().is_recorded(turn)
    assert provider.calls == 1


def test_a_given_up_turn_goes_to_the_backlog_and_the_nightly_retry_adds_its_judgement(tmp_path: Path) -> None:
    """重试账放弃了：先落「没读懂」、这一轮进堆（原因"放弃"）；当晚 ``retry`` 不看写没写过，再描述一次、补一条判断，
    原「没读懂」不动（读的时候被它作废）。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    turn = _turn()
    (outcome,) = asyncio.run(session.abandon((turn,), reason="recovery given up"))
    assert outcome.disposition is TurnDisposition.UNREADABLE and provider.calls == 0
    (item,) = stores.backlog.snapshot().items
    assert item.cause is RetryCause.GIVEN_UP
    assert stores.recorder().is_recorded(turn)  # 覆盖记录会说"写过了"：重试不能走它

    retried = asyncio.run(session.retry(decode_turn(item.payload)))

    assert retried.disposition is TurnDisposition.RECORDED and provider.calls == 1
    behaviors = sorted((judgement["behavior"] is None) for judgement in stores.judgements.list())
    assert behaviors == [False, True]  # 补上的那条 + 原来的「没读懂」


def test_salvaging_a_broken_source_only_marks_the_turns_never_written(tmp_path: Path) -> None:
    """会话源的账坏了（裁定 67 第 5、16 条）：写过的不动——覆盖记录在的、覆盖记录已删但树上从这一轮开口那一刻起有一条的；
    没写过的落「没读懂」，进重试堆；那一天的夜批已经跑过的，堆按天拒收（裁定 68 第 3 条）。不问模型。"""

    from dataclasses import replace
    from datetime import timedelta

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    written, on_tree_only, fresh, settled = (
        split_turns(CONVERSATION, (prompt(n * 10, f"第 {n} 件"), reply(n * 10 + 1, "好的", minutes=n * 10 + 2)))[0]
        for n in range(4)
    )
    asyncio.run(session.consume((written,)))
    earlier = timedelta(days=3)  # 三天前的一轮：那天已经结算过
    settled = replace(
        settled,
        instructed_at=settled.instructed_at - earlier,
        completed_at=settled.completed_at - earlier,
        messages=tuple(replace(message, occurred_at=message.occurred_at - earlier) for message in settled.messages),
    )
    settled_day = settled.instructed_at.astimezone(stores.recorder().zone).date()

    session._landed = lambda turn: turn.key == on_tree_only.key  # type: ignore[method-assign]
    stores.backlog.advance(settled_day)  # 那一天的夜批已经跑过：堆按天拒收（裁定 68 第 3 条）
    outcomes = session.salvage((written, on_tree_only, fresh, settled))

    assert [outcome.disposition for outcome in outcomes] == [
        TurnDisposition.ALREADY_RECORDED,
        TurnDisposition.ALREADY_RECORDED,
        TurnDisposition.UNREADABLE,
        TurnDisposition.UNREADABLE,
    ]
    assert provider.calls == 1  # 只有一开始正常写的那一轮问过模型
    assert [decode_turn(item.payload).key for item in stores.backlog.snapshot().items] == [fresh.key]


def test_a_turn_has_landed_when_something_on_its_branch_starts_at_its_opening_moment(tmp_path: Path) -> None:
    from datetime import UTC
    from types import SimpleNamespace

    from habitus.behavior.tree import BehaviorKind

    stores = Stores(tmp_path)
    session, _provider = lane(stores, [])
    turn = _turn()
    seen: list[object] = []

    class Tree:
        def __init__(self, starts) -> None:  # type: ignore[no-untyped-def]
            self.starts = starts

        def list_day_addresses(self, kind, day):  # type: ignore[no-untyped-def]
            seen.append(day)
            return tuple(SimpleNamespace(started_at=start) for start in self.starts.get(kind, ()))

    assert not session._landed(turn)  # 没给树：只认覆盖记录与判断存储
    session.tree = Tree({BehaviorKind.GAP: (turn.instructed_at.astimezone(UTC),)})  # type: ignore[assignment]
    assert session._landed(turn)  # 换个写法的同一时刻也认
    session.tree = Tree({BehaviorKind.OCCURRENCE: (turn.instructed_at + timedelta(seconds=1),)})  # type: ignore[assignment]
    assert not session._landed(turn)
    assert set(seen) == {turn.instructed_at.astimezone(stores.recorder().zone).date()}


def test_a_retry_killed_after_its_judgement_landed_is_finished_without_asking_again(tmp_path: Path) -> None:
    """裁定 68 第 4 条：重试写到一半被杀（可读判断落了、回执没落）——下一次接着写完，不重问模型（模型第二次的措辞不同，
    会写出第二条行为）。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description(), description("换了说法")])
    turn = _turn()
    asyncio.run(session.abandon((turn,), reason="model unavailable"))
    original = stores.receipts.put
    calls = {"n": 0}

    def killed(receipt):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt("process killed")
        return original(receipt)

    stores.receipts.put = killed  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(session.retry(turn))
    stores.receipts.put = original  # type: ignore[method-assign]
    assert provider.calls == 1

    outcome = asyncio.run(session.retry(turn))

    assert outcome.disposition is TurnDisposition.RECORDED and provider.calls == 1
    readable = [item["behavior"] for item in stores.judgements.list() if item["behavior"] is not None]
    assert readable == ["统计本地项目代码行数"]


def test_a_failed_retry_is_withdrawn_so_nothing_uncovered_is_left_behind(tmp_path: Path) -> None:
    """裁定 68 第 4 条：重试写到一半失败——撤回这一次写下的：不留可读判断的残渣、不把已经归约过的「没读懂」写回判断
    存储；原来的覆盖记录已经随会话源到终态删掉时，连这一次重新落下的凭据也撤掉（不然一组没被覆盖的凭据会把封口视界
    永远钉住）。"""

    from habitus.behavior.fusion.coverage import BehaviorCoverageIndex

    stores = Stores(tmp_path)
    stores.coverage = BehaviorCoverageIndex(tmp_path, settled=lambda _source: True)
    session, _provider = lane(stores, [description()])
    turn = _turn()
    asyncio.run(session.abandon((turn,), reason="model unavailable", sources=("source-1",)))
    (unreadable,) = stores.judgements.list()
    # 白天归约已经把「没读懂」落成空白：判断与凭据随发布释放，会话源到终态后覆盖记录也过期删掉
    stores.judgements.discard(unreadable["judgement_id"])
    for envelope in stores.observations.list():
        stores.observations.discard(envelope.source_id)
    stores.coverage.expire(stores.now + timedelta(days=1))
    assert not stores.recorder().is_recorded(turn)

    def broken(_receipt):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    stores.receipts.put = broken  # type: ignore[method-assign]
    with pytest.raises(OSError):
        asyncio.run(session.retry(turn))

    assert stores.judgements.list() == ()
    assert stores.observations.list() == ()
