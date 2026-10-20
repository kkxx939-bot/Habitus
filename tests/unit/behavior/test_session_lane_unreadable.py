"""会话 lane 确定性地写不成的一轮：写成一段没读懂、凭据被覆盖，不再每次重做（26 号清单 2.4.4）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from habitus.behavior.fusion.errors import BehaviorFusionLimitError, BehaviorFusionTruncatedError
from habitus.behavior.fusion.lanes.session import TurnDisposition, split_turns
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
    (item,) = stores.quarantine.items()
    assert item.meta["kind"] == "session_turn"
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
