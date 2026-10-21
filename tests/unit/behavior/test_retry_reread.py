"""时窗末当场重读（裁定 68 第 1 条）：只重读范围里开口的会话轮；读出来了从堆里删、为真；模型用不了留在堆里、为假。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from habitus.behavior.fusion.lanes.session import split_turns
from habitus.behavior.fusion.lanes.session.model import encode_turn
from habitus.behavior.retry import RetryCause, RetryItem, RetryKind, retry_id
from habitus.behavior.retry.handlers import TurnHandler
from habitus.behavior.retry.reread import Rereader
from habitus.model_client import ModelQuotaError
from tests.unit.behavior.session_fixtures import Stores, at, description, lane, prompt, reply


def _item(text: str, start: int) -> RetryItem:
    (turn,) = split_turns(
        f"c-{text}", (prompt(start, text, minutes=start), reply(start + 1, "好的", minutes=start + 2))
    )
    day = turn.instructed_at.date()
    return RetryItem(
        retry_id(RetryKind.TURN, turn.key), RetryKind.TURN, RetryCause.GIVEN_UP, day, at(0), encode_turn(turn)
    )


def test_only_turns_opened_within_the_range_are_reread_and_a_recovered_one_leaves_the_backlog(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    inside, outside = _item("范围里的", 10), _item("范围外的", 100)
    stores.backlog.put(inside)
    stores.backlog.put(outside)
    reread = Rereader(stores.backlog, TurnHandler(session))

    assert asyncio.run(reread.reread(at(5), at(60)))
    assert provider.calls == 1
    assert [item.item_id for item in stores.backlog.snapshot().items] == [outside.item_id]
    assert not asyncio.run(reread.reread(at(5), at(60)))  # 范围里已经没有了


def test_an_unavailable_model_keeps_the_turn_for_the_night(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, _provider = lane(stores, [])

    async def quota(_turn, _reminders=()):  # type: ignore[no-untyped-def]
        raise ModelQuotaError("quota exhausted")

    session.describer.describe = quota  # type: ignore[method-assign]
    item = _item("范围里的", 10)
    stores.backlog.put(item)
    assert not asyncio.run(Rereader(stores.backlog, TurnHandler(session)).reread(at(5), at(5) + timedelta(hours=1)))
    assert [found.item_id for found in stores.backlog.snapshot().items] == [item.item_id]
