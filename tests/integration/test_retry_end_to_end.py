"""每晚重试的整条链（裁定 67、68；四方评审时测试工程师用真实组件写的用例收进来）：真实会话 lane + 真实归约 + 真实行为树
+ 真实重试编排。白天放弃 → 落「没读懂」、进堆 → 夜里补回 → 树上那条行为、按天读树不再算空白；重试写到一半失败、原覆盖
记录早已删掉时，不让一组没被覆盖的凭据钉住之后的封口。"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from habitus.behavior.fusion.lanes.session import SessionLane, SessionTurnDescriber, SessionTurnRecorder, split_turns
from habitus.behavior.model import BehaviorKind
from habitus.behavior.retry import RetryBranch, RetryKind, RetryStage
from habitus.behavior.retry.handlers import ChainHandler, TurnHandler
from habitus.scene.views import DayIndexCache
from habitus.series.reader import admitted
from tests.unit.behavior import session_fixtures as sf
from tests.unit.behavior.test_reduction_runner import Harness

NAME = "统计本地项目代码行数"


class Site:
    def __init__(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        self.h = Harness(tmp_path, known_kinds=(NAME,))
        self.backlog = self.h.runner.failed_chains.backlog
        self.h.now = sf.at(180)  # 轮在 sf.at(0..)，3 小时后：封口视界已过
        self.recorder = SessionTurnRecorder(
            observations=self.h.observations,
            judgements=self.h.judgements,
            receipts=self.h.receipts,
            coverage=self.h.runner.coverage,
            primary_subject=sf.SUBJECT,
            zone=sf.ZONE,
            clock=lambda: self.h.now,
        )

    def lane(self, bodies):  # type: ignore[no-untyped-def]
        chat, provider = sf.client(bodies)
        session = SessionLane(SessionTurnDescriber(chat, zone=sf.ZONE), self.recorder, backlog=self.backlog)
        return session, provider

    def stage(self, session) -> RetryStage:  # type: ignore[no-untyped-def]
        async def reduce() -> str | None:
            await self.h.runner.run_once()
            return None

        handlers = {RetryKind.TURN: TurnHandler(session), RetryKind.CHAIN: ChainHandler(self.h.runner.failed_chains)}
        return RetryStage([RetryBranch("session", self.backlog, handlers, reduce)])

    def has_gap(self, turn) -> bool:  # type: ignore[no-untyped-def]
        cache = DayIndexCache(self.h.tree, subject=sf.SUBJECT, admits=admitted)
        return cache.has_gap(turn.instructed_at, turn.completed_at)

    def occurrences(self) -> list[str]:
        return [address.name for address in self.h.tree.list_addresses(BehaviorKind.OCCURRENCE)]


def _turn(text: str, start: int, conversation: str | None = None):  # type: ignore[no-untyped-def]
    (turn,) = split_turns(
        conversation or f"c-{text}",
        (sf.prompt(start, text, minutes=start), sf.reply(start + 1, "好的", minutes=start + 2)),
    )
    return turn


def _day(turn):  # type: ignore[no-untyped-def]
    return turn.instructed_at.astimezone(sf.ZONE).date()


def test_a_given_up_turn_is_recovered_at_night_and_the_gap_stops_counting(tmp_path) -> None:  # type: ignore[no-untyped-def]
    site = Site(tmp_path)
    turn = _turn("统计一下行数", 0)
    day_lane, _ = site.lane([])
    asyncio.run(day_lane.abandon([turn], reason="quota"))
    assert asyncio.run(site.h.runner.run_once()).published_gaps == 1
    assert site.has_gap(turn) and site.backlog.count() == 1

    night_lane, provider = site.lane([sf.description(NAME)])
    site.h.now = sf.at(600)
    report = asyncio.run(site.stage(night_lane).run_once(through=_day(turn)))

    assert (report.recovered, report.taken, report.unreduced) == (1, 1, ())
    assert provider.calls == 1 and site.occurrences() == [NAME]
    assert not site.has_gap(turn)  # 当晚结算按天读树看到的是那条行为，不是空白
    assert site.backlog.count() == 0 and site.h.judgements.list() == () and site.h.observations.list() == ()
    after = asyncio.run(site.h.runner.run_once())
    assert (after.published_occurrences, after.published_gaps) == (0, 0)


def test_a_failed_retry_does_not_pin_the_seal_once_the_original_coverage_is_gone(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """评审 B1：会话源已到终态、原覆盖记录随观测释放删掉；夜里重试回执写入一直失败——撤回这一次写下的，第二天别的会话
    照常封口落树。"""

    site = Site(tmp_path)
    site.h.runner.coverage.settled = lambda _source: True
    turn = _turn("统计一下行数", 0)
    day_lane, _ = site.lane([])
    asyncio.run(day_lane.abandon([turn], reason="quota", sources=("session-source-1",)))
    asyncio.run(site.h.runner.run_once())
    asyncio.run(site.h.runner.run_once())  # 第二轮过期时观测已释放 → 覆盖记录被删
    assert not site.recorder.is_recorded(turn)

    original = site.h.receipts.put

    def broken(_receipt):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    site.h.receipts.put = broken  # type: ignore[method-assign]
    night_lane, _ = site.lane([sf.description(NAME)])
    site.h.now = sf.at(600)
    # 磁盘错误不是"内容不合格"：抛出去，夜批记 ``retry_failed``、健康面看得见；这一次写下的已经撤回
    with pytest.raises(OSError):
        asyncio.run(site.stage(night_lane).run_once(through=_day(turn)))
    site.h.receipts.put = original  # type: ignore[method-assign]
    assert site.h.observations.list() == () and site.h.judgements.list() == ()

    tomorrow = _turn("再看一下", 700, conversation="c-next")
    next_lane, _ = site.lane([sf.description(NAME)])
    asyncio.run(next_lane.consume([tomorrow], sources=("session-source-2",)))
    site.h.now = sf.at(1000) + timedelta(hours=1)
    assert asyncio.run(site.h.runner.run_once()).published_occurrences == 1
