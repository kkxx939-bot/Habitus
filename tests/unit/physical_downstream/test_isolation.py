"""第二步丙二阶段 C 补的几条（撤回即失败时没有测试变红的那几处）：两个常驻 worker 一支抛了不挡另一支、一支退避不拖另一支，
一支的预测根租约被占只停这一支，物理一支同一个行为同一刻只放一条请求（S5 的物理那一面）。

worker 的两条用替身支（只认 ``lane`` 与 ``run_once`` / ``drain``、``wait_seconds``），其余用 ``World``。
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from habitus.runtime.foresight import foresight_lease_key
from habitus.runtime.foresight_tick import TickReport
from habitus.runtime.foresight_worker import ForesightWorker, SlotBranch, SlotFailed
from habitus.runtime.model_queue_worker import DrainFailed, ModelQueueWorker
from tests.unit.physical_downstream.site import MEDICINE, Named, World, at, day_delivery

D13 = date(2026, 4, 13)
ZONE = ZoneInfo("Asia/Shanghai")


class _Tick:
    """预测拍替身：记下跑了几次；``broken`` 时每次都抛。"""

    def __init__(self, lane: str, *, broken: bool = False) -> None:
        self.lane = lane
        self.broken = broken
        self.runs = 0

    async def run_once(self, *, scheduled: datetime | None = None, before: Any = None) -> TickReport:
        self.runs += 1
        if self.broken:
            raise RuntimeError(f"{self.lane} tick broke")
        return TickReport(datetime(2026, 4, 14, 9, tzinfo=ZONE), 0, 0)


def _slot_worker(*ticks: _Tick) -> ForesightWorker:
    return ForesightWorker(
        tuple(SlotBranch(tick.lane, tick=tick) for tick in ticks),  # type: ignore[arg-type]
        interval_seconds=900.0,
        shutdown_timeout_seconds=1.0,
        zone=ZONE,
    )


@pytest.mark.parametrize("broken", ["session", "physical"])
def test_a_branch_whose_slot_raises_does_not_stop_the_other_branch_in_the_same_slot(broken: str) -> None:
    ticks = {lane: _Tick(lane, broken=lane == broken) for lane in ("session", "physical")}
    worker = _slot_worker(ticks["session"], ticks["physical"])
    with pytest.raises(SlotFailed) as caught:
        asyncio.run(worker.run_once())
    healthy = "physical" if broken == "session" else "session"
    assert set(caught.value.failures) == {broken}
    assert set(caught.value.reports) == {healthy}
    assert ticks["session"].runs == ticks["physical"].runs == 1


class _Queue:
    """模型队列替身：``broken`` 时每趟都抛；``wait`` 是它说的下一次最早隔多久再看。"""

    def __init__(self, lane: str, *, broken: bool = False, wait: float = 30.0) -> None:
        self.lane = lane
        self.broken = broken
        self.wait = wait
        self.drains = 0

    async def drain(self, *, stop: Any = None) -> Any:
        self.drains += 1
        if self.broken:
            raise RuntimeError(f"{self.lane} queue broke")
        return f"{self.lane} drained"

    def wait_seconds(self) -> float:
        return self.wait


@pytest.mark.parametrize("broken", ["session", "physical"])
def test_a_branch_whose_queue_raises_does_not_stop_the_other_branchs_queue(broken: str) -> None:
    queues = {lane: _Queue(lane, broken=lane == broken) for lane in ("session", "physical")}
    worker = ModelQueueWorker(
        (queues["session"], queues["physical"]),  # type: ignore[arg-type]
        shutdown_timeout_seconds=1.0,
    )
    with pytest.raises(DrainFailed) as caught:
        asyncio.run(worker.drain())
    healthy = "physical" if broken == "session" else "session"
    assert set(caught.value.failures) == {broken}
    assert caught.value.reports == {healthy: f"{healthy} drained"}
    assert queues["session"].drains == queues["physical"].drains == 1


def test_a_branch_backing_off_does_not_make_the_queue_worker_wait_for_the_other_branch() -> None:
    """物理一支整体退避（下一次 600 秒后再看）：会话一支照它自己的间隔（30 秒）再看。"""

    worker = ModelQueueWorker(
        (_Queue("session", wait=30.0), _Queue("physical", wait=600.0)),  # type: ignore[arg-type]
        shutdown_timeout_seconds=1.0,
    )
    assert worker.wait_seconds() == 30.0


def _hold_lease(raw: dict[str, Any]) -> None:
    raw["scene"] = {"enabled": True, "generation_lease_ttl_seconds": 5, "generation_heartbeat_seconds": 2}


def test_c3b_the_session_foresight_root_lease_held_elsewhere_stops_only_the_session_slot(tmp_path: Path) -> None:
    """C3 的预测根租约那一半：会话一支的预测根租约被别的进程占着，会话这一槽到此为止，物理一支照常换代。"""

    world = World(tmp_path, start=at(D13, 0), edit=_hold_lease)
    try:
        world.router.pick_default = Named("写周报")
        world.set_clock(at(D13, 9, 2))
        world.turn("conv", at(D13, 9), "帮我写本周周报")
        world.deliver(day_delivery(D13, [("吃药", 8, 0, 10)]), now=at(D13 + timedelta(days=1), 0, 5))
        lock_store = world.runtime.components.infrastructure.path_lock.lock_store
        token = lock_store.acquire(foresight_lease_key(world.config, "session"), ttl_seconds=3600)
        try:
            world.night(D13 + timedelta(days=1))
        finally:
            lock_store.release(token)
        assert world.marks("session") == []
        assert world.marks("physical") == ["2026-04-14"]
    finally:
        world.close()


def test_s5b_one_physical_behaviour_at_one_moment_puts_one_request_at_the_outlet(tmp_path: Path) -> None:
    """S5 的物理那一面：物理一支同一个行为同一刻的两条预期，出口里只有一条请求，账上只有一条「待取」。"""

    day = date(2026, 4, 14)
    world = World(tmp_path, start=at(day, 0))
    try:
        world.use("physical", MEDICINE)
        world.earn("physical", MEDICINE, at(day, 0))
        window = (at(day, 9), at(day, 9, 30))
        world.expect("physical", "pill-day", MEDICINE, window=window, deadline=at(day, 23, 59))
        world.expect("physical", "pill-ctx", MEDICINE, window=window, deadline=at(day, 23, 59), source="情境")
        assert world.claim(at(day, 8), session="probe") == ()
        world.slots_between(at(day, 9), at(day, 10), reduce=False)
        claimed = world.claim(at(day, 10, 5))
        assert len(claimed) == 1, [item.request_id for item in claimed]
        reasons = world.reasons("physical", day)
        assert sum(1 for made in reasons.values() if "待取" in made) == 1
        assert world.reasons("session", day) == {}
    finally:
        world.close()


# ── 出口两支：「说了」、回话、取走记录、问过的提问都按请求所属那一支 ─────────────────────────────


def _two_branch_outlet(tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    from habitus.runtime.foresight_ledger import ForesightLedgerStore
    from habitus.runtime.foresight_outlet import ReminderOutlet
    from habitus.runtime.foresight_outlet_store import OutletStore

    ledgers = {lane: ForesightLedgerStore(tmp_path / lane) for lane in ("session", "physical")}
    run = ReminderOutlet(
        {lane: OutletStore(tmp_path / lane / "outlet") for lane in ledgers},
        ledgers,
        liveness=timedelta(hours=48),
        lease=timedelta(minutes=30),
        max_attempts=3,
        claim_limit=3,
        similarity=0.6,
    )
    return run, ledgers


def _physical_request(run: Any, ledgers: dict[str, Any]) -> Any:
    from habitus.foresight.ledger import Delivery
    from habitus.runtime.foresight_outlet import OpenRequest
    from tests.unit.runtime.test_foresight_outlet import ASKED, TEXT, expectation

    item = expectation("pill", kind=MEDICINE)
    ledgers["physical"].record_expectation(item)
    ledgers["physical"].record_delivery(item.day, Delivery(item.expectation_id, ASKED, 1, False, "待取", 1.0, TEXT))
    row = OpenRequest(
        item.expectation_id, item.kind_token, TEXT, item.deadline, item.formed_at, 1, item.day, "physical"
    )
    run.publish("physical", [row], now=ASKED + timedelta(minutes=10))
    return item


def test_a_physical_request_spoken_and_answered_with_a_fixed_option_lands_on_the_physical_ledger(
    tmp_path: Path,
) -> None:
    from tests.unit.runtime.test_foresight_outlet import ASKED, TEXT, claude_answer, claude_question, turn

    run, ledgers = _two_branch_outlet(tmp_path)
    item = _physical_request(run, ledgers)
    now = ASKED + timedelta(minutes=10)
    assert [got.request_id for got in run.claim("claude-code", "claude-s1", now, 3)] == ["pill"]
    assert (tmp_path / "physical" / "outlet" / "claims" / "pill.json").is_file()
    assert not (tmp_path / "session" / "outlet" / "claims").exists()
    asked_at = now + timedelta(minutes=2)
    messages = turn(
        claude_question(TEXT, at=asked_at),
        claude_answer(f'User has answered your questions: "{TEXT}"="已经做过了". You can now continue.',
                      at=asked_at + timedelta(seconds=40)),
        at=now + timedelta(seconds=3),
    )  # fmt: skip
    report = run.observe("claude-s1", messages, host="claude-code")
    assert (report.spoken, report.responded) == (1, 1)
    assert [d.reason for d in ledgers["physical"].deliveries_on(item.day) if d.spoken] == ["说了"]
    assert [(r.expectation_id, r.polarity) for r in ledgers["physical"].responses_on(item.day)] == [("pill", "已做")]
    assert ledgers["session"].deliveries_on(item.day) == () and ledgers["session"].responses_on(item.day) == ()


def test_a_physical_question_asked_on_codex_desktop_is_kept_in_the_physical_outlet_and_its_answer_lands_there(
    tmp_path: Path,
) -> None:
    import json

    from habitus.runtime.foresight_outlet import OPTIONS, HostMessage
    from tests.unit.runtime.test_foresight_outlet import ASKED, REPLY_TAG, TEXT

    run, ledgers = _two_branch_outlet(tmp_path)
    item = _physical_request(run, ledgers)
    now = ASKED + timedelta(minutes=10)
    run.claim("codex", "codex-1", now, 3)
    arguments = json.dumps({"questions": [{"title": TEXT, "options": list(OPTIONS)}]}, ensure_ascii=False)
    asked_at = now + timedelta(minutes=1)
    first = (
        HostMessage(1, now + timedelta(seconds=2), "prompt", "帮我看看这个报错"),
        HostMessage(2, asked_at, "tool_call", arguments, tool_name="request_user_input_async", tool_call_id="call_a1"),
        HostMessage(3, asked_at, "tool_result", '{"accepted":true}', tool_name="request_user_input_async",
                    tool_call_id="call_a1"),
        HostMessage(4, asked_at + timedelta(minutes=3), "completion", "报错看完了。"),
    )  # fmt: skip
    assert run.observe("codex-1", first, host="codex").awaiting_reply == 1
    physical = [path.relative_to(tmp_path).as_posix() for path in (tmp_path / "physical").rglob("*")]
    session = [path.relative_to(tmp_path).as_posix() for path in (tmp_path / "session").rglob("*")]
    assert any(name.startswith("physical/outlet/asked") for name in physical), physical
    assert not any("asked" in name for name in session), session
    reply = {
        "questionItemId": json.dumps(["request_user_input_async", "call_a1", 0]),
        "question": TEXT,
        "answer": "现在就做",
    }
    answered_at = asked_at + timedelta(minutes=6)
    second = (
        HostMessage(5, answered_at, "prompt",
                    f"{REPLY_TAG}\n{json.dumps([reply], ensure_ascii=False)}\n</send_user_message_question_reply>\n"),
        HostMessage(6, answered_at + timedelta(seconds=30), "completion", "好。"),
    )  # fmt: skip
    assert run.observe("codex-1", second, host="codex").responded == 1
    assert [(r.expectation_id, r.polarity) for r in ledgers["physical"].responses_on(item.day)] == [("pill", "正")]
    assert ledgers["session"].responses_on(item.day) == ()
