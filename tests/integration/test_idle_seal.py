"""闲置封存（裁定 107）：够不到 token 门槛的会话闲置够了被封存成普通 history 段并建记忆任务；没闲置够、没内容的不动；
没完成的地方由系统收尾后全部封存；迟到的结果丢弃并计数；封存后能接着聊；一个会话失败不挡别的；和正在跑的 Memory 交付
不并发；封完当场试放会话源。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta
from itertools import count
from pathlib import Path

import pytest

from habitus.config import ConversationLifecycleConfig
from habitus.conversation.source import ConversationSourceConsumer, SourceAddress
from habitus.conversation.source.fence import conversation_order_lock_key
from habitus.foundation.observability import ObservationEvent
from habitus.infrastructure.store.contracts import LockToken, PathLock
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationIngressRequest,
    ConversationMessageJournal,
    ConversationWriteConflictError,
)
from habitus.memory.conversation.interruption import INTERRUPTED_COMPLETION, INTERRUPTED_TOOL_RESULT
from habitus.memory.workflow import ConversationMemoryEnqueuer
from habitus.memory.workflow.jobs import MemoryJobStatus, MemoryJobStore
from habitus.pre.conversation import ConversationBatch, ConversationMessageRole, ConversationToolResultStatus
from habitus.runtime.lifecycle_idle import ConversationIdleSealParticipant, fully_sealed
from tests.helpers import BASE_TIME, closed_turn, message, tool_turn
from tests.lock_store import ProcessLocalLockStore

DAY = date(2026, 7, 1)


class _Context:
    _issued = count()

    def __init__(self, lock_store: ProcessLocalLockStore, now: datetime) -> None:
        self._token = lock_store.acquire(f"runtime:lifecycle:idle-test:{next(self._issued)}", ttl_seconds=60)
        self._now = now

    @property
    def now(self) -> datetime:
        return self._now

    def assert_owned(self) -> None:
        return None

    def fence_token(self) -> LockToken:
        return self._token


class _Releaser:
    def __init__(self) -> None:
        self.directories: list[str] = []

    def release_address(self, directory: str, _ctx: object) -> int:
        self.directories.append(directory)
        return 1


class _Events:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


class _Chain:
    def __init__(
        self, tmp_path: Path, *, idle_seconds: int = 3_600, batch: int = 1_000, observer: _Events | None = None
    ) -> None:
        self.lock_store = ProcessLocalLockStore()
        self.path_lock = PathLock(self.lock_store, **({} if observer is None else {"observer": observer}))
        self.journal = ConversationMessageJournal(tmp_path / "conversation", self.path_lock)
        self.jobs = MemoryJobStore(tmp_path / "workflow", self.path_lock, memory_root=tmp_path / "memory")
        # 默认切段配置：token 门槛 12,000，几句话的短对话靠 afterTurn 永远封不了
        self.enqueuer = ConversationMemoryEnqueuer(self.journal, self.jobs)
        self.releaser = _Releaser()
        self.woken = 0
        self.participant = ConversationIdleSealParticipant(
            self.enqueuer,
            self.releaser,  # type: ignore[arg-type]
            self.path_lock,
            ConversationLifecycleConfig(idle_commit_seconds=idle_seconds, idle_scan_batch_conversations=batch),
            lock_ttl_seconds=30,
            wake_memory=self._wake,
        )

    def _wake(self) -> None:
        self.woken += 1

    def talk(self, address: ConversationAddress, *messages) -> None:  # type: ignore[no-untyped-def]
        result = self.enqueuer.append_and_maybe_enqueue(
            address, ConversationBatch(address.conversation_id, messages), after_turn=True
        )
        assert result.jobs == ()  # afterTurn 够不到门槛

    def tick(self, now: datetime):  # type: ignore[no-untyped-def]
        return asyncio.run(self.participant.maintain(_Context(self.lock_store, now)))


def _after(hours: float) -> datetime:
    return BASE_TIME + timedelta(hours=hours)


def test_a_short_idle_conversation_is_sealed_into_history_with_a_memory_job(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)
    address = ConversationAddress("short-1", DAY)
    chain.talk(address, *closed_turn(start_sequence=0), *closed_turn(start_sequence=2))

    report = chain.tick(_after(2))

    assert report.released == 1
    assert chain.journal.read_live(address) is None
    (history,) = chain.journal.list_history(address)
    assert history.segment_id == "000000000000-000000000003"
    (job,) = chain.jobs.list_for_conversation(address)
    assert job.status is MemoryJobStatus.QUEUED
    assert job.segment_id == history.segment_id
    assert chain.woken == 1
    assert chain.releaser.directories == [SourceAddress(address.conversation_id, address.started_on).digest]
    assert fully_sealed(chain.journal)(SourceAddress(address.conversation_id, address.started_on))


def test_a_conversation_not_idle_long_enough_is_left_alone(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)
    address = ConversationAddress("short-1", DAY)
    chain.talk(address, *closed_turn(start_sequence=0))

    report = chain.tick(_after(0.5))

    assert report.released == 0
    assert chain.journal.list_history(address) == ()
    assert chain.jobs.list_for_conversation(address) == ()
    assert chain.releaser.directories == []
    assert not fully_sealed(chain.journal)(SourceAddress(address.conversation_id, address.started_on))


def test_a_fully_sealed_conversation_produces_no_new_segment_or_job(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)
    address = ConversationAddress("short-1", DAY)
    chain.talk(address, *closed_turn(start_sequence=0))
    chain.tick(_after(2))

    report = chain.tick(_after(200))

    assert report.released == 0
    assert len(chain.journal.list_history(address)) == 1
    assert len(chain.jobs.list_for_conversation(address)) == 1
    assert chain.jobs.high_watermark() == 1


@pytest.mark.parametrize("open_tail", ["no_final_reply", "tool_call_without_result"])
def test_an_unfinished_last_turn_is_closed_by_the_system_and_everything_is_sealed(
    tmp_path: Path, open_tail: str
) -> None:
    """裁定 107 第 2 条：闲置到点就当对话到此为止——没有结果的调用补一条失败的终态结果、没有最终回复的轮补一条说明
    中断的最终回复，然后 live 全部封存；段在轮尾结束，记忆任务按完整的一轮抽取。"""

    chain = _Chain(tmp_path)
    address = ConversationAddress("open-1", DAY)
    call = message(
        7, ConversationMessageRole.TOOL_CALL, {"cmd": "pytest"}, tool_call_id="call-open", tool_name="shell.run"
    )
    tail = (
        message(6, ConversationMessageRole.PROMPT, "跑一下测试"),
        *((call,) if open_tail == "tool_call_without_result" else ()),
    )
    chain.enqueuer.append(
        address, ConversationBatch("open-1", (*closed_turn(start_sequence=0), *tool_turn(start_sequence=2), *tail))
    )

    assert chain.tick(_after(2)).released == 1

    (segment,) = chain.journal.list_history(address)
    assert chain.journal.read_live(address) is None
    assert fully_sealed(chain.journal)(SourceAddress(address.conversation_id, address.started_on))
    assert segment.ends_mid_turn is False
    closing = segment.messages[len(tail) + 6 :]
    if open_tail == "tool_call_without_result":
        result, reply = closing
        assert (result.role, result.tool_call_id, result.tool_status, result.content) == (
            ConversationMessageRole.TOOL_RESULT,
            "call-open",
            ConversationToolResultStatus.ERROR,
            INTERRUPTED_TOOL_RESULT,
        )
    else:
        (reply,) = closing
    assert (reply.role, reply.content, reply.occurred_at) == (
        ConversationMessageRole.COMPLETION,
        INTERRUPTED_COMPLETION,
        _after(2),
    )
    turn = chain.journal.read_editor_turn(address, segment)
    assert (turn.unsealed, turn.whole.end_sequence) == ((), segment.end_sequence)
    (job,) = chain.jobs.list_for_conversation(address)
    assert job.segment_id == segment.segment_id


def test_a_late_result_after_the_seal_is_dropped_counted_and_the_rest_renumbered(tmp_path: Path) -> None:
    observer = _Events()
    chain = _Chain(tmp_path, observer=observer)
    address = ConversationAddress("late-1", DAY)
    call = message(1, ConversationMessageRole.TOOL_CALL, {"q": 1}, tool_call_id="call-open", tool_name="shell.run")
    chain.enqueuer.append(
        address, ConversationBatch("late-1", (message(0, ConversationMessageRole.PROMPT, "查一下"), call))
    )
    chain.tick(_after(2))
    sealed_next = chain.journal.read_state(address).next_sequence  # 0、1 加系统补的两条
    assert sealed_next == 4

    late = replace(
        tool_turn(start_sequence=0)[2],
        sequence=4,
        message_id="late-result",
        tool_call_id="call-open",
        tool_name="shell.run",
        occurred_at=_after(3),
    )
    following = tuple(
        replace(item, sequence=item.sequence + 1, occurred_at=_after(3))
        for item in closed_turn(start_sequence=4, prompt="换个话题")
    )
    appended = chain.enqueuer.append(address, ConversationBatch("late-1", (late, *following)))

    assert appended.next_sequence == 6
    live = chain.journal.read_live(address)
    assert live is not None
    assert [(item.sequence, item.message_id) for item in live.messages] == [
        (4, following[0].message_id),
        (5, following[1].message_id),
    ]
    assert [event.attributes for event in observer.events if event.operation == "late_tool_result_dropped"] == [
        {"dropped": 1}
    ]

    only_late = replace(late, sequence=6, message_id="late-again")
    unchanged = chain.enqueuer.append(address, ConversationBatch("late-1", (only_late,)))
    assert (unchanged.appended_count, unchanged.next_sequence) == (0, 6)


def test_a_batch_numbered_before_the_seal_is_shifted_past_the_closing_messages(tmp_path: Path) -> None:
    """插件编好序号、还没送出，会话就闲置封存了：这一批开头落在系统收尾消息上，服务端整批顺延到它们后面，不拒收；
    同一份投递重放时判出同一批，认出是重放。"""

    chain = _Chain(tmp_path)
    address = ConversationAddress("shift-1", DAY)
    chain.enqueuer.append(
        address, ConversationBatch("shift-1", (message(0, ConversationMessageRole.PROMPT, "还在吗"),))
    )
    chain.tick(_after(2))  # 系统补了一条最终回复，占了序号 1
    assert chain.journal.read_state(address).next_sequence == 2
    bound = tuple(replace(item, occurred_at=_after(3)) for item in closed_turn(start_sequence=1, prompt="接着说"))
    ingress = ConversationIngressRequest("e" * 64, "f" * 64)

    appended = chain.enqueuer.append(address, ConversationBatch("shift-1", bound), ingress=ingress)

    assert (appended.appended_count, appended.next_sequence) == (2, 4)
    live = chain.journal.read_live(address)
    assert live is not None
    assert [(item.sequence, item.message_id) for item in live.messages] == [
        (2, bound[0].message_id),
        (3, bound[1].message_id),
    ]
    again = chain.enqueuer.append(address, ConversationBatch("shift-1", bound), ingress=ingress)
    assert (again.appended_count, again.next_sequence) == (0, 4)


def test_a_replay_of_a_delivery_that_had_a_late_result_dropped_is_recognized(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)
    address = ConversationAddress("late-2", DAY)
    call = message(1, ConversationMessageRole.TOOL_CALL, {"q": 1}, tool_call_id="call-open", tool_name="shell.run")
    chain.enqueuer.append(
        address, ConversationBatch("late-2", (message(0, ConversationMessageRole.PROMPT, "查一下"), call))
    )
    chain.tick(_after(2))
    late = replace(
        tool_turn(start_sequence=0)[2],
        sequence=4,
        message_id="late-result",
        tool_call_id="call-open",
        tool_name="shell.run",
        occurred_at=_after(3),
    )
    following = tuple(
        replace(item, sequence=item.sequence + 1, occurred_at=_after(3)) for item in closed_turn(start_sequence=4)
    )
    ingress = ConversationIngressRequest("1" * 64, "2" * 64)
    assert (
        chain.enqueuer.append(address, ConversationBatch("late-2", (late, *following)), ingress=ingress).appended_count
        == 2
    )

    again = chain.enqueuer.append(address, ConversationBatch("late-2", (late, *following)), ingress=ingress)

    assert (again.appended_count, again.next_sequence) == (0, 6)


def test_a_result_for_a_call_that_never_existed_is_still_refused(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)
    address = ConversationAddress("ghost-1", DAY)
    chain.talk(address, *closed_turn(start_sequence=0))
    ghost = replace(
        tool_turn(start_sequence=0)[2],
        sequence=2,
        message_id="ghost",
        tool_call_id="never-called",
        occurred_at=_after(1),
    )

    with pytest.raises(ConversationWriteConflictError, match="known tool_call"):
        chain.enqueuer.append(address, ConversationBatch("ghost-1", (ghost,)))


def test_a_replayed_delivery_is_not_rejudged_after_its_call_was_sealed(tmp_path: Path) -> None:
    """同一份投递重放时有入口回执：不再丢结果，原样认出是重放，不会因为调用已封存而判出另一批。"""

    chain = _Chain(tmp_path)
    address = ConversationAddress("replay-1", DAY)
    call = message(1, ConversationMessageRole.TOOL_CALL, {"q": 1}, tool_call_id="call-1", tool_name="shell.run")
    chain.enqueuer.append(
        address, ConversationBatch("replay-1", (message(0, ConversationMessageRole.PROMPT, "查一下"), call))
    )
    rest = (*tool_turn(start_sequence=0)[2:],)
    rest = tuple(
        replace(
            item,
            sequence=2 + index,
            tool_call_id=item.tool_call_id and "call-1",
            tool_name=item.tool_name and "shell.run",
        )
        for index, item in enumerate(rest)
    )
    ingress = ConversationIngressRequest("a" * 64, "b" * 64)
    chain.enqueuer.append(address, ConversationBatch("replay-1", rest), ingress=ingress)
    chain.tick(_after(2))
    assert chain.journal.read_live(address) is None

    again = chain.enqueuer.append(address, ConversationBatch("replay-1", rest), ingress=ingress)

    assert again.appended_count == 0


def test_the_conversation_continues_after_an_idle_seal_with_contiguous_sequences(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)
    address = ConversationAddress("again-1", DAY)
    chain.talk(address, *closed_turn(start_sequence=0))
    chain.tick(_after(2))

    later = tuple(
        replace(item, occurred_at=item.occurred_at + timedelta(hours=5))
        for item in closed_turn(start_sequence=2, prompt="明天提醒我交周报")
    )
    chain.talk(address, *later)
    assert chain.tick(_after(5.5)).released == 0  # 刚说完，还没闲置够
    assert chain.tick(_after(7)).released == 1

    assert tuple(item.segment_id for item in chain.journal.list_history(address)) == (
        "000000000000-000000000001",
        "000000000002-000000000003",
    )
    assert [job.memory_sequence for job in chain.jobs.list_for_conversation(address)] == [1, 2]


def test_one_conversation_failing_to_seal_does_not_stop_the_others_and_retries_next_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _Chain(tmp_path)
    broken, fine = ConversationAddress("a-broken", DAY), ConversationAddress("b-fine", DAY)
    chain.talk(broken, *closed_turn(start_sequence=0))
    chain.talk(fine, *closed_turn(start_sequence=0))
    original = chain.enqueuer.enqueue_ready_segments

    def failing(address, **kwargs):  # type: ignore[no-untyped-def]
        if address == broken:
            raise OSError("disk hiccup")
        return original(address, **kwargs)

    monkeypatch.setattr(chain.enqueuer, "enqueue_ready_segments", failing)
    report = chain.tick(_after(2))

    assert report.released == 1
    assert report.signals == ("1 idle conversations failed to seal: OSError",)
    assert len(chain.journal.list_history(fine)) == 1
    assert chain.journal.list_history(broken) == ()

    monkeypatch.setattr(chain.enqueuer, "enqueue_ready_segments", original)
    report = chain.tick(_after(3))
    assert report.released == 1
    assert report.signals == ()
    assert len(chain.journal.list_history(broken)) == 1


def test_a_conversation_whose_memory_delivery_is_running_is_skipped_without_failing(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)
    address = ConversationAddress("busy-1", DAY)
    chain.talk(address, *closed_turn(start_sequence=0))
    key = conversation_order_lock_key(address.conversation_id, address.started_on, ConversationSourceConsumer.MEMORY)

    with chain.path_lock.acquire(key, ttl_seconds=30):
        report = chain.tick(_after(2))
    assert (report.released, report.signals) == (0, ())
    assert chain.journal.list_history(address) == ()

    assert chain.tick(_after(3)).released == 1


def test_each_tick_looks_at_a_bounded_batch_and_rotates(tmp_path: Path) -> None:
    chain = _Chain(tmp_path, batch=2)
    addresses = [ConversationAddress(f"c-{index}", DAY) for index in range(3)]
    for address in addresses:
        chain.talk(address, *closed_turn(start_sequence=0))

    assert chain.tick(_after(2)).released == 2
    assert chain.journal.list_history(addresses[2]) == ()
    assert chain.tick(_after(2)).released == 1
    assert all(chain.journal.read_live(address) is None for address in addresses)


def test_idle_time_counts_from_the_last_message_not_the_first(tmp_path: Path) -> None:
    chain = _Chain(tmp_path, idle_seconds=300)
    address = ConversationAddress("slow-1", DAY)
    late = replace(
        message(1, ConversationMessageRole.COMPLETION, "好的。"), occurred_at=BASE_TIME + timedelta(minutes=10)
    )
    chain.talk(address, message(0, ConversationMessageRole.PROMPT, "记一下"), late)

    assert chain.tick(BASE_TIME + timedelta(minutes=12)).released == 0
    assert chain.tick(BASE_TIME + timedelta(minutes=15)).released == 1
