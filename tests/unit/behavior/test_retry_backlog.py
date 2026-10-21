"""每晚重试的堆（裁定 67 第 1、8、13 条）：同一项幂等、满了拒新的、只删取出的那一批、读不出的一起列出来。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from habitus.behavior.fusion.lanes.session import SessionMessage, SessionMessageRole, SessionTurn
from habitus.behavior.fusion.lanes.session.model import decode_turn, encode_turn
from habitus.behavior.retry import (
    Admission,
    RetryBacklogStore,
    RetryCause,
    RetryError,
    RetryItem,
    RetryKind,
    retry_id,
)

AT = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def _item(key: str, *, minutes: int = 0) -> RetryItem:
    return RetryItem(
        retry_id(RetryKind.TURN, key),
        RetryKind.TURN,
        RetryCause.GIVEN_UP,
        date(2026, 10, 21),
        AT + timedelta(minutes=minutes),
        {"key": key},
    )


def test_the_same_item_goes_in_once_and_a_full_backlog_refuses_new_ones(tmp_path) -> None:
    store = RetryBacklogStore(tmp_path, max_items=2)
    assert store.put(_item("a")) is Admission.KEPT
    assert store.put(_item("a")) is Admission.PRESENT  # 同一项再进：当作已在
    assert store.put(_item("b", minutes=1)) is Admission.KEPT
    assert store.put(_item("c", minutes=2)) is Admission.FULL  # 满了：拒新的，不挤旧的
    assert store.put(_item("a")) is Admission.PRESENT  # 已在里面的照样认
    assert store.refused() == {"full": 1, "late": 0}  # 拒收要数得到（裁定 68 第 4 条）
    assert [item.payload["key"] for item in store.snapshot().items] == ["a", "b"]


def test_only_the_batch_taken_at_the_start_of_the_night_is_discarded(tmp_path) -> None:
    store = RetryBacklogStore(tmp_path, max_items=10)
    store.put(_item("b", minutes=1))
    store.put(_item("a"))
    batch = store.snapshot()
    assert [item.payload["key"] for item in batch.items] == ["a", "b"]  # 按进来的先后
    store.put(_item("late", minutes=5))  # 跑批中途新进来的
    assert store.discard(batch.ids) == 2
    assert [item.payload["key"] for item in store.snapshot().items] == ["late"]


def test_an_unreadable_file_is_listed_for_discarding(tmp_path) -> None:
    store = RetryBacklogStore(tmp_path, max_items=10)
    store.put(_item("a"))
    broken = "f" * 32
    (store.directory / f"{broken}.json").write_text("{not json")
    (store.directory / "notes.json").write_text("{}")  # 不是本堆格式的名字：不认识、不碰（评审：原来每晚删除都抛错）
    batch = store.snapshot()
    assert batch.unreadable == (broken,) and len(batch.items) == 1
    store.discard(batch.ids)
    assert store.count() == 0 and (store.directory / "notes.json").exists()


def test_an_item_too_large_is_refused(tmp_path) -> None:
    store = RetryBacklogStore(tmp_path, max_items=10, max_item_bytes=200)
    big = RetryItem(
        retry_id(RetryKind.CHAIN, "x"),
        RetryKind.CHAIN,
        RetryCause.DETERMINISTIC,
        date(2026, 10, 21),
        AT,
        {"x": "y" * 500},
    )
    assert store.put(big) is Admission.FULL
    with pytest.raises(RetryError):
        RetryItem("short", RetryKind.TURN, RetryCause.GIVEN_UP, date(2026, 10, 21), AT, {})


def test_a_session_turn_round_trips_through_its_encoding() -> None:
    at = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
    turn = SessionTurn(
        conversation_id="c1",
        start_sequence=3,
        end_sequence=5,
        instructed_at=at,
        completed_at=at + timedelta(minutes=2),
        instruction="订机票",
        messages=(
            SessionMessage(3, at, SessionMessageRole.PROMPT, "订机票"),
            SessionMessage(4, at + timedelta(minutes=1), SessionMessageRole.TOOL_CALL, "{}", "search", "t1"),
            SessionMessage(5, at + timedelta(minutes=2), SessionMessageRole.TOOL_RESULT, "ok", "search", "t1", True),
        ),
    )
    assert decode_turn(encode_turn(turn)) == turn


def test_items_are_kept_by_day_and_a_day_whose_night_ran_takes_no_more(tmp_path) -> None:
    """裁定 68 第 3 条：当天重试、当天废弃——那一天的夜批跑过之后（``advance``），属于它的再来一律不收；``snapshot`` 只取
    属于截止日及以前的。"""

    store = RetryBacklogStore(tmp_path, max_items=10)
    yesterday = _item("昨天")
    today = RetryItem(retry_id(RetryKind.TURN, "今天"), RetryKind.TURN, RetryCause.GIVEN_UP, date(2026, 10, 22), AT, {})
    store.put(yesterday)
    store.put(today)
    assert [item.item_id for item in store.snapshot(through=date(2026, 10, 21)).items] == [yesterday.item_id]
    store.advance(date(2026, 10, 21))
    store.advance(date(2026, 10, 20))  # 只往前推
    assert store.through() == date(2026, 10, 21)
    assert store.put(_item("昨天又来一轮")) is Admission.LATE
    assert store.refused() == {"full": 0, "late": 1}


def test_listing_retries_when_another_writer_changes_the_directory(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """别处正在写（临时文件来了又走）时列目录撞上"目录在枚举中变了"：重列，不让这一晚整步失败（评审实测 20000 次 224 次）。"""

    import habitus.behavior.retry.store as module
    from habitus.infrastructure.store.filesystem import DurablePathIntegrityError

    store = RetryBacklogStore(tmp_path, max_items=10)
    store.put(_item("a"))
    real = module.list_real_directory
    calls = {"n": 0}

    def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] < 3:
            raise DurablePathIntegrityError("durable directory changed during enumeration")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "list_real_directory", flaky)
    assert len(store.snapshot().items) == 1
