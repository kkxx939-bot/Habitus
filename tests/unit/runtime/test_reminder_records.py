"""提醒记录（裁定 113）：系统每决定要说一次（账里「待取」）就有一条；之后从账重算来自账的那几段，取走那一段由出口写取走记录时
传进来、重算时保留；账删了记录留着；写记录失败不让写账失败；一条记录一把锁，两个线程不丢更新，持着它不读出口。"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from habitus.foresight.ledger import Delivery, Response, Ruling, Settlement
from habitus.runtime.foresight_outlet import ReminderOutlet
from habitus.runtime.foresight_reminder_records import RecordingLedgerStore, RecordingOutletStore
from habitus.runtime.reminder_records import ReminderRecords
from tests.unit.runtime.test_foresight_outlet import (
    ASKED,
    CST,
    FORMED,
    TEXT,
    awaiting,
    claude_answer,
    claude_question,
    expectation,
    request,
    turn,
)

ZONE = CST


def wired(tmp_path):  # type: ignore[no-untyped-def]
    records = ReminderRecords(
        tmp_path / "habitus",
        lane="session",
        max_children=100,
        max_bytes=64_000,
        clock=lambda: datetime(2026, 10, 22, 12, tzinfo=UTC),
    )
    ledger = RecordingLedgerStore(
        tmp_path / "foresight" / "session",
        records=records,
        kind_name=lambda token: {"s-k0002": "写周报"}.get(token),
        zone=ZONE,  # type: ignore[arg-type]
    )
    outlet = ReminderOutlet(
        {"session": RecordingOutletStore(tmp_path / "foresight" / "session" / "outlet", ledger=ledger)},
        {"session": ledger},
        liveness=timedelta(hours=48),
        lease=timedelta(minutes=30),
        max_attempts=3,
        claim_limit=3,
        similarity=0.6,
    )
    return ledger, records, outlet


def fields(records: ReminderRecords, identity: str = "e1") -> dict:  # type: ignore[type-arg]
    stored = records.read(FORMED.date(), identity)
    assert stored is not None
    return stored.fields


def test_one_reminder_from_awaiting_to_its_late_ruling_keeps_every_stage_and_the_claims_after_settling(
    tmp_path,
) -> None:
    ledger, records, outlet = wired(tmp_path)
    item = expectation()
    path = tmp_path / "habitus" / "reminders" / "session" / "2026" / "10" / "22" / "e1.md"
    awaiting(ledger, item)  # 待取
    now = ASKED + timedelta(minutes=10)
    record = fields(records)
    assert path.is_file() and record["text"] == TEXT and record["kind_name"] == "写周报"
    assert record["decided_at"] == ASKED.isoformat() and record["claims"] is None and record["spoken"] is None
    outlet.publish("session", [request(item)], now=now)
    outlet.claim("codex", "codex-1", now, 3)  # 取走
    assert fields(records)["claims"] == {
        "attempts": 0,
        "handoffs": [{"claimed_at": now.isoformat(), "host": "codex", "ended": None}],
    }
    outlet.observe("codex-1", turn(at=now + timedelta(seconds=5)), host="codex")  # 那一轮过了没问：放回
    assert fields(records)["claims"]["attempts"] == 1
    again = now + timedelta(minutes=1)
    outlet.claim("claude-code", "claude-1", again, 3)  # 再取走
    asked_at, answered_at = again + timedelta(minutes=2), again + timedelta(minutes=3)
    outlet.observe(  # 说了，他点了"这次不用"
        "claude-1",
        turn(
            claude_question(TEXT, at=asked_at),
            claude_answer(
                f'User has answered your questions: "{TEXT}"="这次不用". You can now continue.', at=answered_at
            ),
            at=again + timedelta(seconds=3),
        ),
        host="claude-code",
    )
    record = fields(records)
    assert record["spoken"] == {"at": asked_at.isoformat(), "text": TEXT, "via": "claude-code", "turn": "toolu_1"}
    assert [(item["host"], item["ended"]) for item in record["claims"]["handoffs"]] == [
        ("codex", "released"),
        ("claude-code", "spoken"),
    ]
    # 会话一支按模型判的又写了一条回应
    ledger.record_response(item.day, Response("e1", answered_at + timedelta(minutes=5), "中", "turn-7"))
    assert [(item["polarity"], item["turn"]) for item in fields(records)["responses"]] == [
        ("负", "toolu_1"),
        ("中", "turn-7"),
    ]
    # 过了最晚时限：出口删掉取走记录，结算、后补之后取走那一段还在
    outlet.sweep(item.deadline + timedelta(minutes=1))
    assert outlet.store.read_claim("e1") is None
    ledger.record_settlement(
        Settlement("e1", item.kind_token, item.day, item.deadline + timedelta(hours=1), "过期待决")
    )
    ledger.record_ruling(
        item.day, Ruling("e1", item.deadline + timedelta(hours=2), "后补", "habitus://occurrences/x.md", 30)
    )
    record = fields(records)
    assert record["settlement"]["outcome"] == "过期待决" and record["rulings"][0]["delay_minutes"] == 30
    assert record["claims"]["attempts"] == 1 and len(record["claims"]["handoffs"]) == 2
    text = path.read_text(encoding="utf-8")
    assert "## 宿主取没取" in text and "没问出口，放回" in text and "经 claude-code 说出口" in text
    assert records.read(item.day, "e1").metadata.revision >= 7  # type: ignore[union-attr]
    # 「说了」账里是 UTC 写法，记录里统一成本地偏移
    assert record["spoken"]["at"].endswith("+08:00") and record["decided_at"].endswith("+08:00")


def test_a_claim_swept_away_at_the_deadline_is_closed_in_the_record_not_left_in_hand(tmp_path) -> None:
    """取走了、没问出口、过了最晚时限：出口直接删取走记录（不经写取走记录），记录里那一次取走要收尾，不能永远"还在手里"。"""

    ledger, records, outlet = wired(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    outlet.publish("session", [request(item)], now=now)
    outlet.claim("codex", "codex-1", now, 3)
    assert fields(records)["claims"]["handoffs"][-1]["ended"] is None
    outlet.sweep(item.deadline + timedelta(minutes=1))
    assert outlet.store.read_claim("e1") is None
    assert fields(records)["claims"] == {
        "attempts": 0,
        "handoffs": [{"claimed_at": now.isoformat(), "host": "codex", "ended": "removed"}],
    }
    assert records.failures == 0


@pytest.mark.parametrize("reason", ["已发生", "宿主没取", "依据失效", "已停用"])
def test_each_way_of_not_being_spoken_is_written_down(tmp_path, reason: str) -> None:
    ledger, records, _outlet = wired(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    later = ASKED + timedelta(minutes=30)
    ledger.record_delivery(item.day, Delivery("e1", later, 1, False, reason, 0.0))
    assert fields(records)["unspoken"] == {"at": later.isoformat(), "reason": reason}
    assert fields(records)["spoken"] is None


def test_an_expectation_never_awaiting_a_host_gets_no_record_and_its_deliveries_are_not_read(
    tmp_path, monkeypatch
) -> None:
    ledger, records, _outlet = wired(tmp_path)
    item = expectation()
    ledger.record_expectation(item)
    ledger.record_delivery(item.day, Delivery("e1", ASKED, 1, False, "等时窗末", 0.0, TEXT))
    reads: list[object] = []
    original = ledger.deliveries_on
    monkeypatch.setattr(
        ledger, "deliveries_on", lambda *args, **kwargs: reads.append(args) or original(*args, **kwargs)
    )
    ledger.record_settlement(Settlement("e1", item.kind_token, item.day, item.deadline, "过期待决"))
    assert reads == [] and not (tmp_path / "habitus" / "reminders").exists()
    assert records.failures == 0


def test_a_failing_record_write_never_fails_the_ledger_write(tmp_path, monkeypatch) -> None:
    ledger, records, _outlet = wired(tmp_path)
    item = expectation()
    ledger.record_expectation(item)

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(records.files, "write", broken)
    assert ledger.record_delivery(item.day, Delivery("e1", ASKED, 1, False, "待取", 1.0, TEXT)) is True
    assert [delivery.reason for delivery in ledger.deliveries_on(item.day)] == ["待取"]
    assert records.failures == 1 and records.read(item.day, "e1") is None


def test_a_deleted_ledger_day_leaves_the_record_as_it_was(tmp_path) -> None:
    """账约 60 天后按天删：之后任何更新都只会重算出"没有这条"，记录不写、不变空。"""

    ledger, records, _outlet = wired(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    ledger.record_settlement(Settlement("e1", item.kind_token, item.day, item.deadline, "过期待决"))
    path = tmp_path / "habitus" / "reminders" / "session" / "2026" / "10" / "22" / "e1.md"
    before = path.read_bytes()
    ledger.discard_day(item.day)
    assert ledger.expectation(item.day, "e1") is None
    ledger.follow(item.day, "e1")
    ledger.follow(item.day, "e1", start=True, claims={"attempts": 0, "handoffs": []})
    assert path.read_bytes() == before and records.failures == 0


def test_two_threads_updating_one_record_never_lose_an_update(tmp_path, monkeypatch) -> None:
    """确定的交错：线程 A（出口写取走记录，带来新的取走那一段）读完账、写回之前停住；线程 B 这时写一条回话并更新同一条记录。
    有锁时 B 等 A 写完再读旧记录与账，最后的记录里取走那一段和回话都在；没有锁时 B 先写完，A 拿它读到的旧账盖掉——回话丢了。"""

    ledger, records, _outlet = wired(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    from habitus.runtime import foresight_reminder_records as hooks

    computed, release, asked = threading.Event(), threading.Event(), threading.Event()
    original_fields, original_lock = hooks.reminder_fields, records._lock

    def slow_fields(*args, **kwargs):  # type: ignore[no-untyped-def]
        found = original_fields(*args, **kwargs)
        if threading.current_thread().name == "A":
            computed.set()
            release.wait(5)
        return found

    def lock(identity: str) -> threading.Lock:
        if threading.current_thread().name == "B":
            asked.set()
        return original_lock(identity)

    monkeypatch.setattr(hooks, "reminder_fields", slow_fields)
    monkeypatch.setattr(records, "_lock", lock)
    claims = {"attempts": 0, "handoffs": [{"claimed_at": ASKED.isoformat(), "host": "codex", "ended": None}]}
    first = threading.Thread(target=ledger.follow, args=(item.day, "e1"), kwargs={"start": True, "claims": claims}, name="A")
    first.start()
    assert computed.wait(5)
    second = threading.Thread(
        target=ledger.record_response, args=(item.day, Response("e1", ASKED + timedelta(hours=1), "正", "t1")), name="B"
    )
    second.start()
    asked.wait(2)
    release.set()
    first.join(5)
    second.join(5)
    record = fields(records)
    assert [response["polarity"] for response in record["responses"]] == ["正"] and record["claims"] == claims
    assert records.failures == 0


def test_the_record_modules_never_reach_the_outlet() -> None:
    """持着记录的锁不读出口：更新记录的两处不引用出口（出口持着自己的锁来写账，反过来拿会和它先后相反）。"""

    from pathlib import Path

    from tests.architecture.test_dependency_boundaries import imported_modules

    root = Path(__file__).resolve().parents[3] / "habitus" / "runtime"
    for name in ("reminder_records.py", "foresight_reminder_records.py", "record_files.py"):
        assert "habitus.runtime.foresight_outlet" not in imported_modules(root / name), name
