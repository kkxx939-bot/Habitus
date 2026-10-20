"""归约封口时盖 ``reminded`` 的查法：同一类、说出口的递送在这条记录开始之前、记录落在那条预期说中的范围里。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.ledger import Delivery, Expectation
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_reminded import RemindedLookup

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 10, 1, 9, 0, tzinfo=CST)


def expectation(identity: str, kind: str = "s-k0002") -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=FORMED.date(),
        formed_at=FORMED,
        source="事件",
        trigger="rel",
        trigger_instance="behavior://a.md",
        window=(FORMED + timedelta(minutes=10), FORMED + timedelta(minutes=40)),
        deadline=FORMED + timedelta(hours=2),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=FORMED,
    )


def test_reminded_means_a_spoken_delivery_of_that_class_before_the_occurrence_started(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    lookup = RemindedLookup(ledger)
    ledger.record_expectation(expectation("e1"))
    ledger.record_expectation(expectation("e2", kind="s-k0003"))
    ledger.record_delivery(FORMED.date(), Delivery("e1", FORMED + timedelta(minutes=15), 1, True, "说了", 1.0, "话"))
    ledger.record_delivery(FORMED.date(), Delivery("e2", FORMED + timedelta(minutes=15), 1, False, "不可达", 0.0))
    assert lookup("s-k0002", FORMED + timedelta(minutes=30))  # 说了之后开始
    assert not lookup("s-k0002", FORMED + timedelta(minutes=12))  # 说之前就开始了
    assert not lookup("s-k0002", FORMED + timedelta(hours=3))  # 过了最晚时限：不是那条预期说中的
    assert not lookup("s-k0003", FORMED + timedelta(minutes=30))  # 这一类的决定是「不可达」，没说出口
    assert not lookup("s-k0009", FORMED + timedelta(minutes=30))
    with pytest.raises(ForesightError):
        lookup("", FORMED)
    with pytest.raises(ForesightError):
        lookup("s-k0002", FORMED.replace(tzinfo=None))
    # 账本读不出来：抛，不许静默记成没提醒
    (tmp_path / "ledger" / "expectations" / "2026" / "10" / "01" / "e1.json").write_text("{", encoding="utf-8")
    lookup = RemindedLookup(ledger)  # 新的查法（读过的那天记住了：只增的记录读一遍就够）
    with pytest.raises(ForesightError, match="could not be read"):
        lookup("s-k0002", FORMED + timedelta(minutes=30))
    with pytest.raises(TypeError):
        RemindedLookup(object())  # type: ignore[arg-type]


def test_reminded_recognises_the_class_through_the_same_mapping_as_settlement(tmp_path) -> None:
    """词表把 s-k0002 拆成 s-k0020 之后：预期记的是旧编号，这条记录是新编号，结算按映射认它，提醒标记也要认。"""

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    ledger.record_expectation(expectation("e1"))
    ledger.record_delivery(FORMED.date(), Delivery("e1", FORMED + timedelta(minutes=15), 1, True, "说了", 1.0, "话"))
    split = RemindedLookup(ledger, current=lambda token: ("s-k0020",) if token == "s-k0002" else (token,))
    assert split("s-k0020", FORMED + timedelta(minutes=30))
    assert not RemindedLookup(ledger)("s-k0020", FORMED + timedelta(minutes=30))


def test_a_bad_file_someone_else_read_does_not_fail_the_reminder_check(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    ledger.record_expectation(expectation("e1"))
    other = tmp_path / "ledger" / "settlements" / "2026" / "10" / "01"
    other.mkdir(parents=True)
    (other / "bad.json").write_text("{", encoding="utf-8")
    ledger.settlements_on(FORMED.date())  # 别的读口读到了坏文件（记在共享的那一份里）
    assert not RemindedLookup(ledger)("s-k0002", FORMED + timedelta(minutes=30))
    assert len(ledger.take_unreadable()) == 1  # 共享的那一份还在，留给读到它的那一拍去报


def test_a_day_that_is_finished_leaves_the_memory_cache(tmp_path) -> None:
    """26 号清单 6.13：结完的日子不会再被问到，缓存跟着淘汰。"""

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    lookup = RemindedLookup(ledger)
    ledger.record_expectation(expectation("e1"))
    ledger.record_delivery(FORMED.date(), Delivery("e1", FORMED + timedelta(minutes=15), 1, True, "说了", 1.0, "话"))
    assert lookup("s-k0002", FORMED + timedelta(minutes=30)) and FORMED.date() in lookup._spoken
    ledger.mark_final(FORMED.date(), at=FORMED + timedelta(days=1))
    assert not lookup("s-k0002", FORMED + timedelta(minutes=30)) and lookup._spoken == {}


def test_a_finished_day_leaves_the_cache(tmp_path) -> None:
    """6.13：结完的日子不会再被问到，它的缓存淘汰掉——常驻的查法不随历史一直长（L10.14 补测：原先没有测试守着）。"""

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    lookup = RemindedLookup(ledger)
    ledger.record_expectation(expectation("e1"))
    ledger.record_delivery(FORMED.date(), Delivery("e1", FORMED + timedelta(minutes=15), 1, True, "说了", 1.0, "话"))
    assert lookup("s-k0002", FORMED + timedelta(minutes=30))
    assert FORMED.date() in lookup._spoken

    ledger.mark_final(FORMED.date(), at=FORMED + timedelta(hours=3))
    assert not lookup("s-k0002", FORMED + timedelta(minutes=30))

    assert lookup._spoken == {}
