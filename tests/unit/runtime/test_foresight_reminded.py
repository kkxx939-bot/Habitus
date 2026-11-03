"""归约封口时盖 ``reminded`` 的查法：同一类、说出口的递送在这条记录开始之前、记录落在那条预期说中的范围里。"""

from __future__ import annotations

from dataclasses import replace
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


@pytest.mark.parametrize("bad", ["expectations", "deliveries"])
def test_a_bad_ledger_file_is_skipped_and_counted_not_raised(tmp_path, bad) -> None:
    """段 6 Q3：坏的预期 / 递送文件不让查法抛（抛的话会话 lane 归约每轮失败、那天定稿不了、结算等定稿，互等成死锁）——
    跳过、计入观测；坏的那一份里的「说了」盖不上 reminded，读得出的照常认。"""

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    ledger.record_expectation(expectation("e1"))
    ledger.record_expectation(expectation("e2", kind="s-k0003"))
    ledger.record_delivery(FORMED.date(), Delivery("e1", FORMED + timedelta(minutes=15), 1, True, "说了", 1.0, "话"))
    ledger.record_delivery(FORMED.date(), Delivery("e2", FORMED + timedelta(minutes=15), 1, True, "说了", 1.0, "话"))
    directory = tmp_path / "ledger" / bad / "2026" / "10" / "01"
    broken = sorted(directory.glob("e1*.json"))[0]
    broken.write_text('{"schema_version": 8}', encoding="utf-8")
    lookup = RemindedLookup(ledger)
    assert not lookup("s-k0002", FORMED + timedelta(minutes=30))  # 坏的那一份：盖不上
    assert lookup("s-k0003", FORMED + timedelta(minutes=30))  # 同一天读得出的照常认
    assert ledger.take_unreadable() == (str(broken),)


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


@pytest.mark.parametrize("bad", ["old_schema_delivery", "truncated_expectation"])
def test_the_reduction_keeps_landing_rows_and_closing_days_when_a_ledger_file_is_bad(tmp_path, bad) -> None:
    """段 6 Q3 端到端：真实归约 runner 接真实查法，还要看的那天有一份坏递送 / 坏预期——每轮照常落树、天照常定稿
    （原来每轮抛、树上 0 条、``closed_days`` 一直是空）。"""

    import asyncio
    import json

    from habitus.behavior.model import BehaviorKind
    from tests.unit.behavior.reduction_fixtures import at
    from tests.unit.behavior.test_reduction_runner import OBS_A, OBS_B, OBS_C, Harness, seed_wash

    harness = Harness(tmp_path)
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    formed = at(0) - timedelta(hours=1)
    item = replace(
        expectation("e1", kind="s-k0001"),
        day=formed.date(),
        formed_at=formed,
        observed_from=formed,
        window=(formed + timedelta(minutes=4), formed + timedelta(minutes=51)),
        deadline=formed + timedelta(hours=30),
    )
    ledger.record_expectation(item)
    ledger.record_delivery(item.day, Delivery("e1", formed + timedelta(minutes=5), 1, True, "说了", 1.0, "要洗手吗"))
    day = f"{item.day.year:04d}/{item.day.month:02d}/{item.day.day:02d}"
    if bad == "old_schema_delivery":
        directory = tmp_path / "ledger" / "deliveries" / day
        raw = json.loads(next(directory.iterdir()).read_text(encoding="utf-8"))
        (directory / "e0--old.json").write_text(json.dumps({**raw, "schema_version": "foresight_ledger_v8"}))
    else:
        (tmp_path / "ledger" / "expectations" / day / "e2.json").write_text("{", encoding="utf-8")
    harness.runner.attach_reminder_lookup(RemindedLookup(ledger))
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    closed = []
    for _round in range(3):
        harness.now = harness.now + timedelta(days=1)
        asyncio.run(harness.runner.run_once())
        closed.append(len(harness.runner.closed_days()))
    assert len(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1
    assert closed[-1] > closed[0] >= 1
    assert ledger.take_unreadable()
