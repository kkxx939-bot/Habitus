"""Q17：预期的 ``level`` 按定义填"形成时这本账所在的级"。

场景号（L1…L9）对应丁批测试输入的场景表。"账上的级"= 形成那一刻这本账最近一条级的变化，没有记录是 0。现场走真实的
``DayExpectationStage``、``ForesightTick``、``DeliveryStage`` 和账本；账身份先在同样摆法的另一份现场跑一遍取得。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

from habitus.foresight import NoUnsealed
from habitus.foresight.ledger import Expectation
from habitus.foresight.ledger.model import AccountLevel, UnitSetting
from habitus.runtime.foresight_days import DayExpectationStage
from tests.unit.runtime.test_foresight_delivery import DAY0, earn, expectation, stage
from tests.unit.scene.fixtures import publish
from tests.unit.stale_table.harness import BASE, CODE, DISCUSS, DOCS, TOLERANCE, ZONE, World, days_after, world

#: 现场连着五天，第六天这一代建日子预期。
GENERATION_DAY = BASE + timedelta(days=5)
#: 级的变化发生在现场开始之前，早于任何一条预期形成。
BEFORE = datetime.combine(BASE, time(3), tzinfo=ZONE)


def clock(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=ZONE)


def five_mornings(tmp: Path) -> World:
    w = world(tmp)
    w.mornings(days_after(BASE, 5))
    return w


def day_accounts(tmp: Path) -> dict[str, str]:
    """在同样摆法的另一份现场跑一遍换代，取各类日子预期的账身份。"""

    probe = five_mornings(tmp / "probe")
    probe.generation(GENERATION_DAY)
    return {item.kind_token: item.account for item in probe.expectations(GENERATION_DAY, source="日子")}


def set_level(w: World, account: str, level: int, at: datetime = BEFORE) -> None:
    w.session.ledger.record_level(AccountLevel(account, at, level))


def by_kind(items: list[Expectation]) -> dict[str, Expectation]:
    return {item.kind_token: item for item in items}


def file_of(w: World, item: Expectation) -> Path:
    (path,) = (w.session.root / "expectations").rglob(f"{item.expectation_id}.json")
    return path


# ── 换代入口 ──────────────────────────────────────────────────────────────


def test_l1_an_account_that_never_changed_level_forms_at_level_0(tmp_path: Path) -> None:
    """L1：账从没有级变化记录，换代建的日子预期 ``level`` 是 0。"""

    w = five_mornings(tmp_path / "site")
    w.generation(GENERATION_DAY)
    built = w.expectations(GENERATION_DAY, source="日子")
    assert {item.kind_token for item in built} == {DISCUSS, CODE, DOCS}
    assert [item.level for item in built] == [0, 0, 0]


def test_l2_the_generation_records_the_level_of_each_account(tmp_path: Path) -> None:
    """L2：账最近一条记录是 1，换代建的日子预期 ``level`` 是 1，盘上 JSON 也是 1；同一代别的账仍是 0。"""

    accounts = day_accounts(tmp_path)
    w = five_mornings(tmp_path / "site")
    set_level(w, accounts[DISCUSS], 1)

    w.generation(GENERATION_DAY)

    built = by_kind(w.expectations(GENERATION_DAY, source="日子"))
    assert built[DISCUSS].level == 1
    assert built[CODE].level == 0
    assert built[DOCS].level == 0
    stored = json.loads(file_of(w, built[DISCUSS]).read_text(encoding="utf-8"))
    assert stored["level"] == 1


def test_l3_an_account_back_at_level_0_forms_at_level_0(tmp_path: Path) -> None:
    """L3：账升到 1 后又降回 0，按最近一条算，``level`` 是 0。"""

    accounts = day_accounts(tmp_path)
    w = five_mornings(tmp_path / "site")
    set_level(w, accounts[DISCUSS], 1)
    set_level(w, accounts[DISCUSS], 0, BEFORE + timedelta(days=1))

    w.generation(GENERATION_DAY)

    assert by_kind(w.expectations(GENERATION_DAY, source="日子"))[DISCUSS].level == 0


# ── 预测拍入口 ────────────────────────────────────────────────────────────


def morning_with_discussion(tmp: Path) -> tuple[World, date]:
    """今天这一代已做成，今天 10:00 刚讨论完方案。"""

    w = world(tmp)
    w.mornings(days_after(BASE, 4))
    today = BASE + timedelta(days=4)
    w.generation(today)
    publish(w.site.behavior_tree, today, "讨论方案", 10, 0)
    return w, today


def test_l4_the_tick_records_the_level_on_an_event_expectation(tmp_path: Path) -> None:
    """L4：账在 1 级，预测拍建的事件预期 ``level`` 是 1；同一类的日子预期是另一本账，仍是 0。"""

    probe, today = morning_with_discussion(tmp_path / "probe")
    probe.tick(clock(today, 10, 15), deliver=False)
    (seen,) = probe.expectations(today, source="事件")

    w, _ = morning_with_discussion(tmp_path / "site")
    set_level(w, seen.account, 1)
    w.tick(clock(today, 10, 15), deliver=False)

    (event,) = w.expectations(today, source="事件")
    assert event.account == seen.account
    assert event.level == 1
    assert by_kind(w.expectations(today, source="日子"))[event.kind_token].level == 0


def every_other_day(tmp: Path) -> tuple[World, date, date]:
    """写文档隔天做、讨论方案和修改代码每天做；昨天那一代完成了，今天的还没成，所以写文档的日子预期从旧表建。"""

    w = world(tmp)
    w.mornings(days_after(BASE, 6), docs=False)
    for offset in (0, 2, 4):
        publish(w.site.behavior_tree, BASE + timedelta(days=offset), "写文档", 12, 0)
    today = BASE + timedelta(days=6)
    night = today - timedelta(days=1)
    w.table(night)
    w.mark(night)
    return w, night, today


def test_l5_the_tick_records_the_level_on_a_day_expectation_from_an_old_table(tmp_path: Path) -> None:
    """L5：账在 1 级，预测拍用旧表建的日子预期 ``level`` 是 1。"""

    probe, _, today = every_other_day(tmp_path / "probe")
    probe.tick(clock(today, 9), deliver=False)
    (seen,) = probe.expectations(today, source="日子")

    w, night, _ = every_other_day(tmp_path / "site")
    set_level(w, seen.account, 1)
    w.tick(clock(today, 9), deliver=False)

    (built,) = w.expectations(today, source="日子")
    assert (built.kind_token, built.generation) == (DOCS, night.isoformat())
    assert built.level == 1


# ── 写一次不改 ────────────────────────────────────────────────────────────


def test_l6_an_existing_expectation_is_not_rewritten_when_the_account_rises_later(tmp_path: Path) -> None:
    """L6：预期建好后账才升到 1，同一身份再算一遍：算得出、不写，文件逐字节不变，仍是 0，不报错。"""

    w = five_mornings(tmp_path / "site")
    w.generation(GENERATION_DAY)
    first = by_kind(w.expectations(GENERATION_DAY, source="日子"))[DISCUSS]
    path = file_of(w, first)
    before = path.read_bytes()
    set_level(w, first.account, 1, clock(GENERATION_DAY, 2))

    report = asyncio.run(
        DayExpectationStage(
            w.session.ledger,
            w.session.opportunities,
            w.read_side(),
            NoUnsealed(),
            lane="session",
            vocabulary_version=lambda: 7,
            recent=35,
            tolerance=TOLERANCE,
            clock=lambda: clock(GENERATION_DAY, 3),
        ).run_once()
    )

    assert report.formed.formed == 3
    assert report.formed.recorded == 0
    assert path.read_bytes() == before
    assert by_kind(w.expectations(GENERATION_DAY, source="日子"))[DISCUSS].level == 0


def test_l7_a_rise_during_delivery_shows_on_the_next_expectation_only(tmp_path: Path) -> None:
    """L7：账在当天的递送里才升到 1：当天那条预期仍是 0，次日同一本账的新预期是 1。"""

    w = world(tmp_path / "site")
    w.mornings(days_after(BASE, 6), docs=False)
    today = BASE + timedelta(days=6)
    w.generation(today)
    current = by_kind(w.expectations(today, source="日子"))[DISCUSS]
    w.earn(current)

    w.ticks(today, time(9, 45), time(10, 30))

    risen = w.session.ledger.level_of(current.account)
    assert risen is not None and risen.level == 1, "现场没有走到递送升级"
    assert by_kind(w.expectations(today, source="日子"))[DISCUSS].level == 0

    tomorrow = today + timedelta(days=1)
    w.generation(tomorrow)
    following = by_kind(w.expectations(tomorrow, source="日子"))[DISCUSS]
    assert following.account == current.account
    assert following.expectation_id != current.expectation_id
    assert following.level == 1


# ── 不该变的 ──────────────────────────────────────────────────────────────


def test_l8_a_silenced_behavior_forms_no_expectation_at_any_level(tmp_path: Path) -> None:
    """L8：账在 1 级，这个行为设了「以后别提了」：账本里没有这条预期，别的行为照建。"""

    accounts = day_accounts(tmp_path)
    w = five_mornings(tmp_path / "site")
    set_level(w, accounts[DISCUSS], 1)
    w.session.ledger.record_setting(UnitSetting(DISCUSS, BEFORE, "以后别提了"))

    w.generation(GENERATION_DAY)

    assert {item.kind_token for item in w.expectations(GENERATION_DAY, source="日子")} == {CODE, DOCS}


def test_l9_delivery_does_not_depend_on_the_expectation_level(tmp_path: Path) -> None:
    """L9：两份账本只差预期的 ``level``（0 和 1），跑同样的递送：递送记录逐项相同，账的级也相同。"""

    results = []
    for level in (0, 1):
        run, ledger = stage(tmp_path / f"level-{level}")
        account = earn(ledger)
        ledger.record_expectation(dataclasses.replace(expectation("earned", DAY0), level=level))
        ledger.record_expectation(
            dataclasses.replace(expectation("shadow", DAY0, kind="s-k0009", trigger="other"), level=level)
        )
        for minutes in (16, 41):
            asyncio.run(run.run_once(DAY0 + timedelta(minutes=minutes)))
        deliveries = sorted(
            ledger.deliveries_on(DAY0.date()), key=lambda item: (item.expectation_id, item.decided_at, item.reason)
        )
        results.append((deliveries, ledger.level_of(account)))

    assert results[0][0], "现场没有产生递送记录"
    assert results[0] == results[1]
