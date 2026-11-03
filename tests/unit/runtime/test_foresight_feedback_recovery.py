"""裁定 91 第 3 条①的回归（由 ``37-材料/readme/repro-91`` 的复现脚本改成，断言已翻转）：提醒反馈的坏笔要能恢复。

提醒反馈读数的遗忘时钟按这个行为单元历史里的每一次机会推（做了的也算，``foresight/feedback.py:feedback_reading``）；原来只有
"没来"的那几刻推，他照常天天做时名次不动，几笔坏就一直压着单元。走真实链路（同 ``test_foresight_streak.py`` 第 2 节）：真实
行为树、每晚建机会表、``HistoryStage`` 按当晚的表重标单元历史、白天每 15 分钟一拍 ``ForesightTick`` + ``DeliveryStage``、每晚
按组合根接法 ``SettlementStage``（带 ``opportunities``、``tolerance``）。档不种，从影子自己升上来。出口替身：宿主一直活着
（``Outlet``）；"问出口"照 ``ReminderOutlet._record_spoken`` 追加一条「说了」（09:30），"回这次不用"照 ``_record_response``
追加一条 ``Response(…, "负", call_id)``（09:31）。

场景：第 0–39 天天天做（档自己升上来）；第 40 起连着 N 天（N = 2 / 3）被问、都回"这次不用"、没做；之后恢复天天做 20 天；
其中第 14 天（真漏那天）没做、也没人问——那天要进「待取」。对照：同样的漏法，但前 N 天宿主没问出口（没有提醒反馈）。
按纯函数（H = 8、M = 35、θ = 0.67）两笔坏约 4 次机会、三笔坏约 8 次机会回到线上；单元历史那一刻要等那天定稿才进来，
所以链路上回到提醒档比纯函数各晚一两天。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

from habitus.foresight.ledger import SPOKEN, Delivery, Response
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_history import HistoryStage
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import build_timelines
from habitus.series.reader import read_series
from tests.unit.runtime.test_foresight_daily_miss_and_silent_day import (
    CONFIG,
    HABIT,
    OTHER,
    TICK,
    World,
    interval_items,
    morning,
    reasons,
    settle,
)
from tests.unit.runtime.test_foresight_daily_miss_and_silent_day import nightly as plain_nightly
from tests.unit.runtime.test_foresight_delivery import EVIDENCE, LOSS, Outlet
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import Site, at, publish

START = date(2026, 10, 2)
TRAIN = 40  # 第 0–39 天天天做
RECOVER = 20  # 之后恢复天天做的天数
REAL_MISS = 14  # 恢复后第几天真漏


def _live(tmp: Path) -> tuple[World, HistoryStage]:
    site = Site(tmp, now=at(START, 3, 0))
    root = tmp / "foresight" / "session"
    ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    store = HistoryStore(root)
    delivery = DeliveryStage(
        ledger,
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=3,
        tick_seconds=TICK.total_seconds(),
        outlet=Outlet(alive=True),
        history=store,
    )
    stage = HistoryStage(ledger, store, evidence=EVIDENCE, statuses=("established",))
    return World(site, ledger, opportunities, delivery, ConceptSet([base(HABIT)])), stage


def _nightly(w: World, stage: HistoryStage, night: date) -> None:
    """夜批同序：建机会表之后按当晚的表与时间线重标单元历史（``runtime.foresight_nightly``）。"""

    plain_nightly(w, night)
    table = w.opportunities.latest("session")
    if table is None:
        return
    lines = build_timelines(read_series(w.site.behavior_tree, cutoff=night), w.concepts, {}, CONFIG)
    stage.run(night=night, table=table, line=lines["session"])


def _ask_and_decline(w: World, day: date) -> bool:
    """今天的「待取」被问出口（09:30「说了」）、他点了"这次不用"（09:31 回话 负）。返回问没问。"""

    asked = False
    for open_day in w.ledger.open_days():
        made = w.ledger.deliveries_on(open_day)
        for item in made:
            if item.reason == "待取" and not any(x.spoken for x in made if x.expectation_id == item.expectation_id):
                zone = item.decided_at.tzinfo
                spoken_at = datetime.combine(day, time(9, 30), tzinfo=zone)
                w.ledger.record_delivery(
                    open_day,
                    Delivery(
                        item.expectation_id,
                        spoken_at,
                        item.level,
                        True,
                        SPOKEN,
                        1.0,
                        "该吃维生素了",
                        via="codex",
                        turn=f"call-{day}",
                    ),
                )
                w.ledger.record_response(
                    open_day, Response(item.expectation_id, spoken_at + timedelta(minutes=1), "负", f"call-{day}")
                )
                asked = True
    return asked


def _state(w: World, day: date) -> dict[str, object]:
    unit = w.ledger.unit_level_of(w.kind)
    table = w.opportunities.latest("session")
    fb = w.delivery._feedback(w.kind, table)  # noqa: SLF001 — 递送那一拍读的同一个数
    items = interval_items(w, day)
    return {
        "day": day.isoformat(),
        "reasons": reasons(w, day, items[0].expectation_id) if items else [],
        "unit_level": unit.level if unit is not None else 0,
        "feedback": None
        if fb is None
        else {"k": round(fb.k, 3), "n": round(fb.n, 3), "upper": round(fb.upper, 3), "lower": round(fb.lower, 3)},
        "feedback_rows": [(f.at.date().isoformat(), f.good, f.polarity) for f in w.ledger.feedback_of(w.kind)],
    }


def _run(tmp: Path, declines: int, *, ask: bool) -> tuple[World, list[dict[str, object]], dict[str, int]]:
    w, stage = _live(tmp)
    plan = [True] * TRAIN + [False] * declines + [index != REAL_MISS for index in range(RECOVER)]
    rows: list[dict[str, object]] = []
    marks = {"first_decline": TRAIN, "back": TRAIN + declines, "real_miss": TRAIN + declines + REAL_MISS}
    for index, did in enumerate(plan):
        day = START + timedelta(days=index)
        if index >= 2:
            settle(w, day - timedelta(days=1))
        _nightly(w, stage, day)
        if index >= 2:
            morning(w, day, done=did)
            if ask and TRAIN <= index < TRAIN + declines:
                assert _ask_and_decline(w, day)
            if index >= TRAIN - 2:
                rows.append({"index": index, **_state(w, day)})
        elif did:
            publish(w.site.behavior_tree, day, HABIT, 8, 30)
        publish(w.site.behavior_tree, day, OTHER, 15, 0)
    settle(w, START + timedelta(days=len(plan)))
    return w, rows, marks


def _reminded(row: dict[str, object]) -> bool:
    return "待取" in row["reasons"]  # type: ignore[operator]


def _row(rows: list[dict[str, object]], index: int) -> dict[str, object]:
    return next(row for row in rows if row["index"] == index)


def _quiet_days(rows: list[dict[str, object]], marks: dict[str, int]) -> list[int]:
    return [
        int(r["index"]) for r in rows if marks["back"] <= int(r["index"]) < marks["real_miss"] and r["unit_level"] == 0
    ]  # type: ignore[call-overload]


def test_two_declines_are_forgotten_while_he_keeps_doing_it(tmp_path: Path) -> None:
    _, rows, marks = _run(tmp_path, 2, ask=True)
    assert _row(rows, TRAIN - 1)["unit_level"] == 1  # 升档了
    assert all(_reminded(_row(rows, marks["first_decline"] + i)) for i in range(2))  # 两天都被问
    # 两笔坏把单元压回影子（扣分），他照常做的每一次机会都把它们往后推：第 4 次推完上界回到线上（0.663 → 0.688），
    # 第二天单元回提醒档。改前读数一动不动，影子一直到真漏那天之后
    quiet = _quiet_days(rows, marks)
    assert quiet == list(range(marks["back"] + 1, marks["back"] + 6))
    uppers = [_row(rows, i)["feedback"]["upper"] for i in range(marks["back"] + 1, marks["back"] + 7)]  # type: ignore[index]
    assert uppers == sorted(uppers) and len(set(uppers)) == len(uppers)  # 每天都在恢复，不是停住
    miss = _row(rows, marks["real_miss"])
    assert _reminded(miss) and miss["unit_level"] == 1
    assert miss["feedback"]["upper"] >= LOSS.threshold  # type: ignore[index]
    assert miss["feedback_rows"] == [
        (f"{_row(rows, marks['first_decline'] + i)['day']}", False, "负") for i in range(2)
    ]


def test_three_declines_take_longer_but_still_come_back(tmp_path: Path) -> None:
    _, rows, marks = _run(tmp_path, 3, ask=True)
    quiet = _quiet_days(rows, marks)
    # 三笔坏：约 8 次机会才回到线上（比两笔晚），之后照常开口
    assert quiet == list(range(marks["back"], marks["back"] + 10))
    miss = _row(rows, marks["real_miss"])
    assert _reminded(miss) and miss["unit_level"] == 1
    assert miss["feedback"]["upper"] >= LOSS.threshold  # type: ignore[index]


def test_control_same_misses_without_asking_reminds_on_the_real_miss(tmp_path: Path) -> None:
    """对照：前 2 天同样没做、只记「待取」没问出口（没有提醒反馈）：单元一直在提醒档，真漏那天照常进「待取」。"""

    _, rows, marks = _run(tmp_path, 2, ask=False)
    miss = _row(rows, marks["real_miss"])
    assert miss["feedback"] is None
    assert _reminded(miss)
    assert _quiet_days(rows, marks) == []
