"""结算不重复读账（丙批处二 B1、B2、B3、B4、B13）：同一个文件没变就不再打开，结算与诊断的结果一个字节都不变。

读文件的次数在 ``JsonFiles.load`` 这一层按路径数（包一层、照常调原函数），不假设账本内部怎么留住读过的内容。
B1、B2、B3 在改动前的代码上失败（那正是要修的重复读），B4、B13 改动前后都通过。
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.foresight import Evidence, Tolerance
from habitus.foresight.diagnosis import Diagnosis
from habitus.foresight.ledger import AccountFact, Delivery, Expectation, Response, Settlement
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_files import JsonFiles
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_settlement_diagnosis import SettlementDiagnosis
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_settlement import Recorder, a_session, expectation
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

DAY = date(2026, 12, 10)
NEXT = DAY + timedelta(days=1)
HOTEL, CARD, LATE = kind_id("订酒店"), kind_id("还信用卡"), kind_id("交房租")
TOLERANCE = Tolerance(line=2 / 3)
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
#: 整份替换的可变小文件：内容会变，每次都要看，不算"重复读记录"。
WATERMARKS = frozenset({"final_through.json", "omissions_scanned.json"})


class Opened:
    """账本目录下每个文件被 ``JsonFiles.load`` 打开了几次（相对账本根的路径 → 次数），按阶段分开数。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.phase = "settlement"
        self.counts: dict[str, Counter[str]] = {"settlement": Counter(), "diagnosis": Counter()}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = JsonFiles.load
        watcher = self

        def counting(files: JsonFiles, path: Path) -> dict[str, Any]:
            watcher.saw(path)
            return original(files, path)

        monkeypatch.setattr(JsonFiles, "load", counting)

    def saw(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.is_relative_to(self.root):
            self.counts[self.phase][resolved.relative_to(self.root).as_posix()] += 1

    def reset(self) -> None:
        self.counts = {"settlement": Counter(), "diagnosis": Counter()}

    def everything(self) -> Counter[str]:
        return self.counts["settlement"] + self.counts["diagnosis"]

    def records(self) -> set[str]:
        """这一段打开过的文件，水位除外。"""

        return {path for path in self.everything() if path not in WATERMARKS}

    def under(self, top: str, *, phase: str | None = None) -> Counter[str]:
        found = self.everything() if phase is None else self.counts[phase]
        return Counter({path: count for path, count in found.items() if path.startswith(f"{top}/")})


def watch_diagnosis(opened: Opened, monkeypatch: pytest.MonkeyPatch) -> None:
    """诊断那一段单独数（实盘量级按结算、诊断两段报）。"""

    original = SettlementDiagnosis.diagnose

    def phased(self: SettlementDiagnosis, **kwargs: Any) -> Diagnosis:
        opened.phase = "diagnosis"
        try:
            return original(self, **kwargs)
        finally:
            opened.phase = "settlement"

    monkeypatch.setattr(SettlementDiagnosis, "diagnose", phased)


def read_side(site: Site) -> ForesightReadSide:
    return ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))


def settlement_stage(
    ledger: ForesightLedgerStore,
    site: Site,
    closed: Callable[[], tuple[date, ...]],
    *,
    now: Callable[[], datetime],
    tolerance: Tolerance | None = TOLERANCE,
    observer: Recorder | None = None,
) -> SettlementStage:
    return SettlementStage(
        ledger,
        read_side(site),
        closed_days=closed,
        evidence=EVIDENCE,
        observer=observer,
        clock=now,
        tolerance=tolerance,
    )


# ── 处二 B1、B2：同一天多条预期、下一代不重读 ─────────────────────────────────


def five_expectations_forty_deliveries(tmp_path: Path) -> tuple[Site, ForesightLedgerStore]:
    """一天 5 条到期的预期、40 条递送（每条 8 个决定）。"""

    site = Site(tmp_path, now=at(DAY, 23, 0))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    formed = at(DAY, 14, 7)
    for index in range(5):
        identity = f"e{index}"
        item = expectation(identity, kind_id(f"行为{index}"), formed, window_minutes=(4, 51), deadline_hours=3)
        ledger.record_expectation(item)
        for step in range(8):
            ledger.record_delivery(DAY, Delivery(identity, formed + timedelta(minutes=step), 0, False, "影子", 0.0))
    return site, ledger


@pytest.mark.parametrize("tolerance", [None, TOLERANCE], ids=["without_diagnosis", "with_diagnosis"])
def test_b1_each_delivery_file_is_opened_at_most_once_in_a_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tolerance: Tolerance | None
) -> None:
    """处二 B1：一天 5 条到期预期、40 条递送跑一次结算：这 40 个递送文件每个最多打开 1 次（改动前每条预期读一遍，是 5 次；
    带诊断再多读一遍，是 6 次）。"""

    site, ledger = five_expectations_forty_deliveries(tmp_path)
    opened = Opened(ledger.root)
    opened.install(monkeypatch)
    moment = at(NEXT, 4, 0).astimezone(UTC)
    report = settlement_stage(ledger, site, lambda: (DAY,), now=lambda: moment, tolerance=tolerance).run_once()
    assert report.settled == 5
    deliveries = opened.under("deliveries")
    assert len(deliveries) == 40
    assert max(deliveries.values()) <= 1, f"递送文件最多被打开 {max(deliveries.values())} 次"


def test_b2_the_next_generation_does_not_open_what_the_first_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """处二 B2：接着再跑一代、账上没有新文件：第二代（含诊断）不再打开第一代读过的文件（水位是可变文件，不算）。"""

    site, ledger = five_expectations_forty_deliveries(tmp_path)
    opened = Opened(ledger.root)
    opened.install(monkeypatch)
    moment = at(NEXT, 4, 0).astimezone(UTC)
    stage = settlement_stage(ledger, site, lambda: (DAY,), now=lambda: moment)
    stage.run_once()
    first = opened.records()
    assert {path.split("/")[0] for path in first} >= {"deliveries", "expectations", "settlements", "accounts"}
    opened.reset()
    stage.run_once()
    again = sorted(opened.records() & first)
    assert again == [], f"第二代重新打开了 {len(again)} 个第一代读过的文件，例如 {again[:3]}"


# ── 处二 B3：实盘量级 ─────────────────────────────────────────────────────────


#: 实盘第 81–90 代的量级：每天 27 条预期、183 条递送，一共 8 本账；诊断窗走满 30 天。
EXPECTATIONS_A_DAY, DELIVERIES_A_DAY, ACCOUNTS = 27, 183, 8
HISTORY_DAYS = 32
FIRST_GENERATION = date(2026, 12, 1)


def _plain_create(path: Path, data: bytes, *, artifact_root: Path) -> bool:
    """铺底时跳过逐文件落盘（几千个文件的 fsync 与测的东西无关）：路径与字节都是账本自己算的。"""

    del artifact_root
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


@contextmanager
def quick_seeding(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with monkeypatch.context() as patch:
        patch.setattr("habitus.runtime.foresight_files.atomic_create_bytes", _plain_create)
        patch.setattr("habitus.runtime.foresight_ledger.atomic_create_bytes", _plain_create)
        yield


def scaled_expectation(day: date, index: int) -> Expectation:
    formed = at(day, 9, 0)
    account = index % ACCOUNTS
    return replace(
        expectation(
            f"{day.isoformat()}-e{index:02d}",
            kind_id(f"行为{account}"),
            formed,
            window_minutes=(4, 51),
            deadline_hours=3,
        ),
        source="日子",
        trigger=f"rel-{account}",
    )


def seed_day(ledger: ForesightLedgerStore, day: date, *, settled: bool) -> None:
    """铺一天的账；``settled`` 的是诊断窗里已经结完的历史日子。"""

    items = [scaled_expectation(day, index) for index in range(EXPECTATIONS_A_DAY)]
    for item in items:
        ledger.record_expectation(item)
    start = at(day, 9, 0)
    for index in range(DELIVERIES_A_DAY):
        item = items[index % EXPECTATIONS_A_DAY]
        decided = start + timedelta(seconds=index)
        ledger.record_delivery(day, Delivery(item.expectation_id, decided, 0, False, "影子", 0.0))
    if not settled:
        return
    moment = at(day + timedelta(days=1), 4, 0)
    for item in items:
        ledger.append_fact(AccountFact(item.account, item.formed_at, True, 1.0, "准时", item.expectation_id))
        ledger.record_settlement(
            Settlement(item.expectation_id, item.kind_token, day, moment, "准时", "behavior://done.md")
        )
    ledger.mark_final(day, at=moment)


def test_b3_at_production_scale_a_generation_reads_only_what_is_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """处二 B3：实盘量级（第 81–90 代那样的账）：一代在结算里打开递送文件的次数从约 4,950 降到不超过 320，诊断打开文件的
    次数从约 6,970 降到不超过 260。第一代是冷启动（什么都要读一遍），从第二代起算平均。"""

    site = Site(tmp_path, now=at(FIRST_GENERATION, 23, 0))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    writer = ForesightLedgerStore(tmp_path / "ledger")  # 白天的那几拍写账的一方
    with quick_seeding(monkeypatch):
        for back in range(HISTORY_DAYS, 0, -1):
            seed_day(writer, FIRST_GENERATION - timedelta(days=back), settled=True)
    opened = Opened(ledger.root)
    opened.install(monkeypatch)
    watch_diagnosis(opened, monkeypatch)
    closed: list[date] = []
    clock = [at(FIRST_GENERATION, 4, 0)]
    stage = settlement_stage(ledger, site, lambda: tuple(closed), now=lambda: clock[0].astimezone(UTC))
    deliveries, diagnosis = [], []
    for offset in range(4):
        day = FIRST_GENERATION + timedelta(days=offset)
        with quick_seeding(monkeypatch):
            seed_day(writer, day, settled=False)
        closed.append(day)
        clock[0] = at(day + timedelta(days=1), 4, 0)
        opened.reset()
        assert stage.run_once().settled == EXPECTATIONS_A_DAY
        deliveries.append(sum(opened.under("deliveries", phase="settlement").values()))
        diagnosis.append(sum(opened.counts["diagnosis"].values()))
    settled_reads, diagnosis_reads = sum(deliveries[1:]) / 3, sum(diagnosis[1:]) / 3
    assert settled_reads <= 320 and diagnosis_reads <= 260, (
        f"每代结算打开递送 {settled_reads:.0f} 次（逐代 {deliveries}），诊断打开文件 {diagnosis_reads:.0f} 次（逐代 {diagnosis}）"
    )


# ── 处二 B4、B13：结果不变 ────────────────────────────────────────────────────


@dataclass
class Ground:
    """同一份账本的现场：准时的、说了没做的（落空）、后补的、漏报的都有。"""

    site: Site
    ledger: ForesightLedgerStore
    root: Path


def ground(tmp_path: Path) -> Ground:
    site = Site(tmp_path, now=at(DAY, 23, 0))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    a_session(site, DAY, NEXT)
    formed = at(DAY, 14, 7)
    items = (
        expectation("hotel", HOTEL, formed, window_minutes=(4, 51), deadline_hours=3),
        expectation("card", CARD, formed, window_minutes=(4, 51), deadline_hours=3),
        expectation("late", LATE, at(DAY, 9, 0), window_minutes=(4, 51), deadline_hours=3),
    )
    for item in items:
        ledger.record_expectation(replace(item, trigger_instance=f"behavior://{item.expectation_id}.md"))
    ledger.record_delivery(DAY, Delivery("card", at(DAY, 14, 20), 1, True, "说了", 1.0, "该还信用卡了"))
    ledger.record_response(DAY, Response("card", at(DAY, 14, 25), "负", "turn-1"))
    publish(site.behavior_tree, DAY, "订酒店", 14, 28, kind="订酒店")
    publish(site.behavior_tree, DAY, "订机票", 8, 0, kind="订机票")  # 没有预期盖着：漏报
    publish(site.behavior_tree, NEXT, "交房租", 9, 0, kind="交房租")  # 过了时限才做：后补
    return Ground(site, ledger, tmp_path / "ledger")


FROZEN = at(NEXT + timedelta(days=1), 4, 0).astimezone(UTC)


def snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def read_everything(ledger: ForesightLedgerStore) -> None:
    for day in (DAY, NEXT):
        ledger.expectations_on(day)
        ledger.deliveries_on(day)
        ledger.responses_on(day)
        ledger.settlements_on(day)
        ledger.rulings_on(day)
        ledger.omissions_on(day)
        ledger.unsettled_expectations(day)
    for account in ledger.accounts():
        ledger.facts_of(account)


def test_b4_a_warmed_up_instance_and_a_fresh_one_write_the_same_bytes(tmp_path: Path) -> None:
    """处二 B4：同一份账本，一次用读过的实例结算（先跑一代什么都结不了的、再把账读一遍），一次用全新实例结算：结算、事实、裁决、
    反馈、漏报文件逐字节相同。"""

    warm = ground(tmp_path / "warm")
    fresh = ground(tmp_path / "fresh")
    closed: list[date] = []
    stage = settlement_stage(warm.ledger, warm.site, lambda: tuple(closed), now=lambda: FROZEN)
    stage.run_once()
    read_everything(warm.ledger)
    closed.extend((DAY, NEXT))
    stage.run_once()
    settlement_stage(ForesightLedgerStore(fresh.root), fresh.site, lambda: (DAY, NEXT), now=lambda: FROZEN).run_once()
    written = snapshot(warm.root)
    assert written == snapshot(fresh.root)
    tops = {path.split("/")[0] for path in written}
    assert tops >= {"settlements", "rulings", "omissions", "accounts", "units"}, "现场要把这几种记录都造出来"
    assert any("/feedback/" in path for path in written) and any("/facts/" in path for path in written)


def diagnosis_values(found: Diagnosis) -> tuple[Any, ...]:
    return (
        dict(found.counts()),
        dict(found.counts(spoken_only=True)),
        found.within_budget(),
        found.within_budget(spoken_only=True),
        found.calibration,
    )


def observed(recorder: Recorder) -> dict[str, Any]:
    return {
        key: value
        for key, value in recorder.events[-1].attributes.items()
        if key.startswith("wrong_") or key in ("calibration_off", "diagnosis_failed")
    }


def test_b13_the_diagnosis_values_do_not_depend_on_what_the_instance_has_read(tmp_path: Path) -> None:
    """处二 B13：同一份账本，读过的实例跑第二代、全新实例跑第二代：观测里的 ``wrong_*``、``calibration_off`` 相同。"""

    site = ground(tmp_path / "site")
    warm_recorder, fresh_recorder = Recorder(), Recorder()
    stage = settlement_stage(site.ledger, site.site, lambda: (DAY, NEXT), now=lambda: FROZEN, observer=warm_recorder)
    stage.run_once()
    copied = tmp_path / "copied"
    shutil.copytree(site.root, copied)
    stage.run_once()
    settlement_stage(
        ForesightLedgerStore(copied), site.site, lambda: (DAY, NEXT), now=lambda: FROZEN, observer=fresh_recorder
    ).run_once()
    values = observed(warm_recorder)
    assert values == observed(fresh_recorder)
    assert "diagnosis_failed" not in values and "calibration_off" in values
    assert any(key != "wrong_within_budget" and key.startswith("wrong_") and count for key, count in values.items())


def test_b13_a_fact_written_by_another_instance_changes_the_diagnosis_of_the_warmed_up_one(tmp_path: Path) -> None:
    """处二 B13：诊断读过之后，另一个实例给说错的那条写了改判：读过的实例再诊断，结果与全新实例相同，且确实变了。"""

    site = ground(tmp_path)
    stage = settlement_stage(site.ledger, site.site, lambda: (DAY, NEXT), now=lambda: FROZEN)
    stage.run_once()
    before = diagnosis_values(stage.diagnose(now=FROZEN))
    other = ForesightLedgerStore(site.root)
    account = next(item.account for item in other.expectations_on(DAY) if item.expectation_id == "card")
    missed = next(fact for fact in other.facts_of(account) if fact.expectation_id == "card")
    other.append_fact(replace(missed, good=True, strength=1.0, kind="准时", written_at=FROZEN))
    warm = diagnosis_values(stage.diagnose(now=FROZEN))
    fresh_stage = settlement_stage(ForesightLedgerStore(site.root), site.site, lambda: (DAY, NEXT), now=lambda: FROZEN)
    assert warm == diagnosis_values(fresh_stage.diagnose(now=FROZEN))
    assert warm != before
