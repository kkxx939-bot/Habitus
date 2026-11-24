"""账本读到的永远是盘上此刻的样子（丙批处二 B5–B12）：不重复读文件是提速，不许因此读到旧内容。

两个 ``ForesightLedgerStore`` 实例开在同一个目录上，模拟两个进程：``reader`` 先读过（该留的都留下了），``writer`` 或文件操作
再改盘，``reader`` 下一次读必须看到改动。这些场景在改动前的代码上都通过，守的是改动之后不退化。
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.config.foresight import ForesightStoreConfig
from habitus.foresight import Evidence
from habitus.foresight.ledger import (
    AccountFact,
    AccountLevel,
    Delivery,
    Expectation,
    Omission,
    Response,
    Ruling,
    Settlement,
    UnitFeedback,
    UnitLevel,
    UnitSetting,
    account_key,
)
from habitus.foresight.ledger.compaction import AccountCompaction
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_files import JsonFiles, LedgerFiles
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import SettlementStage
from tests.unit.kind_ids import kind_id
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 12, 10, 14, 7, tzinfo=CST)
DAY = FORMED.date()
KIND = "s-k0002"
ACCOUNT = account_key(KIND, "日子", "rel")


def expectation(identity: str = "e1", *, hits: int = 10) -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=KIND,
        day=DAY,
        formed_at=FORMED,
        source="日子",
        trigger="rel",
        trigger_instance=f"occ:{identity}.md",
        window=(FORMED + timedelta(minutes=4), FORMED + timedelta(minutes=51)),
        deadline=FORMED + timedelta(hours=3),
        basis=(),
        opportunities=13,
        hits=hits,
        observed_from=FORMED,
    )


def delivery(identity: str = "e1", *, text: str = "甲乙丙丁", minute: int = 1) -> Delivery:
    return Delivery(identity, FORMED + timedelta(minutes=minute), 1, True, "说了", 1.0, text)


def settlement(identity: str = "e1", *, uri: str = "occ:hotel-a.md") -> Settlement:
    return Settlement(identity, KIND, DAY, FORMED + timedelta(days=1), "准时", uri)


@dataclass(frozen=True)
class Shape:
    """一种按目录读的记录：两份同名不同内容的样本、怎么写、怎么读回。"""

    name: str
    first: Any
    second: Any
    write: Callable[[ForesightLedgerStore, Any], Any]
    read: Callable[[ForesightLedgerStore], tuple[Any, ...]]


def _one(found: Any) -> tuple[Any, ...]:
    return () if found is None else (found,)


SHAPES = (
    Shape(
        "expectation",
        expectation(hits=10),
        expectation(hits=11),
        lambda store, item: store.record_expectation(item),
        lambda store: store.expectations_on(DAY),
    ),
    Shape(
        "delivery",
        delivery(text="甲乙丙丁"),
        delivery(text="戊己庚辛"),
        lambda store, item: store.record_delivery(DAY, item),
        lambda store: store.deliveries_on(DAY),
    ),
    Shape(
        "response",
        Response("e1", FORMED + timedelta(minutes=2), "正", "turn-1"),
        Response("e1", FORMED + timedelta(minutes=2), "负", "turn-1"),
        lambda store, item: store.record_response(DAY, item),
        lambda store: store.responses_on(DAY),
    ),
    Shape(
        "settlement",
        settlement(uri="occ:hotel-a.md"),
        settlement(uri="occ:hotel-b.md"),
        lambda store, item: store.record_settlement(item),
        lambda store: store.settlements_on(DAY),
    ),
    Shape(
        "ruling",
        Ruling("e1", FORMED + timedelta(days=2), "后补", "occ:late.md", 90),
        Ruling("e1", FORMED + timedelta(days=2), "后补", "occ:late.md", 91),
        lambda store, item: store.record_ruling(DAY, item),
        lambda store: store.rulings_on(DAY),
    ),
    Shape(
        "omission",
        Omission("occ:other.md", KIND, DAY, FORMED + timedelta(days=1), False),
        Omission("occ:other.md", KIND, DAY, FORMED + timedelta(days=1), True),
        lambda store, item: store.record_omission(item),
        lambda store: store.omissions_on(DAY),
    ),
    Shape(
        "fact",
        AccountFact(ACCOUNT, FORMED, True, 1.0, "准时", "e1"),
        AccountFact(ACCOUNT, FORMED, True, 0.5, "准时", "e1"),
        lambda store, item: store.append_fact(item),
        lambda store: store.facts_of(ACCOUNT),
    ),
    Shape(
        "account_level",
        AccountLevel(ACCOUNT, FORMED, 1),
        AccountLevel(ACCOUNT, FORMED, 0),
        lambda store, item: store.record_level(item),
        lambda store: store.levels_of(ACCOUNT),
    ),
    Shape(
        "latest_account_level",
        AccountLevel(ACCOUNT, FORMED, 1),
        AccountLevel(ACCOUNT, FORMED, 0),
        lambda store, item: store.record_level(item),
        lambda store: _one(store.level_of(ACCOUNT)),
    ),
    Shape(
        "unit_level",
        UnitLevel(KIND, FORMED, 1),
        UnitLevel(KIND, FORMED, 0),
        lambda store, item: store.record_unit_level(item),
        lambda store: _one(store.unit_level_of(KIND)),
    ),
    Shape(
        "setting",
        UnitSetting(KIND, FORMED, "以后别提了"),
        UnitSetting(KIND, FORMED, "照常"),
        lambda store, item: store.record_setting(item),
        lambda store: store.settings_of(KIND),
    ),
    Shape(
        "latest_setting",
        UnitSetting(KIND, FORMED, "以后别提了"),
        UnitSetting(KIND, FORMED, "照常"),
        lambda store, item: store.record_setting(item),
        lambda store: _one(store.setting_of(KIND)),
    ),
    Shape(
        "feedback",
        UnitFeedback(KIND, FORMED, True, "e1"),
        UnitFeedback(KIND, FORMED, False, "e1"),
        lambda store, item: store.record_feedback(item),
        lambda store: store.feedback_of(KIND),
    ),
)
BY_NAME = [shape.name for shape in SHAPES]


def stores(tmp_path: Path) -> tuple[ForesightLedgerStore, ForesightLedgerStore]:
    """同一个目录上的两个实例：两个进程。"""

    return ForesightLedgerStore(tmp_path / "ledger"), ForesightLedgerStore(tmp_path / "ledger")


def files_of(root: Path) -> set[Path]:
    return {path for path in root.rglob("*.json") if path.is_file()}


def write_and_locate(store: ForesightLedgerStore, shape: Shape, item: Any) -> Path:
    """写一条，返回这一条落在哪个文件（写之前没有、写之后多出来的、不是账的 ``account.json``）。"""

    before = files_of(store.root)
    shape.write(store, item)
    (path,) = {
        found
        for found in files_of(store.root) - before
        if found.name != "account.json" and found.parent.name != "expectations"  # 账的说明与预期的索引不是这一条本身
    }
    return path


def bytes_of(tmp_path: Path, shape: Shape, item: Any) -> bytes:
    """这一条规范编码之后的字节（写进一个旁边的账本取回来，不自己拼）。"""

    return write_and_locate(ForesightLedgerStore(tmp_path / "scratch" / shape.name), shape, item).read_bytes()


def swap(path: Path, payload: bytes, *, keep_times: bool = True) -> None:
    """原子替换成另一份内容：先写同目录临时文件再改名（``keep_times`` 让修改时刻与原来一字不差，大小相同时只有内容变了）。"""

    held = path.stat()
    temporary = path.with_name(f".{path.name}.swap.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    if keep_times:
        os.utime(path, ns=(held.st_atime_ns, held.st_mtime_ns))


# ── 处二 B5：结算到一半有新递送 ───────────────────────────────────────────────


def test_b5_a_delivery_written_after_a_read_is_seen_by_the_next_read(tmp_path: Path) -> None:
    """处二 B5：读过这一天之后，另一个实例写入一条「说了」，同一个实例再读这一天读得到。"""

    reader, writer = stores(tmp_path)
    reader.record_expectation(expectation())
    assert reader.deliveries_on(DAY) == ()
    spoken = delivery()
    writer.record_delivery(DAY, spoken)
    assert reader.deliveries_on(DAY) == (spoken,)


def test_b5_a_delivery_written_halfway_through_a_settlement_counts_for_the_expectations_after_it(
    tmp_path: Path,
) -> None:
    """处二 B5：真实结算走到一半（第一条结完），另一个实例给第二条写入「说了」：第二条按说过结——和现在每条现读一样，
    一次结算里不许拿开头读到的那一份递送结到底。"""

    site = Site(tmp_path, now=at(DAY, 23, 0))
    reader, writer = stores(tmp_path)
    hotel = kind_id("订酒店")
    for identity in ("a-first", "b-second"):
        reader.record_expectation(replace(expectation(identity), kind_token=hotel))
    publish(site.behavior_tree, DAY, "订酒店", 14, 30, kind="订酒店")
    spoken = Delivery("b-second", at(DAY, 14, 20), 1, True, "说了", 1.0, "要订酒店吗")
    record = reader.record_settlement

    def halfway(item: Settlement) -> bool:
        written = record(item)
        if item.expectation_id == "a-first":
            writer.record_delivery(DAY, spoken)
        return written

    reader.record_settlement = halfway  # type: ignore[method-assign]
    SettlementStage(
        reader,
        ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai")),
        closed_days=lambda: (DAY,),
        evidence=Evidence(alpha=0.05, half_life=8.0),
        clock=lambda: at(DAY + timedelta(days=1), 4, 0).astimezone(UTC),
    ).run_once()
    outcomes = {item.expectation_id: item.outcome for item in writer.settlements_on(DAY)}
    assert outcomes == {"a-first": "准时", "b-second": "提醒后发生"}


# ── 处二 B6：别的进程写 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", SHAPES, ids=BY_NAME)
def test_b6_a_record_another_instance_writes_is_read_next_time(tmp_path: Path, shape: Shape) -> None:
    """处二 B6：这个实例读过（是空的），另一个实例写了一条，下一次读到。"""

    reader, writer = stores(tmp_path)
    assert shape.read(reader) == ()
    shape.write(writer, shape.first)
    assert shape.read(reader) == (shape.first,)


def test_b6_a_settlement_another_instance_writes_takes_the_expectation_out_of_the_unsettled(tmp_path: Path) -> None:
    """处二 B6：另一个实例写了新结算，这个实例下一次问"这一天还有哪些没结"就不再有它。"""

    reader, writer = stores(tmp_path)
    item = expectation()
    reader.record_expectation(item)
    assert reader.unsettled_expectations(DAY) == (item,)
    writer.record_settlement(settlement())
    assert reader.unsettled_expectations(DAY) == ()


def test_b6_a_day_another_instance_opens_appears_among_the_open_days(tmp_path: Path) -> None:
    """处二 B6：另一个实例在新的一天写了预期，这个实例的 ``open_days`` 里马上有这一天。"""

    reader, writer = stores(tmp_path)
    reader.record_expectation(expectation())
    assert reader.open_days() == (DAY,)
    later = DAY + timedelta(days=1)
    shift = timedelta(days=1)
    writer.record_expectation(
        replace(
            expectation("e2"),
            day=later,
            formed_at=FORMED + shift,
            window=(FORMED + shift + timedelta(minutes=4), FORMED + shift + timedelta(minutes=51)),
            deadline=FORMED + shift + timedelta(hours=3),
            observed_from=FORMED + shift,
        )
    )
    assert reader.open_days() == (DAY, later)
    assert [item.expectation_id for item in reader.expectations_on(later)] == ["e2"]


def test_b6_a_final_mark_another_instance_writes_closes_the_day(tmp_path: Path) -> None:
    """处二 B6：另一个实例标了这一天结完，这个实例的 ``open_days`` 里不再有它。"""

    reader, writer = stores(tmp_path)
    reader.record_expectation(expectation())
    assert reader.open_days() == (DAY,)
    writer.mark_final(DAY, at=FORMED + timedelta(days=1))
    assert reader.open_days() == () and reader.is_final(DAY)


def test_b6_watermarks_another_instance_moves_are_read_next_time(tmp_path: Path) -> None:
    """处二 B6：水位是整份替换的可变文件：另一个实例推进之后，这个实例读到的是新水位。"""

    reader, writer = stores(tmp_path)
    reader.record_expectation(expectation())
    assert reader.final_through() is None
    writer.mark_final(DAY, at=FORMED + timedelta(days=1))
    assert reader.final_through() == DAY
    assert reader.scan_watermark(start=DAY).through == DAY - timedelta(days=1)
    writer.mark_scanned(DAY, start=DAY)
    assert reader.scan_watermark(start=DAY).through == DAY


def test_b6_a_compaction_another_instance_writes_hides_the_facts_it_covers(tmp_path: Path) -> None:
    """处二 B6：压实摘要写在事实目录之外：另一个实例写了摘要，这个实例读事实时压实盖住的那几笔不再算，事实目录本身一个文件都没变。"""

    reader, writer = stores(tmp_path)
    covered = AccountFact(ACCOUNT, FORMED, True, 1.0, "准时", "e1")
    kept = AccountFact(ACCOUNT, FORMED + timedelta(days=1), True, 1.0, "准时", "e2")
    reader.append_fact(covered)
    reader.append_fact(kept)
    assert set(reader.facts_of(ACCOUNT)) == {covered, kept}
    writer.write_compaction(AccountCompaction(ACCOUNT, (FORMED, "e1"), (), (), None, ()))
    assert reader.facts_of(ACCOUNT) == (kept,)


# ── 处二 B7：同名换内容 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", SHAPES, ids=BY_NAME)
def test_b7_a_file_atomically_replaced_by_different_content_is_read_anew(tmp_path: Path, shape: Shape) -> None:
    """处二 B7：同名文件被原子替换成内容不同的一份（修改时刻保持原样），读到新内容。"""

    reader, _writer = stores(tmp_path)
    path = write_and_locate(reader, shape, shape.first)
    assert shape.read(reader) == shape.read(reader) == (shape.first,)
    swap(path, bytes_of(tmp_path, shape, shape.second))
    assert shape.read(reader) == (shape.second,)


def test_b7_same_size_and_same_modified_time_still_reads_the_new_content(tmp_path: Path) -> None:
    """处二 B7：最难认的一种：换上去的那份字节数相同、修改时刻也相同，只有内容不同。"""

    shape = next(item for item in SHAPES if item.name == "delivery")
    reader, _writer = stores(tmp_path)
    path = write_and_locate(reader, shape, shape.first)
    payload = bytes_of(tmp_path, shape, shape.second)
    assert len(payload) == path.stat().st_size and payload != path.read_bytes()
    assert reader.deliveries_on(DAY) == (shape.first,)
    held = path.stat().st_mtime_ns
    swap(path, payload)
    assert path.stat().st_mtime_ns == held
    assert reader.deliveries_on(DAY) == (shape.second,)


def test_b7_a_file_rewritten_in_place_with_the_same_size_is_read_anew(tmp_path: Path) -> None:
    """处二 B7：同一个文件被就地改写成大小相同、内容不同（inode 和大小都不变，修改时刻变了）：下一次读到新内容。"""

    shape = next(item for item in SHAPES if item.name == "delivery")
    reader, _writer = stores(tmp_path)
    path = write_and_locate(reader, shape, shape.first)
    payload = bytes_of(tmp_path, shape, shape.second)
    assert reader.deliveries_on(DAY) == (shape.first,)
    held = path.stat()
    with path.open("r+b") as handle:
        handle.write(payload)
    os.utime(path, ns=(held.st_atime_ns, held.st_mtime_ns + 1_000_000_000))
    now = path.stat()
    assert (now.st_ino, now.st_size) == (held.st_ino, held.st_size) and now.st_mtime_ns != held.st_mtime_ns
    assert reader.deliveries_on(DAY) == (shape.second,)


def test_b7_a_replaced_expectation_is_seen_by_the_point_read_and_the_unsettled_list(tmp_path: Path) -> None:
    """处二 B7：预期被换了内容：按日读、点读、"没结的"三个读口都是新内容。"""

    shape = next(item for item in SHAPES if item.name == "expectation")
    reader, _writer = stores(tmp_path)
    path = write_and_locate(reader, shape, shape.first)
    assert reader.expectation(DAY, "e1") == shape.first and reader.unsettled_expectations(DAY) == (shape.first,)
    swap(path, bytes_of(tmp_path, shape, shape.second))
    assert reader.expectation(DAY, "e1") == shape.second
    assert reader.unsettled_expectations(DAY) == (shape.second,)
    assert reader.expectations_on(DAY) == (shape.second,)


# ── 处二 B8：文件被删 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", SHAPES, ids=BY_NAME)
def test_b8_a_file_another_process_deletes_is_no_longer_read(tmp_path: Path, shape: Shape) -> None:
    """处二 B8：另一个进程删了文件，不再读到。"""

    reader, _writer = stores(tmp_path)
    path = write_and_locate(reader, shape, shape.first)
    assert shape.read(reader) == (shape.first,)
    path.unlink()
    assert shape.read(reader) == ()
    assert reader.take_unreadable() == ()


@pytest.mark.parametrize("released_by", ["same", "other"])
def test_b8_a_released_day_is_no_longer_read(tmp_path: Path, released_by: str) -> None:
    """处二 B8：整天释放（``discard_day``，这个实例自己或另一个实例做的）之后，这一天什么都读不到。"""

    reader, writer = stores(tmp_path)
    reader.record_expectation(expectation())
    reader.record_delivery(DAY, delivery())
    reader.record_settlement(settlement())
    assert reader.expectations_on(DAY) and reader.deliveries_on(DAY) and reader.settlements_on(DAY)
    (reader if released_by == "same" else writer).discard_day(DAY)
    assert reader.expectations_on(DAY) == () and reader.deliveries_on(DAY) == () and reader.settlements_on(DAY) == ()
    assert reader.unsettled_expectations(DAY) == () and reader.open_days() == ()
    assert reader.day_of("e1") is None and not reader.has_expectation("e1")


def test_b8_a_released_day_written_again_reads_the_new_records(tmp_path: Path) -> None:
    """处二 B8：同名的文件释放之后重新写成另一份内容，读到的是新写的，不是释放前的。"""

    reader, writer = stores(tmp_path)
    reader.record_expectation(expectation(hits=10))
    reader.record_delivery(DAY, delivery(text="甲乙丙丁"))
    assert reader.deliveries_on(DAY) == (delivery(text="甲乙丙丁"),)
    writer.discard_day(DAY)
    writer.record_expectation(expectation(hits=11))
    writer.record_delivery(DAY, delivery(text="戊己庚辛"))
    assert reader.expectations_on(DAY) == (expectation(hits=11),)
    assert reader.deliveries_on(DAY) == (delivery(text="戊己庚辛"),)


# ── 处二 B9：坏文件 ───────────────────────────────────────────────────────────


BAD = {
    "half": b'{"record": "deliv',
    "not_object": b"[]\n",
    "wrong_shape": b'{"record": "delivery", "schema_version": "foresight_ledger_v5"}\n',
}


@pytest.mark.parametrize("how", sorted(BAD))
def test_b9_a_bad_file_is_skipped_and_reported_on_every_read(tmp_path: Path, how: str) -> None:
    """处二 B9：一份递送坏了：每次读都跳过它、每次都记进 ``take_unreadable``，好的那份照常读到（不能只有第一次读报）。"""

    reader, _writer = stores(tmp_path)
    good = delivery("e2")
    reader.record_delivery(DAY, delivery("e1"))
    reader.record_delivery(DAY, good)
    path = next(found for found in files_of(reader.root) if found.name.startswith("e1--"))
    path.write_bytes(BAD[how])
    for _round in range(3):
        assert reader.deliveries_on(DAY) == (good,)
        assert reader.take_unreadable() == (str(path),)
    assert reader.take_unreadable() == ()


def test_b9_reading_twice_before_taking_reports_the_bad_file_once(tmp_path: Path) -> None:
    """处二 B9：两次读之间没取走：同一路径只记一条，条数与现在相同。"""

    reader, _writer = stores(tmp_path)
    reader.record_delivery(DAY, delivery())
    (path,) = [found for found in files_of(reader.root)]
    path.write_bytes(BAD["half"])
    assert reader.deliveries_on(DAY) == reader.deliveries_on(DAY) == ()
    assert reader.take_unreadable() == (str(path),)


def test_b9_a_callers_own_list_gets_the_bad_file_on_every_read(tmp_path: Path) -> None:
    """处二 B9：调用方传自己的列表：每次读都记进这一次的列表，不经共享的那一份。"""

    reader, _writer = stores(tmp_path)
    reader.record_delivery(DAY, delivery())
    (path,) = [found for found in files_of(reader.root)]
    path.write_bytes(BAD["half"])
    for _round in range(2):
        mine: list[str] = []
        assert reader.deliveries_on(DAY, unreadable=mine) == ()
        assert mine == [str(path)]
    assert reader.take_unreadable() == ()


def test_b9_a_bad_settlement_file_still_counts_as_settled_on_every_read(tmp_path: Path) -> None:
    """处二 B9：结算文件坏了：这一条算结过（不重结），每次问都一样，每次都报坏文件。"""

    reader, _writer = stores(tmp_path)
    reader.record_expectation(expectation())
    reader.record_settlement(settlement())
    path = next(found for found in files_of(reader.root / "settlements"))
    path.write_bytes(BAD["half"])
    for _round in range(2):
        assert reader.unsettled_expectations(DAY) == ()
        assert reader.take_unreadable() == (str(path),)


# ── 处二 B10：坏文件修好 / 好文件变坏 ─────────────────────────────────────────


@pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "in_place"])
@pytest.mark.parametrize("shape", SHAPES, ids=BY_NAME)
def test_b10_a_repaired_file_is_read_again(tmp_path: Path, shape: Shape, atomic: bool) -> None:
    """处二 B10：坏文件修好（原子替换或就地写）：修好后读到，不再报坏文件。"""

    reader, _writer = stores(tmp_path)
    path = write_and_locate(reader, shape, shape.first)
    good = path.read_bytes()
    path.write_bytes(BAD["half"])
    assert shape.read(reader) == ()
    assert reader.take_unreadable() == (str(path),)
    if atomic:
        swap(path, good, keep_times=False)
    else:
        path.write_bytes(good)
    assert shape.read(reader) == (shape.first,)
    assert reader.take_unreadable() == ()


@pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "in_place"])
@pytest.mark.parametrize("shape", SHAPES, ids=BY_NAME)
def test_b10_a_file_gone_bad_is_skipped_and_reported_not_served_from_memory(
    tmp_path: Path, shape: Shape, atomic: bool
) -> None:
    """处二 B10：好文件变坏（原子替换或就地写）：跳过并记下，不给读过的旧内容。"""

    reader, _writer = stores(tmp_path)
    path = write_and_locate(reader, shape, shape.first)
    assert shape.read(reader) == (shape.first,)
    if atomic:
        swap(path, BAD["half"], keep_times=False)
    else:
        path.write_bytes(BAD["half"])
    assert shape.read(reader) == ()
    assert reader.take_unreadable() == (str(path),)


# ── 处二 B11：重启 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", SHAPES, ids=BY_NAME)
def test_b11_a_new_instance_reads_what_is_on_the_disk(tmp_path: Path, shape: Shape) -> None:
    """处二 B11：重启（新建账本实例）：读到的与盘上一致，包括重启前换过内容的那一份。"""

    before, _writer = stores(tmp_path)
    path = write_and_locate(before, shape, shape.first)
    assert shape.read(before) == (shape.first,)
    assert shape.read(ForesightLedgerStore(tmp_path / "ledger")) == (shape.first,)
    swap(path, bytes_of(tmp_path, shape, shape.second))
    assert shape.read(ForesightLedgerStore(tmp_path / "ledger")) == (shape.second,)
    path.unlink()
    assert shape.read(ForesightLedgerStore(tmp_path / "ledger")) == ()


# ── 处二 B12：写到一半崩溃 ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        ".e9--20261210T060800000000.json.0123456789abcdef0123456789abcdef.tmp",
        ".e9.json.tmp",
        ".e9.json",
        "_e9.json",
        "e9.json.tmp",
    ],
)
def test_b12_a_leftover_temporary_file_is_not_a_record(tmp_path: Path, name: str) -> None:
    """处二 B12：目录里留下写到一半的临时文件（内容是半截）：不当记录读，也不算坏文件；读过之后才出现的也一样。"""

    reader, _writer = stores(tmp_path)
    kept = delivery()
    reader.record_delivery(DAY, kept)
    assert reader.deliveries_on(DAY) == (kept,)
    (path,) = [found for found in files_of(reader.root)]
    (path.parent / name).write_bytes(BAD["half"])
    for _round in range(2):
        assert reader.deliveries_on(DAY) == (kept,)
    assert reader.take_unreadable() == ()


def test_b12_a_temporary_file_renamed_into_place_becomes_a_record(tmp_path: Path) -> None:
    """处二 B12：临时文件写完改名成正式文件（另一个进程的一次完整写入）：改名之前读不到，改名之后读到。"""

    reader, _writer = stores(tmp_path)
    first = delivery()
    reader.record_delivery(DAY, first)
    (path,) = [found for found in files_of(reader.root)]
    second = delivery("e2", minute=2)
    payload = bytes_of(tmp_path, next(item for item in SHAPES if item.name == "delivery"), second)
    temporary = path.parent / ".e2.json.0123456789abcdef0123456789abcdef.tmp"
    temporary.write_bytes(payload)
    assert reader.deliveries_on(DAY) == (first,)
    os.replace(temporary, path.parent / "e2--20261210T060900000000.json")
    assert set(reader.deliveries_on(DAY)) == {first, second}
    assert reader.take_unreadable() == ()


# ── 缓存容量 ──────────────────────────────────────────────────────────────────


def test_a_cache_of_one_record_still_reads_everything_right(tmp_path: Path) -> None:
    """``read_cache_max_records`` 配成 1：缓存一直在淘汰，读到的内容、顺序、换过内容的那一份照样全对。"""

    limits = ForesightStoreConfig(read_cache_max_records=1)
    reader = ForesightLedgerStore(tmp_path / "ledger", limits=limits)
    items = tuple(delivery(f"e{number}", minute=number) for number in range(1, 6))
    for item in items:
        reader.record_delivery(DAY, item)
    for _round in range(3):
        assert reader.deliveries_on(DAY) == items
    shape = next(item for item in SHAPES if item.name == "delivery")
    path = next(found for found in files_of(reader.root) if found.name.startswith("e3--"))
    changed = delivery("e3", text="戊己庚辛", minute=3)
    swap(path, bytes_of(tmp_path, shape, changed))
    assert reader.deliveries_on(DAY) == (*items[:2], changed, *items[3:])
    assert reader.take_unreadable() == ()


def _as_upper(raw: dict[str, Any]) -> str:
    return str(raw["value"]).upper()


def _as_length(raw: dict[str, Any]) -> int:
    return len(str(raw["value"]))


def test_two_decoders_reading_the_same_directory_keep_their_own_results(tmp_path: Path) -> None:
    """同一份记录文件先后被两个解码函数读：各拿各的解码结果，回头再用第一个读，仍是第一个的结果。"""

    files = LedgerFiles(tmp_path / "ledger", ForesightStoreConfig())
    directory = files.root / "records"
    files.replace(directory / "one.json", {"value": "abc"})
    assert list(files.cached_decoded(directory, _as_upper)) == [("one", "ABC")]
    assert list(files.cached_decoded(directory, _as_length)) == [("one", 3)]
    assert list(files.cached_decoded(directory, _as_upper)) == [("one", "ABC")]


def test_the_read_cache_really_stops_at_its_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``read_cache_max_records`` 配成 2、目录里 3 个文件：读第三个时最早读的那个被淘汰，第二遍必须重新打开它，内容照样全对。"""

    files = LedgerFiles(tmp_path / "ledger", ForesightStoreConfig(read_cache_max_records=2))
    directory = files.root / "records"
    expected = [("a", "A"), ("b", "B"), ("c", "C")]
    for name, _decoded in expected:
        files.replace(directory / f"{name}.json", {"value": name})
    opened: Counter[str] = Counter()
    original = JsonFiles.load

    def counting(self: JsonFiles, path: Path) -> dict[str, Any]:
        opened[path.name] += 1
        return original(self, path)

    monkeypatch.setattr(JsonFiles, "load", counting)
    assert list(files.cached_decoded(directory, _as_upper)) == expected
    assert opened == Counter({"a.json": 1, "b.json": 1, "c.json": 1})
    opened.clear()
    assert list(files.cached_decoded(directory, _as_upper)) == expected
    assert opened["a.json"] == 1, f"被淘汰的文件第二遍没有重新打开：{dict(opened)}"
