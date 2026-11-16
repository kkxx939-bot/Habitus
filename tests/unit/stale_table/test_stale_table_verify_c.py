"""预测不等换代（2026-11-16）阶段 C 补的验收：用户在阶段 A 之后定的两条（旧表下日子预期加检查、换代只等跨零点的空白）、
"说两遍"、对抗盘面。盘面见 ``harness``；``xfail(strict=True)`` 的是阶段 C 确认的问题（见 ``out/报告-验证.md``），修好了会变红提醒删掉标记。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from habitus.config.foresight import ForesightConfig
from habitus.foresight import UnsealedRow
from habitus.foresight.hierarchy import KindHierarchy
from habitus.foresight.opportunities import OpportunityTable
from habitus.runtime.foresight_forming import AHEAD_TABLE, Recent
from habitus.runtime.foresight_tick import ForesightTick, TickReport
from habitus.runtime.generation_watermark import (
    WATERMARK_FILENAME,
    TickTrack,
    TickWatermark,
    TickWatermarkError,
    TickWatermarkStore,
)
from habitus.scene.relations.conditions import Condition, Template
from tests.integration.test_runtime_assembly import _fully_enabled_runtime
from tests.unit.scene.fixtures import publish
from tests.unit.stale_table.harness import (
    BASE,
    CODE,
    DISCUSS,
    DOCS,
    SLOT,
    TOLERANCE,
    ZONE,
    World,
    chain,
    days_after,
    world,
)


def clock(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=ZONE)


class Unsealed:
    """还没封口的判断替身：给定的几条，按时段过滤。"""

    def __init__(self, rows: tuple[UnsealedRow, ...]) -> None:
        self._rows = rows

    async def prepare(self, *, until: datetime) -> None:
        return None

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
        return tuple(row for row in self._rows if since <= row.started_at < until)


def tick_with(w: World, now: datetime, unsealed: Unsealed) -> TickReport:
    """同 ``World.tick``，只是还没封口的判断换成 ``unsealed``。"""

    state = w.session
    run = ForesightTick(
        state.ledger,
        state.opportunities,
        w.read_side(),
        unsealed,
        lane="session",
        vocabulary_version=lambda: 7,
        observer=state.recorder,
        clock=lambda: now,
        delivery=state.delivery,
        tick_seconds=SLOT.total_seconds(),
        recent=35,
        tolerance=TOLERANCE,
        track=TickTrack(
            TickWatermarkStore(state.root),
            slot=SLOT,
            tolerance_slots=ForesightConfig().unwatched_tolerance_slots,
        ),
        finished=state.marks.finished,
    )
    return asyncio.run(run.run_once())


def every_other_day(tmp: Path) -> tuple[World, date, date]:
    """BASE … BASE+5 每天讨论方案、修改代码；写文档只在 BASE、BASE+2、BASE+4。昨天那一代（G = BASE+5）完成了，今天 BASE+6 的还没成。
    从 G 到昨天只有 G 这一天：讨论方案、修改代码出现过，写文档没出现过。"""

    w = world(tmp)
    w.mornings(days_after(BASE, 6), docs=False)
    for offset in (0, 2, 4):
        publish(w.site.behavior_tree, BASE + timedelta(days=offset), "写文档", 12, 0)
    today = BASE + timedelta(days=6)
    night = today - timedelta(days=1)
    w.table(night)
    w.mark(night)
    return w, night, today


# ── 第 1 条：旧表下的日子预期，加一道检查后建 ─────────────────────────────────


def test_c2_a_kind_not_seen_since_the_old_table_gets_its_day_expectation_from_it_and_is_said(tmp_path: Path) -> None:
    """C2（新口径）前一半：写文档从旧表那一晚起没再出现过 → 预测拍用旧表把它今天的日子预期建上（记着旧表那一晚），出现过的两类不建
    （``days_held`` = 2）；到时窗照常说。今天这一代做成后换代那一步按身份查重，不再写第二份。"""

    w, night, today = every_other_day(tmp_path)
    report = w.tick(clock(today, 9))
    days = w.expectations(today, source="日子")
    assert [(item.kind_token, item.generation) for item in days] == [(DOCS, night.isoformat())]
    assert report.table_night == night and report.days_held == 2
    w.earn(days[0])
    w.ticks(today, time(9, 15), time(12, 30))
    assert "待取" in w.reasons(today, days[0].expectation_id)
    w.generation(today)
    after = w.expectations(today, source="日子")
    assert sorted(item.kind_token for item in after) == sorted([CODE, DISCUSS, DOCS])
    (docs,) = [item for item in after if item.kind_token == DOCS]
    assert docs.expectation_id == days[0].expectation_id and docs.generation == night.isoformat()


def test_c2_a_vocabulary_changed_since_the_old_table_holds_every_day_expectation(tmp_path: Path) -> None:
    """词表在旧表那一代之后改过版本（删词、在用的词变了，旧记录会改读成别的类）：旧表的历史与今天的表不同，这一拍不建日子
    预期（``days_held`` 认不出）；那一代的完成标记读不了也一样。版本相同照建。事件、情境不受影响。"""

    from habitus.runtime.generation_marks import GenerationMarkError

    w, night, today = every_other_day(tmp_path)
    changed = w.tick(clock(today, 9), vocabulary_of=lambda day: 6 if day == night else None)
    assert changed.table_night == night and changed.days_held is None
    assert w.expectations(today, source="日子") == []

    def broken(day: date) -> int | None:
        raise GenerationMarkError("generation mark is corrupt")

    assert w.tick(clock(today, 9, 15), vocabulary_of=broken).days_held is None
    assert w.expectations(today, source="日子") == []
    same = w.tick(clock(today, 9, 30), vocabulary_of=lambda day: 7)
    assert same.days_held == 2 and [item.kind_token for item in w.expectations(today, source="日子")] == [DOCS]


def test_c2_the_old_and_todays_table_give_the_same_day_identity_and_window_for_an_unseen_kind(tmp_path: Path) -> None:
    """用户第 1 条的前提：一次都没再出现过的类，旧表与今天的表算出的身份与时窗相同（两条 lane 用两张表各建一遍比对）。"""

    w, night, today = every_other_day(tmp_path)
    w.tick(clock(today, 9))
    (old,) = w.expectations(today, source="日子")
    fresh = world(tmp_path / "fresh")
    fresh.mornings(days_after(BASE, 6), docs=False)
    for offset in (0, 2, 4):
        publish(fresh.site.behavior_tree, BASE + timedelta(days=offset), "写文档", 12, 0)
    fresh.generation(today)
    (new,) = [item for item in fresh.expectations(today, source="日子") if item.kind_token == DOCS]
    assert (old.expectation_id, old.window, old.deadline) == (new.expectation_id, new.window, new.deadline)


def test_c2_a_child_seen_since_the_old_table_holds_its_parent_too(tmp_path: Path) -> None:
    """检查连上级一起算：修改代码挂在写文档下面时，G 之后出现了修改代码，写文档也不在旧表下建。"""

    w, night, today = every_other_day(tmp_path)
    tick = ForesightTick(
        w.session.ledger,
        w.session.opportunities,
        w.read_side(),
        Unsealed(()),
        lane="session",
        vocabulary_version=lambda: 7,
        recent=35,
        tolerance=TOLERANCE,
    )
    plain = asyncio.run(tick.forming.read(clock(today, 9)))
    table = w.session.opportunities.latest("session")
    assert table is not None and table.night == night
    flat = tick.forming.days_unchanged(plain, table)
    assert flat is not None and [kind.kind_token for kind in flat.kinds] == [DOCS]
    nested = Recent(plain.now, plain.cache, plain.unsealed, plain.seen, KindHierarchy({DOCS: frozenset({CODE})}))
    held = tick.forming.days_unchanged(nested, table)
    assert held is not None and [kind.kind_token for kind in held.kinds] == []


def test_c2_an_unsealed_record_without_a_kind_since_the_old_table_holds_every_day_expectation(tmp_path: Path) -> None:
    """那几天里有还没定是哪一类的判断（还没选上词）→ 认不出，这一拍不建日子预期（``days_held`` 为 None），事件照建。"""

    w, night, today = every_other_day(tmp_path)
    started = clock(night, 15)
    rows = (UnsealedRow("写点东西", None, started, started + timedelta(minutes=5), None),)
    report = tick_with(w, clock(today, 9), Unsealed(rows))
    assert report.days_held is None
    assert w.expectations(today, source="日子") == []
    assert report.table_night == night


def test_c2_a_kind_seen_only_in_an_unsealed_record_since_the_old_table_is_held(tmp_path: Path) -> None:
    """G 之后写文档只出现在还没封口的判断里（还没落树）：照样算出现过，不建。"""

    w, night, today = every_other_day(tmp_path)
    started = clock(night, 15)
    rows = (UnsealedRow("写文档", DOCS, started, started + timedelta(minutes=5), None),)
    report = tick_with(w, clock(today, 9), Unsealed(rows))
    assert report.days_held == 3
    assert w.expectations(today, source="日子") == []


def test_c2_today_records_do_not_hold_the_old_table(tmp_path: Path) -> None:
    """检查只看表那一晚到昨天：今天零点后做过的不挡（今天的表同样不含今天），照建、照"今天已做"算。"""

    w, night, today = every_other_day(tmp_path)
    publish(w.site.behavior_tree, today, "写文档", 8, 0)
    report = w.tick(clock(today, 9))
    assert report.days_held == 2
    assert [item.kind_token for item in w.expectations(today, source="日子")] == [DOCS]


# ── "说两遍" ──────────────────────────────────────────────────────────────────


def test_twice_a_relation_swapped_for_a_conditioned_one_is_still_said_once_a_day(tmp_path: Path) -> None:
    """今天这一代把"讨论方案 → 修改代码"换成带条件（时间线：之前写过文档）的版本：换表那一拍补建出另一个身份的事件预期，账上
    同一天两条修改代码的事件预期；说出口的修改代码今天只有一次。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    for day in days_after(BASE, 4):
        publish(w.site.behavior_tree, day, "写文档", 9, 30)
    today = BASE + timedelta(days=4)
    w.table(today - timedelta(days=1))
    w.mark(today - timedelta(days=1))
    publish(w.site.behavior_tree, today, "写文档", 9, 30)
    publish(w.site.behavior_tree, today, "讨论方案", 10, 0)
    w.tick(clock(today, 10, 15))
    conditioned = dataclasses.replace(
        chain(), key=dataclasses.replace(chain().key, condition=Condition(Template.TIMELINE, DOCS).text)
    )
    w.generation(today, relations=(conditioned,))
    w.ticks(today, time(10, 30), time(13, 0))
    events = w.expectations(today, source="事件")
    assert len({item.expectation_id for item in events}) == 2  # 两个身份
    for item in w.expectations(today):
        w.earn(item)
    w.ticks(today, time(13, 15), time(23, 45))
    spoken = w.spoken_kinds(today)
    assert spoken.count(CODE) <= 1, spoken


def _late_record(tmp: Path, docs: tuple[int, ...], late: int, today_offset: int) -> tuple[World, date, date]:
    """模拟：表 G 写成之后，G 之前那一天才多出一条写文档（表 G 当时读到的那天的类，之后变了——例如那天的记录归到的词被删、读到
    上级，``KindHierarchy`` 换算口变了）。检查要比对表那一晚之前几天树上的日子与表里记的，对不上就挡下这一类。"""

    w = world(tmp)
    w.mornings(days_after(BASE, today_offset), docs=False)
    for offset in docs:
        publish(w.site.behavior_tree, BASE + timedelta(days=offset), "写文档", 12, 0)
    today = BASE + timedelta(days=today_offset)
    night = today - timedelta(days=1)
    w.table(night)
    w.mark(night)
    publish(w.site.behavior_tree, BASE + timedelta(days=late), "写文档", 12, 0)
    return w, night, today


def test_twice_a_record_before_the_old_table_that_changed_after_it_gives_one_day_expectation(tmp_path: Path) -> None:
    w, _night, today = _late_record(tmp_path, (0, 1, 2, 3), 4, 6)
    w.tick(clock(today, 9))
    w.generation(today)
    docs = [item for item in w.expectations(today, source="日子") if item.kind_token == DOCS]
    assert len(docs) == 1, [(item.trigger_instance.rsplit("/", 1)[-1], item.generation) for item in docs]


def test_twice_a_record_before_the_old_table_that_changed_after_it_is_not_said_from_the_old_table(
    tmp_path: Path,
) -> None:
    w, _night, today = _late_record(tmp_path, (0, 3, 6), 7, 9)
    w.tick(clock(today, 9))
    w.generation(today)
    for item in w.expectations(today, source="日子"):
        w.earn(item)
    w.ticks(today, time(9, 15), time(23, 45))
    assert DOCS not in w.spoken_kinds(today)


# ── 对抗盘面 ──────────────────────────────────────────────────────────────────


def _yesterday_done(tmp: Path) -> tuple[World, date, date]:
    w = world(tmp)
    w.mornings(days_after(BASE, 4))
    today = BASE + timedelta(days=4)
    night = today - timedelta(days=1)
    w.table(night)
    w.mark(night)
    publish(w.site.behavior_tree, today, "讨论方案", 10, 0)
    return w, night, today


def _latest(w: World) -> OpportunityTable:
    table = w.session.opportunities.latest("session")
    assert table is not None
    return table


def test_adversarial_the_latest_table_corrupt_falls_back_to_the_previous_finished_one_and_counts_it(
    tmp_path: Path,
) -> None:
    """最近一张表（今天的、也有完成标记）坏了：退到上一张做完了的，这一拍的观测计着跳过的坏表（``unreadable``）。"""

    w, night, today = _yesterday_done(tmp_path)
    w.table(today)
    path = w.session.opportunities.write(_latest(w))
    w.mark(today)
    path.write_text("{", encoding="utf-8")
    report = w.tick(clock(today, 10, 15))
    assert report.table_night == night and report.skipped is None
    assert report.unreadable >= 1
    assert w.session.recorder.ticks()[-1].attributes.get("unreadable") == report.unreadable


def test_adversarial_a_mark_without_its_table_falls_back_to_the_previous_table(tmp_path: Path) -> None:
    w, night, today = _yesterday_done(tmp_path)
    w.mark(today)  # 今天的完成标记在、今天的表不在
    report = w.tick(clock(today, 10, 15))
    assert report.table_night == night and report.recorded >= 1


def test_adversarial_a_mark_naming_another_day_is_not_used(tmp_path: Path) -> None:
    """今天的表在、今天的完成标记文件里写的是别的日子：标记读不了，不用今天的表，退到上一张。"""

    w, night, today = _yesterday_done(tmp_path)
    w.table(today)
    w.mark(today)
    path = w.session.marks.path(today)
    path.write_text(path.read_text(encoding="utf-8").replace(today.isoformat(), night.isoformat()), encoding="utf-8")
    report = w.tick(clock(today, 10, 15))
    assert report.table_night == night


def test_adversarial_a_table_dated_in_the_future_forms_nothing_and_says_why(tmp_path: Path) -> None:
    """最晚那张表在今天之后（往西改了时区）：这一拍不建，原因明说；递送照用最晚那张。"""

    w, _night, today = _yesterday_done(tmp_path)
    w.table(today + timedelta(days=1))
    w.mark(today + timedelta(days=1))
    report = w.tick(clock(today, 10, 15))
    assert report.skipped == AHEAD_TABLE and report.table_night is None


def _write_v1_watermark(root: Path, at: datetime) -> Path:
    path = root / WATERMARK_FILENAME
    root.mkdir(parents=True, exist_ok=True)
    stamp = at.isoformat()
    body = {
        "schema": "tick_watermark_v1",
        "ticked_at": stamp,
        "unwatched_until": None,
        "formed_through": stamp,
        "formed_generation": at.date().isoformat(),
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_adversarial_a_v1_watermark_stops_both_the_tick_and_the_generation(tmp_path: Path) -> None:
    """上一个版本形状的水位（升级后盘上留着的 ``tick_watermark_v1``）：预测拍与换代的触发都读它，一直抛——明确报错，不悄悄当空。"""

    w, night, today = _yesterday_done(tmp_path)
    _write_v1_watermark(w.session.root, clock(today, 9))
    with pytest.raises(TickWatermarkError):
        w.tick(clock(today, 10, 15))
    with pytest.raises(TickWatermarkError):
        TickWatermarkStore(w.session.root).read()


def test_adversarial_a_v1_watermark_error_names_the_file(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "foresight"
    path = _write_v1_watermark(root, clock(BASE, 9))
    with pytest.raises(TickWatermarkError) as caught:
        TickWatermarkStore(root).read()
    assert str(path) in str(caught.value) or WATERMARK_FILENAME in str(caught.value)


# ── 第 2 条：换代只等跨零点的空白（C10 见 ``test_stale_table_config``）────────────────


def test_c10_a_stretch_that_starts_today_is_not_overnight_but_one_from_before_midnight_is() -> None:
    """``TickWatermark.unwatched``：从零点之前开始的空白记成跨夜那一段；今天零点之后开始的只推 ``unwatched_until``。"""

    today = BASE + timedelta(days=4)
    mark = TickWatermark().unwatched(clock(today - timedelta(days=1), 23), clock(today, 9))
    assert mark.overnight_until == clock(today, 9)
    later = mark.unwatched(clock(today, 10), clock(today, 11))
    assert later.unwatched_until == clock(today, 11) and later.overnight_until == clock(today, 9)
    edge = TickWatermark().unwatched(clock(today - timedelta(days=1), 23, 30), clock(today, 0))
    assert edge.overnight_until == clock(today, 0)


def test_c11_the_generation_waits_for_an_overnight_stretch_written_by_the_first_tick_of_the_day(tmp_path: Path) -> None:
    """C11（单元一半）：昨晚 23:00 合盖、今天 09:00 开盖。开盖前换代不触发（水位停在昨晚）；09:00 那一拍写下跨夜空白后，等到 09:30
    才触发——给归约留了 2 槽把这段落树。重放一半见 ``summary.json`` 的 ``unreduced_gaps``。"""

    runtime = _fully_enabled_runtime(tmp_path)
    pipeline = runtime.components.pipeline("session")
    assert pipeline is not None
    generation = pipeline.generation.generation
    zone = runtime.config.locale.zone()
    today = date(2026, 10, 20)
    generation.closed_days = lambda: (today - timedelta(days=2), today - timedelta(days=1))
    closed = datetime.combine(today - timedelta(days=1), time(23), tzinfo=zone)
    opened = datetime.combine(today, time(9), tzinfo=zone)
    steps = [
        (opened, TickWatermark(ticked_at=closed)),
        (opened + timedelta(minutes=15), TickWatermark(ticked_at=opened).unwatched(closed, opened)),
        (
            opened + timedelta(minutes=30),
            TickWatermark(ticked_at=opened + timedelta(minutes=15)).unwatched(closed, opened),
        ),
        (
            opened + timedelta(minutes=45),
            TickWatermark(ticked_at=opened + timedelta(minutes=30)).unwatched(closed, opened),
        ),
    ]
    found = []
    for now, mark in steps:
        generation.clock = lambda now=now: now + timedelta(seconds=1)
        generation.watermark = lambda mark=mark: mark
        found.append(generation.ready().day)
    assert found == [None, None, None, today]


def test_health_the_reminder_basis_sentence_for_the_physical_branch_has_no_internal_words(tmp_path: Path) -> None:
    """D3 按支报：物理一支旧了也报一句大白话（会话一支今天做成了，不报）。"""

    from tests.unit.stale_table.harness import BANNED_WORDS
    from tests.unit.stale_table.test_stale_table_health import Disk

    disk = Disk(tmp_path)
    disk.closed_through(disk.today - timedelta(days=1))
    disk.generation(disk.today - timedelta(days=4), lane="physical")
    disk.generation(disk.today, lane="session")
    checks = {check.name: check for check in disk.checks()}
    physical = checks["reminder_basis_physical"]
    assert "4 天" in physical.detail and not any(word in physical.detail for word in BANNED_WORDS)
    assert checks["reminder_basis_session"].status.value == "healthy"
