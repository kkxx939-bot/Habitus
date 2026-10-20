"""每晚只新增：复现够了才建类、同一件事的四种说法聚成一类、像已有类的组交白天归类重判一次、坏输出不建类。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date
from typing import Any

from habitus.behavior.kinds.changes import apply
from habitus.behavior.kinds.classify import DaytimeClassifier, OccurrenceContent
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.nightly import NightlyGrower, PendingRecheck
from habitus.behavior.kinds.pending import PendingEntry, PendingPool
from tests.unit.behavior.kinds_fixtures import EDIT, NOW, RESEARCH, answer, edit_class, research_class, scripted_caller

STATS = ("统计代码行数", "统计代码规模", "统计代码量", "查询代码统计")


def vocabulary() -> Vocabulary:
    return Vocabulary(version=1, classes={EDIT: edit_class(), RESEARCH: research_class()})


def pool() -> PendingPool:
    entries = [
        PendingEntry(f"behavior://s{i}", Lane.SESSION, date(2026, 9, 1 + i), name, f"数 MemoryOS 的{name}")
        for i, name in enumerate(STATS)
    ]
    entries.append(PendingEntry("behavior://z-once", Lane.SESSION, date(2026, 9, 2), "核对时间", "看现在几点"))
    return PendingPool({entry.occurrence: entry for entry in entries})


def group(members: list[str], name: str, *, existing: str | None = None, goes_to: str | None = None) -> dict[str, Any]:
    return {
        "members": members,
        "name": name,
        "criterion": f"{name}的判据",
        "reminder": f"该{name}了",
        "excludes": [{"text": "动手改代码", "goes_to": goes_to}] if goes_to else [],
        "examples": ["统计代码行数", "统计代码行数"],
        "existing": existing,
    }


def content_of(occurrence: str) -> OccurrenceContent | None:
    return None if occurrence.endswith("gone") else OccurrenceContent(occurrence)


def grow(  # type: ignore[no-untyped-def]
    body: object, *rechecks: object, config: BehaviorKindConfig | None = None, entries: PendingPool | None = None
):
    caller, provider = scripted_caller([body, *rechecks], config=config)
    grower = NightlyGrower(caller, PendingRecheck(DaytimeClassifier(caller)))
    result = asyncio.run(
        grower.grow(entries or pool(), vocabulary(), now=NOW, content_of=content_of, checkpoint=lambda: None)
    )
    return result, provider


def test_recurring_group_becomes_one_new_class_and_its_entries_move() -> None:
    result, provider = grow(
        {"groups": [group(["P1", "P2", "P3", "P4"], "统计代码规模", goes_to="C1"), group(["P5"], "核对时间")]}
    )
    record = result.record
    assert record is not None and record.version == 2 and provider.calls == 1
    created = apply(vocabulary(), record).get(ClassId(Lane.SESSION, 3))
    assert created.name == "统计代码规模" and created.excludes[0].goes_to == EDIT
    assert created.examples == ("统计代码行数",)  # 重复的例子去重
    assert {(m.occurrence, m.source, m.target) for m in record.moves} == {
        (f"behavior://s{i}", "s-待定", "s-k0003") for i in range(4)
    }
    assert any("kind_pending_group_waiting session '核对时间'" in signal for signal in result.signals)


def test_recurrence_needs_both_count_and_days() -> None:
    config = BehaviorKindConfig(recurrence_min_count=4, recurrence_min_days=4)
    same_day = PendingPool(
        {
            f"behavior://d{i}": PendingEntry(f"behavior://d{i}", Lane.SESSION, date(2026, 9, 1), "统计", "数行数")
            for i in range(5)
        }
    )
    result, _ = grow(
        {"groups": [group(["P1", "P2", "P3", "P4", "P5"], "统计代码规模")]}, config=config, entries=same_day
    )
    assert result.record is None


def test_group_that_is_an_existing_class_is_rechecked_by_the_daytime_classifier() -> None:
    """白天归类判进类的迁进那一类；仍判「都不是」的放行（复现不够就等着），并报给归约侧打"已重判"标记。"""

    # 门槛按 3 条跨 3 天配，留下的两条才会"复现不够"；默认 2 / 2 下它们会长成新类（见下一条）。
    result, provider = grow(
        {"groups": [group(["P1", "P2", "P3", "P4"], "修改代码", existing="C1"), group(["P5"], "核对时间")]},
        answer(("R1", "C1", None), ("R2", "C2", None), ("R3", "都不是", "统计"), ("R4", "不是一件事", None)),
        config=BehaviorKindConfig(recurrence_min_count=3, recurrence_min_days=3),
    )
    record = result.record
    assert provider.calls == 2 and result.model_calls == 2
    assert record is not None and record.operations == ()
    assert {(m.occurrence, m.source, m.target) for m in record.moves} == {
        ("behavior://s0", "s-待定", str(EDIT)),
        ("behavior://s1", "s-待定", str(RESEARCH)),  # 以白天归类为准，不是每晚新增说的那一类
    }
    assert result.rechecked == {"behavior://s2", "behavior://s3"}
    assert any("kind_pending_group_waiting session '修改代码'" in signal for signal in result.signals)


def test_entries_rechecked_before_are_not_asked_again_and_can_grow_a_class() -> None:
    """只问一次：以前重判过的不再交白天归类，直接放行；复现够了照常长成新类。"""

    marked = PendingPool({key: replace(entry, rechecked=True) for key, entry in pool().entries.items()})
    result, provider = grow(
        {"groups": [group(["P1", "P2", "P3", "P4"], "统计代码规模", existing="C1"), group(["P5"], "核对时间")]},
        entries=marked,
    )
    assert provider.calls == 1 and result.rechecked == frozenset()
    record = result.record
    assert record is not None and apply(vocabulary(), record).get(ClassId(Lane.SESSION, 3)).name == "统计代码规模"
    assert len(record.moves) == 4


def test_entry_missing_from_the_tree_is_left_alone() -> None:
    entries = PendingPool(
        {
            "behavior://gone": PendingEntry("behavior://gone", Lane.SESSION, date(2026, 9, 1), "统计", "数行数"),
        }
    )
    result, provider = grow({"groups": [group(["P1"], "修改代码", existing="C1")]}, entries=entries)
    assert provider.calls == 1 and result.record is None and result.rechecked == frozenset()
    assert any(signal.startswith("kind_pending_recheck_missing") for signal in result.signals)


def test_invalid_output_adds_nothing() -> None:
    missing_one = {"groups": [group(["P1", "P2", "P3", "P4"], "统计代码规模")]}
    result, _ = grow(missing_one)
    assert result.record is None and any(signal.startswith("kind_nightly_rejected") for signal in result.signals)


def test_new_class_cannot_reuse_an_active_name() -> None:
    result, _ = grow({"groups": [group(["P1", "P2", "P3", "P4"], "调研"), group(["P5"], "核对时间")]})
    assert result.record is None and any(signal.startswith("kind_pending_group_invalid") for signal in result.signals)


def test_empty_pool_calls_nothing() -> None:
    result, provider = grow({"groups": []}, entries=PendingPool())
    assert result.record is None and result.model_calls == 0 and provider.calls == 0


def test_default_threshold_is_two_entries_on_two_days() -> None:
    """裁定 59：2 条跨 2 天就建类；同一天的 2 条（当天追问几轮）不算复现。"""

    assert (BehaviorKindConfig().recurrence_min_count, BehaviorKindConfig().recurrence_min_days) == (2, 2)
    two_days = PendingPool(
        {
            f"behavior://t{i}": PendingEntry(f"behavior://t{i}", Lane.SESSION, date(2026, 9, 1 + i), "看牙", "约牙医")
            for i in range(2)
        }
    )
    result, _ = grow({"groups": [group(["P1", "P2"], "约牙医")]}, entries=two_days)
    assert result.record is not None and len(result.record.operations) == 1
    one_day = PendingPool(
        {
            f"behavior://u{i}": PendingEntry(f"behavior://u{i}", Lane.SESSION, date(2026, 9, 1), "写 PPT", "改例会 PPT")
            for i in range(2)
        }
    )
    result, _ = grow({"groups": [group(["P1", "P2"], "写例会 PPT")]}, entries=one_day)
    assert result.record is None


# --- 只交新条目 + 召回（裁定 60 第 2 条；26 号清单 6.6）--------------------------------------------------------


def _old(index: int, name: str, content: str, *, day: date = date(2026, 3, 1)) -> PendingEntry:
    return PendingEntry(f"behavior://old{index:04d}", Lane.SESSION, day, name, content, grouped_on=day)


def test_a_full_pool_sends_only_new_entries_and_what_they_recall() -> None:
    """池里 3,000 条旧条目：提示词里只有新条目与召回的几条，不再整池进去。"""

    old = [_old(index, f"杂事{index}", f"一次性的事第{index}件：整理{index}号文件") for index in range(3000)]
    old.append(_old(9999, "订去杭州的机票", "订 10 月去杭州出差的机票"))
    new = [
        PendingEntry("behavior://new1", Lane.SESSION, date(2026, 9, 1), "订杭州出差的机票", "给下个月杭州出差订机票")
    ]
    entries = PendingPool({entry.occurrence: entry for entry in (*old, *new)})
    result, provider = grow({"groups": [group(["P1", "P2"], "订机票")]}, answer(), entries=entries)
    prompt = provider.prompts[0]  # 聚组那一次（建类之后的召回重判另算）
    assert "订杭州出差的机票" in prompt and "订去杭州的机票" in prompt  # 不同月份、措辞相近：召回、成组
    assert prompt.count("一次性的事") <= BehaviorKindConfig().recall_per_entry
    assert {"behavior://new1", "behavior://old9999"} <= result.grouped
    assert len(result.grouped) <= 1 + BehaviorKindConfig().recall_per_entry  # 交过去的只有新条目与召回的
    record = result.record
    assert record is not None
    assert {move.occurrence for move in record.moves} == {"behavior://new1", "behavior://old9999"}


def test_entries_sent_on_a_night_the_model_failed_stay_new() -> None:
    result, _ = grow({"groups": "不是数组"})
    assert result.grouped == frozenset() and result.record is None
    assert any("kind_nightly_rejected" in signal for signal in result.signals)


def test_a_night_takes_at_most_its_cap_of_new_entries_and_old_ones_alone_do_not_call_the_model() -> None:
    config = BehaviorKindConfig(nightly_max_new=2)
    result, provider = grow({"groups": [group(["P1", "P2"], "统计代码规模")]}, answer(), config=config)
    assert len(result.grouped) == 2 and any("kind_pending_deferred session: 3" in s for s in result.signals)
    quiet = PendingPool({key: replace(entry, grouped_on=date(2026, 9, 9)) for key, entry in pool().entries.items()})
    silent, silent_provider = grow({"groups": []}, entries=quiet)
    assert silent_provider.calls == 0 and silent.record is None


def test_a_new_class_recalls_the_similar_old_entries_grouping_did_not_reach() -> None:
    """新类建出来后按成员召回旧条目、交白天归类按带新类的词表重判：聚组时每条新条目只召回一条，多出来的同一件事
    （旧条目）在建类之后召回、归进新类。"""

    similar = [_old(index, "统计代码行数", f"数 MemoryOS 的统计代码行数（第 {index} 次）") for index in range(6)]
    entries = PendingPool({entry.occurrence: entry for entry in (*pool().entries.values(), *similar)})
    result, _ = grow(
        {
            "groups": [
                group(["P1", "P2", "P3", "P4", "P6", "P7", "P8", "P9"], "统计代码规模"),
                group(["P5"], "核对时间"),
            ]
        },
        answer(("R1", "C3", None), ("R2", "C3", None)),
        entries=entries,
        config=BehaviorKindConfig(recall_per_entry=1),
    )
    record = result.record
    assert record is not None
    moved = {move.occurrence for move in record.moves if move.target == "s-k0003"}
    assert {entry.occurrence for entry in similar} <= moved  # 聚组时召回的 4 条 + 建类后召回的 2 条


def test_entries_gone_from_the_tree_leave_the_pool_when_rechecked() -> None:
    gone = PendingEntry("behavior://s-gone", Lane.SESSION, date(2026, 9, 7), "修改代码", "改代码")
    entries = PendingPool({entry.occurrence: entry for entry in (*pool().entries.values(), gone)})
    result, _ = grow(
        {
            "groups": [
                group(["P1"], "修改代码", existing="C1"),  # s-gone 排在 s0 前面
                group(["P2", "P3", "P4", "P5"], "统计代码规模"),
                group(["P6"], "核对时间"),
            ]
        },
        entries=entries,
    )
    assert result.missing == {"behavior://s-gone"}
