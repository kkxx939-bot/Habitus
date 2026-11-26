"""习惯卡片（裁定 113）：数从复现来、只放日期与整数；每一代按词表此刻的状态对一遍目录——在用的更新、停用的留着标日期、
删了的跟着删；字段没变不写；换代里这一步尽力而为。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from habitus.behavior.kinds.active import count_active
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary, Word, WordOrigin
from habitus.runtime.generation_result import DEGRADED, FALLBACK_HABITS, HABITS, MARK, RELATION_CARDS
from habitus.runtime.habit_cards import ACTIVE, EXITED, HabitCards
from habitus.runtime.record_files import RecordFileError
from habitus.scene.habits import HabitNumbers, habit_numbers
from habitus.scene.recurrence import Recurrence
from habitus.temporal import DayInterval, OccurrenceDay
from tests.unit.temporal.fixtures import at, record, series

NOW = datetime(2026, 3, 20, 12, tzinfo=UTC)
DAY = date(2026, 3, 20)
DRINK, RUN, SIP = ClassId(Lane.SESSION, 1), ClassId(Lane.SESSION, 2), ClassId(Lane.SESSION, 3)


def words(*, grown_sip: bool = True) -> tuple[Word, ...]:
    found = [
        Word(DRINK, 1, "喝水", None, "喝水这件事", "测试"),
        Word(RUN, 1, "跑步", None, "跑步这件事", "测试"),
    ]
    if grown_sip:
        found.append(Word(SIP, 2, "小口喝", DRINK, "小口喝水", "测试", WordOrigin.GROWN, NOW - timedelta(days=40)))
    return tuple(found)


def vocabulary(active: dict[ClassId, date], exited: dict[ClassId, date] | None = None, **options) -> Vocabulary:  # type: ignore[no-untyped-def]
    return Vocabulary(Lane.SESSION, words(**options), active, revision=1, exited=exited or {})


def numbers(days: int = 5, last: date = date(2026, 3, 18), median: int | None = 2) -> HabitNumbers:
    return HabitNumbers(date(2026, 3, 2), last, days, 1, days - 1, 3, 1, median)


def cards(tmp_path) -> HabitCards:  # type: ignore[no-untyped-def]
    return HabitCards(tmp_path / "habitus", lane="session", max_children=100, max_bytes=64_000)


def test_numbers_come_from_the_recurrence_and_below_only_days_are_counted_from_the_series() -> None:
    """复现里"只有下级记录"的日子和"被提醒才做的"分不出来：拿序列现数（s-k0003 是 s-k0001 的下级）。"""

    first, second, third = date(2026, 3, 2), date(2026, 3, 4), date(2026, 3, 7)
    read = series(
        [record(at(first, 9)), record(at(second, 9), "s-k0003"), record(at(third, 9))], cutoff=date(2026, 3, 10)
    )
    days = tuple(
        OccurrenceDay(
            day,
            f"occ:{day}",
            at(day, 9),
            1,
            natural,
            natural_uri=f"occ:{day}" if natural else None,
            natural_at=at(day, 9) if natural else None,
        )
        for day, natural in ((first, 1), (second, 0), (third, 1))
    )
    intervals = (
        DayInterval(days[0], days[1], 2, 1.0, True),
        DayInterval(days[1], days[2], 3, 0.5, False),
    )
    found = habit_numbers(read, {"s-k0001": Recurrence("s-k0001", "session", days, intervals)})
    assert found["s-k0001"] == HabitNumbers(first, third, 3, 1, 2, 1, 1, 3)


def test_an_active_word_gets_a_card_and_a_rerun_with_the_same_numbers_writes_nothing(tmp_path) -> None:
    store = cards(tmp_path)
    state = vocabulary({DRINK: date(2026, 3, 10)})
    assert store.sync(state, {"s-k0001": numbers()}, now=NOW).written == 1
    path = store.directory / "s-k0001.md"
    before = path.read_bytes()
    card = store.read("s-k0001")
    assert card is not None and card.metadata.revision == 1
    assert card.fields["status"] == ACTIVE and card.fields["exited_on"] is None and card.fields["days"] == 5
    assert "# 喝水" in before.decode() and "常见间隔 2 天" in before.decode()
    # 同一代重跑（数没变）：一张都不重写，版本与字节都不动
    again = store.sync(state, {"s-k0001": numbers()}, now=NOW + timedelta(hours=1))
    assert again.written == 0 and path.read_bytes() == before
    # 数变了才改写
    assert store.sync(state, {"s-k0001": numbers(days=6)}, now=NOW + timedelta(days=1)).written == 1
    assert store.read("s-k0001").metadata.revision == 2  # type: ignore[union-attr]


def test_exiting_keeps_the_card_and_its_numbers_then_coming_back_resumes(tmp_path) -> None:
    store = cards(tmp_path)
    store.sync(vocabulary({DRINK: date(2026, 3, 10)}), {"s-k0001": numbers()}, now=NOW)
    exited = vocabulary({}, {DRINK: date(2026, 3, 19)})
    # 停用之后这一代的数变了也不更新：数保持最后一次的值
    assert store.sync(exited, {"s-k0001": numbers(days=9)}, now=NOW).written == 1
    card = store.read("s-k0001")
    assert card is not None and (card.fields["status"], card.fields["exited_on"], card.fields["days"]) == (
        EXITED,
        "2026-03-19",
        5,
    )
    assert "已停用（2026-03-19 起）" in (store.directory / "s-k0001.md").read_text(encoding="utf-8")
    back = vocabulary({DRINK: date(2026, 3, 25)})
    store.sync(back, {"s-k0001": numbers(days=7)}, now=NOW + timedelta(days=5))
    card = store.read("s-k0001")
    assert card is not None and (card.fields["status"], card.fields["exited_on"], card.fields["days"]) == (
        ACTIVE,
        None,
        7,
    )


def test_a_word_deleted_between_generations_loses_its_card_on_the_next_one(tmp_path) -> None:
    """删除不只发生在换代里（选词时候选位置满了会挤掉一个）：按目录对词表此刻的状态，不靠这一代的删除清单。"""

    store = cards(tmp_path)
    store.sync(vocabulary({DRINK: DAY, SIP: DAY}), {"s-k0001": numbers(), "s-k0003": numbers()}, now=NOW)
    assert sorted(path.name for path in store.directory.iterdir()) == ["s-k0001.md", "s-k0003.md"]
    gone = vocabulary({DRINK: DAY}).without(SIP, on=DAY)  # 换代之外删掉的：词表里已经没有它
    synced = store.sync(gone, {"s-k0001": numbers()}, now=NOW)
    assert synced.deleted == 1 and sorted(path.name for path in store.directory.iterdir()) == ["s-k0001.md"]


def test_a_grown_leaf_expiring_is_deleted_outright_and_its_card_goes_with_it(tmp_path) -> None:
    """自己长的、下面没有下级的词到期是直接删，不经过停用（照规则，不是缺陷）；初始词到期是停用、卡片留着。"""

    store = cards(tmp_path)
    state = vocabulary({DRINK: date(2026, 2, 1), SIP: date(2026, 2, 1)})
    store.sync(state, {"s-k0001": numbers(), "s-k0003": numbers()}, now=NOW)
    counted = count_active(state, (), today=DAY, min_days=3, cap=10, retire_days=20)
    assert counted.deleted == (SIP,) and counted.exited == (DRINK,)
    store.sync(counted.vocabulary, None, now=NOW)
    assert [path.name for path in store.directory.iterdir()] == ["s-k0001.md"]
    assert store.read("s-k0001").fields["status"] == EXITED  # type: ignore[union-attr]


def test_a_generation_without_numbers_clears_nothing_and_still_reconciles(tmp_path) -> None:
    store = cards(tmp_path)
    store.sync(vocabulary({DRINK: DAY}), {"s-k0001": numbers()}, now=NOW)
    # 序列里没有记录 / 语义树没交出材料：不清数；在用却从没有过数的词不建卡片
    synced = store.sync(vocabulary({DRINK: DAY, RUN: DAY}), None, now=NOW)
    assert (synced.written, synced.deleted, synced.waiting) == (0, 0, 1)
    assert store.read("s-k0001").fields["days"] == 5 and store.read("s-k0002") is None  # type: ignore[union-attr]
    # 删除和停用照对
    store.sync(vocabulary({}, {DRINK: DAY}), None, now=NOW)
    assert store.read("s-k0001").fields["status"] == EXITED  # type: ignore[union-attr]


def test_a_hand_edited_card_is_refused_not_silently_rewritten(tmp_path) -> None:
    store = cards(tmp_path)
    store.sync(vocabulary({DRINK: DAY}), {"s-k0001": numbers()}, now=NOW)
    path = store.directory / "s-k0001.md"
    path.write_text(path.read_text(encoding="utf-8").replace("一共做过 5 天", "一共做过 50 天"), encoding="utf-8")
    with pytest.raises(RecordFileError):
        store.read("s-k0001")
    (store.directory / "notes.md").write_text("x", encoding="utf-8")
    path.unlink()
    with pytest.raises(RecordFileError, match="named after a word"):
        store.sync(vocabulary({DRINK: DAY}), None, now=NOW)


def test_the_habit_step_failing_degrades_the_generation_which_still_writes_its_mark(tmp_path) -> None:
    from tests.unit.runtime.test_generation import assembled, generate

    class Broken:
        def sync(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("disk")

    _site, run = assembled(tmp_path, habits=Broken())
    run.vocabulary.store = type("Store", (), {"read": lambda self: vocabulary({})})()  # type: ignore[attr-defined]
    result = generate(run)
    step = result.step(HABITS)
    assert step is not None and step.outcome == DEGRADED and "习惯卡片出错" in (step.reason or "")
    assert result.fallbacks[FALLBACK_HABITS] == 1 and result.complete
    assert run.marks.read(run.today()).steps[HABITS]["outcome"] == DEGRADED  # type: ignore[union-attr]
    assert [item.step for item in result.steps][-3:] == [HABITS, RELATION_CARDS, MARK]
    # 同一天不为它重做：完成标记在，这一天就不再做
    assert run.ready().day is None
