"""Q13：转在用与退出同一把尺子。用户定的口径：按"退出"的口径现在就该退出的词，不转在用。

场景号（R1…R15）对应丁批测试输入的场景表。这一代的日子默认是 ``TODAY``（05-03），"差 N 天"= ``TODAY`` 减最后一次直接选中；
入口是真实的 ``count_active`` 与 ``VocabularyCounter.count``，断言只看词表、计数结果和落盘状态。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from habitus.behavior.kinds.active import ActiveCount, count_active
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary, Word, WordOrigin
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.runtime.generation_vocabulary import VocabularyCounter
from habitus.series import EventSeries
from habitus.series.model import SeriesRecord

TODAY = date(2026, 5, 3)
RETIRE_DAYS = (7, 20)

PARENT = ClassId(Lane.SESSION, 1)
WORD = ClassId(Lane.SESSION, 2)
OTHER = ClassId(Lane.SESSION, 3)
CHILD = ClassId(Lane.SESSION, 11)
GRANDCHILD = ClassId(Lane.SESSION, 12)


def day(month: int, number: int) -> date:
    return date(2026, month, number)


def moment(on: date) -> datetime:
    return datetime(on.year, on.month, on.day, 10, 0, tzinfo=UTC)


def initial(word_id: ClassId) -> Word:
    return Word(word_id, 1, f"初始词{word_id.number}", None, "测试用的初始词", "测试")


def grown(
    word_id: ClassId,
    parent: ClassId,
    *,
    level: int = 2,
    written: date = date(2026, 3, 1),
    last_picked: date | None = None,
) -> Word:
    return Word(
        word_id,
        level,
        f"长出来的词{word_id.number}",
        parent,
        "测试用的自己长的词",
        "测试",
        origin=WordOrigin.GROWN,
        created_at=moment(written),
        last_picked=None if last_picked is None else moment(last_picked),
    )


def words(
    *items: Word,
    active: dict[ClassId, date] | None = None,
    exited: dict[ClassId, date] | None = None,
) -> Vocabulary:
    return Vocabulary(Lane.SESSION, items, active or {}, 0, (), exited or {})


def picks(word_id: ClassId, *days: date) -> list[tuple[str, date]]:
    return [(str(word_id), item) for item in days]


def count(
    vocabulary: Vocabulary,
    records: Iterable[tuple[str, date]],
    *,
    today: date = TODAY,
    cap: int = 100,
    retire_days: int = 20,
) -> ActiveCount:
    return count_active(vocabulary, records, today=today, min_days=2, cap=cap, retire_days=retire_days)


def generation(vocabulary: Vocabulary, records: list[tuple[str, date]], today: date) -> ActiveCount:
    """按换代的口径数一代：序列只含这一代的日子之前的记录。"""

    return count(vocabulary, [item for item in records if item[1] < today], today=today)


def assert_stays_out(result: ActiveCount, before: Vocabulary) -> None:
    assert result.added == ()
    assert result.waiting == ()
    assert dict(result.vocabulary.active) == dict(before.active)
    assert result.vocabulary.revision == before.revision
    assert dict(result.vocabulary.exited) == dict(before.exited)


# ── 边界日 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seen",
    [
        pytest.param((day(4, 12), day(4, 13)), id="R1-差20"),
        pytest.param((day(4, 11), day(4, 12)), id="R2-差21"),
    ],
)
def test_r1_r2_a_word_already_due_to_retire_does_not_enter(seen: tuple[date, date]) -> None:
    """R1、R2：最后一次直接选中差 20、21 天，按退出的口径已经到期：不转在用、不等名额，修订号和退出记录都不动。"""

    before = words(initial(WORD))
    assert_stays_out(count(before, picks(WORD, *seen)), before)


def test_r1_a_due_word_keeps_its_earlier_exit_record() -> None:
    """R1："退出记录不变"：退出过的词退出后又凑够两天，但已经到期，原来的退出日要留着。"""

    before = words(initial(WORD), exited={WORD: day(4, 1)})
    result = count(before, picks(WORD, day(4, 12), day(4, 13)))
    assert_stays_out(result, before)
    assert dict(result.vocabulary.exited) == {WORD: day(4, 1)}


def test_r1_the_next_generation_has_no_enter_then_exit() -> None:
    """R1：连跑两代，不出现"这一代转在用、下一代退出"，也不留下新的退出记录。"""

    records = picks(WORD, day(4, 12), day(4, 13))
    first = generation(words(initial(WORD)), records, TODAY)
    second = generation(first.vocabulary, records, TODAY + timedelta(days=1))
    assert first.added == () and second.added == ()
    assert second.exited == ()
    assert dict(second.vocabulary.active) == {}
    assert dict(second.vocabulary.exited) == {}
    assert second.vocabulary.revision == 0


def test_r3_one_day_short_still_enters_and_retires_on_the_next_generation() -> None:
    """R3：差 19 天仍转在用；下一代差 20 天、其间没再被选中，照常退出。"""

    records = picks(WORD, day(4, 13), day(4, 14))
    first = generation(words(initial(WORD)), records, TODAY)
    assert first.added == (WORD,)
    assert dict(first.vocabulary.active) == {WORD: TODAY}
    assert first.vocabulary.revision == 1

    tomorrow = TODAY + timedelta(days=1)
    second = generation(first.vocabulary, records, tomorrow)
    assert second.exited == (WORD,)
    assert dict(second.vocabulary.active) == {}
    assert dict(second.vocabulary.exited) == {WORD: tomorrow}
    assert second.vocabulary.revision == 2


def test_r4_a_recent_word_enters() -> None:
    """R4：最后一次差 1 天，转在用。"""

    result = count(words(initial(WORD)), picks(WORD, day(4, 30), day(5, 2)))
    assert result.added == (WORD,)
    assert result.waiting == ()
    assert dict(result.vocabulary.active) == {WORD: TODAY}


def full_house() -> tuple[Vocabulary, list[tuple[str, date]]]:
    """``cap=1`` 已被另一个新鲜的词占满。"""

    vocabulary = words(initial(OTHER), initial(WORD), active={OTHER: day(5, 1)})
    return vocabulary, picks(OTHER, day(5, 2))


def test_r5_a_due_word_does_not_wait_for_a_seat() -> None:
    """R5：满额时差 20 天的词不进 ``waiting``。"""

    vocabulary, held = full_house()
    result = count(vocabulary, [*held, *picks(WORD, day(4, 12), day(4, 13))], cap=1)
    assert result.waiting == ()
    assert result.added == ()
    assert dict(result.vocabulary.active) == {OTHER: day(5, 1)}


def test_r6_a_word_one_day_short_waits_for_a_seat() -> None:
    """R6：满额时差 19 天的词照常进 ``waiting``。"""

    vocabulary, held = full_house()
    result = count(vocabulary, [*held, *picks(WORD, day(4, 13), day(4, 14))], cap=1)
    assert result.waiting == (WORD,)
    assert result.added == ()
    assert dict(result.vocabulary.active) == {OTHER: day(5, 1)}


def test_r7_a_due_word_does_not_take_the_seat_of_a_fresh_one() -> None:
    """R7：只有一个空位，A 差 31 天、B 差 1 天：到期的 A 不占名额，B 转在用，没有人等着。"""

    before = words(initial(WORD), initial(OTHER))
    result = count(before, [*picks(WORD, day(4, 1), day(4, 2)), *picks(OTHER, day(5, 1), day(5, 2))], cap=1)
    assert result.added == (OTHER,)
    assert result.waiting == ()
    assert dict(result.vocabulary.active) == {OTHER: TODAY}


@pytest.mark.parametrize("retire_days", RETIRE_DAYS)
def test_r1_boundary_follows_the_configured_retire_days(retire_days: int) -> None:
    """R1（换一个 ``retire_days``）：差正好 ``retire_days`` 天不转在用，防止写死 20。"""

    last = TODAY - timedelta(days=retire_days)
    before = words(initial(WORD))
    result = count(before, picks(WORD, last - timedelta(days=1), last), retire_days=retire_days)
    assert_stays_out(result, before)


@pytest.mark.parametrize("retire_days", RETIRE_DAYS)
def test_r3_boundary_follows_the_configured_retire_days(retire_days: int) -> None:
    """R3（换一个 ``retire_days``）：差 ``retire_days - 1`` 天仍转在用。"""

    last = TODAY - timedelta(days=retire_days - 1)
    result = count(words(initial(WORD)), picks(WORD, last - timedelta(days=1), last), retire_days=retire_days)
    assert result.added == (WORD,)
    assert dict(result.vocabulary.active) == {WORD: TODAY}


# ── 已经在用的词：退出行为不变 ────────────────────────────────────────────


@pytest.mark.parametrize("retire_days", RETIRE_DAYS)
def test_r8_an_active_word_at_the_limit_retires(retire_days: int) -> None:
    """R8：在用的词最后一次差 ``retire_days`` 天：退出，记这一代的日子，修订号加一。"""

    last = TODAY - timedelta(days=retire_days)
    before = words(initial(WORD), active={WORD: last - timedelta(days=30)})
    result = count(before, picks(WORD, last), retire_days=retire_days)
    assert result.exited == (WORD,)
    assert result.added == ()
    assert dict(result.vocabulary.active) == {}
    assert dict(result.vocabulary.exited) == {WORD: TODAY}
    assert result.vocabulary.revision == before.revision + 1


@pytest.mark.parametrize("retire_days", RETIRE_DAYS)
def test_r9_an_active_word_one_day_short_stays(retire_days: int) -> None:
    """R9：在用的词最后一次差 ``retire_days - 1`` 天：留在用，什么都不变。"""

    last = TODAY - timedelta(days=retire_days - 1)
    since = last - timedelta(days=30)
    before = words(initial(WORD), active={WORD: since})
    result = count(before, picks(WORD, last), retire_days=retire_days)
    assert result.exited == ()
    assert dict(result.vocabulary.active) == {WORD: since}
    assert dict(result.vocabulary.exited) == {}
    assert result.vocabulary.revision == before.revision


# ── 候选下级、自己长的词、退出过的词 ──────────────────────────────────────


def test_r10_a_parent_never_picked_directly_enters_on_borrowed_days() -> None:
    """R10：初始词从没被直接选中，候选下级借来两天：上级和下级都转在用（没有"最后一次"，不算到期）。"""

    before = words(initial(PARENT), grown(CHILD, PARENT, last_picked=day(5, 2)))
    result = count(before, picks(CHILD, day(5, 1), day(5, 2)))
    assert set(result.added) == {PARENT, CHILD}
    assert dict(result.vocabulary.active) == {PARENT: TODAY, CHILD: TODAY}


def test_r10b_a_parent_last_picked_long_ago_does_not_enter_on_borrowed_days() -> None:
    """R10b：同 R10，但初始词 03-06 被直接选中过（差 58 天）：上级不转在用，下级照转。"""

    before = words(initial(PARENT), grown(CHILD, PARENT, last_picked=day(5, 2)))
    result = count(before, [*picks(PARENT, day(3, 6)), *picks(CHILD, day(5, 1), day(5, 2))])
    assert result.added == (CHILD,)
    assert result.waiting == ()
    assert dict(result.vocabulary.active) == {CHILD: TODAY}


def test_r11_a_due_candidate_with_a_live_child_stays_and_its_parent_enters() -> None:
    """R11：自己长的候选差 20 天、下面有活着的下级：它不转在用、不删；从没被直接选中的初始上级照转。"""

    before = words(
        initial(PARENT),
        grown(CHILD, PARENT, last_picked=day(4, 13)),
        grown(GRANDCHILD, CHILD, level=3, written=day(5, 1)),
    )
    result = count(before, picks(CHILD, day(4, 12), day(4, 13)))
    assert result.added == (PARENT,)
    assert result.waiting == ()
    assert result.deleted == ()
    assert CHILD in result.vocabulary
    assert GRANDCHILD in result.vocabulary
    assert dict(result.vocabulary.active) == {PARENT: TODAY}


def test_r12_a_due_candidate_is_deleted_and_its_days_do_not_lift_the_parent() -> None:
    """R12：自己长的候选差 20 天、没有下级：删掉；旧日子换算到上级后，上级最后一次也差 20 天，不转在用。"""

    before = words(initial(PARENT), grown(CHILD, PARENT, last_picked=day(4, 13)))
    result = count(before, picks(CHILD, day(4, 12), day(4, 13)))
    assert result.deleted == (CHILD,)
    assert CHILD not in result.vocabulary
    assert_stays_out(result, before)


def test_r13_the_latest_of_series_and_last_picked_counts_as_the_last_time() -> None:
    """R13：序列里的日子很旧，但状态里 ``last_picked`` 是 05-02：最后一次取最晚的，转在用。"""

    before = words(initial(PARENT), grown(CHILD, PARENT, last_picked=day(5, 2)))
    result = count(before, picks(CHILD, day(3, 2), day(3, 3)))
    assert set(result.added) == {PARENT, CHILD}
    assert dict(result.vocabulary.active) == {PARENT: TODAY, CHILD: TODAY}


def test_r14_days_before_the_exit_stay_void() -> None:
    """R14：04-20 退出过，退出后只有 05-01 一天：退出前的日子仍作废，不转在用。"""

    before = words(initial(WORD), exited={WORD: day(4, 20)})
    result = count(before, picks(WORD, day(4, 10), day(4, 11), day(5, 1)))
    assert_stays_out(result, before)


def test_r15_two_days_after_the_exit_enter_again() -> None:
    """R15：04-20 退出过，退出后 04-25、05-01 两天：转在用，退出记录清掉，修订号加一。"""

    before = words(initial(WORD), exited={WORD: day(4, 20)})
    result = count(before, picks(WORD, day(4, 25), day(5, 1)))
    assert result.added == (WORD,)
    assert dict(result.vocabulary.active) == {WORD: TODAY}
    assert dict(result.vocabulary.exited) == {}
    assert result.vocabulary.revision == before.revision + 1


# ── 实盘例子的摆法：逐代连跑 ──────────────────────────────────────────────


def test_budget_case_waits_for_a_fresh_direct_pick_and_keeps_the_older_days() -> None:
    """做预算（R10b 的实盘例子）：03-27 直接选中一次，04-27 候选下级借来一次。04-28 起每一代都不转在用、也不退出；
    05-27 再直接选中一次，05-28 转在用（03-27、04-27 没有作废，否则一天不够数）。"""

    records = [*picks(PARENT, day(3, 27), day(5, 27)), *picks(CHILD, day(4, 27))]
    vocabulary = words(initial(PARENT), grown(CHILD, PARENT, written=day(4, 20), last_picked=day(4, 27)))
    today = day(4, 28)
    while today < day(5, 28):
        result = generation(vocabulary, records, today)
        assert result.added == (), f"{today} 不该转在用"
        assert result.exited == (), f"{today} 不该退出"
        assert result.waiting == (), f"{today} 不该等名额"
        vocabulary = result.vocabulary
        today += timedelta(days=1)

    final = generation(vocabulary, records, day(5, 28))
    assert final.added == (PARENT,)
    assert dict(final.vocabulary.active) == {PARENT: day(5, 28)}


def test_writing_plan_case_enters_only_after_a_fresh_direct_pick() -> None:
    """写方案（R12 的实盘例子）：下级这一代被删，旧日子换算到上级后差 20 天：上级不转在用；05-06 直接选中一次，下一代转在用。"""

    records = [*picks(CHILD, day(4, 12), day(4, 13)), *picks(PARENT, day(5, 6))]
    vocabulary = words(initial(PARENT), grown(CHILD, PARENT, last_picked=day(4, 13)))

    deleted = generation(vocabulary, records, TODAY)
    assert deleted.deleted == (CHILD,)
    assert deleted.added == ()
    vocabulary = deleted.vocabulary

    today = TODAY + timedelta(days=1)
    while today < day(5, 7):
        result = generation(vocabulary, records, today)
        assert result.added == (), f"{today} 不该转在用"
        assert result.exited == (), f"{today} 不该退出"
        vocabulary = result.vocabulary
        today += timedelta(days=1)

    final = generation(vocabulary, records, day(5, 7))
    assert final.added == (PARENT,)
    assert dict(final.vocabulary.active) == {PARENT: day(5, 7)}


# ── 走换代那一步：真实读写词表状态 ────────────────────────────────────────


def kind_store(root: Path) -> BehaviorKindStore:
    return BehaviorKindStore(
        root,
        lane=Lane.SESSION,
        lock_store=SQLiteLockStore(root / "locks.sqlite3"),
        lock_ttl_seconds=30,
    )


def series_of(token: str, *days: date) -> EventSeries:
    return EventSeries(
        cutoff=TODAY,
        records=tuple(
            SeriesRecord(
                uri=f"occ:{item.isoformat()}.md",
                name="测试行为",
                day=item,
                started_at=moment(item),
                last_observed_at=moment(item) + timedelta(minutes=10),
                kind_token=token,
                lane=Lane.SESSION.value,
                classified=True,
                reminded=False,
                summary="测试行为",
            )
            for item in days
        ),
    )


@pytest.mark.parametrize("retire_days", RETIRE_DAYS)
def test_r1_through_the_generation_step_leaves_the_state_on_disk_unchanged(tmp_path: Path, retire_days: int) -> None:
    """R1（走 ``VocabularyCounter.count``）：差 ``retire_days`` 天的词不转在用，重新读盘在用集合和修订号都没变。"""

    store = kind_store(tmp_path)
    word = store.read().words[0].id
    last = TODAY - timedelta(days=retire_days)
    counter = VocabularyCounter(store, BehaviorKindConfig(retire_days=retire_days))

    counted = counter.count(series_of(str(word), last - timedelta(days=1), last), TODAY)

    assert counted.added == 0
    assert counted.waiting == 0
    reread = kind_store(tmp_path).read()
    assert dict(reread.active) == {}
    assert reread.revision == 0


@pytest.mark.parametrize("retire_days", RETIRE_DAYS)
def test_r3_through_the_generation_step_writes_the_active_word(tmp_path: Path, retire_days: int) -> None:
    """R3（走 ``VocabularyCounter.count``，对照）：差 ``retire_days - 1`` 天的词转在用，重新读盘能读到。"""

    store = kind_store(tmp_path)
    word = store.read().words[0].id
    last = TODAY - timedelta(days=retire_days - 1)
    counter = VocabularyCounter(store, BehaviorKindConfig(retire_days=retire_days))

    counted = counter.count(series_of(str(word), last - timedelta(days=1), last), TODAY)

    assert counted.added == 1
    assert counted.waiting == 0
    reread = kind_store(tmp_path).read()
    assert dict(reread.active) == {word: TODAY}
    assert reread.revision == 1
