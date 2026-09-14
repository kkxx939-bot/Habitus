"""还立着的前提：一轮扫一次规律级投影成一张表；按前提记账，没有窗口也没有过期。

这一组同时是旧实现四处毛病的回归：粒度按行为、消费者掉出窗口会复活、窗口两头都错、过期判断
长在语义层。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.uri import BehaviorURI
from habitus.scene.association.premises import Premise, PremiseTable
from habitus.scene.model import AssociationAddress
from habitus.scene.regularity import AssociationDocument, RegularityTree

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 30, tzinfo=UTC)
JUNE = date(2026, 6, 8)
SEPTEMBER = date(2026, 9, 11)


def moment(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def occurrence(name: str, at: datetime) -> str:
    return str(
        BehaviorURI.from_address(
            BehaviorAddress(kind=BehaviorKind.OCCURRENCE, occurred_on=at.date(), name=name, started_at=at)
        )
    )


def store(tmp_path) -> RegularityTree:
    tree = RegularityTree(tmp_path / "regularity")
    tree.initialize()
    return tree


def write(
    tree: RegularityTree,
    kind: str,
    name: str,
    at: datetime,
    *,
    left: tuple[tuple[str, str], ...] = (),
    consumed: tuple[tuple[str, str], ...] = (),
    complete: bool = True,
) -> AssociationDocument:
    document = tree.write(
        AssociationDocument(
            address=AssociationAddress(kind, at.date(), name, at),
            created_at=NOW,
            occurrence_uri=occurrence(name, at),
            context="一句话",
            left=left,
            consumed=consumed,
        )
    )
    if complete:
        tree.complete_day(kind, at.date(), records=len(tree.read_day(kind, at.date())), completed_at=NOW)
    return document


def standing(table: PremiseTable, kind: str, *, limit: int = 10, before: datetime | None = None) -> list[str]:
    premises, _signals = table.waiting_for(kind, before=before or moment(SEPTEMBER, 19), limit=limit)
    return [premise.text for premise in premises]


# ── 旧实现四处毛病的回归 ─────────────────────────────────────────────────────


def test_a_premise_nobody_consumed_is_still_standing(tmp_path) -> None:
    tree = store(tmp_path)
    write(tree, "买菜", "去超市买菜", moment(JUNE, 15), left=(("家里有今晚要用的食材", "做饭"),))

    table, signals = PremiseTable.scan(tree)

    assert signals == ()
    (premise,), _ = table.waiting_for("做饭", before=moment(SEPTEMBER, 19), limit=10)
    assert (premise.text, premise.waits_for, premise.created_on) == ("家里有今晚要用的食材", "做饭", JUNE)
    assert premise.standing and premise.consumptions == 0


def test_a_premise_from_three_months_ago_is_visible(tmp_path) -> None:
    """旧实现按 90 天的日历窗收集，一个月前的预约正是最该看的那种，却看不见。"""

    tree = store(tmp_path)
    write(tree, "挂号", "网上挂号", moment(JUNE, 9), left=(("挂了下周的号", "就诊"),))

    assert standing(PremiseTable.scan(tree)[0], "就诊", before=moment(SEPTEMBER, 9)) == ["挂了下周的号"]


def test_consuming_one_premise_leaves_the_others_from_the_same_behaviour(tmp_path) -> None:
    """旧实现按"整条行为被指过一次"判兑现：用掉菜，灯泡那条跟着消失。"""

    tree = store(tmp_path)
    shopping = write(
        tree, "买菜", "去超市买菜", moment(JUNE, 15), left=(("家里有菜", "做饭"), ("买到了灯泡", "换灯泡"))
    )
    write(tree, "做饭", "做晚饭", moment(JUNE, 19), consumed=((shopping.occurrence_uri, "家里有菜"),))

    table, _signals = PremiseTable.scan(tree)

    assert standing(table, "做饭") == []
    assert standing(table, "换灯泡") == ["买到了灯泡"]


def test_a_consumer_far_in_the_past_does_not_let_the_premise_come_back(tmp_path) -> None:
    """旧实现只在窗口内收集兑现：就诊那天滑出窗口，挂号那条前提会复活。"""

    tree = store(tmp_path)
    booking = write(tree, "挂号", "网上挂号", moment(JUNE, 9), left=(("挂了号", "就诊"),))
    write(tree, "就诊", "去医院", moment(JUNE, 10, 30), consumed=((booking.occurrence_uri, "挂了号"),))

    table, _signals = PremiseTable.scan(tree)

    assert standing(table, "就诊", before=moment(SEPTEMBER, 9)) == []
    (premise,) = table.all_premises()
    assert premise.consumptions == 1 and not premise.standing


# ── 取材口径 ─────────────────────────────────────────────────────────────────


def test_records_on_a_day_that_is_not_complete_yet_still_count(tmp_path) -> None:
    """记录先落盘、完成标记最后写。只认标记的话，那天写下的 left 永远看不见，而它 consumed
    掉的前提会一直算"还立着"，下一轮被另一条发生第二次用掉。"""

    tree = store(tmp_path)
    booking = write(tree, "挂号", "网上挂号", moment(JUNE, 9), left=(("挂了号", "就诊"),))
    write(tree, "就诊", "去医院", moment(JUNE, 10, 30), consumed=((booking.occurrence_uri, "挂了号"),), complete=False)

    assert tree.days_for("就诊") == frozenset()
    table, _signals = PremiseTable.scan(tree)
    assert standing(table, "就诊") == []


def test_one_unreadable_day_does_not_bring_down_the_whole_scan(tmp_path) -> None:
    """这个方法是整轮的第一条语句。任意一个杂文件掀翻整轮是不可接受的。"""

    tree = store(tmp_path)
    write(tree, "买菜", "去超市买菜", moment(JUNE, 15), left=(("家里有菜", "做饭"),))
    write(tree, "打球", "去球场", moment(JUNE, 19))
    junk = tree.root / "kinds" / "打球" / "2026" / "06" / "19" / "note.txt"
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_text("不是记录", encoding="utf-8")

    table, signals = PremiseTable.scan(tree)

    assert standing(table, "做饭") == ["家里有菜"]
    assert any("premises_skipped" in note and "打球" in note for note in signals)


def test_a_consumption_of_something_never_produced_is_reported(tmp_path) -> None:
    tree = store(tmp_path)
    ghost = occurrence("从没关联过的行为", moment(JUNE, 9))
    write(tree, "就诊", "去医院", moment(JUNE, 10, 30), consumed=((ghost, "挂了号"),))

    _table, signals = PremiseTable.scan(tree)

    assert any("premises_unmatched" in note for note in signals)


# ── 兑现是事实 ───────────────────────────────────────────────────────────────


def test_consumption_is_a_fact_that_survives_rather_than_deleting_the_premise() -> None:
    """默认不再摆出来，但"一条前提能不能服务多次"由取回口径决定，不由存储决定。"""

    card = occurrence("办健身卡", moment(JUNE, 10))
    table = PremiseTable([Premise(card, "办了健身卡", "健身", JUNE, moment(JUNE, 10))])
    table.consume(card, "办了健身卡")
    table.consume(card, "办了健身卡")

    (premise,) = table.all_premises()
    assert premise.consumptions == 2 and premise.text == "办了健身卡"


def test_a_consumption_naming_an_unknown_premise_is_reported_not_silently_dropped() -> None:
    buy = occurrence("去超市买菜", moment(JUNE, 15))
    table = PremiseTable([Premise(buy, "家里有菜", "做饭", JUNE, moment(JUNE, 15))])
    assert table.consume(buy, "家里有菜") is True
    assert table.consume(buy, "从没建立过的前提") is False


# ── 取回口径 ─────────────────────────────────────────────────────────────────


def test_only_premises_established_before_this_moment_are_offered() -> None:
    table = PremiseTable(
        [
            Premise(occurrence("甲", moment(JUNE, 15)), "更早建立的", "做饭", JUNE, moment(JUNE, 15)),
            Premise(occurrence("乙", moment(SEPTEMBER, 20)), "还没建立的", "做饭", SEPTEMBER, moment(SEPTEMBER, 20)),
        ]
    )

    assert standing(table, "做饭") == ["更早建立的"]


def test_the_waiting_kind_is_matched_by_canonical_identity() -> None:
    table = PremiseTable([Premise(occurrence("买肉", moment(JUNE, 15)), "买了肉", "Cooking", JUNE, moment(JUNE, 15))])

    assert standing(table, "cooking") == ["买了肉"]
    assert standing(table, "做饭") == []


def test_a_consumed_by_that_no_kind_matches_is_kept_not_judged() -> None:
    """对不上的只是不会被摆出来。拿词表把它判掉，就是在规定现实该长什么样。"""

    table = PremiseTable(
        [Premise(occurrence("网上挂号", moment(JUNE, 15)), "挂了下周的号", "从没见过的行为", JUNE, moment(JUNE, 15))]
    )

    assert standing(table, "就诊") == []
    assert [premise.text for premise in table.all_premises()] == ["挂了下周的号"]


def test_truncation_says_how_much_it_left_out(tmp_path) -> None:
    """单纯截断等于一条没写进配置里的过期：被挤出去的从此不再进任何一次提示词，
    也就永远不可能被兑现，而用户完全看不见自己放弃了什么。"""

    table = PremiseTable(
        [
            Premise(occurrence(f"第{n}件事", moment(JUNE, 8 + n)), f"第 {n} 条", "做饭", JUNE, moment(JUNE, 8 + n))
            for n in range(6)
        ]
    )

    premises, signals = table.waiting_for("做饭", before=moment(SEPTEMBER, 19), limit=4)

    assert len(premises) == 4
    assert any("premises_truncated: 2 more" in note for note in signals)


def test_truncation_keeps_the_oldest_as_well_as_the_newest() -> None:
    """只留最近的几条会把低频、长跨度的前提系统性挤掉——恰好是最值得关联的那一类。"""

    table = PremiseTable(
        [
            Premise(occurrence(f"第{n}件事", moment(JUNE, 8 + n)), f"第 {n} 条", "做饭", JUNE, moment(JUNE, 8 + n))
            for n in range(6)
        ]
    )

    assert standing(table, "做饭", limit=4) == ["第 5 条", "第 4 条", "第 1 条", "第 0 条"]


def test_a_short_list_is_returned_whole_without_a_signal() -> None:
    table = PremiseTable(
        [
            Premise(occurrence(f"第{n}件事", moment(JUNE, 8 + n)), f"第 {n} 条", "做饭", JUNE, moment(JUNE, 8 + n))
            for n in range(3)
        ]
    )

    premises, signals = table.waiting_for("做饭", before=moment(SEPTEMBER, 19), limit=4)

    assert [premise.text for premise in premises] == ["第 2 条", "第 1 条", "第 0 条"] and signals == ()


@pytest.mark.parametrize("limit", [0, -1, True])
def test_a_non_positive_limit_is_refused(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PremiseTable().waiting_for("做饭", before=NOW, limit=limit)  # type: ignore[arg-type]


def test_scanning_something_that_is_not_a_regularity_tree_is_refused() -> None:
    with pytest.raises(TypeError, match="RegularityTree"):
        PremiseTable.scan(object())  # type: ignore[arg-type]


def test_an_empty_tree_projects_an_empty_table(tmp_path) -> None:
    table, signals = PremiseTable.scan(store(tmp_path))
    assert table.all_premises() == () and signals == ()


# ── 自由文本的句子不许被当成路径段 ───────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["冰箱里有菜/水果", "明天 9:00 要复诊", "灯泡买好了.", "买了 A|B 两种", "con"],
)
def test_an_ordinary_sentence_is_a_usable_premise(tmp_path, text: str) -> None:
    """这些都是模型会写出来的正常句子。拿路径段校验器去卡它们，一句人话就能把整条流程卡死。"""

    tree = store(tmp_path)
    write(tree, "挂号", "网上挂号", moment(JUNE, 9), left=((text, "就诊"),))

    table, signals = PremiseTable.scan(tree)

    assert standing(table, "就诊") == [text] and signals == ()
