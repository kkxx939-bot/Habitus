"""概念命中过哪些 kind：叶子概念一对一，聚合概念跨几个；跨得多要报出来（分辨率信号）。"""

from __future__ import annotations

import pytest

from habitus.scene.occurrences import ConceptHit

from habitus.scene.views.kinds import concept_kinds, concept_overlap, kinds_by_concept
from tests.unit.scene.fixtures import DAY1, DAY2
from tests.unit.scene.ledger_fixtures import CONCEPTS, record


def test_a_leaf_concept_maps_to_one_kind_and_an_aggregate_concept_spans_several() -> None:
    """机会口要的就是这张表：概念 → 要相加的那几条曲线的键。跨 1 个时"相加"退化成"就用那条"。"""

    records = [
        # 「早餐」只被 kind「吃饭」命中过 → 叶子，一对一
        record(DAY1, "吃了碗面", 7, 30, "早餐", kind="吃饭"),
        record(DAY2, "吃早饭", 7, 40, "早餐", kind="吃饭"),
        # 「运动」被两个 kind 命中过 → 聚合，机会口要把两条曲线相加
        record(DAY1, "和同事打羽毛球", 19, 0, "运动", kind="打羽毛球"),
        record(DAY2, "晨跑", 6, 30, "运动", kind="跑步"),
        record(DAY2, "又打球", 19, 0, "运动", kind="打羽毛球"),
    ]
    spreads = {item.concept: item for item in concept_kinds(records, CONCEPTS)}

    breakfast = spreads["早餐"]
    assert breakfast.kinds == ("吃饭",) and breakfast.spread == 1 and breakfast.occurrences == 2

    exercise = spreads["运动"]
    # 次数多的在前：打羽毛球 2 次 > 跑步 1 次
    assert exercise.kinds == ("打羽毛球", "跑步") and exercise.spread == 2 and exercise.occurrences == 3

    assert kinds_by_concept(concept_kinds(records, CONCEPTS))["运动"] == ("打羽毛球", "跑步")


def test_hits_are_counted_as_written_without_walking_the_parent_chain() -> None:
    """只数记录上写着的命中，不沿 parent 链聚合——聚合是读侧的事，而这张表要喂给机会口。

    「打球」的上级是「运动」，但一条只命中「打球」的记录不会让「运动」也出现在表里。
    """

    records = [record(DAY1, "和同事打羽毛球", 19, 0, "打球", kind="打羽毛球")]
    spreads = {item.concept for item in concept_kinds(records, CONCEPTS)}
    assert spreads == {"打球"}


def test_concepts_that_no_longer_exist_are_skipped() -> None:
    """概念删了，它的历史命中不该复活（否则机会口会去问树一个已经没人定义的东西）。"""

    from habitus.scene.concepts import ConceptSet

    without = ConceptSet([CONCEPTS[identity] for identity in CONCEPTS if identity != "早餐"])
    records = [record(DAY1, "吃了碗面", 7, 30, "早餐", "咖啡", kind="吃饭")]
    spreads = {item.concept for item in concept_kinds(records, without)}
    assert spreads == {"咖啡"}


def test_the_overlap_between_concepts_is_counted_and_names_the_crowded_pairs() -> None:
    """概念之间重叠了多少：一条 occurrence 平均命中几个**叶子**，以及最常一起命中的那几对。

    2026-09-29 探针实测：16 个平级近义概念（0 个 parent）让平均命中数到 3.6，于是同一次前件命中开出
    几倍的承诺、清楚的因果被切成几条更薄的账。最常同时命中的那几对就是该合并或该挂同一上级的线索。
    """

    records = [
        record(DAY1, "改代码", 10, 0, ConceptHit("晚睡"), ConceptHit("早餐")),
        record(DAY1, "又改代码", 11, 0, ConceptHit("晚睡"), ConceptHit("早餐")),
        record(DAY1, "打球", 18, 0, ConceptHit("打球")),
    ]
    overlap = concept_overlap(records, CONCEPTS)
    assert overlap.records == 3 and overlap.hits == 5
    assert overlap.mean_hits == pytest.approx(5 / 3) and not overlap.diluted  # 1.67 还在 2 以内
    assert overlap.crowded() == ((("早餐", "晚睡"), 2),)  # 对按身份排序，不按命中顺序
    assert "平均命中 1.7 个概念" in overlap.render() and "早餐 + 晚睡 共 2 次" in overlap.render()
    # 平均命中数过高就报"在互相稀释"
    crowded = [record(DAY1, f"第{i}条", 10 + i, 0, ConceptHit("晚睡"), ConceptHit("早餐"), ConceptHit("打球")) for i in range(3)]
    assert concept_overlap(crowded, CONCEPTS).diluted
