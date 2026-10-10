"""节律口：七个周几逐槽平均后取峰、逐日数"哪几天有"、复发间隔取最短的那个类；给模型看的是类名。"""

from __future__ import annotations

import pytest

from habitus.runtime.scene_rhythms import TreeRhythms
from tests.unit.runtime.prediction_tree_fixtures import curve, tree, with_recurrences


def test_the_typical_day_averages_the_weekdays_and_days_with_peaks_counts_them_one_by_one() -> None:
    """节律是"平常一天什么样"，所以七天逐槽平均；而分型问的是"是不是天天有机会"，所以那一项逐日数。"""

    # 周一到周五 07:00–08:30 有峰，周六周日没有曲线 → 平均之后峰还在，days_with_peaks 是 5
    curves = {(weekday, "吃饭"): curve({14: 0.2, 15: 0.5, 16: 0.1}) for weekday in range(5)}
    built = tree(curves)
    rhythm = TreeRhythms(built, {"早餐": ("吃饭",)}).rhythm_of("早餐")
    assert rhythm.days_with_peaks == 5 and rhythm.has_rhythm
    assert [(peak.ordinal, peak.label) for peak in rhythm.peaks] == [(1, "07:00–08:30")]
    assert rhythm.peaks[0].probability == pytest.approx(0.8)
    assert "一天 1 个机会" in rhythm.render() and "有节律" in rhythm.render()


def test_a_behaviour_that_only_shows_up_on_two_weekdays_is_not_rhythmic() -> None:
    """只在周末冒头的（打球）落到无节律型那一套：不数机会、没有时效，读兑现率。"""

    built = tree({(weekday, "打球"): curve({30: 0.4, 31: 0.3}) for weekday in (5, 6)})
    rhythm = TreeRhythms(built, {"打球": ("打球",)}).rhythm_of("打球")
    assert rhythm.peaks and rhythm.days_with_peaks == 2 and not rhythm.has_rhythm
    assert "按无节律写" in rhythm.render()


def test_the_recurrence_median_is_the_shortest_of_its_classes_and_a_missing_curve_gives_an_empty_rhythm() -> None:
    """汇总概念的任一成员类发生它就发生，所以间隔不长于最短那个（上界估计）。"""

    built = with_recurrences(
        tree({(weekday, kind): curve({20: 0.5}) for weekday in range(7) for kind in ("修改代码", "审查代码")}),
        {"修改代码": 2.0, "审查代码": 24.0},
    )
    provider = TreeRhythms(built, {"写代码": ("修改代码", "审查代码"), "就诊": ()})
    assert provider.rhythm_of("写代码").recurrence_hours == pytest.approx(2.0)
    # 一条曲线都没有 → 空节律（不是报错）：承诺仍然能以无节律型开着。
    empty = provider.rhythm_of("就诊")
    assert empty.peaks == () and empty.days_with_peaks == 0 and not empty.has_rhythm
    assert set(provider.rhythms()) == {"写代码", "就诊"}


def test_a_base_concept_is_keyed_by_its_class_id_but_rendered_with_its_class_name() -> None:
    """基础概念的身份是类编号；节律交给提示词时显示类名（裁定 20：给模型看的一律是名字）。"""

    built = tree({(weekday, "s-k0007"): curve({14: 0.2, 15: 0.5}) for weekday in range(7)})
    provider = TreeRhythms(built, {"s-k0007": ("s-k0007",), "s-k0008": ()}, {"s-k0007": "早餐", "s-k0008": "就诊"})
    rhythm = provider.rhythm_of("s-k0007")
    assert rhythm.concept == "s-k0007" and rhythm.render().startswith("早餐：")
    assert provider.rhythm_of("s-k0008").render().startswith("就诊：")  # 空节律也用名字
    assert TreeRhythms(built, {"s-k0007": ("s-k0007",)}).rhythm_of("s-k0007").render().startswith("s-k0007：")
