"""留出回测、校准曲线、等渗校准与置换检验。

这是定档十三个参数唯一的仪器，所以它自己必须先站得住：**能把学到东西和没学到东西分开**，
并且不会靠"只在发生槽上评估"给自己刷分（那正是方法论教训里点名的虚高来源）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from prediction import evaluation
from prediction.errors import PredictionTreeError
from prediction.model import ObservedAction, ObservedGap
from prediction.source import BehaviorSnapshot
from tests.unit.prediction.prediction_fixtures import config

CST = timezone(timedelta(hours=8))
FIRST = date(2026, 6, 1)  # 周一
BUILT_AT = datetime(2026, 8, 16, 23, 0, tzinfo=CST)


def at(day_offset: int, hour: int, minute: int = 0) -> datetime:
    day = FIRST + timedelta(days=day_offset)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def snapshot(actions, gaps=()) -> BehaviorSnapshot:
    ordered = tuple(sorted(actions, key=lambda item: item.started_at))
    return BehaviorSnapshot(
        actions=ordered, gaps=tuple(gaps), concurrent=(), skipped_duplicates=0
    )


def act(name: str, day_offset: int, hour: int, minute: int = 0) -> ObservedAction:
    moment = at(day_offset, hour, minute)
    return ObservedAction(action=name, started_at=moment, day=moment.date())


def daily(name: str, days: int, *, hour: int) -> list[ObservedAction]:
    return [act(name, offset, hour) for offset in range(days)]


# --- 切分 ---------------------------------------------------------------------------------


def test_split_puts_the_cutoff_day_in_the_training_half() -> None:
    data = snapshot(daily("吃药", 10, hour=7))
    train, holdout = evaluation.split(data, cutoff=FIRST + timedelta(days=4))
    assert [item.day for item in train.actions][-1] == FIRST + timedelta(days=4)
    assert [item.day for item in holdout.actions][0] == FIRST + timedelta(days=5)
    assert len(train.actions) + len(holdout.actions) == 10


def test_split_drops_parallel_pairs_that_straddle_the_cutoff() -> None:
    """并行对是按下标存的；切开之后指向留出侧的那一半已经没有端点可指。"""

    actions = [act("吃饭", 0, 18), act("看手机", 0, 18, 5), act("吃饭", 9, 18), act("看手机", 9, 18, 5)]
    data = BehaviorSnapshot(
        actions=tuple(actions), gaps=(), concurrent=((0, 1), (2, 3)), skipped_duplicates=0
    )
    train, holdout = evaluation.split(data, cutoff=FIRST + timedelta(days=4))
    assert train.concurrent == ((0, 1),)
    assert holdout.concurrent == ()


# --- 评估集 -------------------------------------------------------------------------------


def test_every_checked_slot_enters_the_evaluation_set_not_only_the_ones_that_fired() -> None:
    """只在发生槽上评估会让高频行为靠自相关刷虚高，而且真实数据上看不出来。"""

    data = snapshot(daily("吃药", 40, hour=7))
    pairs = _pairs(data, cutoff=FIRST + timedelta(days=29), through=FIRST + timedelta(days=39))
    # 10 个留出日 × 96 个槽 × 1 个已知动作
    assert len(pairs) == 10 * 96
    assert sum(1 for _p, actual in pairs if actual) == 10


def test_days_with_no_records_at_all_still_enter_the_evaluation_set() -> None:
    """"那天什么都没做"是最该被预测到的阴性样本；按"有记录的天"收集会把它整天丢掉。"""

    actions = daily("吃药", 30, hour=7) + [act("吃药", offset, 7) for offset in (30, 31, 33, 35, 37)]
    pairs = _pairs(
        snapshot(actions), cutoff=FIRST + timedelta(days=29), through=FIRST + timedelta(days=39)
    )
    assert len(pairs) == 10 * 96  # 不是只有 5 个有记录的日子
    base_rate = sum(1 for _p, actual in pairs if actual) / len(pairs)
    assert base_rate == pytest.approx(5 / 960, rel=0.01)  # 不是 5/480 的两倍虚高


def test_slots_hidden_by_a_gap_do_not_enter_the_evaluation_set() -> None:
    """那段时间他做了我们也看不见；当成"没发生"就是在惩罚观测质量。"""

    data = snapshot(
        daily("吃药", 40, hour=7),
        [ObservedGap(started_at=at(35, 0), ended_at=at(36, 0))],  # 留出期内整整一天没在看
    )
    pairs = _pairs(data, cutoff=FIRST + timedelta(days=29), through=FIRST + timedelta(days=39))
    assert len(pairs) == 9 * 96


def test_a_gap_straddling_the_cutoff_is_split_across_both_halves() -> None:
    """整段留给训练侧会让留出首日看起来"全程在看"，没观测到的槽被当成阴性打了分。"""

    data = snapshot(
        daily("吃药", 40, hour=7),
        [ObservedGap(started_at=at(29, 23), ended_at=at(30, 9))],  # 跨过截止日
    )
    train, holdout = evaluation.split(data, cutoff=FIRST + timedelta(days=29))
    assert len(train.gaps) == 1 and len(holdout.gaps) == 1
    assert train.gaps[0].ended_at == at(30, 0)
    assert holdout.gaps[0].started_at == at(30, 0)
    # 留出首日 00:00–09:00 的 36 个槽因此不进评估集
    pairs = _pairs(data, cutoff=FIRST + timedelta(days=29), through=FIRST + timedelta(days=39))
    assert len(pairs) == 10 * 96 - 36


def _pairs(data, *, cutoff, through):
    from prediction import builder

    cfg = config()
    train, holdout = evaluation.split(data, cutoff=cutoff)
    tree = builder.build(train, config=cfg, reference=cutoff, built_at=BUILT_AT)
    return evaluation.samples(
        tree, holdout, config=cfg, since=cutoff + timedelta(days=1), through=through
    )


# --- 成绩单 -------------------------------------------------------------------------------


def test_a_learnable_habit_beats_the_time_blind_baseline() -> None:
    """每天固定时刻吃药：树必须显著赢过"只看整体基础率"。"""

    report = evaluation.backtest(
        snapshot(daily("吃药", 60, hour=7)),
        config=config(),
        cutoff=FIRST + timedelta(days=49),
        built_at=BUILT_AT,
    )
    assert report.samples == 10 * 96
    assert report.positives == 10
    assert report.log_loss < report.baseline_log_loss
    assert report.bits_gained > 0.0


def test_pure_noise_does_not_beat_the_baseline() -> None:
    """时刻完全随机的行为学不出东西——仪器必须如实报告这一点，而不是总给正分。"""

    import random

    generator = random.Random(7)
    actions = [act("发呆", offset, generator.randrange(24)) for offset in range(60)]
    report = evaluation.backtest(
        snapshot(actions),
        config=config(),
        cutoff=FIRST + timedelta(days=49),
        built_at=BUILT_AT,
    )
    assert report.bits_gained < 0.02  # 与基线打平，不该刷出可观的提升


def test_calibration_bins_report_what_was_said_against_what_happened() -> None:
    pairs = [(0.9, True)] * 90 + [(0.9, False)] * 10 + [(0.1, True)] * 10 + [(0.1, False)] * 90
    curve = evaluation.calibration(pairs, bins=10)
    high = next(item for item in curve if item.lower == pytest.approx(0.9))
    low = next(item for item in curve if item.lower == pytest.approx(0.1))
    assert high.observed == pytest.approx(0.9)
    assert low.observed == pytest.approx(0.1)
    assert max(item.error for item in curve) < 0.01


def test_expected_calibration_error_catches_systematic_overconfidence() -> None:
    """说 0.9 结果只发生了一成——ECE 必须把这件事叫出来。"""

    honest = [(0.9, True)] * 9 + [(0.9, False)]
    overconfident = [(0.9, True)] + [(0.9, False)] * 9
    assert evaluation._expected_calibration_error(honest, 10) < 0.05
    assert evaluation._expected_calibration_error(overconfident, 10) > 0.7


# --- 等渗 ---------------------------------------------------------------------------------


def test_isotonic_repairs_calibration_without_reordering() -> None:
    """等渗只重排数值不改排序：校准修好了，区分度一点没损失。"""

    pairs = [(0.2, False)] * 8 + [(0.2, True)] * 2 + [(0.8, False)] * 3 + [(0.8, True)] * 7
    curve = evaluation.isotonic(pairs)
    low = evaluation.apply(curve, 0.2)
    high = evaluation.apply(curve, 0.8)
    assert low == pytest.approx(0.2, abs=0.01)
    assert high == pytest.approx(0.7, abs=0.01)
    assert low < high  # 排序没被动过


def test_isotonic_is_monotone_even_when_the_raw_scores_are_not() -> None:
    pairs = [(0.1, True), (0.2, False), (0.3, False), (0.4, True), (0.5, True)]
    curve = evaluation.isotonic(pairs)
    values = [evaluation.apply(curve, threshold) for threshold, _ in curve]
    assert values == sorted(values)


def test_isotonic_of_nothing_is_the_identity() -> None:
    assert evaluation.isotonic(()) == ()
    assert evaluation.apply((), 0.42) == 0.42


# --- 置换检验 -----------------------------------------------------------------------------


def test_permutation_test_finds_a_real_weekly_effect() -> None:
    """每周二打球：打乱"哪天"之后成绩应当明显变差，p 值因此很小。"""

    actions = [
        act("打球", offset, 19)
        for offset in range(70)
        if (FIRST + timedelta(days=offset)).weekday() == 1
    ]
    actions.extend(daily("吃药", 70, hour=7))  # 给树一点与周几无关的背景
    p_value = evaluation.permutation_test(
        snapshot(actions),
        config=config(),
        cutoff=FIRST + timedelta(days=55),
        built_at=BUILT_AT,
        rounds=30,
        seed=3,
    )
    assert p_value < 0.2


def test_permutation_test_is_deterministic_for_a_given_seed() -> None:
    data = snapshot(daily("吃药", 60, hour=7))
    kwargs = dict(
        config=config(), cutoff=FIRST + timedelta(days=49), built_at=BUILT_AT, rounds=5, seed=11
    )
    assert evaluation.permutation_test(data, **kwargs) == evaluation.permutation_test(data, **kwargs)


# --- 输入纪律 -----------------------------------------------------------------------------


def test_an_empty_holdout_window_fails_loudly() -> None:
    data = snapshot(daily("吃药", 10, hour=7))
    with pytest.raises(PredictionTreeError, match="holdout half"):
        evaluation.backtest(
            data, config=config(), cutoff=FIRST + timedelta(days=30), built_at=BUILT_AT
        )
