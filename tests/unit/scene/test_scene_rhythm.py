"""节律由 temporal 算（裁定 59）：做的那天几点做、多久一次；汇总概念是成员类合起来的发生。"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import RhythmRulers, rhythms_of
from tests.unit.scene.concept_fixtures import BALL, EXERCISE, RUNNING, cid
from tests.unit.temporal.fixtures import at, record, series

RULERS = RhythmRulers(
    slot_minutes=15,
    merge_window_seconds=2700,
    pool_half_width=3,
    shrink_slot_to_pool=5.0,
    shrink_pool_to_all_day=5.0,
    min_peak_mass=0.10,
    recent=35,
)
START = date(2026, 3, 2)


def test_rhythm_of_a_weekly_kind_and_its_group() -> None:
    ball = [record(at(START + timedelta(days=7 * week), 19, 0), cid("打球")) for week in range(8)]
    run = [record(at(START + timedelta(days=7 * week + 3), 7, 0), cid("跑步")) for week in range(8)]
    found = rhythms_of(
        series(ball + run, cutoff=START + timedelta(days=56)),
        ConceptSet((BALL, RUNNING, EXERCISE)),
        RULERS,
        anchors={cid("打球"): ("周一",)},
    )
    weekly = found[cid("打球")]
    assert len(weekly.peaks) == 1 and weekly.peaks[0].start_minute <= 19 * 60 < weekly.peaks[0].end_minute
    assert weekly.days_per_occurrence == 7.0  # 56 天里 8 次
    assert weekly.render().startswith("打球：做的那天多在 ") and "锚在周一" in weekly.render()
    group = found["运动"]
    assert len(group.peaks) == 2  # 早上跑步、晚上打球：两个峰
    assert group.days_per_occurrence is not None and group.days_per_occurrence < weekly.days_per_occurrence


def test_a_concept_without_records_has_no_rhythm() -> None:
    found = rhythms_of(series([], cutoff=START), ConceptSet((BALL,)), RULERS)
    assert dict(found) == {}
