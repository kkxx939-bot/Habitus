"""事件序列（语义树新方案 ``13`` 第四节）：行为树的读法只写一处、截止日是类型的一部分、两棵树读同一份。

CI 钉的两条都在这里：
- 截止日之后追加任意数据，第 N 晚的序列与预测树逐字节不变；
- 两棵树对同一条记录的读法一致（预测树的动作 / 占位与语义树映射的记录，是同一批 uri）。
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from habitus.behavior import BehaviorLinkType
from habitus.behavior.kinds.ids import Lane, not_event_token, pending_token
from habitus.prediction import builder, codec, source
from habitus.prediction.errors import PredictionTreeError
from habitus.scene.concepts import ConceptSet
from habitus.scene.occurrences import ConceptHitStore
from habitus.scene.occurrences.mapper import ConceptMapper, MapperConfig, map_closed_day
from habitus.series import EventSeries, GapKind, SeriesError, SkipReason
from habitus.series.reader import read_series
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.prediction.prediction_fixtures import config
from tests.unit.scene.concept_fixtures import BEDTIME, BREAKFAST, CODING
from tests.unit.scene.fixtures import DAY1, SUBJECT, Site, publish, publish_gap

NOW = datetime(2026, 9, 1, 3, 0, tzinfo=UTC)
PENDING = pending_token(Lane.SESSION)
NOT_EVENT = not_event_token(Lane.SESSION)


def day(offset: int) -> date:
    return DAY1 + timedelta(days=offset)


def a_week(site: Site) -> None:
    """一周的会话：每天写代码、晚上睡觉，夹着待定、非事件、撞车重复、被提醒过的一条与两种空白。"""

    for offset in range(7):
        publish(site.behavior_tree, day(offset), "改代码", 10, 0, kind="修改代码")
        publish(site.behavior_tree, day(offset), "上床", 23, 30, kind="就寝", lasts_minutes=20)
    publish(site.behavior_tree, day(1), "说不清的一件事", 15, 0, kind=PENDING)
    publish(site.behavior_tree, day(1), "看了眼手机", 15, 30, kind=NOT_EVENT)
    publish(site.behavior_tree, day(2), "改代码-2", 10, 0, kind="修改代码", original_name="改代码")
    publish_gap(site.behavior_tree, day(3), (12, 0), (13, 0), kind="没读懂")
    publish_gap(site.behavior_tree, day(4), (23, 55), (0, 5), kind="未观测", end_day=day(5))


def test_the_reading_rules_live_in_one_place(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    a_week(site)
    meal = publish(site.behavior_tree, day(2), "吃早饭", 8, 0, kind="早餐", reminded=True, goal="吃饱")

    series = read_series(site.behavior_tree, cutoff=day(5))
    assert series.cutoff == day(5) and series.last_day == day(4)
    assert {record.day for record in series.records} == {day(offset) for offset in range(5)}
    pending = [record for record in series.records if not record.classified]
    assert [(record.name, record.kind_token) for record in pending] == [("说不清的一件事", PENDING)]
    assert {record.lane for record in series.records} == {"session"}
    # 非事件与撞车重复不进序列，只留可观测量
    assert series.skipped_on(day(1), SkipReason.NOT_EVENT) == 1 and series.skipped_on(day(2), SkipReason.DUPLICATE) == 1
    assert all(record.kind_token != NOT_EVENT for record in series.records)
    # 被提醒过的原样带着，目标、概要也在
    (reminded,) = [record for record in series.records if record.uri == meal]
    assert reminded.reminded and reminded.goal == "吃饱" and reminded.summary
    # 两种空白；跨过截止时刻的截到那一刻（第 4 天 23:55 起的未观测本来到第 5 天 00:05）
    kinds = {(gap.started_at.date(), gap.kind) for gap in series.gaps}
    assert kinds == {(day(3), GapKind.UNREADABLE), (day(4), GapKind.UNOBSERVED)}
    crossing = next(gap for gap in series.gaps if gap.kind is GapKind.UNOBSERVED)
    assert crossing.ended_at == series.boundary_for(crossing.started_at)
    assert not crossing.watched and next(gap for gap in series.gaps if gap.watched)


def test_cutting_a_series_equals_reading_at_that_cutoff(tmp_path) -> None:
    """回放、回测只读一次行为树，再一天一天往前截：截出来的必须与直接按那个截止日读的逐值相等。"""

    site = Site(tmp_path, now=NOW)
    a_week(site)
    latest = read_series(site.behavior_tree, cutoff=day(7))
    for offset in range(8):
        assert latest.until(day(offset)) == read_series(site.behavior_tree, cutoff=day(offset))
    with pytest.raises(SeriesError, match="past its own cutoff"):
        latest.until(day(8))


def test_appending_data_from_the_cutoff_on_changes_nothing_on_night_n(tmp_path) -> None:
    """第 N 晚只用 N 之前的数据：在 N 当天及以后追加任意记录与空白，第 N 晚的序列和建出来的预测树逐字节不变。"""

    site = Site(tmp_path, now=NOW)
    a_week(site)
    tonight = day(5)

    def night_n() -> tuple[EventSeries, str]:
        series = read_series(site.behavior_tree, cutoff=tonight)
        tree = builder.build(source.snapshot_of(series), config=config(), reference=series.last_day, built_at=NOW)
        return series, json.dumps(codec.encode(tree), ensure_ascii=False, sort_keys=True)

    before = night_n()
    # 第 4 天 23:30 的就寝之后 00:10 才来的那条：窗口跨过截止时刻，第 N 晚记删失，不算"没有后继"
    publish(site.behavior_tree, tonight, "又改代码", 0, 10, kind="修改代码")
    publish(site.behavior_tree, tonight, "说不清", 9, 0, kind=PENDING)
    publish(site.behavior_tree, day(9), "吃早饭", 8, 0, kind="早餐")
    publish_gap(site.behavior_tree, tonight, (1, 0), (6, 0), kind="没读懂")
    after = night_n()
    assert after[0] == before[0]
    assert after[1] == before[1]
    # 第 N+1 晚就看得到了
    assert read_series(site.behavior_tree, cutoff=day(6)) != before[0]


def test_the_reference_day_must_be_sealed() -> None:
    series = EventSeries(cutoff=day(3))
    with pytest.raises(PredictionTreeError, match="sealed day"):
        builder.build(source.snapshot_of(series), config=config(), reference=day(3), built_at=NOW)


def test_both_trees_read_the_same_records(tmp_path) -> None:
    """同一条记录两棵树读法一致：预测树的动作 / 占位与语义树映射下来的记录是同一批（同一份序列给的）。

    第二轮评审 D5 就是两处各写一遍写反了（「待定」在预测树里是占位、在语义树里被当成"没来"）。
    """

    site = Site(tmp_path, now=NOW)
    a_week(site)
    pending = publish(site.behavior_tree, day(2), "说不清的另一件事", 16, 0, kind=PENDING)
    paired = publish(site.behavior_tree, day(2), "边吃边看", 12, 0, kind="早餐")
    publish(
        site.behavior_tree,
        day(2),
        "看文档",
        12,
        5,
        kind="修改代码",
        links=((BehaviorLinkType.CONCURRENT_WITH.value, paired),),
    )
    series = read_series(site.behavior_tree, cutoff=day(7))
    snapshot = source.snapshot_of(series)

    hits = ConceptHitStore(tmp_path / "scene")
    client, _provider = recording_client([])
    mapper = ConceptMapper(
        client,
        ConceptSet([BEDTIME, BREAKFAST, CODING]),  # 只有基础概念：不问模型
        config=MapperConfig(transient_retry_delay_seconds=0.0),
        subject=SUBJECT,
        clock=lambda: NOW,
    )
    for offset in range(7):
        asyncio.run(
            map_closed_day(
                site.behavior_tree,
                hits,
                mapper,
                day(offset),
                now=NOW,
                series=series,
                situation_for=lambda _d: (),
                baseline_for=lambda _d: {},
            )
        )
    mapped = {record.occurrence_uri: record for offset in range(7) for record in hits.read_day(day(offset))}
    assert set(mapped) == {record.uri for record in series.records}
    # 类 / 待定两边分得一样：预测树的动作数 = 语义树里有类的记录数，占位数 = 待定记录数
    assert len(snapshot.actions) == sum(record.classified for record in mapped.values())
    assert len(snapshot.unnamed) == sum(not record.classified for record in mapped.values())
    assert not mapped[pending].classified and mapped[pending].kind_token == PENDING
    # 编号与 lane 一致
    by_uri = {record.uri: record for record in series.records}
    for uri, record in mapped.items():
        assert (record.kind_token, record.lane) == (by_uri[uri].kind_token, by_uri[uri].lane)
    # 并行：两边读的是同一对
    assert len(series.concurrent) == 1 and len(snapshot.concurrent) == 1
    assert paired in series.concurrent[0]


def test_a_reminded_record_is_refused_by_the_prediction_tree_until_randomisation_is_recorded(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, day(0), "吃早饭", 8, 0, kind="早餐", reminded=True)
    series = read_series(site.behavior_tree, cutoff=day(1))
    assert series.records[0].reminded
    with pytest.raises(PredictionTreeError, match="reminded"):
        source.snapshot_of(series)


def test_a_series_refuses_what_it_cannot_hold() -> None:
    with pytest.raises(SeriesError, match="cutoff must be a date"):
        EventSeries(cutoff=datetime(2026, 1, 1, tzinfo=UTC))  # type: ignore[arg-type]
    with pytest.raises(SeriesError, match="concurrent pairs"):
        EventSeries(cutoff=day(1), concurrent=(("a", "b"),))


def test_a_disproved_unreadable_gap_is_dropped_once_for_every_reader(tmp_path) -> None:
    """「没读懂」10:15–10:30 里读出了 10:20 那条行为的开始：这段空白被证伪、整段作废。序列这一处消解，预测树与关系检验读到
    同一份——关系时间线上 10:15 那一槽算人在场（第四轮评审 E2：以前关系时间线不作废，"同一条链"丢掉的恰好是后果出现的那一次）。
    「未观测」不被证伪（留给预测树当矛盾报）。"""

    from habitus.scene.relations import RelationConfig, build_timelines
    from tests.unit.scene.concept_fixtures import concept_set

    site = Site(tmp_path, now=datetime(2026, 8, 20, tzinfo=UTC))
    publish(site.behavior_tree, DAY1, "写代码", 10, 0, kind="修改代码")
    publish(site.behavior_tree, DAY1, "又写代码", 10, 20, kind="修改代码")
    publish_gap(site.behavior_tree, DAY1, (10, 15), (10, 30), kind="没读懂")
    publish_gap(site.behavior_tree, DAY1, (6, 0), (7, 0), kind="未观测")
    series = read_series(site.behavior_tree, cutoff=DAY1 + timedelta(days=1))
    assert [gap.kind for gap in series.gaps] == [GapKind.UNOBSERVED]
    line = build_timelines(
        series, concept_set(), {}, RelationConfig(slot_minutes=15, transition_window_slots=3, recurrence_window_days=90)
    )["session"]
    assert 41 in line.present[DAY1]
