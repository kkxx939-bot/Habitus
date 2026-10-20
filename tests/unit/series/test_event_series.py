"""事件序列（语义树新方案 ``13`` 第四节）：行为树的读法只写一处、截止日是类型的一部分、各个读者读同一份。

CI 钉的两条都在这里：
- 截止日之后追加任意数据，第 N 晚的序列与从它算出的复现（``temporal``，预测层「离上次够久」读它）都不变；
- 语义树映射与复现对同一条记录的读法一致（有类的进复现、待定的不进，是同一批 uri；预测树删了，裁定 59）。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from habitus.behavior import BehaviorLinkType
from habitus.behavior.kinds.ids import Lane, not_event_token, pending_token
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.occurrences import ConceptHitStore
from habitus.scene.occurrences.mapper import ConceptMapper, MapperConfig, map_closed_day
from habitus.scene.recurrence import recurrences_of
from habitus.series import EventSeries, GapKind, SeriesError, SkipReason
from habitus.series.reader import read_series
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.scene.concept_fixtures import BEDTIME, BREAKFAST, CODING
from tests.unit.scene.fixtures import DAY1, SUBJECT, Site, publish, publish_gap

NOW = datetime(2026, 9, 1, 3, 0, tzinfo=UTC)
PENDING = pending_token(Lane.SESSION)
RULERS = RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35)
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
    """第 N 晚只用 N 之前的数据：在 N 当天及以后追加任意记录与空白，第 N 晚的序列和从它算出的复现都不变。"""

    site = Site(tmp_path, now=NOW)
    a_week(site)
    tonight = day(5)

    def night_n() -> tuple[EventSeries, dict]:  # type: ignore[type-arg]
        series = read_series(site.behavior_tree, cutoff=tonight)
        return series, dict(recurrences_of(series, RULERS))

    before = night_n()
    publish(site.behavior_tree, tonight, "又改代码", 0, 10, kind="修改代码")
    publish(site.behavior_tree, tonight, "说不清", 9, 0, kind=PENDING)
    publish(site.behavior_tree, day(9), "吃早饭", 8, 0, kind="早餐")
    publish_gap(site.behavior_tree, tonight, (1, 0), (6, 0), kind="没读懂")
    after = night_n()
    assert after[0] == before[0]
    assert after[1] == before[1]
    # 第 N+1 晚就看得到了
    assert read_series(site.behavior_tree, cutoff=day(6)) != before[0]


def test_the_semantic_layer_and_the_recurrences_read_the_same_records(tmp_path) -> None:
    """同一条记录各个读者读法一致：语义树映射下来的有类记录，就是复现里数到的那几类；待定的两边都不当成一类的发生
    （同一份序列给的）。第二轮评审 D5 就是两处各写一遍写反了（「待定」在一处是占位、在另一处被当成"没来"）。
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
    recurrences = recurrences_of(series, RULERS)

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
    # 类 / 待定两边分得一样：复现里的类 = 语义树里有类的记录的类，待定的不在复现里
    assert set(recurrences) == {record.kind_token for record in mapped.values() if record.classified}
    assert PENDING not in recurrences and NOT_EVENT not in recurrences
    assert not mapped[pending].classified and mapped[pending].kind_token == PENDING
    # 编号与 lane 一致
    by_uri = {record.uri: record for record in series.records}
    for uri, record in mapped.items():
        assert (record.kind_token, record.lane) == (by_uri[uri].kind_token, by_uri[uri].lane)
    # 并行：序列里读到的就是那一对
    assert len(series.concurrent) == 1 and paired in series.concurrent[0]


def test_a_reminded_record_is_carried_and_the_recurrence_counts_it_as_done_but_not_natural(tmp_path) -> None:
    """序列原样带着 ``reminded``；复现里那一天算做了（"离上次多久"按全部发生算），但不是自然发生（不进命中，裁定 54）。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, day(0), "吃早饭", 8, 0, kind="早餐", reminded=True)
    series = read_series(site.behavior_tree, cutoff=day(1))
    assert series.records[0].reminded
    (breakfast,) = recurrences_of(series, RULERS).values()
    (only,) = breakfast.days
    assert (only.count, only.natural, only.natural_uri) == (1, 0, None)


def test_a_series_refuses_what_it_cannot_hold() -> None:
    with pytest.raises(SeriesError, match="cutoff must be a date"):
        EventSeries(cutoff=datetime(2026, 1, 1, tzinfo=UTC))  # type: ignore[arg-type]
    with pytest.raises(SeriesError, match="concurrent pairs"):
        EventSeries(cutoff=day(1), concurrent=(("a", "b"),))


def test_a_disproved_unreadable_gap_is_dropped_once_for_every_reader(tmp_path) -> None:
    """「没读懂」10:15–10:30 里读出了 10:20 那条行为的开始：这段空白被证伪、整段作废。序列这一处消解，temporal 与关系检验读到
    同一份——关系时间线上 10:15 那一槽算人在场（第四轮评审 E2：以前关系时间线不作废，"同一条链"丢掉的恰好是后果出现的那一次）。
    「未观测」不被证伪（留给消费方当矛盾报）。"""

    from habitus.scene.relations import RelationConfig, build_timelines
    from tests.unit.scene.concept_fixtures import concept_set

    site = Site(tmp_path, now=datetime(2026, 8, 20, tzinfo=UTC))
    publish(site.behavior_tree, DAY1, "写代码", 10, 0, kind="修改代码")
    publish(site.behavior_tree, DAY1, "又写代码", 10, 20, kind="修改代码")
    publish_gap(site.behavior_tree, DAY1, (10, 15), (10, 30), kind="没读懂")
    publish_gap(site.behavior_tree, DAY1, (6, 0), (7, 0), kind="未观测")
    series = read_series(site.behavior_tree, cutoff=DAY1 + timedelta(days=1))
    assert [gap.kind for gap in series.gaps] == [GapKind.UNOBSERVED]
    line = build_timelines(series, concept_set(), {}, RelationConfig(slot_minutes=15, transition_window_slots=3))[
        "session"
    ]
    assert 41 in line.present[DAY1]


def test_who_prompted_it_is_read_into_the_series_and_old_records_without_it_read_as_unknown(tmp_path) -> None:
    """行为树 ``prompted_by``（裁定 54）：写了就带进序列；没写的（已写好的旧记录）是 None。"""

    site = Site(tmp_path, now=datetime.combine(day(1), datetime.min.time(), UTC))
    publish(site.behavior_tree, day(0), "查快递", 16, 30, prompted_by="host_agent")
    publish(site.behavior_tree, day(0), "查快递", 18, 45)
    series = read_series(site.behavior_tree, cutoff=day(1))
    assert [record.prompted_by for record in series.records] == ["host_agent", None]


def test_a_record_backfilled_three_days_late_shows_up_the_next_night(tmp_path) -> None:
    """26 号清单 6.7：序列按天缓存、每晚在内存里重新拼；补发进 3 天前的一条，归约把那一天记进脏日日志，下一晚重读那一天。
    没记脏的日子读缓存——同一棵树上改了却不记，下一晚看不见（缓存确实在起作用、靠的就是这份日志）。"""

    from habitus.behavior.tree.dirty_days import DirtyDayLog
    from habitus.series.reader import SeriesReader

    site = Site(tmp_path, now=NOW)
    a_week(site)
    log = DirtyDayLog(tmp_path / "reduction")
    log.record({day(0)})  # 第一次读之前记的：缓存本来就是空的，跳过
    reader = SeriesReader(site.behavior_tree, (log,))
    first = reader.read(cutoff=day(7))
    late = publish(site.behavior_tree, day(4), "补发的一条", 16, 0, kind="修改代码")
    log.record({day(4)})
    unlogged = publish(site.behavior_tree, day(3), "改了却没记", 17, 0, kind="修改代码")
    second = reader.read(cutoff=day(8))
    uris = {record.uri for record in second.records}
    assert late in uris and late not in {record.uri for record in first.records}
    assert unlogged not in uris  # 没记脏：读的是缓存
    assert log.after(0) == (log.last(), frozenset())  # 读过的截掉了
    assert (
        read_series(site.behavior_tree, cutoff=day(8)).records
        == SeriesReader(site.behavior_tree).read(cutoff=day(8)).records
    )
    assert unlogged in {record.uri for record in read_series(site.behavior_tree, cutoff=day(8)).records}


def test_a_reader_without_a_dirty_day_log_never_serves_a_stale_day(tmp_path) -> None:
    """没有脏日日志就不知道哪天变过：不缓存，每次照读（L10.3；原先永久缓存，改过的日子一直读成旧的）。"""

    from habitus.series.reader import SeriesReader

    site = Site(tmp_path, now=NOW)
    a_week(site)
    reader = SeriesReader(site.behavior_tree)
    reader.read(cutoff=day(7))
    changed = publish(site.behavior_tree, day(3), "改过的一天", 17, 0, kind="修改代码")

    assert changed in {record.uri for record in reader.read(cutoff=day(8)).records}
