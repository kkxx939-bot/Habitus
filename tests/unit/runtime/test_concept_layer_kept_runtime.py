"""概念层只留"词表节点 + 日历"（2026-11-14）：换代的语义树一步封口、建命中记录（预期表 S4、S5、C2、C3、C7）。

只通过会留下来的口子驱动：组合根装语义树 → 换代 → 读命中记录与日完成章。改动之后命中记录在封口时由代码直接生成，所以一次换代
就该把已封口的日子封好。断言只碰会留下来的：每条命中记录的编号、lane、是不是已归类、成立与判过的日历情境；日完成章盖在哪些天、
章上的条数；对话模型一次都没被调用。

这两条在改动前的代码（``9f23a13``）上是红的：那时命中记录只由模型队列的"映射"一类建（``runtime.scene_night_mapping``），换代
只封口不建命中记录，没映射过的日子不盖章。下面写死的读数是在 ``9f23a13`` 上走"换代 → 按条映射 → 换代"取的（同一个现场，
测试报告里附了取数脚本与输出），就是改动前的行为：改动只去掉了映射这一段，这些读数不该变。
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

from habitus.behavior.kinds.ids import Lane, pending_token
from habitus.conditions import DAY_OF_MONTH_KEY, WEEKDAY_KEY
from habitus.runtime.scene_night import SceneNightlyRun
from habitus.scene.concepts.calendar_situations import calendar_key
from habitus.series.reader import read_series
from tests.unit.runtime.concept_layer_site import (
    CLASSES,
    CST,
    DIARY,
    EXPENSE,
    RUN,
    ModelMustNotBeCalled,
    generate,
    hits_on,
    scene_run,
)
from tests.unit.scene.fixtures import Site, publish

TUESDAY, WEDNESDAY = date(2026, 4, 21), date(2026, 4, 22)
NIGHT, NEXT_NIGHT = date(2026, 4, 23), date(2026, 4, 24)
PENDING = pending_token(Lane.SESSION)


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=CST)


def two_days(tmp_path: Path) -> tuple[Site, dict[str, str]]:
    site = Site(tmp_path, now=at(NIGHT, 1))
    uris = {
        "tue_diary": publish(site.behavior_tree, TUESDAY, "写日记", 21, 0, kind=DIARY),
        "run": publish(site.behavior_tree, WEDNESDAY, "晨跑", 6, 10, kind=RUN, lasts_minutes=40),
        "expense": publish(site.behavior_tree, WEDNESDAY, "补记一笔支出", 12, 0, kind=EXPENSE),
        "pending": publish(site.behavior_tree, WEDNESDAY, "说不清在做什么", 15, 0, kind=PENDING),
    }
    return site, uris


def assert_sealed_days_are_whole(run: SceneNightlyRun, night: date) -> None:
    """已盖章的每一天：行为树上那一天的每条记录都有一份命中记录、没有多出来的，章上的条数与它对得上。"""

    series = read_series(run.behavior_tree, cutoff=night)
    for day in run.hits.days_done():
        expected = {record.uri for record in series.on(day)}
        found = {record.occurrence_uri for record in hits_on(run, day)}
        assert found == expected, day
        marker = run.hits.read_marker(day)
        assert marker is not None and marker.records == len(expected), day


def test_s4_s5_c2_c3_c7_one_generation_seals_each_day_with_class_lane_and_calendar_without_the_model(
    tmp_path: Path,
) -> None:
    site, uris = two_days(tmp_path)
    client = ModelMustNotBeCalled()
    run = scene_run(tmp_path, site, client)

    report = generate(run, NIGHT, at(NIGHT, 1))
    concepts = run.concepts.read_all()
    # 同步词表：每个类一个行为概念 + 数据里出现过的日历取值（04-21 周二 21 号、04-22 周三 22 号）
    assert sorted(concepts.behaviors()) == sorted(CLASSES)
    calendar = {concepts[i].name: key for i in concepts if (key := calendar_key(concepts, i)) is not None}
    assert calendar == SNAPSHOT_CALENDAR
    # S5：一次换代就把截止日之前的两天封好、盖章
    assert run.hits.days_done() == frozenset({TUESDAY, WEDNESDAY})
    assert report.sealed_days == 2
    assert_sealed_days_are_whole(run, NIGHT)

    by_uri = {record.occurrence_uri: record for day in (TUESDAY, WEDNESDAY) for record in hits_on(run, day)}
    assert set(by_uri) == set(uris.values())
    reading = {
        name: (
            by_uri[uri].kind_token,
            by_uri[uri].lane,
            by_uri[uri].classified,
            tuple(sorted(hit.concept for hit in by_uri[uri].situation_hits)),
            tuple(sorted(by_uri[uri].situations_checked)),
        )
        for name, uri in uris.items()
    }
    assert reading == SNAPSHOT_RECORDS
    # C2：建命中记录、封口整个过程对话模型一次都没调
    assert client.names == []


def test_s5_after_each_generation_every_sealed_day_has_a_hit_record_per_record_and_the_seal_counts_them(
    tmp_path: Path,
) -> None:
    site = two_days(tmp_path)[0]
    client = ModelMustNotBeCalled()
    run = scene_run(tmp_path, site, client)
    generate(run, NIGHT, at(NIGHT, 1))
    assert run.hits.days_done() == frozenset({TUESDAY, WEDNESDAY})
    assert_sealed_days_are_whole(run, NIGHT)

    # 盖章之后 04-22 又补进一条记录：下一次换代把这一天按 4 条重新封好，章上的条数跟着变
    publish(site.behavior_tree, WEDNESDAY, "夜跑", 20, 0, kind=RUN)
    site.now = at(NEXT_NIGHT, 1)
    generate(run, NEXT_NIGHT, at(NEXT_NIGHT, 1))
    assert run.hits.days_done() == frozenset({TUESDAY, WEDNESDAY})
    assert_sealed_days_are_whole(run, NEXT_NIGHT)
    marker = run.hits.read_marker(WEDNESDAY)
    assert marker is not None and marker.records == 4
    assert client.names == []


#: 基线 ``9f23a13`` 上的结果（LocalCalendar 不知道日型，只给周几、几号；月末不出现）。
SNAPSHOT_CALENDAR: dict[str, object] = {
    "21号": (DAY_OF_MONTH_KEY, ("21",)),
    "22号": (DAY_OF_MONTH_KEY, ("22",)),
    "周三": (WEEKDAY_KEY, ("周三",)),
    "周二": (WEEKDAY_KEY, ("周二",)),
}
_CHECKED = ("21号", "22号", "周三", "周二")
#: 每条记录：编号、lane、是否已归类、成立的日历情境、判过的日历情境。「待定」那条有命中记录、没有类，日历情境照常（C7）。
SNAPSHOT_RECORDS: dict[str, object] = {
    "expense": (EXPENSE, "session", True, ("22号", "周三"), _CHECKED),
    "pending": (PENDING, "session", False, ("22号", "周三"), _CHECKED),
    "run": (RUN, "session", True, ("22号", "周三"), _CHECKED),
    "tue_diary": (DIARY, "session", True, ("21号", "周二"), _CHECKED),
}
