"""概念层只留"词表节点 + 日历"（2026-11-14）的特征测试：钉住改动前后都应当不变的行为（预期表 S2、S3、S4/C3、S6/C4、S7）。

只通过会留下来的接口断言：概念的身份、显示名、判据、lane、停用；日历情境的键与取值；情境命中的名字；关系键与当晚量出来的数；
日子预期的类、时窗、最晚时限。不断言细分、汇总、派生情境、常态、未决这些要去掉的东西。数值快照在基线提交 ``9f23a13`` 上取得。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from habitus.conditions import DAY_OF_MONTH_KEY, DAY_TYPE_KEY, WEEKDAY_KEY, LocalCalendar
from habitus.foresight import Moment
from habitus.foresight.opportunities import build_table
from habitus.foresight.triggers import day_drafts
from habitus.scene.concepts.calendar_situations import (
    calendar_concepts,
    calendar_key,
    most_specific,
    value_set,
    value_sets_to_add,
)
from habitus.scene.concepts.catalog import CatalogWord, CatalogWords
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.concepts.sync import plan_sync
from habitus.scene.occurrences.situations import SituationInputs, situation_hits
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import LaneTests, RelationConfig, build_timelines
from habitus.scene.relations.spans import Segment
from habitus.series.reader import read_series
from habitus.temporal import Rulers
from tests.unit.scene.fixtures import Site, publish

ZONE = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 4, 28, 1, 0, tzinfo=ZONE)
RUN, DIARY, MINUTES, WEEKLY = "s-k0023", "s-k0031", "s-k0040", "s-k0041"
#: 在用的词（词表方案 2026-11-14 起语义树只拿在用的词，没有停用的类）。
CLASSES = (
    CatalogWord(RUN, "跑步", "户外或跑步机上跑步", "session"),
    CatalogWord(DIARY, "写日记", "写当天的日记", "session"),
    CatalogWord(MINUTES, "写会议纪要", "整理会议的纪要", "session"),
    CatalogWord(WEEKLY, "写周报", "写这一周的工作周报", "session"),
    CatalogWord("p-k0007", "开车", "驾驶汽车", "physical"),
)
RULERS = Rulers(15, 2700, 3, 5.0, 5.0, 35)
RELATIONS = RelationConfig(slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2)


def synced(observed: tuple[tuple[str, str], ...] = ()) -> ConceptSet:
    """空概念集按 ``CLASSES`` 同步一次后的概念集。"""

    plan = plan_sync(ConceptSet(()), CatalogWords(1, CLASSES), now=NOW, observed=observed)
    return ConceptSet(plan.write)


# ── S2：每个词表类一个行为概念 ────────────────────────────────────────────────


def test_s2_sync_writes_one_behaviour_concept_per_class_with_identity_label_criterion_and_lane() -> None:
    concepts = synced()
    behaviours = {identity: concepts[identity] for identity in concepts.behaviors()}
    assert set(behaviours) == {item.id for item in CLASSES}
    for item in CLASSES:
        concept = behaviours[item.id]
        assert concept.name == item.id
        assert concept.label == item.name
        assert concept.definition == item.definition
        assert concept.lane == item.lane
        assert not concept.retired
        assert concepts.base_for(item.id) == item.id
    assert concepts.class_label(MINUTES) == "写会议纪要"


def test_s2_a_renamed_word_rewrites_its_concept_under_the_same_identity() -> None:
    concepts = synced()
    changed = tuple(
        CatalogWord(item.id, "长跑", "跑十公里以上", item.lane) if item.id == RUN else item for item in CLASSES
    )
    plan = plan_sync(concepts, CatalogWords(2, changed), now=NOW + timedelta(days=1))
    rewritten = {item.name: item for item in plan.write}
    assert set(rewritten) == {RUN}
    assert (rewritten[RUN].label, rewritten[RUN].definition, rewritten[RUN].retired) == ("长跑", "跑十公里以上", False)
    # 没变的词不再写
    assert plan_sync(ConceptSet((*concepts.values(),)), CatalogWords(1, CLASSES), now=NOW).write == ()


def test_s2_behaviour_concepts_survive_a_store_round_trip(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "scene")
    store.initialize()
    for definition in synced().values():
        store.write(definition)
    read = store.read_all()
    assert sorted(read.behaviors()) == sorted(item.id for item in CLASSES)
    assert {(read[i].label, read[i].definition, read[i].lane, read[i].retired) for i in read.behaviors()} == {
        (item.name, item.definition, item.lane, False) for item in CLASSES
    }


# ── S3：日历情境 ─────────────────────────────────────────────────────────────


def _calendar_of(concepts: ConceptSet) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {concepts[identity].name: key for identity in concepts if (key := calendar_key(concepts, identity))}


def test_s3_sync_gives_calendar_situations_only_for_values_seen_in_the_data() -> None:
    observed = ((WEEKDAY_KEY, "周三"), (DAY_TYPE_KEY, "工作日"), (DAY_OF_MONTH_KEY, "22"))
    concepts = synced(observed)
    assert _calendar_of(concepts) == {
        "周三": (WEEKDAY_KEY, ("周三",)),
        "工作日": (DAY_TYPE_KEY, ("工作日",)),
        "22号": (DAY_OF_MONTH_KEY, ("22",)),
    }
    # 再同步一次、数据里多了一个周四：只补周四
    plan = plan_sync(concepts, CatalogWords(1, CLASSES), now=NOW, observed=(*observed, (WEEKDAY_KEY, "周四")))
    assert [item.name for item in plan.write] == ["周四"]


def test_s3_single_values_value_sets_and_most_specific() -> None:
    every = ConceptSet(calendar_concepts(now=NOW))
    names = {item.name for item in every.values()}
    assert {"周一", "周日", "1号", "31号", "月末", "工作日"} <= names
    grouped = value_set(WEEKDAY_KEY, ("周六", "周一", "周三"), now=NOW)
    assert grouped.name == "周一、周三、周六"
    assert calendar_key(ConceptSet((grouped,)), grouped.name) == (WEEKDAY_KEY, ("周一", "周三", "周六"))
    added = value_sets_to_add(every, {RUN: ("周一", "周三", "周六", "工作日")}, now=NOW)
    assert [item.name for item in added] == ["周一、周三、周六"]
    with_set = ConceptSet((*every.values(), *added))
    assert most_specific(with_set, ("周一", "周三", "周一、周三、周六", "工作日")) == ("周一、周三、周六", "工作日")
    assert most_specific(every, ("31号", "月末")) == ("月末",)


# ── S4 / C3：一条记录成立的日历情境、判过的日历情境 ────────────────────────────


def test_s4_c3_a_wednesday_workday_record_holds_exactly_the_calendar_situations_in_the_set() -> None:
    observed = ((WEEKDAY_KEY, "周三"), (WEEKDAY_KEY, "周四"), (DAY_TYPE_KEY, "工作日"), (DAY_OF_MONTH_KEY, "22"))
    concepts = synced(observed)
    moment = datetime(2026, 4, 22, 6, 10, tzinfo=ZONE)
    calendar = {WEEKDAY_KEY: "周三", DAY_TYPE_KEY: "工作日", DAY_OF_MONTH_KEY: "22"}
    outcome = situation_hits(concepts, SituationInputs(day=moment.date(), calendar=calendar))
    assert sorted(hit.concept for hit in outcome.hits) == ["22号", "周三", "工作日"]
    assert sorted(outcome.checked) == ["22号", "周三", "周四", "工作日"]
    # 没有日历可读：一个都不判（不知道 ≠ 否）
    blind = situation_hits(concepts, SituationInputs(day=moment.date(), calendar=None))
    assert blind.hits == () and blind.checked == ()


def test_s4_the_local_calendar_feeds_the_same_situations() -> None:
    """日历由外部条件包给（``LocalCalendar``）：2026-04-22 是周三、22 号。"""

    day = date(2026, 4, 22)
    given = dict(LocalCalendar().of_day(day))
    concepts = synced(tuple(given.items()))
    outcome = situation_hits(concepts, SituationInputs(day=day, calendar=given))
    held = {hit.concept for hit in outcome.hits}
    assert {"周三", "22号"} <= held
    assert held == set(outcome.checked)


# ── S6 / C4：两个词表类之间的关系键与当晚量出来的数 ──────────────────────────


def _minutes_then_weekly(tmp_path: Path) -> Site:
    """X =「写会议纪要」在 20 个不同的日子 09:00 做，其中 18 天 09:30 接着「写周报」；「写周报」另在两个没写纪要的日子 16:00
    出现。每天 08:00 跑步、11:00 写日记、18:00 再跑一次，让这一天从早到晚都在场（对照要整段在场才配得上）。"""

    site = Site(tmp_path, now=NOW)
    start = date(2026, 3, 1)
    for offset in range(30):
        day = start + timedelta(days=offset)
        if offset < 20:
            publish(site.behavior_tree, day, "写会议纪要", 9, 0, kind=MINUTES, lasts_minutes=20)
            if offset not in (6, 13):
                publish(site.behavior_tree, day, "写周报", 9, 30, kind=WEEKLY, lasts_minutes=20)
        if offset in (22, 27):
            publish(site.behavior_tree, day, "写周报", 16, 0, kind=WEEKLY, lasts_minutes=20)
        publish(site.behavior_tree, day, "晨跑", 8, 0, kind=RUN, lasts_minutes=20)
        publish(site.behavior_tree, day, "写日记", 11, 0, kind=DIARY, lasts_minutes=10)
        publish(site.behavior_tree, day, "夜跑", 18, 0, kind=RUN, lasts_minutes=20)
    return site


def test_s6_c4_the_minutes_to_weekly_report_key_exists_and_measures_the_same(tmp_path: Path) -> None:
    site = _minutes_then_weekly(tmp_path)
    series = read_series(site.behavior_tree, cutoff=date(2026, 3, 31))
    concepts = synced()
    line = build_timelines(series, concepts, {}, RELATIONS)["session"]
    tests = LaneTests(line, concepts, RELATIONS, series.cutoff, established=())
    keys = {key for key in tests.keys() if not key.condition}
    wanted = [key for key in keys if (key.antecedent, key.consequent) == (MINUTES, WEEKLY)]
    assert {key.segment for key in wanted} >= {Segment.CHAIN}
    night = tests.night()
    found = {
        (test.key.antecedent, test.key.consequent, test.key.segment, test.key.condition): test for test in night.tests
    }
    test = found[(MINUTES, WEEKLY, Segment.CHAIN, "")]
    reading = (
        test.verdict.value,
        test.antecedents,
        test.blocks,
        round(test.effect.treated_rate, 6),
        round(test.effect.control_rate, 6),
        None if test.interval is None else tuple(round(value, 6) for value in test.interval),
    )
    assert reading == SNAPSHOT_C4


#: 基线 ``9f23a13`` 上量出来的数（裁定式快照：改动后必须逐项相同）。
SNAPSHOT_C4: tuple[object, ...] = ("significant", 20, 20, 0.9, 0.0, (0.75, 1.0))


# ── S7：只由一个类自己的复现建出来的"日子"预期 ────────────────────────────────


def test_s7_a_daily_habit_gives_interval_day_expectations_from_its_own_recurrence(tmp_path: Path) -> None:
    site = Site(tmp_path, now=NOW)
    start = date(2026, 3, 1)
    for offset in range(15):
        day = start + timedelta(days=offset)
        publish(site.behavior_tree, day, "跑步", 8, 30, kind=RUN, lasts_minutes=30)
        publish(site.behavior_tree, day, "写日记", 21, 0, kind=DIARY, lasts_minutes=10)
    last = start + timedelta(days=14)
    concepts = synced()
    found = []
    for elapsed in (1, 2, 3):
        today = last + timedelta(days=elapsed)
        series = read_series(site.behavior_tree, cutoff=today)
        lines = build_timelines(series, concepts, {}, RELATIONS)
        table = build_table(
            lines["session"],
            (),
            concepts,
            night=today,
            relation_night=today,
            slot_minutes=15,
            window_slots=3,
            recurrences=recurrences_of(series, RULERS),
        )
        moment = Moment(at=datetime.combine(today, time(8, 0), tzinfo=ZONE), slot=32)
        drafts = [
            draft
            for draft in day_drafts(table, moment, done_today=frozenset(), recent=35, line=2 / 3)
            if draft.trigger == "interval"
        ]
        found.append(
            sorted(
                (
                    draft.kind_token,
                    draft.window[0].astimezone(ZONE).strftime("%m-%d %H:%M"),
                    draft.window[1].astimezone(ZONE).strftime("%m-%d %H:%M"),
                    draft.deadline.astimezone(ZONE).strftime("%m-%d %H:%M"),
                )
                for draft in drafts
            )
        )
    assert found == SNAPSHOT_S7


#: 基线 ``9f23a13`` 上的结果：离上次 1、2、3 天各在 08:00 读一次，两个天天做的类各一条，时窗只在当天、最晚时限是次日零点。
SNAPSHOT_S7: list[object] = [
    [(RUN, "03-16 08:30", "03-16 08:45", "03-17 00:00"), (DIARY, "03-16 21:00", "03-16 21:15", "03-17 00:00")],
    [(RUN, "03-17 08:30", "03-17 08:45", "03-18 00:00"), (DIARY, "03-17 21:00", "03-17 21:15", "03-18 00:00")],
    [(RUN, "03-18 08:30", "03-18 08:45", "03-19 00:00"), (DIARY, "03-18 21:00", "03-18 21:15", "03-19 00:00")],
]


def test_c1_one_class_and_three_calendar_values_sync_to_exactly_four_concepts() -> None:
    """预期表 C1：词表只有在用的 ``s-k0023``（跑步），数据里出现过周三、工作日、22 号——同步之后概念集恰好是这四个。"""

    run_only = CatalogWords(1, (CatalogWord(RUN, "跑步", "户外或跑步机上跑步", "session"),))
    observed = ((WEEKDAY_KEY, "周三"), (DAY_TYPE_KEY, "工作日"), (DAY_OF_MONTH_KEY, "22"))
    concepts = ConceptSet(plan_sync(ConceptSet(()), run_only, now=NOW, observed=observed).write)
    assert concepts.behaviors() == (RUN,)
    assert concepts.base_for(RUN) == RUN
    assert sorted(concepts[name].name for name in concepts.situations()) == ["22号", "周三", "工作日"]
    assert len(concepts) == 4
