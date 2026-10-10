"""① 概念定义：身份、机械判据与档、四种概念（基础 / 细分 / 汇总 / 情境）的形状、概念集的查询、落盘回读、
存储的容量 / 规范叶名 / 同步版本，以及同步词表的规划与词表口的形状（裁定 20）。"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from habitus.behavior.kinds.ids import Lane
from habitus.scene.codec import SceneRecordError
from habitus.scene.concepts import (
    BaselineKey,
    BaselineStatistic,
    BaselineWindow,
    ConceptDefinition,
    ConceptError,
    ConceptGrade,
    ConceptKind,
    ConceptLookupError,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
    ConceptStore,
    ConceptStoreError,
    ContextScope,
    GradeMeasure,
    MechanicalRule,
    circular_offset,
    concept_identity,
    parse_baseline_value,
)
from habitus.scene.concepts.catalog import CatalogChanges, CatalogClass, MovedOccurrence
from habitus.scene.concepts.document import decode, encode
from habitus.scene.concepts.situation import SituationBasis, SituationRule
from habitus.scene.concepts.sync import base_concept, plan_sync
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import (
    ALL_CONCEPTS,
    BALL,
    BEDTIME,
    BEDTIME_KEY,
    BEDTIME_KEY_ALL,
    BREAKFAST,
    CODING,
    CREATED,
    EAT_ON_WAKING,
    EXERCISE,
    LATE_GRADES,
    LATE_RULE,
    REWORK,
    RUNNING,
    SLEEP_LATE,
    TRAVELLING,
    WEEKEND,
    cid,
    concept_set,
    group,
    refinement,
    situation,
)

START = GradeMeasure.START_MINUTE_OF_DAY
DURATION = GradeMeasure.DURATION_MINUTES
LATER = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)

# ── 身份 ──────────────────────────────────────────────────────────────────────


def test_identity_folds_case_and_unicode_and_rejects_the_leaf_separators() -> None:
    assert concept_identity("Late Night") == concept_identity("late night") == "late night"
    assert concept_identity(kind_id("就寝")) == kind_id("就寝")  # 类编号本身就是规范身份
    for bad in (".hidden", "晚睡.md", "出差中+晚睡", "晚睡@重", "a--b"):
        with pytest.raises(ConceptError):
            concept_identity(bad)
    with pytest.raises(ConceptError, match="safe non-empty path segment"):
        ConceptGrade("轻/早", DURATION, 0, 10)
    with pytest.raises(ConceptError, match="must not contain"):
        ConceptGrade("a+b", DURATION, 0, 10)


# ── 机械判据与档 ───────────────────────────────────────────────────────────────


def test_the_clock_difference_wraps_around_midnight() -> None:
    """整个语义树只有这一处环形差（判据偏移、常态中位数、漂移都用它）：02:10 比 23:30 晚 160 分钟。

    读侧不需要它——时刻读数比的是绝对时刻（``observed_at − 机会.at``），跨午夜天然就对，所以
    ``views/stats.py`` 里那个同名函数是零调用者，已删（评审 C-16）。
    """

    assert circular_offset(2 * 60 + 10, 23 * 60 + 30) == 160.0
    assert circular_offset(22 * 60, 23 * 60 + 30) == -90.0
    assert circular_offset(0, 12 * 60) == -720.0  # 正好半天：落在负侧，钟面上两边等价


def test_a_relative_rule_is_decided_by_the_algorithm_with_a_circular_offset() -> None:
    """02:10 比常态 23:30 晚 160 分钟（不是早 1280 分钟）→ 命中「晚睡」，落在「轻」；04:10 → 重；22:00 → 不命中。"""

    baseline = {BEDTIME_KEY: "23:30"}
    late = SLEEP_LATE.decide({START: 2 * 60 + 10}, baseline)
    assert (late.hit, late.value) == (True, 160.0)
    assert SLEEP_LATE.grade_for({START: 2 * 60 + 10}, baseline) == "轻"
    assert SLEEP_LATE.grade_for({START: 4 * 60 + 10}, baseline) == "重"
    early = SLEEP_LATE.decide({START: 22 * 60}, baseline)
    assert (early.hit, early.value) == (False, -90.0)
    # 材料缺：常态没给、常态解不出、量没给——都是"没判"，不是 false。
    assert SLEEP_LATE.decide({START: 130}, {}).hit is None
    assert SLEEP_LATE.decide({START: 130}, {BEDTIME_KEY: "深夜"}).hit is None
    assert SLEEP_LATE.decide({DURATION: 30}, baseline).hit is None
    assert SLEEP_LATE.required_baseline_keys == (BEDTIME_KEY,)
    assert LATE_RULE.criterion() == f"开始时刻相对常态（{BEDTIME_KEY}）的偏移 +120–… 分钟"
    assert LATE_GRADES[0].criterion() == "开始时刻相对常态的偏移 +120–+240 分钟"
    with pytest.raises(ConceptError, match="no mechanical rule"):
        REWORK.decide({START: 0}, {})


def test_baseline_values_are_parsed_strictly() -> None:
    assert parse_baseline_value(START, "23:30") == 23 * 60 + 30
    assert parse_baseline_value(START, " 07:05 ") == 7 * 60 + 5
    assert parse_baseline_value(START, "7:5") is None and parse_baseline_value(START, "25:00") is None
    assert parse_baseline_value(DURATION, "45") == 45.0 and parse_baseline_value(DURATION, "-1") is None
    assert parse_baseline_value(DURATION, None) is None


def test_absolute_grades_may_wrap_midnight_and_grading_is_mechanical() -> None:
    grades = (ConceptGrade("晚", START, 22 * 60, 2 * 60), ConceptGrade("很晚", START, 2 * 60, 6 * 60))
    item = refinement("夜间", "夜里开始的", "就寝", grades=grades)
    assert item.grade_for({START: 23 * 60 + 30}) == "晚"
    assert item.grade_for({START: 2 * 60 + 10}) == "很晚"
    assert item.grade_for({START: 12 * 60}) is None  # 落在所有档之外：命中仍算命中，不带档
    assert BREAKFAST.grade_for({START: 0}) is None  # 没有档
    assert grades[0].criterion() == "开始时刻 22:00–02:00"
    assert ConceptGrade("长", DURATION, 30, 90).criterion() == "时长 30–90 分钟"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(grades=(ConceptGrade("只有一档", DURATION, 0, 10),)), "2–3 grades"),
        (dict(grades=(ConceptGrade("a", DURATION, 0, 10), ConceptGrade("b", START, 0, 10))), "same measure"),
        (dict(grades=(ConceptGrade("a", DURATION, 0, 10), ConceptGrade("b", DURATION, 5, 15))), "overlap"),
        (dict(grades=(ConceptGrade("a", START, 22 * 60, 2 * 60), ConceptGrade("b", START, 60, 180))), "overlap"),
        (dict(grades=(ConceptGrade("a", DURATION, 0, 10), ConceptGrade("A", DURATION, 10, 20))), "distinct"),
        (
            dict(
                grades=(
                    ConceptGrade("a", DURATION, 0, 10, relative=True),
                    ConceptGrade("b", DURATION, 10, 20, relative=True),
                )
            ),
            "relative grades need",
        ),
        (
            dict(rule=LATE_RULE, grades=(ConceptGrade("a", START, 0, 10), ConceptGrade("b", START, 10, 20))),
            "relative offsets",
        ),
        (
            dict(
                rule=LATE_RULE,
                grades=(
                    ConceptGrade("a", DURATION, 0, 10, relative=True),
                    ConceptGrade("b", DURATION, 10, 20, relative=True),
                ),
            ),
            "same quantity",
        ),
        (dict(rule=LATE_RULE, context=ContextScope.DAY), "computed from the occurrence alone"),
        (dict(rule=LATE_RULE, context=ContextScope.RECENT), "computed from the occurrence alone"),
        (dict(rule=LATE_RULE, baseline_keys=(BEDTIME_KEY,)), "do not repeat"),
        (dict(definition="两行\n判据"), "single printable line"),
    ],
)
def test_a_refinement_is_self_consistent(kwargs, message) -> None:
    with pytest.raises(ConceptError, match=message):
        refinement("x", kwargs.pop("definition", "y"), "就寝", **kwargs)


def test_a_baseline_key_names_a_window_and_a_rule_may_only_compare_against_the_recent_one() -> None:
    """键是三段 ``<概念>:<统计>:<窗>``；概念可以是类编号（基础概念）。判据比近期常态，历来常态只用来读漂移。"""

    key = BaselineKey.parse(BEDTIME_KEY)
    assert (key.concept, key.statistic, key.window) == (
        BEDTIME.name,
        BaselineStatistic.USUAL_START,
        BaselineWindow.RECENT,
    )
    assert key.label == f"{BEDTIME.name}的近期常态时刻"
    assert BaselineKey.parse(BEDTIME_KEY_ALL).window is BaselineWindow.ALL
    with pytest.raises(ConceptError, match="is not of the form"):
        BaselineKey.parse("就寝:usual_start")  # 两段的旧写法不再接受
    with pytest.raises(ConceptError, match="unknown statistic or window"):
        BaselineKey.parse("就寝:usual_start:last_year")
    with pytest.raises(ConceptError, match="compares against the recent baseline"):
        MechanicalRule(START, lower=120, upper=None, relative_to=BEDTIME_KEY_ALL)
    # 判据句引用历来常态是允许的（那是给模型读的材料，不是数值判据）
    assert refinement("x", "y", "就寝", baseline_keys=(BEDTIME_KEY_ALL,)).required_baseline_keys == (BEDTIME_KEY_ALL,)


def test_rule_and_grade_bounds_are_checked() -> None:
    with pytest.raises(ConceptError):
        MechanicalRule(DURATION, None, None)
    with pytest.raises(ConceptError):
        MechanicalRule(DURATION, 20, 10)
    with pytest.raises(ConceptError):
        MechanicalRule(START, 0, 25 * 60)
    with pytest.raises(ConceptError):
        ConceptGrade("x", DURATION, 10, 10)
    with pytest.raises(ConceptError):
        ConceptGrade("x", START, 0, 25 * 60)
    with pytest.raises(ConceptError):
        ConceptGrade("x", START, 240, 120, relative=True)


# ── 四种概念的形状 ────────────────────────────────────────────────────────────


def test_a_base_concept_is_its_class_titled_with_the_class_name() -> None:
    assert BEDTIME.name == kind_id("就寝") and BEDTIME.identity == kind_id("就寝")
    assert (BEDTIME.kind, BEDTIME.classes, BEDTIME.lane, BEDTIME.title) == (
        ConceptKind.BASE,
        (kind_id("就寝"),),
        "session",
        "就寝",
    )
    assert BEDTIME.label == "就寝" and SLEEP_LATE.label == "晚睡"  # 显示名：基础概念是类名，其余是自己的名字
    assert not BEDTIME.is_judged and SLEEP_LATE.is_judged and not EXERCISE.is_judged and not WEEKEND.is_judged
    with pytest.raises(ConceptError, match="named by its one class id"):
        replace(BEDTIME, name="就寝")  # 身份必须是类编号
    with pytest.raises(ConceptError, match="named by its one class id"):
        replace(BEDTIME, title=None)
    with pytest.raises(ConceptError, match="carries no rule, grade or material"):
        replace(BEDTIME, context=ContextScope.DAY)
    with pytest.raises(ConceptError, match="carries no rule, grade or material"):
        replace(BEDTIME, rule=MechanicalRule(DURATION, 0, 10))


def test_refinement_group_and_situation_concepts_keep_their_shapes() -> None:
    assert SLEEP_LATE.classes == (kind_id("就寝"),) and SLEEP_LATE.kind is ConceptKind.REFINEMENT
    assert EXERCISE.classes == (kind_id("打球"), kind_id("跑步")) and EXERCISE.kind is ConceptKind.GROUP
    with pytest.raises(ConceptError, match="exactly one source class"):
        replace(SLEEP_LATE, classes=(kind_id("就寝"), kind_id("早餐")))
    with pytest.raises(ConceptError, match="exactly one source class"):
        replace(SLEEP_LATE, title="晚睡")
    with pytest.raises(ConceptError, match="at least two member classes"):
        replace(EXERCISE, classes=(kind_id("打球"),))
    with pytest.raises(ConceptError, match="never judged"):
        replace(EXERCISE, context=ContextScope.DAY)
    with pytest.raises(ConceptError, match="names each class once"):
        replace(EXERCISE, classes=(kind_id("打球"), kind_id("打球")))
    with pytest.raises(ConceptError, match="relates to the vocabulary"):
        replace(SLEEP_LATE, kind=None, classes=(), lane=None)
    with pytest.raises(ConceptError, match="not tied to vocabulary classes"):
        replace(TRAVELLING, classes=(kind_id("就寝"),))
    with pytest.raises(ConceptError, match="only a base concept retires"):
        replace(SLEEP_LATE, retired=True)
    with pytest.raises(ConceptError, match="only a situation concept"):
        replace(SLEEP_LATE, situation=WEEKEND.situation)


# ── 概念集 ────────────────────────────────────────────────────────────────────


def test_a_concept_set_answers_the_class_questions() -> None:
    concepts = concept_set()
    assert set(concepts.behaviors()) == {
        cid("就寝"),
        "晚睡",
        cid("早餐"),
        "起床就吃",
        cid("修改代码"),
        "返工",
        cid("打球"),
        cid("跑步"),
        "运动",
    }
    assert set(concepts.situations()) == {"出差中", "周末"}
    assert concepts.base_for(kind_id("就寝")) == BEDTIME.identity and concepts.base_for(kind_id("吃药")) is None
    assert concepts.refinements_on(kind_id("就寝")) == ("晚睡",) and concepts.refinements_on(kind_id("打球")) == ()
    assert concepts.classes_of("运动") == (kind_id("打球"), kind_id("跑步")) and concepts.classes_of("周末") == ()
    assert concepts.class_label(kind_id("就寝")) == "就寝" and concepts.class_label(kind_id("吃药")) == kind_id("吃药")
    assert (
        concepts.label_of(cid("早餐")) == "早餐"
        and concepts.label_of("返工") == "返工"
        and concepts.label_of("咖啡") == "咖啡"
    )
    assert concepts[cid("打球")].role is ConceptRole.BEHAVIOR
    # Mapping 契约：查不到是 KeyError，``get`` 照常给默认值——账本按名字查 role 时不该炸。
    assert concepts.get("咖啡") is None and concepts.get("a/b") is None and ".hidden" not in concepts
    with pytest.raises(ConceptLookupError):
        concepts["咖啡"]


def test_groups_aggregate_their_member_classes_and_overlap_is_a_tautology() -> None:
    concepts = concept_set()
    assert (
        concepts.ancestors(cid("打球")) == ("运动",)
        and concepts.ancestors("晚睡") == ()
        and concepts.ancestors("运动") == ()
    )
    assert concepts.with_ancestors((cid("打球"), "晚睡")) == {cid("打球"), "运动", "晚睡"}
    # 读同一批记录的两个行为概念连起来是重言；细分概念量的是另一条记录，不算
    assert concepts.overlaps("运动", cid("打球")) and concepts.overlaps(cid("跑步"), "运动")
    assert not concepts.overlaps("晚睡", cid("就寝")) and not concepts.overlaps(cid("打球"), cid("跑步"))


def physical_base(title: str) -> ConceptDefinition:
    """物理 lane 的基础概念（概念夹具只造会话 lane 的）。"""

    class_id = kind_id(title, Lane.PHYSICAL)
    return ConceptDefinition(
        name=class_id,
        definition=f"{title}这件事",
        role=ConceptRole.BEHAVIOR,
        source=ConceptSource(ConceptOrigin.VOCABULARY),
        created_at=CREATED,
        kind=ConceptKind.BASE,
        classes=(class_id,),
        lane="physical",
        title=title,
    )


CRUNCH = situation(
    "连续运动中",
    "往前连着三个 24 小时都在运动",
    role=ConceptRole.DERIVED,
    rule=SituationRule(SituationBasis.STREAK, concept="运动", days=3),
)


def test_lane_of_follows_behaviours_and_the_concept_a_derived_situation_watches() -> None:
    """行为概念是自己的 lane；派生情境跟它盯的概念走；日历 / 对象情境不属于任何 lane（两边都能用）。"""

    meal = physical_base("吃饭")
    concepts = ConceptSet([*ALL_CONCEPTS, CRUNCH, meal])
    assert concepts.lane_of(cid("打球")) == "session" and concepts.lane_of("晚睡") == "session"
    assert concepts.lane_of("运动") == "session" and concepts.lane_of(meal.name) == "physical"
    assert concepts.lane_of("连续运动中") == "session"  # 盯的「运动」在会话 lane
    assert concepts.lane_of("周末") is None and concepts.lane_of("出差中") is None
    assert concepts.lane_of("不在集里") is None


def test_relatable_to_keeps_one_lane_plus_lane_free_situations_and_never_itself() -> None:
    meal = physical_base("吃饭")
    concepts = ConceptSet([*ALL_CONCEPTS, CRUNCH, meal])
    related = concepts.relatable_to(cid("早餐"))
    assert cid("早餐") not in related and meal.name not in related
    assert {cid("打球"), "晚睡", "运动", "连续运动中", "周末", "出差中"} <= set(related)
    assert concepts.relatable_to(meal.name) == ("出差中", "周末")  # 物理 lane 只有它自己 + 不属于任何 lane 的情境
    assert list(related) == sorted(related)


def test_labels_are_unique_within_a_lane_but_may_repeat_across_lanes() -> None:
    """给模型看的是名字（裁定 21-1）：同一条 lane 能一起出现的概念不许同名；两条 lane 各有一个「打球」可以。"""

    clash = refinement("打球", "和人约着打的那几次", "跑步")
    with pytest.raises(ConceptError, match="both show as"):
        ConceptSet([*ALL_CONCEPTS, clash])
    # 不属于任何 lane 的情境与会话 lane 的类同名也算撞
    with pytest.raises(ConceptError, match="both show as"):
        ConceptSet([*ALL_CONCEPTS, situation("打球", "在球场上")])
    both = ConceptSet([*ALL_CONCEPTS, physical_base("打球")])
    assert both.label_of(kind_id("打球", Lane.PHYSICAL)) == both.label_of(cid("打球")) == "打球"


def test_only_judged_concepts_enter_the_set_fingerprint() -> None:
    concepts = concept_set()
    assert len(concepts.fingerprint) == 16 and ConceptSet(ALL_CONCEPTS).fingerprint == concepts.fingerprint
    renamed = ConceptSet([replace(BEDTIME, title="睡觉"), *ALL_CONCEPTS[1:]])
    assert renamed.fingerprint == concepts.fingerprint  # 类改名不必重映射
    regrouped = ConceptSet([*ALL_CONCEPTS[:8], group("运动", "另一句", "打球", "跑步", "修改代码"), *ALL_CONCEPTS[9:]])
    assert regrouped.fingerprint == concepts.fingerprint  # 汇总改成员不必重映射
    changed = ConceptSet(
        [
            BEDTIME,
            refinement("晚睡", "入睡晚于常态三小时", "就寝", rule=LATE_RULE, grades=LATE_GRADES),
            *ALL_CONCEPTS[2:],
        ]
    )
    assert changed.fingerprint != concepts.fingerprint  # 改区别 → 指纹变 → 映射口径变


def test_a_concept_set_refuses_duplicates_and_unwatched_situations() -> None:
    with pytest.raises(ConceptError, match="appears twice"):
        ConceptSet([EXERCISE, group("运动", "另一句", "打球", "跑步")])
    watcher = situation(
        "连着打",
        "连着三天打球",
        role=ConceptRole.DERIVED,
        rule=SituationRule(SituationBasis.STREAK, concept=cid("打球"), days=3),
    )
    with pytest.raises(ConceptError, match="not in the set"):
        ConceptSet([watcher])
    assert "连着打" in ConceptSet([BALL, watcher])


# ── 落盘格式 ──────────────────────────────────────────────────────────────────


def test_a_concept_document_round_trips_and_rejects_a_tampered_body() -> None:
    text = encode(SLEEP_LATE)
    assert "# 晚睡" in text and "判据：开始时刻比近期常态晚两小时以上" in text
    assert f"判法：算法按区别规则判——开始时刻相对常态（{BEDTIME_KEY}）的偏移 +120–… 分钟" in text
    assert f"材料：只看这一条；常态 {BEDTIME_KEY}" in text
    assert "档：轻 = 开始时刻相对常态的偏移 +120–+240 分钟 · 重 = 开始时刻相对常态的偏移 +240–+720 分钟" in text
    assert "类别：行为 · 细分概念 · lane session" in text and f"类：{kind_id('就寝')}" in text
    assert decode(text) == SLEEP_LATE
    for item in ALL_CONCEPTS:
        assert decode(encode(item)) == item
    assert "材料：要看当天时间线" in encode(EAT_ON_WAKING) and "判法：模型按区别判据判" in encode(EAT_ON_WAKING)
    assert "材料：要看近几天同一个类的记录" in encode(REWORK)
    assert "# 早餐" in encode(BREAKFAST) and "判法：看记录的类编号（不判）" in encode(BREAKFAST)
    assert "来源：同步词表自动生成" in encode(BREAKFAST) and "来源：模型写" in encode(SLEEP_LATE)
    assert "判法：读时按成员类聚合（不判）" in encode(EXERCISE)
    retired = replace(BEDTIME, retired=True)
    assert "类已停用" in encode(retired) and decode(encode(retired)) == retired
    with pytest.raises(SceneRecordError, match="canonical rendering"):
        decode(text.replace("判据：开始", "判据：随便"))
    with pytest.raises(SceneRecordError, match="identity"):
        decode(text, expected_identity="早餐")
    with pytest.raises(SceneRecordError):
        decode(text.replace('"record_type":"concept"', '"record_type":"other"'))
    with pytest.raises(SceneRecordError, match="must be a boolean"):
        decode(text.replace('"retired":false', '"retired":"no"'))


# ── 存储 ──────────────────────────────────────────────────────────────────────


def test_store_writes_reads_lists_and_rebuilds_a_consistent_set(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    for definition in ALL_CONCEPTS:
        store.write(definition)
    assert set(store.identities()) == {item.identity for item in ALL_CONCEPTS}
    assert store.read("晚睡") == SLEEP_LATE and store.read(kind_id("早餐")) == BREAKFAST
    assert store.exists(cid("打球")) and not store.exists("吃药")
    assert set(store.read_all()) == set(concept_set())
    # .DS_Store、原子写临时文件、同步版本点文件与概念文件同目录：全是点文件，列目录时跳过。
    (store.directory / ".DS_Store").write_bytes(b"\x00")
    store.write_synced_version(3)
    assert "晚睡" in store.identities()


def test_store_overwrites_by_identity(tmp_path) -> None:
    """同步词表改类名、改判据就是这样覆写基础概念的：身份是编号，不变。"""

    store = ConceptStore(tmp_path / "scene")
    store.write(BALL)
    revised = replace(BALL, title="打篮球", definition="打一场篮球")
    store.write(revised)
    assert store.read(cid("打球")) == revised and len(store.identities()) == 1


def test_store_remembers_the_vocabulary_version_it_synced_to(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    assert store.synced_version() == 0  # 从没同步过
    store.write_synced_version(7)
    assert store.synced_version() == 7
    store.write_synced_version(9)
    assert store.synced_version() == 9
    with pytest.raises(ConceptStoreError, match="non-negative integer"):
        store.write_synced_version(-1)
    with pytest.raises(ConceptStoreError, match="non-negative integer"):
        store.write_synced_version(True)
    (store.directory / ".vocabulary-version").write_text("seven\n", encoding="utf-8")
    with pytest.raises(ConceptStoreError, match="corrupt"):
        store.synced_version()


def test_store_enforces_directory_capacity_before_writing(tmp_path) -> None:
    """写成一个列不出来的目录比拒绝写更糟；同身份覆写不占新位。"""

    store = ConceptStore(tmp_path / "scene", max_directory_entries=2)
    store.write(EXERCISE)
    store.write(SLEEP_LATE)
    store.write(group("运动", "改一句", "打球", "跑步"))  # 覆写
    with pytest.raises(ConceptStoreError, match="capacity"):
        store.write(BREAKFAST)


def test_store_reports_corrupt_and_non_canonical_records_and_refuses_a_symlinked_root(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    store.write(EXERCISE)
    path = store.path_for("运动")
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ConceptStoreError, match="corrupt"):
        store.read("运动")
    store.write(EXERCISE)
    # 非规范叶名（大小写不同）在 APFS 上会与规范名静默互认，在大小写敏感的机器上又变成"不存在"——统一报错。
    (store.directory / "Nap.md").write_text(encode(situation("nap", "小睡")), encoding="utf-8")
    with pytest.raises(ConceptStoreError, match="canonical identity"):
        store.identities()
    os.symlink(tmp_path / "scene", tmp_path / "link")
    with pytest.raises(ConceptStoreError, match="symbolic link"):
        ConceptStore(tmp_path / "link")


# ── 词表口与同步词表 ─────────────────────────────────────────────────────────


def catalog_class(
    title: str, *, criterion: str | None = None, active: bool = True, lane: str = "session"
) -> CatalogClass:
    return CatalogClass(
        id=kind_id(title), name=title, criterion=criterion or f"{title}这件事", lane=lane, active=active
    )


def test_catalog_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        CatalogClass(id="", name="x", criterion="y", lane="session", active=True)
    with pytest.raises(ValueError, match="boolean"):
        CatalogClass(id="s-k0001", name="x", criterion="y", lane="session", active=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot end before"):
        CatalogChanges(since=5, version=4)
    nothing = CatalogChanges(since=3, version=3)
    assert dict(nothing.split_from) == {} and nothing.moved == ()
    moved = CatalogChanges(since=0, version=1, moved=(MovedOccurrence("behavior://x", "s-待定", "s-k0001"),))
    assert moved.moved[0].target == "s-k0001"
    with pytest.raises(TypeError):
        moved.split_from["a"] = "b"  # type: ignore[index]


def test_the_first_sync_generates_a_base_concept_for_every_class() -> None:
    classes = (
        catalog_class("就寝", criterion="上床睡觉"),
        catalog_class("早餐"),
        catalog_class("看文档", active=False),
    )
    plan = plan_sync(ConceptSet([]), classes, CatalogChanges(since=0, version=4), now=LATER)
    assert plan.version == 4
    assert [item.name for item in plan.write] == [kind_id("就寝"), kind_id("早餐"), kind_id("看文档")]
    first = plan.write[0]
    assert (first.kind, first.title, first.definition, first.lane, first.retired) == (
        ConceptKind.BASE,
        "就寝",
        "上床睡觉",
        "session",
        False,
    )
    assert first.source.origin is ConceptOrigin.VOCABULARY and first.created_at == LATER
    assert plan.write[2].retired  # 停用的类也有基础概念，标成停用
    assert base_concept(classes[1], now=LATER) == plan.write[1]
    # 第二次同步：什么都不用写
    again = plan_sync(ConceptSet(plan.write), classes, CatalogChanges(since=4, version=4), now=LATER)
    assert again.write == () and again.moved == ()


def test_sync_follows_renames_criteria_and_retirement_keeping_identity_and_birth() -> None:
    existing = ConceptSet([BEDTIME, SLEEP_LATE, BALL])
    classes = (
        catalog_class("就寝", criterion="躺下准备睡"),
        CatalogClass(id=kind_id("打球"), name="打篮球", criterion=BALL.definition, lane="session", active=False),
    )
    plan = plan_sync(existing, classes, CatalogChanges(since=2, version=3), now=LATER)
    by_name = {item.name: item for item in plan.write}
    assert set(by_name) == {kind_id("就寝"), kind_id("打球")}
    assert by_name[kind_id("就寝")].definition == "躺下准备睡" and by_name[kind_id("就寝")].created_at == CREATED
    assert by_name[kind_id("打球")].title == "打篮球" and by_name[kind_id("打球")].retired
    # 一个不是基础概念的概念占了类编号这个名字：同步拒绝
    squatter = refinement(kind_id("早餐"), "y", "就寝")
    with pytest.raises(ValueError, match="is not the base concept"):
        plan_sync(
            ConceptSet([BEDTIME, squatter]), (catalog_class("早餐"),), CatalogChanges(since=0, version=1), now=LATER
        )


def test_sync_writes_the_classes_a_split_or_merge_touched_and_passes_the_moves_on() -> None:
    existing = ConceptSet([CODING, BALL])
    classes = (
        catalog_class("修改代码", criterion=CODING.definition),
        catalog_class("改提示词"),
        catalog_class("打球", criterion=BALL.definition, active=False),
        catalog_class("跑步"),
    )
    moves = (MovedOccurrence("behavior://occurrences/2026/08/15/x@0900.md", kind_id("修改代码"), kind_id("改提示词")),)
    changes = CatalogChanges(
        since=3,
        version=5,
        split_from={kind_id("改提示词"): kind_id("修改代码")},
        merged_into={kind_id("打球"): kind_id("跑步")},
        moved=moves,
    )
    plan = plan_sync(existing, classes, changes, now=LATER)
    assert plan.version == 5 and plan.moved == moves
    assert {item.name for item in plan.write} == {kind_id("改提示词"), kind_id("打球"), kind_id("跑步")}
    with pytest.raises(TypeError):
        plan_sync([CODING], classes, changes, now=LATER)  # type: ignore[arg-type]


def test_sync_rewrites_group_members_to_the_current_classes() -> None:
    """汇总概念的成员跟着拆分 / 合并改写成现编号：拆出来的加进来；并掉的换成目标类（去重）；不足两个就不动。"""

    classes = (
        catalog_class("打球", criterion=BALL.definition),
        catalog_class("跑步", criterion=RUNNING.definition),
        catalog_class("打羽毛球"),
        catalog_class("快走"),
    )
    existing = ConceptSet([BALL, RUNNING, EXERCISE])
    split = plan_sync(
        existing,
        classes,
        CatalogChanges(since=1, version=2, split_from={kind_id("打羽毛球"): kind_id("打球")}),
        now=LATER,
    )
    (regrouped,) = [item for item in split.write if item.name == "运动"]
    assert regrouped.classes == (cid("打球"), cid("跑步"), kind_id("打羽毛球"))
    assert regrouped.created_at == EXERCISE.created_at and regrouped.definition == EXERCISE.definition

    # 合并：跑步并进快走 → 成员换成快走
    merged = plan_sync(
        existing, classes, CatalogChanges(since=1, version=2, merged_into={cid("跑步"): kind_id("快走")}), now=LATER
    )
    (regrouped,) = [item for item in merged.write if item.name == "运动"]
    assert regrouped.classes == (cid("打球"), kind_id("快走"))

    # 并进的目标本来就在成员里 → 去重后只剩一个成员，保持原样不写
    collapsed = plan_sync(
        existing, classes, CatalogChanges(since=1, version=2, merged_into={cid("跑步"): cid("打球")}), now=LATER
    )
    assert "运动" not in {item.name for item in collapsed.write}
    # 没有拆分合并，汇总概念不写
    assert "运动" not in {
        item.name for item in plan_sync(existing, classes, CatalogChanges(since=1, version=1), now=LATER).write
    }
