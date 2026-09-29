"""① 概念定义：身份、机械判据与档、概念集自洽与层级、落盘回读、存储的层级/容量/规范叶名约束。"""

from __future__ import annotations

import os

import pytest

from habitus.scene.codec import SceneRecordError
from habitus.scene.concepts import (
    BaselineKey,
    BaselineStatistic,
    BaselineWindow,
    ConceptError,
    ConceptGrade,
    ConceptLookupError,
    ConceptRole,
    ConceptSet,
    ConceptStore,
    ConceptStoreError,
    ContextScope,
    GradeMeasure,
    MechanicalRule,
    circular_offset,
    concept_identity,
    parse_baseline_value,
)
from habitus.scene.concepts.document import decode, encode
from tests.unit.scene.concept_fixtures import (
    ALL_CONCEPTS,
    BALL,
    BEDTIME_KEY,
    BEDTIME_KEY_ALL,
    BREAKFAST,
    EXERCISE,
    LATE_GRADES,
    LATE_RULE,
    SLEEP_LATE,
    concept,
    concept_set,
)

START = GradeMeasure.START_MINUTE_OF_DAY
DURATION = GradeMeasure.DURATION_MINUTES

# ── 身份 ──────────────────────────────────────────────────────────────────────


def test_identity_folds_case_and_unicode_and_rejects_the_leaf_separators() -> None:
    assert concept_identity("Late Night") == concept_identity("late night") == "late night"
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
        BALL.decide({START: 0}, {})


def test_baseline_values_are_parsed_strictly() -> None:
    assert parse_baseline_value(START, "23:30") == 23 * 60 + 30
    assert parse_baseline_value(START, " 07:05 ") == 7 * 60 + 5
    assert parse_baseline_value(START, "7:5") is None and parse_baseline_value(START, "25:00") is None
    assert parse_baseline_value(DURATION, "45") == 45.0 and parse_baseline_value(DURATION, "-1") is None
    assert parse_baseline_value(DURATION, None) is None


def test_absolute_grades_may_wrap_midnight_and_grading_is_mechanical() -> None:
    grades = (ConceptGrade("晚", START, 22 * 60, 2 * 60), ConceptGrade("很晚", START, 2 * 60, 6 * 60))
    item = concept("夜间", "夜里开始的行为", grades=grades)
    assert item.grade_for({START: 23 * 60 + 30}) == "晚"
    assert item.grade_for({START: 2 * 60 + 10}) == "很晚"
    assert item.grade_for({START: 12 * 60}) is None  # 落在所有档之外：命中仍算命中，不带档
    assert concept("x", "y").grade_for({START: 0}) is None  # 没有档
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
        (dict(grades=(ConceptGrade("a", DURATION, 0, 10, relative=True), ConceptGrade("b", DURATION, 10, 20, relative=True))), "relative grades need"),
        (dict(rule=LATE_RULE, grades=(ConceptGrade("a", START, 0, 10), ConceptGrade("b", START, 10, 20))), "relative offsets"),
        (dict(rule=LATE_RULE, grades=(ConceptGrade("a", DURATION, 0, 10, relative=True), ConceptGrade("b", DURATION, 10, 20, relative=True))), "same quantity"),
        (dict(rule=LATE_RULE, context=ContextScope.DAY), "does not take day context"),
        (dict(rule=LATE_RULE, baseline_keys=(BEDTIME_KEY,)), "do not repeat"),
        (dict(definition="两行\n判据"), "single printable line"),
        (dict(parent="X"), "own parent"),
    ],
)
def test_a_definition_is_self_consistent(kwargs, message) -> None:
    with pytest.raises(ConceptError, match=message):
        concept("x", kwargs.pop("definition", "y"), **kwargs)


def test_a_baseline_key_names_a_window_and_a_rule_may_only_compare_against_the_recent_one() -> None:
    """键是三段 ``<概念>:<统计>:<窗>``；判据比近期常态（2026-09-27 裁定），历来常态只用来读漂移。"""

    key = BaselineKey.parse(BEDTIME_KEY)
    assert (key.concept, key.statistic, key.window) == ("就寝", BaselineStatistic.USUAL_START, BaselineWindow.RECENT)
    assert key.label == "就寝的近期常态时刻"
    assert BaselineKey.parse(BEDTIME_KEY_ALL).window is BaselineWindow.ALL
    with pytest.raises(ConceptError, match="is not of the form"):
        BaselineKey.parse("就寝:usual_start")  # 两段的旧写法不再接受
    with pytest.raises(ConceptError, match="unknown statistic or window"):
        BaselineKey.parse("就寝:usual_start:last_year")
    with pytest.raises(ConceptError, match="compares against the recent baseline"):
        MechanicalRule(START, lower=120, upper=None, relative_to=BEDTIME_KEY_ALL)
    # 判据句引用历来常态是允许的（那是给模型读的材料，不是数值判据）
    assert concept("x", "y", baseline_keys=(BEDTIME_KEY_ALL,)).required_baseline_keys == (BEDTIME_KEY_ALL,)


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


# ── 概念集 ────────────────────────────────────────────────────────────────────


def test_a_concept_set_knows_leaves_ancestors_and_read_time_aggregation() -> None:
    concepts = concept_set()
    assert set(concepts.behaviors()) == {"运动", "打球", "晚睡", "早餐"}
    assert set(concepts.behavior_leaves()) == {"打球", "晚睡", "早餐"}  # 「运动」有子概念，不参与映射
    assert set(concepts.situations()) == {"出差中", "周末"}
    assert concepts.ancestors("打球") == ("运动",) and concepts.children("运动") == ("打球",)
    assert concepts.is_ancestor("运动", "打球") and not concepts.is_ancestor("打球", "运动")
    assert concepts.with_ancestors(("打球", "晚睡")) == {"打球", "运动", "晚睡"}
    assert "打球" in concepts and "Ball" not in concepts and ".hidden" not in concepts
    assert concepts["打球"].role is ConceptRole.BEHAVIOR
    # Mapping 契约：查不到是 KeyError，``get`` 照常给默认值——账本按名字查 role 时不该炸。
    assert concepts.get("咖啡") is None and concepts.get("a/b") is None
    with pytest.raises(ConceptLookupError):
        concepts["咖啡"]
    with pytest.raises(KeyError):
        concepts.ancestors("咖啡")
    assert len(concepts.fingerprint) == 16
    assert ConceptSet(ALL_CONCEPTS).fingerprint == concepts.fingerprint
    changed = ConceptSet([*ALL_CONCEPTS[:2], concept("晚睡", "入睡晚于常态三小时", rule=LATE_RULE, grades=LATE_GRADES), *ALL_CONCEPTS[3:]])
    assert changed.fingerprint != concepts.fingerprint  # 改判据句 → 指纹变 → 映射口径变


def test_a_concept_set_refuses_unknown_parents_duplicates_and_cycles() -> None:
    with pytest.raises(ConceptError, match="unknown parent"):
        ConceptSet([BALL])
    with pytest.raises(ConceptError, match="appears twice"):
        ConceptSet([EXERCISE, concept("运动", "另一句")])
    with pytest.raises(ConceptError, match="cycle"):
        ConceptSet([concept("a", "x", parent="b"), concept("b", "y", parent="a")])


# ── 落盘格式 ──────────────────────────────────────────────────────────────────


def test_a_concept_document_round_trips_and_rejects_a_tampered_body() -> None:
    text = encode(SLEEP_LATE)
    assert "判据：入睡时刻晚于常态两小时以上" in text
    assert f"判法：算法按规则判——开始时刻相对常态（{BEDTIME_KEY}）的偏移 +120–… 分钟" in text
    assert f"材料：只看这一条；常态 {BEDTIME_KEY}" in text
    assert "档：轻 = 开始时刻相对常态的偏移 +120–+240 分钟 · 重 = 开始时刻相对常态的偏移 +240–+720 分钟" in text
    assert decode(text) == SLEEP_LATE
    assert "材料：要看当天时间线" in encode(BREAKFAST) and "判法：模型按判据句判" in encode(BREAKFAST)
    assert decode(encode(BREAKFAST)) == BREAKFAST
    with pytest.raises(SceneRecordError, match="canonical rendering"):
        decode(text.replace("判据：入睡", "判据：随便"))
    with pytest.raises(SceneRecordError, match="identity"):
        decode(text, expected_identity="早餐")
    with pytest.raises(SceneRecordError):
        decode(text.replace('"record_type":"concept"', '"record_type":"other"'))
    with pytest.raises(SceneRecordError, match="non-canonical value"):
        decode(text.replace('"parent":null', '"parent":NaN'))  # json.loads 收 NaN，规范形式不收：报本层错误


# ── 存储 ──────────────────────────────────────────────────────────────────────


def test_store_writes_reads_lists_and_rebuilds_a_consistent_set(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    for definition in ALL_CONCEPTS:
        store.write(definition)
    assert set(store.identities()) == {"打球", "周末", "出差中", "晚睡", "运动", "早餐"}
    assert store.read("晚睡") == SLEEP_LATE
    assert store.exists("打球") and not store.exists("吃药")
    assert set(store.read_all()) == set(concept_set())
    # 旁册、.DS_Store、原子写临时文件与概念文件同目录：全是点文件，列目录时跳过。
    (store.directory / ".vectors.json").write_text("{}", encoding="utf-8")
    (store.directory / ".DS_Store").write_bytes(b"\x00")
    assert "早餐" in store.identities()


def test_store_requires_the_parent_first_and_overwrites_by_identity(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    with pytest.raises(ConceptStoreError, match="parent"):
        store.write(BALL)
    store.write(EXERCISE)
    store.write(BALL)
    revised = concept("打球", "和别人一起打一场球", parent="运动")
    store.write(revised)
    assert store.read("打球") == revised and len(store.identities()) == 2


def test_store_refuses_to_close_a_cycle_when_a_parent_is_rewritten(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    store.write(EXERCISE)
    store.write(BALL)
    with pytest.raises(ConceptStoreError, match="cycle"):
        store.write(concept("运动", EXERCISE.definition, parent="打球"))


def test_store_enforces_directory_capacity_before_writing(tmp_path) -> None:
    """写成一个列不出来的目录比拒绝写更糟；同身份覆写不占新位。"""

    store = ConceptStore(tmp_path / "scene", max_directory_entries=2)
    store.write(EXERCISE)
    store.write(SLEEP_LATE)
    store.write(concept("运动", "改一句"))  # 覆写
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
    (store.directory / "Nap.md").write_text(encode(concept("nap", "小睡")), encoding="utf-8")
    with pytest.raises(ConceptStoreError, match="canonical identity"):
        store.identities()
    os.symlink(tmp_path / "scene", tmp_path / "link")
    with pytest.raises(ConceptStoreError, match="symbolic link"):
        ConceptStore(tmp_path / "link")
