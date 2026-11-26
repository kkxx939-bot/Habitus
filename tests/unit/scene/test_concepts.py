"""① 概念定义：身份、两种概念（词表类的行为概念 / 日历情境）的形状、概念集的查询、落盘回读与严格拒收、
存储的容量 / 规范叶名 / 同步版本，以及同步词表的规划与词表口的形状（裁定 20、59）。"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from habitus.behavior.kinds.ids import Lane
from habitus.scene.codec import SceneRecordError
from habitus.scene.concepts import (
    ConceptDefinition,
    ConceptError,
    ConceptLookupError,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
    concept_identity,
)
from habitus.scene.concepts.calendar_situations import calendar_concepts
from habitus.scene.concepts.catalog import CatalogWord, CatalogWords
from habitus.scene.concepts.document import decode, encode
from habitus.scene.concepts.store import ConceptStore, ConceptStoreError
from habitus.scene.concepts.sync import base_concept, plan_sync
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import (
    ALL_CONCEPTS,
    BALL,
    BEDTIME,
    BREAKFAST,
    CREATED,
    WEEKEND,
    base,
    calendar,
    cid,
    concept_set,
)

LATER = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)

# ── 身份 ──────────────────────────────────────────────────────────────────────


def test_identity_folds_case_and_unicode_and_rejects_the_leaf_separators() -> None:
    assert concept_identity("Late Night") == concept_identity("late night") == "late night"
    assert concept_identity(kind_id("就寝")) == kind_id("就寝")  # 类编号本身就是规范身份
    for bad in (".hidden", "晚睡.md", "出差中+晚睡", "晚睡@重", "a--b"):
        with pytest.raises(ConceptError):
            concept_identity(bad)


# ── 两种概念的形状 ─────────────────────────────────────────────────────────────


def test_a_behaviour_concept_is_its_class_titled_with_the_class_name() -> None:
    assert BEDTIME.name == kind_id("就寝") and BEDTIME.identity == kind_id("就寝")
    assert (BEDTIME.role, BEDTIME.classes, BEDTIME.lane, BEDTIME.title) == (
        ConceptRole.BEHAVIOR,
        (kind_id("就寝"),),
        "session",
        "就寝",
    )
    assert BEDTIME.label == "就寝" and WEEKEND.label == "周末"  # 显示名：行为概念是类名，日历情境是自己的名字
    with pytest.raises(ConceptError, match="named by its own class id"):
        replace(BEDTIME, name="就寝")  # 身份必须是类编号
    with pytest.raises(ConceptError, match="named by its own class id"):
        replace(BEDTIME, title=None)
    # 管的编号第一个是自己，后面是下级（丙一 2026-11-15，原来要求恰好一个）
    assert replace(BEDTIME, classes=(kind_id("就寝"), kind_id("早餐"))).classes == (kind_id("就寝"), kind_id("早餐"))
    with pytest.raises(ConceptError, match="named by its own class id"):
        replace(BEDTIME, classes=(kind_id("早餐"), kind_id("就寝")))
    with pytest.raises(ConceptError, match="carries no situation"):
        replace(BEDTIME, situation=WEEKEND.situation)
    with pytest.raises(ConceptError, match="carries no situation"):
        replace(BEDTIME, source=ConceptSource(ConceptOrigin.CALENDAR))


def test_a_calendar_situation_is_generated_from_the_calendar_and_tied_to_no_class() -> None:
    assert WEEKEND.situation is not None and WEEKEND.situation.values == ("周六", "周日")
    assert WEEKEND.lane is None and WEEKEND.classes == () and not WEEKEND.retired
    with pytest.raises(ConceptError, match="not tied to vocabulary classes"):
        replace(WEEKEND, classes=(kind_id("就寝"),))
    with pytest.raises(ConceptError, match="never retires"):
        replace(WEEKEND, retired=True)
    with pytest.raises(ConceptError, match="carries its calendar rule"):
        replace(WEEKEND, situation=None)
    with pytest.raises(ConceptError, match="carries its calendar rule"):
        replace(WEEKEND, source=ConceptSource(ConceptOrigin.VOCABULARY))


# ── 概念集 ────────────────────────────────────────────────────────────────────


def test_a_concept_set_answers_the_class_questions() -> None:
    concepts = concept_set()
    assert set(concepts.behaviors()) == {cid("就寝"), cid("早餐"), cid("修改代码"), cid("打球"), cid("跑步")}
    assert set(concepts.situations()) == {"周末"}
    assert concepts.base_for(kind_id("就寝")) == BEDTIME.identity and concepts.base_for(kind_id("吃药")) is None
    assert concepts[cid("打球")].classes == (kind_id("打球"),) and concepts["周末"].classes == ()
    assert concepts.class_label(kind_id("就寝")) == "就寝" and concepts.class_label(kind_id("吃药")) == kind_id("吃药")
    assert concepts.label_of(cid("早餐")) == "早餐" and concepts.label_of("周末") == "周末"
    assert concepts.label_of("咖啡") == "咖啡"
    assert concepts[cid("打球")].role is ConceptRole.BEHAVIOR
    # Mapping 契约：查不到是 KeyError，``get`` 照常给默认值——账本按名字查 role 时不该炸。
    assert concepts.get("咖啡") is None and concepts.get("a/b") is None and ".hidden" not in concepts
    with pytest.raises(ConceptLookupError):
        concepts["咖啡"]


def physical_base(title: str) -> ConceptDefinition:
    """物理 lane 的行为概念（概念夹具只造会话 lane 的）。"""

    class_id = kind_id(title, Lane.PHYSICAL)
    return ConceptDefinition(
        name=class_id,
        definition=f"{title}这件事",
        role=ConceptRole.BEHAVIOR,
        source=ConceptSource(ConceptOrigin.VOCABULARY),
        created_at=CREATED,
        classes=(class_id,),
        lane="physical",
        title=title,
    )


def test_lane_of_a_behaviour_is_its_own_and_a_calendar_situation_belongs_to_no_lane() -> None:
    meal = physical_base("吃饭")
    concepts = ConceptSet([*ALL_CONCEPTS, meal])
    assert concepts.lane_of(cid("打球")) == "session" and concepts.lane_of(meal.name) == "physical"
    assert concepts.lane_of("周末") is None and concepts.lane_of("不在集里") is None


def test_labels_are_unique_within_a_lane_but_may_repeat_across_lanes() -> None:
    """给模型看的是名字（裁定 21-1）：同一条 lane 能一起出现的概念不许同名；两条 lane 各有一个「打球」可以。"""

    with pytest.raises(ConceptError, match="both show as"):
        ConceptSet([*ALL_CONCEPTS, replace(base("另一个打球"), title="打球")])
    # 不属于任何 lane 的日历情境与会话 lane 的类同名也算撞
    with pytest.raises(ConceptError, match="both show as"):
        ConceptSet([*ALL_CONCEPTS, calendar("打球", "日历.周几", "周一")])
    both = ConceptSet([*ALL_CONCEPTS, physical_base("打球")])
    assert both.label_of(kind_id("打球", Lane.PHYSICAL)) == both.label_of(cid("打球")) == "打球"


def test_a_retired_concept_may_share_its_label_with_one_in_use() -> None:
    """与词表同一口径：只有在用的不许同名。并掉的「打球」停用后又长出一个「打球」，两个都留着，按名字只找得到在用的。"""

    retired = replace(BALL, retired=True)
    reborn = replace(base("另一个打球"), title="打球")
    both = ConceptSet([*(item for item in ALL_CONCEPTS if item.name != BALL.name), retired, reborn])
    assert both.label_of(retired.name) == both.label_of(reborn.name) == "打球"
    with pytest.raises(ConceptError, match="both show as"):
        ConceptSet([*ALL_CONCEPTS, reborn])


def test_sync_refuses_before_writing_a_word_that_would_show_like_a_concept_in_use() -> None:
    """写盘前整体核对：新在用的词叫「就寝」撞了在用的词、词叫「周二」撞了日历情境——那两份不进 ``write``，带着理由进
    ``rejected``；不撞的照常写。"""

    clash = CatalogWord(id=kind_id("另一个就寝"), name="就寝", definition="躺下", lane="session")
    classes = (catalog_word("就寝"), catalog_word("早餐"), clash, catalog_word("周二"), catalog_word("午睡"))
    existing = ConceptSet([base("就寝"), base("早餐"), *calendar_concepts(now=LATER, observed=[("日历.周几", "周二")])])
    plan = plan_sync(existing, CatalogWords(2, classes), now=LATER)
    assert {item.name for item in plan.write} == {kind_id("午睡")}
    assert {item.name for item, _reason in plan.rejected} == {kind_id("另一个就寝"), kind_id("周二")}
    assert all("both show as" in reason for _item, reason in plan.rejected)
    # 计划里剩下的写完，概念集仍然自洽
    ConceptSet([*existing.values(), *plan.write])


def test_lookups_by_identity_do_not_revalidate_the_identity_string(monkeypatch) -> None:
    """概念集不可变：按规范身份查、问身份都只算一次（两两检验每晚按名字问几十万次，重新校验身份串曾占折叠的大半时间）。"""

    import habitus.scene.concepts.model as model

    concepts = concept_set()
    names = list(concepts)
    for name in names:
        concepts[name].identity  # noqa: B018 - 先算一次身份
    calls: list[object] = []
    real = model.concept_identity

    def counting(value: object) -> str:
        calls.append(value)
        return real(value)

    monkeypatch.setattr(model, "concept_identity", counting)
    for _ in range(3):
        for name in names:
            assert concepts[name].identity == name and name in concepts and not concepts[name].retired
    assert calls == []


def test_a_concept_set_refuses_duplicates() -> None:
    with pytest.raises(ConceptError, match="appears twice"):
        ConceptSet([BALL, replace(BALL, definition="另一句")])


def test_the_fingerprint_follows_what_counts_as_a_hit_not_the_label() -> None:
    assert len(WEEKEND.fingerprint) == 16
    assert replace(BEDTIME, title="睡觉").fingerprint == BEDTIME.fingerprint  # 类改名不改指纹
    assert (
        replace(WEEKEND, situation=calendar("周末", "日历.周几", "周日").situation).fingerprint != WEEKEND.fingerprint
    )


# ── 落盘格式 ──────────────────────────────────────────────────────────────────


def test_a_concept_document_round_trips_and_rejects_a_tampered_body() -> None:
    text = encode(BEDTIME)
    assert "# 就寝" in text and "判据：上床睡觉" in text and "判法：看记录的类编号（不判）" in text
    assert "类别：行为 · lane session" in text and f"类：{kind_id('就寝')}" in text
    assert "来源：同步词表自动生成" in text
    for item in ALL_CONCEPTS:
        assert decode(encode(item)) == item
    assert "判法：算法按日历算——日历的「日历.周几」是 周六/周日" in encode(WEEKEND)
    assert "类别：情境·日历" in encode(WEEKEND) and "来源：日历机械生成" in encode(WEEKEND)
    retired = replace(BEDTIME, retired=True)
    assert "类已停用" in encode(retired) and decode(encode(retired)) == retired
    with pytest.raises(SceneRecordError, match="canonical rendering"):
        decode(text.replace("判据：上床", "判据：随便"))
    with pytest.raises(SceneRecordError, match="identity"):
        decode(text, expected_identity="早餐")
    with pytest.raises(SceneRecordError):
        decode(text.replace('"record_type":"concept"', '"record_type":"other"'))
    with pytest.raises(SceneRecordError, match="must be a boolean"):
        decode(text.replace('"retired":false', '"retired":"no"'))


@pytest.mark.parametrize(
    ("concept", "old", "new", "message"),
    [
        (BEDTIME, '"lane":"session"', '"lane":5', "not a valid definition: concept lane must be text"),
        (BREAKFAST, '"title":"早餐"', '"title":["早餐"]', "not a valid definition: concept title must be text"),
        # 2026-11-14 收口修补之前写的概念文件 source 里带着 "note":null：现在多了这个键，明确拒（不静默忽略）
        (
            BEDTIME,
            '"source":{"origin":"vocabulary"}',
            '"source":{"note":null,"origin":"vocabulary"}',
            "concept source must be an object with only an origin",
        ),
        (WEEKEND, '"value":"日历.周几"', '"value":["日历.周几"]', "value must be text"),
        (WEEKEND, '"situation":{"value"', '"situation":{"basis":"calendar","value"', "value and values"),
    ],
)
def test_a_concept_document_with_wrong_types_is_rejected_when_decoded(
    concept: ConceptDefinition, old: str, new: str, message: str
) -> None:
    """概念文件注释里类型错的字段在解码时就拒（W16：类型检查在边界）。"""

    text = encode(concept)
    assert old in text
    with pytest.raises(SceneRecordError, match=message):
        decode(text.replace(old, new))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"classes":', '"baseline_keys":[],"classes":'),
        ('"classes":', '"context":"occurrence","classes":'),
        ('"lane":', '"kind":"base","lane":'),
        ('"retired":', '"grades":[],"retired":'),
        ('"role":', '"role":"behavior","rule":null,"x":'),
    ],
)
def test_a_concept_document_with_fields_of_the_old_shape_is_refused(old: str, new: str) -> None:
    """概念文件的字段集合是严格的：旧盘面上的种类、档、规则、材料、常态键一出现就拒，不静默忽略、不自动转换。"""

    text = encode(BEDTIME)
    assert old in text
    with pytest.raises(SceneRecordError):
        decode(text.replace(old, new, 1))


# ── 存储 ──────────────────────────────────────────────────────────────────────


def test_store_writes_reads_lists_and_rebuilds_a_consistent_set(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    for definition in ALL_CONCEPTS:
        store.write(definition)
    assert set(store.identities()) == {item.identity for item in ALL_CONCEPTS}
    assert store.read("周末") == WEEKEND and store.read(kind_id("早餐")) == BREAKFAST
    assert store.exists(cid("打球")) and not store.exists("吃药")
    assert set(store.read_all()) == set(concept_set())
    # .DS_Store、原子写临时文件、同步版本点文件与概念文件同目录：全是点文件，列目录时跳过。
    (store.directory / ".DS_Store").write_bytes(b"\x00")
    store.write_synced_version(3)
    assert "周末" in store.identities()


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
    (store.directory / ".vocabulary-version").write_text("seven\n", encoding="utf-8")
    with pytest.raises(ConceptStoreError, match="corrupt"):
        store.synced_version()


def test_store_enforces_directory_capacity_before_writing(tmp_path) -> None:
    """写成一个列不出来的目录比拒绝写更糟；同身份覆写不占新位。"""

    store = ConceptStore(tmp_path / "scene", max_directory_entries=2)
    store.write(BALL)
    store.write(WEEKEND)
    store.write(replace(BALL, definition="改一句"))  # 覆写
    with pytest.raises(ConceptStoreError, match="capacity"):
        store.write(BREAKFAST)


def test_store_reports_corrupt_and_non_canonical_records_and_refuses_a_symlinked_root(tmp_path) -> None:
    store = ConceptStore(tmp_path / "scene")
    store.write(WEEKEND)
    path = store.path_for("周末")
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ConceptStoreError, match="corrupt"):
        store.read("周末")
    store.write(WEEKEND)
    # 非规范叶名（大小写不同）在 APFS 上会与规范名静默互认，在大小写敏感的机器上又变成"不存在"——统一报错。
    (store.directory / "Nap.md").write_text(encode(calendar("nap", "日历.周几", "周一")), encoding="utf-8")
    with pytest.raises(ConceptStoreError, match="canonical identity"):
        store.identities()
    os.symlink(tmp_path / "scene", tmp_path / "link")
    with pytest.raises(ConceptStoreError, match="symbolic link"):
        ConceptStore(tmp_path / "link")


# ── 词表口与同步词表（词表方案 2026-11-14：只交在用的词）─────────────────────────────────────────────────────────


def catalog_word(title: str, *, definition: str | None = None, lane: str = "session") -> CatalogWord:
    return CatalogWord(id=kind_id(title), name=title, definition=definition or f"{title}这件事", lane=lane)


def test_catalog_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        CatalogWord(id="", name="x", definition="y", lane="session")
    with pytest.raises(ValueError, match="non-negative"):
        CatalogWords(-1, ())
    with pytest.raises(ValueError, match="repeat"):
        CatalogWords(1, (catalog_word("就寝"), catalog_word("就寝")))


def test_the_first_sync_generates_a_behaviour_concept_for_every_word_in_use() -> None:
    classes = (catalog_word("就寝", definition="上床睡觉"), catalog_word("早餐"))
    observed = [
        ("日历.周几", "周三"),
        ("日历.几号", "8"),
        ("日历.月末", "是"),
        ("日历.月末", "否"),
        ("日历.日型", "工作日"),
    ]
    plan = plan_sync(ConceptSet([]), CatalogWords(4, classes), now=LATER, observed=observed)
    assert plan.version == 4
    written = _behaviours(plan.write)
    assert [item.name for item in written] == [kind_id("就寝"), kind_id("早餐")]
    calendar = [item for item in plan.write if item.source.origin is ConceptOrigin.CALENDAR]
    # 日历情境只长数据里出现过的取值（裁定 59 第 7 条）：没碰到过的"调休上班"不生成
    assert {item.name for item in calendar} == {"周三", "8号", "月末", "工作日"}
    first = written[0]
    assert (first.role, first.title, first.definition, first.lane, first.retired) == (
        ConceptRole.BEHAVIOR,
        "就寝",
        "上床睡觉",
        "session",
        False,
    )
    assert first.source.origin is ConceptOrigin.VOCABULARY and first.created_at == LATER
    assert base_concept(classes[1], now=LATER) == written[1]
    # 第二次同步：什么都不用写
    again = plan_sync(ConceptSet(plan.write), CatalogWords(4, classes), now=LATER)
    assert again.write == ()


def test_sync_follows_names_and_definitions_keeping_identity_and_birth() -> None:
    existing = ConceptSet([BEDTIME, BALL])
    classes = (
        catalog_word("就寝", definition="躺下准备睡"),
        CatalogWord(id=kind_id("打球"), name="打篮球", definition=BALL.definition, lane="session"),
    )
    plan = plan_sync(existing, CatalogWords(3, classes), now=LATER)
    by_name = {item.name: item for item in _behaviours(plan.write)}
    assert set(by_name) == {kind_id("就寝"), kind_id("打球")}
    assert by_name[kind_id("就寝")].definition == "躺下准备睡" and by_name[kind_id("就寝")].created_at == CREATED
    assert by_name[kind_id("打球")].title == "打篮球" and not by_name[kind_id("打球")].retired
    # 一个不是行为概念的概念（日历情境）占了词编号这个名字：同步拒绝
    squatter = calendar(kind_id("早餐"), "日历.周几", "周一")
    with pytest.raises(ValueError, match="is not the behaviour concept"):
        plan_sync(ConceptSet([BEDTIME, squatter]), CatalogWords(1, (catalog_word("早餐"),)), now=LATER)


def _behaviours(items):  # type: ignore[no-untyped-def]
    """同步计划里除日历情境之外的那些（日历情境每次在概念集里没有时都会补上）。"""

    return [item for item in items if item.source.origin is not ConceptOrigin.CALENDAR]


def test_sync_refuses_a_non_calendar_concept_under_a_calendar_name() -> None:
    squatter = ConceptDefinition(
        name="周三",
        definition="一个叫周三的类",
        role=ConceptRole.BEHAVIOR,
        source=ConceptSource(ConceptOrigin.VOCABULARY),
        created_at=CREATED,
        classes=("周三",),
        lane="session",
        title="周三",
    )
    with pytest.raises(ValueError, match="calendar situation"):
        plan_sync(ConceptSet([squatter]), CatalogWords(1, ()), now=LATER, observed=[("日历.周几", "周三")])
