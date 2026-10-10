"""触点①（裁定 20）：只写细分 / 汇总 / 情境概念；基础概念由同步词表生成，这里只读不写。

核对的都是"单个概念立不住就丢那一个"：跨 lane 汇总、与已有重名（含与类名同名）、形状不自洽、引用了不存在的概念（连带丢）；
整批否决只留给答复本身不成形。模型看到的是类名与判据，看不到次数，也看不到编号（裁定 21-1）：
它答的是类名，程序换回编号——成员类、常态键的"谁"、情境说明盯的概念都一样。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from habitus.behavior.kinds.ids import Lane
from habitus.scene.concepts import (
    ConceptDefinition,
    ConceptKind,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ContextScope,
)
from habitus.scene.concepts.author import (
    CONCEPT_AUTHOR_VERSION,
    ClassBrief,
    ConceptAuthor,
    ConceptAuthorError,
    assemble_concepts,
    build_concept_request,
    class_labels,
    concept_author_json_schema,
)
from habitus.scene.concepts.rhythm import Rhythm, RhythmPeak
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import CREATED, VOCABULARY, base, refinement

NOW = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)

CODE, DEBUG, TEST, SLEEP = (kind_id(name) for name in ("修改代码", "排查问题", "验证测试", "就寝"))
MEAL = kind_id("吃饭", Lane.PHYSICAL)


def physical_base(title: str) -> ConceptDefinition:
    class_id = kind_id(title, Lane.PHYSICAL)
    return ConceptDefinition(
        name=class_id,
        definition=f"{title}这件事",
        role=ConceptRole.BEHAVIOR,
        source=VOCABULARY,
        created_at=CREATED,
        kind=ConceptKind.BASE,
        classes=(class_id,),
        lane="physical",
        title=title,
    )


BRIEFS = (
    ClassBrief(CODE, "修改代码", "改动代码文件", "session", ("调整 Tagent 的记忆召回代码：把重复分支合成一个函数",)),
    ClassBrief(DEBUG, "排查问题", "为找出故障原因而调查", "session"),
    ClassBrief(TEST, "验证测试", "跑测试确认改动没坏", "session"),
    ClassBrief(SLEEP, "就寝", "上床睡觉", "session", ("关灯躺下",)),
    ClassBrief(MEAL, "吃饭", "吃一顿饭", "physical"),
)
#: 同步词表之后、触点① 之前的概念集：每个类一个基础概念。
EXISTING = ConceptSet(
    (
        base("修改代码", "改动代码文件"),
        base("排查问题"),
        base("验证测试"),
        base("就寝", "上床睡觉"),
        physical_base("吃饭"),
    )
)
BEDTIME_KEY = f"{SLEEP}:usual_start:recent"
#: 模型写的常态键：用类名（程序换成编号就是 ``BEDTIME_KEY``）。
SHOWN_BEDTIME_KEY = "就寝:usual_start:recent"


def entry(name: str, definition: str, **fields: object) -> dict[str, object]:
    """严格模式的答复形状：每个键都在。"""

    return {
        "name": name,
        "definition": definition,
        "role": "behavior",
        "kind": None,
        "classes": [],
        "context": "occurrence",
        "baseline_keys": [],
        "rule": None,
        "grades": [],
        "situation": None,
        "why": "",
        **fields,
    }


LATE = entry(
    "晚睡",
    "开始时刻比近期常态晚两小时以上",
    kind="refinement",
    classes=["就寝"],
    rule={
        "measure": "start_minute_of_day",
        "lower_minutes": 120,
        "upper_minutes": None,
        "relative_to": SHOWN_BEDTIME_KEY,
    },
    grades=[
        {"name": "轻", "lower_minutes": 120, "upper_minutes": 240},
        {"name": "重", "lower_minutes": 240, "upper_minutes": 720},
    ],
    why="晚睡的提醒句仍是该睡了，只是条件不同",
)
REWORK = entry("返工", "改的是前两天刚改过的同一处", kind="refinement", classes=["修改代码"], context="recent")
CODING = entry(
    "写代码",
    "以改动代码库为目的的工作",
    kind="group",
    classes=["修改代码", "排查问题", "验证测试"],
    why="三类一起看才有意义",
)
CRUNCH = entry("赶工中", "这段时间有一个明确的交付期限压着", role="state")
STREAK = entry(
    "连日改代码",
    "最近连着三天都在改代码",
    role="derived",
    situation={"basis": "streak", "weekdays": [], "value": None, "concept": "修改代码", "grade": None, "days": 3},
)
GOOD_ANSWER: dict[str, object] = {"concepts": [LATE, REWORK, CODING, CRUNCH, STREAK]}


def answer(*concepts: dict[str, object]) -> dict[str, object]:
    return {"concepts": list(concepts)}


def replaced(original: dict[str, object], **fields: object) -> dict[str, object]:
    return {**original, **fields}


def test_refinements_groups_and_situations_come_out_ready_to_write() -> None:
    definitions, dropped = assemble_concepts(GOOD_ANSWER, BRIEFS, EXISTING, now=NOW)
    assert dropped == ()
    by_name = {item.name: item for item in definitions}
    assert set(by_name) == {"晚睡", "返工", "写代码", "赶工中", "连日改代码"}
    late = by_name["晚睡"]
    assert (late.kind, late.classes, late.lane) == (ConceptKind.REFINEMENT, (SLEEP,), "session")
    assert late.rule is not None and late.rule.relative_to == BEDTIME_KEY
    assert [grade.name for grade in late.grades] == ["轻", "重"] and all(grade.relative for grade in late.grades)
    assert by_name["返工"].context is ContextScope.RECENT and by_name["返工"].rule is None
    assert (by_name["写代码"].kind, by_name["写代码"].classes) == (ConceptKind.GROUP, (CODE, DEBUG, TEST))
    assert (
        by_name["赶工中"].role is ConceptRole.STATE
        and by_name["赶工中"].kind is None
        and by_name["赶工中"].lane is None
    )
    streak = by_name["连日改代码"]
    assert streak.situation is not None and streak.situation.criterion() == f"「{CODE}」往前连着 3 个 24 小时都命中"
    assert late.source.origin is ConceptOrigin.AUTHOR
    assert CONCEPT_AUTHOR_VERSION in (late.source.note or "") and "提醒句" in (late.source.note or "")
    # 写出来的与已有的拼得成一个自洽的概念集
    merged = ConceptSet((*(EXISTING[identity] for identity in EXISTING), *definitions))
    assert merged.refinements_on(SLEEP) == ("晚睡",) and merged.ancestors(CODE) == ("写代码",)


def test_the_request_shows_class_names_and_criteria_never_counts() -> None:
    authored = ConceptSet(
        (*(EXISTING[identity] for identity in EXISTING), refinement("夜宵", "睡前两小时内吃东西", "就寝"))
    )
    rhythm = Rhythm(SLEEP, (RhythmPeak(1, 1380, 1440, 0.8),), 7, 24.0, label="就寝")
    rendered = build_concept_request(BRIEFS, authored, {SLEEP: rhythm}).messages[-1].content or ""
    assert "- 修改代码（session）：改动代码文件｜例：调整 Tagent 的记忆召回代码" in rendered
    assert "- 吃饭（physical）：吃一顿饭｜例：（没给例子）" in rendered
    # 编号只给程序用，不进模型看的材料（裁定 21-1）
    assert all(class_id not in rendered for class_id in (CODE, DEBUG, TEST, SLEEP, MEAL))
    # 已有的只列模型写的（细分 / 汇总 / 情境），显示名字和它挂的类名；基础概念不再列一遍
    assert (
        "已经有的概念（不要重写、不要重名）" in rendered
        and "- 夜宵（refinement：就寝）：睡前两小时内吃东西" in rendered
    )
    assert "就寝：一天 1 个机会" in rendered  # 节律按类名显示
    assert "次、跨" not in rendered and "→ 给它一个概念" not in rendered  # 不看次数（裁定七：按次数定粒度太死板）
    without = build_concept_request(BRIEFS[:1], EXISTING).messages[-1].content or ""
    assert "已经有的概念" not in without and "作息" not in without
    with pytest.raises(ConceptAuthorError, match="at least one class brief"):
        build_concept_request((), EXISTING)


def test_the_schema_pins_class_names_and_briefs_are_checked() -> None:
    schema = concept_author_json_schema(BRIEFS)
    item = schema["properties"]["concepts"]["items"]
    assert item["properties"]["classes"]["items"]["enum"] == ["修改代码", "排查问题", "验证测试", "就寝", "吃饭"]
    assert item["properties"]["kind"]["enum"] == ["refinement", "group", None]
    assert set(item["required"]) == set(item["properties"])  # 严格模式：每个键都要在 required 里
    with pytest.raises(ConceptAuthorError, match="at least one"):
        concept_author_json_schema(())
    with pytest.raises(ConceptAuthorError, match="name"):
        ClassBrief(CODE, " ", "判据", "session")
    many = ClassBrief(CODE, "修改代码", "改动代码文件", "session", tuple(f"例{index}" for index in range(9)) + (" ",))
    assert len(many.examples) == 5


@pytest.mark.parametrize(
    ("concept", "reason"),
    [
        (replaced(CODING, classes=["修改代码", "吃饭"]), "跨了 lane"),
        (replaced(LATE, name=SLEEP), "已经有同名概念"),
        (replaced(LATE, name="就寝"), "与类名同名"),
        (replaced(REWORK, classes=["修改代码", "排查问题"]), "exactly one source class"),
        (replaced(CODING, classes=["修改代码"]), "at least two member classes"),
        (replaced(CODING, context="day"), "never judged"),
        (replaced(LATE, context="day"), "computed from the occurrence alone"),
        (replaced(CRUNCH, classes=["修改代码"]), "not tied to vocabulary classes"),
        (replaced(REWORK, kind=None), "relates to the vocabulary"),
        (replaced(STREAK, situation=None), "derived situation concept carries the rule"),
    ],
)
def test_one_unusable_concept_is_dropped_and_the_rest_of_the_batch_is_kept(concept, reason) -> None:
    """单个概念不合格就丢那一个，**不否决整批**（2026-09-29 探针：一个坏的不该把另外 24 个合格的连同那次调用一起废掉）。"""

    template = next(
        item
        for item in GOOD_ANSWER["concepts"]
        if item["why"] == concept["why"] and item["definition"] == concept["definition"]
    )  # type: ignore[index]
    others = [item for item in GOOD_ANSWER["concepts"] if item is not template]  # type: ignore[union-attr]
    definitions, dropped = assemble_concepts(answer(*others, concept), BRIEFS, EXISTING, now=NOW)
    assert len(dropped) == 1 and reason in dropped[0] and dropped[0].startswith(f"dropped: {concept['name']}")
    survivors = {item.name for item in definitions}
    # 盯着「写代码」的「连日写代码」在「写代码」被丢时会连带丢——这里每组只丢一个，所以其余都在
    assert survivors == {item["name"] for item in others}  # type: ignore[index]


def test_references_to_a_dropped_or_unknown_concept_cascade() -> None:
    """引用了被丢概念的也丢，并且说清是因为谁（算法的展开不许算成模型答错）。"""

    cross_lane = replaced(CODING, classes=["修改代码", "吃饭"])
    watcher = replaced(STREAK, name="连日写代码", situation={**STREAK["situation"], "concept": "写代码"})  # type: ignore[dict-item]
    definitions, dropped = assemble_concepts(answer(LATE, cross_lane, watcher), BRIEFS, EXISTING, now=NOW)
    assert {item.name for item in definitions} == {"晚睡"}
    assert dropped[0].startswith("dropped: 写代码") and "跨了 lane" in dropped[0]
    assert dropped[1].startswith("dropped: 连日写代码") and "「写代码」" in dropped[1]
    # 常态键的主不是已有、也不是这批留下的概念
    orphan = replaced(LATE, rule={**LATE["rule"], "relative_to": "打太极:usual_start:recent"})  # type: ignore[dict-item]
    definitions, dropped = assemble_concepts(answer(orphan, REWORK), BRIEFS, EXISTING, now=NOW)
    assert {item.name for item in definitions} == {"返工"} and "「打太极」" in dropped[0] and "不是已有" in dropped[0]
    # 判据句引用的常态键同理；引用这批里留下的细分概念则可以
    keyed = replaced(REWORK, baseline_keys=["晚睡:usual_start:recent"])
    definitions, dropped = assemble_concepts(answer(LATE, keyed), BRIEFS, EXISTING, now=NOW)
    assert dropped == () and {item.name for item in definitions} == {"晚睡", "返工"}


@pytest.mark.parametrize(
    ("concepts", "message"),
    [
        ([replaced(REWORK, classes=["打太极"])], "must name the given classes"),
        ([replaced(REWORK, classes=[CODE])], "must name the given classes"),  # 给编号不算：选项是类名
        ([REWORK, REWORK], "share one name"),
        ([replaced(REWORK, kind="leaf")], "kind must be one of"),
        ([replaced(REWORK, classes="修改代码")], "classes must be a list"),
        ([replaced(REWORK, grades=[{"name": "轻", "lower_minutes": 0, "upper_minutes": 10}])], "grades need a rule"),
    ],
)
def test_a_malformed_answer_is_refused_as_a_whole(concepts, message) -> None:
    with pytest.raises(ValueError, match=message):
        assemble_concepts(answer(*concepts), BRIEFS, EXISTING, now=NOW)
    with pytest.raises(ValueError, match="concepts list"):
        assemble_concepts({"concepts": None}, BRIEFS, EXISTING, now=NOW)
    with pytest.raises(ValueError, match="at most"):
        assemble_concepts({"concepts": [REWORK] * 41}, BRIEFS, EXISTING, now=NOW)


def test_a_failed_answer_writes_nothing_and_an_empty_answer_is_legitimate() -> None:
    """模型没答成、或答复两轮都不成形 → 整批一个都不写、留信号；"这批没有值得写的"是合法答复。"""

    malformed = answer(REWORK, REWORK)  # schema 收得下、核对不过：两个概念同名
    client, provider = recording_client([malformed])
    proposal = asyncio.run(ConceptAuthor(client, clock=lambda: NOW).propose(BRIEFS, EXISTING))
    assert proposal.wrote_nothing and provider.calls == 2  # 结构层纠正重试了一轮，仍不合格
    assert any(signal.startswith("model: ") and "share one name" in signal for signal in proposal.signals)

    idle, idle_provider = recording_client([answer()])
    quiet = asyncio.run(ConceptAuthor(idle, clock=lambda: NOW).propose(BRIEFS, EXISTING))
    assert idle_provider.calls == 1 and quiet.wrote_nothing
    assert any("没有写成的概念" in signal for signal in quiet.signals)


def test_the_author_returns_what_it_wrote_with_its_dropped_reasons() -> None:
    client, provider = recording_client([answer(LATE, replaced(CODING, classes=["修改代码", "吃饭"]))])
    author = ConceptAuthor(client, clock=lambda: NOW)
    proposal = asyncio.run(author.propose(BRIEFS, EXISTING))
    assert provider.calls == 1 and [item.name for item in proposal.definitions] == ["晚睡"]
    assert proposal.definitions[0].created_at == NOW
    assert proposal.dropped and proposal.dropped[0].startswith("dropped: 写代码")
    assert proposal.signals[0] == f"author: {author.version}" and proposal.dropped[0] in proposal.signals
    with pytest.raises(TypeError):
        asyncio.run(author.propose(BRIEFS, ()))  # type: ignore[arg-type]


def test_the_same_class_given_twice_is_refused() -> None:
    with pytest.raises(ConceptAuthorError, match="once"):
        concept_author_json_schema((BRIEFS[0], BRIEFS[0]))


def test_answers_in_class_names_are_mapped_back_to_class_ids() -> None:
    """成员类、规则比的常态键、判据句引用的常态键、情境说明盯的概念：模型写类名，落盘是编号。"""

    keyed = replaced(REWORK, baseline_keys=["修改代码:usual_duration:recent"])
    definitions, dropped = assemble_concepts(answer(LATE, keyed, CODING, STREAK), BRIEFS, EXISTING, now=NOW)
    assert dropped == ()
    by_name = {item.name: item for item in definitions}
    assert by_name["晚睡"].classes == (SLEEP,) and by_name["晚睡"].rule.relative_to == BEDTIME_KEY  # type: ignore[union-attr]
    assert by_name["返工"].baseline_keys == (f"{CODE}:usual_duration:recent",)
    assert by_name["写代码"].classes == (CODE, DEBUG, TEST)
    assert by_name["连日改代码"].situation.concept == CODE  # type: ignore[union-attr]


def test_classes_with_the_same_name_in_two_lanes_carry_their_lane() -> None:
    """两条 lane 各有一个「调研」：模型看到、答的都是"调研（session）"/"调研（physical）"，各自换回自己的编号。"""

    briefs = (
        ClassBrief(kind_id("调研"), "调研", "查资料", "session"),
        ClassBrief(kind_id("调研", Lane.PHYSICAL), "调研", "实地看看", "physical"),
    )
    assert dict(class_labels(briefs)) == {
        "调研（session）": kind_id("调研"),
        "调研（physical）": kind_id("调研", Lane.PHYSICAL),
    }
    schema = concept_author_json_schema(briefs)
    assert schema["properties"]["concepts"]["items"]["properties"]["classes"]["items"]["enum"] == [
        "调研（session）",
        "调研（physical）",
    ]
    rendered = build_concept_request(briefs, ConceptSet(())).messages[-1].content or ""
    assert "- 调研（session）：查资料" in rendered and "（session）（session）" not in rendered
    picked = replaced(REWORK, classes=["调研（physical）"])
    (definition,), _ = assemble_concepts(answer(picked), briefs, ConceptSet(()), now=NOW)
    assert definition.classes == (kind_id("调研", Lane.PHYSICAL),) and definition.lane == "physical"
