"""触点①：粒度三档的三条核对、层级排序、失败整批不写、长尾一个调用都不花。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from habitus.scene.concepts import ConceptOrigin, ConceptRole, ConceptSet
from habitus.scene.concepts.author import (
    CONCEPT_AUTHOR_VERSION,
    ConceptAuthor,
    ConceptAuthorError,
    KindBrief,
    KindTier,
    assemble_concepts,
    build_concept_request,
    concept_author_json_schema,
)
from habitus.scene.concepts.rhythm import Rhythm, RhythmPeak
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.scene.concept_fixtures import concept_set

NOW = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)
EMPTY = ConceptSet(())

BRIEFS = (
    KindBrief("修改代码", 98, 40, ("修改代码：把重复的分支合并成一个函数",)),
    KindBrief("审查代码", 26, 20, ("审查代码：读同事的改动并提了两条意见",)),
    KindBrief("排查CI失败原因", 5, 5, ("排查CI失败原因：流水线在打包那一步挂了",)),
    KindBrief("排查性能问题", 4, 4, ()),
    KindBrief("排查构建问题", 1, 1, ()),
    KindBrief("睡觉", 12, 12, ("睡觉：关灯躺下",)),
    KindBrief("查询论文", 2, 2, ()),
)

GOOD_ANSWER: dict[str, object] = {
    "concepts": [
        {"name": "写代码", "definition": "一次以改动代码库为目的的工作", "role": "behavior", "parent": None, "claims": [], "why": "两类细的上级"},
        {"name": "修改代码", "definition": "一次为实现或修复功能而改动代码的行为", "role": "behavior", "parent": "写代码", "claims": ["修改代码"]},
        {"name": "审查代码", "definition": "一次阅读并评价已写代码的行为", "role": "behavior", "parent": "写代码", "claims": ["审查代码"]},
        {
            "name": "排查问题",
            "definition": "一次为找出某个故障的原因而展开的调查",
            "role": "behavior",
            "claims": ["排查CI失败原因", "排查性能问题", "排查构建问题"],
            "why": "三种排查是同一件事，合起来才够样本",
        },
        {"name": "就寝", "definition": "一次躺下准备入睡的行为", "role": "behavior", "claims": ["睡觉"]},
        {
            "name": "晚睡",
            "definition": "入睡时刻晚于常态两小时以上",
            "role": "behavior",
            "claims": [],
            "rule": {"measure": "start_minute_of_day", "lower_minutes": 120, "upper_minutes": None, "relative_to": "就寝:usual_start:recent"},
            "grades": [{"name": "轻", "lower_minutes": 120, "upper_minutes": 240}, {"name": "重", "lower_minutes": 240, "upper_minutes": 720}],
        },
        {"name": "赶工中", "definition": "这段时间有一个明确的交付期限压着", "role": "state", "claims": []},
        {
            "name": "连日写代码",
            "definition": "最近连着三天都在改代码",
            "role": "derived",
            "claims": [],
            "situation": {"basis": "streak", "concept": "修改代码", "days": 3},
        },
    ],
    "skipped": [{"kind": "查询论文", "why": "45 天只 2 次，留在残差"}],
}


def test_the_three_tiers_and_the_hierarchy_come_out_as_definitions_ready_to_write() -> None:
    """一批的成品：≥10 次的各自一个概念、3–9 次的凑成一个、长尾并进那个凑出来的、上级排在子概念前。"""

    definitions, claims, dropped = assemble_concepts(GOOD_ANSWER, BRIEFS, EMPTY, now=NOW)
    assert dropped == ()
    names = [item.name for item in definitions]
    assert names.index("写代码") < names.index("修改代码") and names.index("写代码") < names.index("审查代码")
    assert claims["排查问题"] == ("排查CI失败原因", "排查性能问题", "排查构建问题")  # 5 + 4 + 1 = 10
    assert "查询论文" not in {kind for kinds in claims.values() for kind in kinds}
    by_name = {item.name: item for item in definitions}
    assert by_name["赶工中"].role is ConceptRole.STATE and "赶工中" not in claims
    # 派生情境带着算法说明：盯谁、连几天；「赶工中」这种算不出的状态不写说明（等事实门）
    streak = by_name["连日写代码"]
    assert streak.situation is not None and streak.situation.criterion() == "「修改代码」连着 3 天命中"
    assert by_name["赶工中"].situation is None
    # 「晚睡」不认领任何 kind：它靠数值规则判别人的 occurrence，档跟着规则一起走。
    late = by_name["晚睡"]
    assert late.rule is not None and late.rule.relative_to == "就寝:usual_start:recent"
    assert [grade.name for grade in late.grades] == ["轻", "重"] and all(grade.relative for grade in late.grades)
    assert by_name["排查问题"].source.origin is ConceptOrigin.BASELINE
    assert CONCEPT_AUTHOR_VERSION in (by_name["排查问题"].source.note or "") and "同一件事" in (by_name["排查问题"].source.note or "")


def test_the_tiers_are_computed_from_the_counts_and_the_prompt_carries_the_instruction() -> None:
    assert [brief.tier for brief in BRIEFS[:5]] == [KindTier.LEAF, KindTier.LEAF, KindTier.GROUP, KindTier.GROUP, KindTier.RESIDUE]
    request = build_concept_request(BRIEFS, concept_set(), {"早餐": Rhythm("早餐", (RhythmPeak(1, 420, 510, 0.8),), 7, 24.0)})
    rendered = request.messages[-1].content or ""
    assert "修改代码：45 天里 98 次、跨 40 天 → 给它一个概念" in rendered
    assert "排查构建问题：45 天里 1 次、跨 1 天 → 不要定义，留在残差" in rendered
    assert "已经有的概念（不要重写" in rendered and "一天 1 个机会" in rendered
    # kind 名字钉进 enum：模型认领不了材料里没有的名字。
    schema = concept_author_json_schema(BRIEFS)
    assert schema["properties"]["concepts"]["items"]["properties"]["claims"]["items"]["enum"] == [brief.kind_token for brief in BRIEFS]
    with pytest.raises(ConceptAuthorError, match="once"):
        concept_author_json_schema((BRIEFS[0], BRIEFS[0]))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda entries: [entry for entry in entries if entry["name"] != "修改代码"], "its own"),
        (
            lambda entries: [
                {**entry, "claims": ["修改代码", "审查代码"]} if entry["name"] == "修改代码" else entry
                for entry in entries
                if entry["name"] != "审查代码"
            ],
            "own concept",
        ),
        (
            lambda entries: [
                {**entry, "situation": {"basis": "streak", "concept": "修改代码", "days": 1}} if entry["name"] == "连日写代码" else entry
                for entry in entries
            ],
            "not computable",
        ),
        (
            lambda entries: [
                {**entry, "situation": {"basis": "streak", "concept": "打太极", "days": 3}} if entry["name"] == "连日写代码" else entry
                for entry in entries
            ],
            "watches",
        ),
        (
            lambda entries: [{**entry, "situation": None} if entry["name"] == "连日写代码" else entry for entry in entries],
            "derived situation concept carries the rule",
        ),
        (
            lambda entries: [
                {**entry, "rule": {**entry["rule"], "relative_to": "打球:usual_start:recent"}} if entry["name"] == "晚睡" else entry for entry in entries
            ],
            "not a concept here",
        ),
    ],
)
def test_the_answer_is_refused_when_the_whole_batch_does_not_hold(mutate, message) -> None:
    """**整批**重答只留给整体不自洽：≥10 次的 kind 没有独占概念、上级/常态键指向不存在的概念。"""

    answer = {**GOOD_ANSWER, "concepts": mutate(GOOD_ANSWER["concepts"])}
    with pytest.raises(ValueError, match=message):
        assemble_concepts(answer, BRIEFS, EMPTY, now=NOW)


@pytest.mark.parametrize(
    ("mutate", "name", "reason"),
    [
        (
            lambda entries: [{**entry, "claims": ["排查CI失败原因", "排查性能问题"]} if entry["name"] == "排查问题" else entry for entry in entries],
            "排查问题",
            "凑出来只有 9 次",
        ),
        (
            lambda entries: [{**entry, "claims": ["查询论文"]} if entry["name"] == "赶工中" else entry for entry in entries],
            "赶工中",
            "情境概念不认领 kind",
        ),
        (
            lambda entries: [{**entry, "rule": None, "grades": []} if entry["name"] == "晚睡" else entry for entry in entries],
            "晚睡",
            "永远不会命中",
        ),
    ],
)
def test_one_unusable_concept_is_dropped_and_the_rest_of_the_batch_is_kept(mutate, name, reason) -> None:
    """单个概念不合格就丢那一个，**不否决整批**（2026-09-29 探针：真实模型一次写 25 个，其中一个凑到 8 次，
    老写法把另外 24 个合格的连同那次调用一起废掉）。它认领的 kind 退回残差——那正是残差的定义。"""

    answer = {**GOOD_ANSWER, "concepts": mutate(GOOD_ANSWER["concepts"])}
    definitions, claims, dropped = assemble_concepts(answer, BRIEFS, EMPTY, now=NOW)
    assert name not in {item.name for item in definitions} and name not in claims
    assert len(dropped) == 1 and reason in dropped[0] and dropped[0].startswith(f"dropped: {name}")
    assert len(definitions) == len(GOOD_ANSWER["concepts"]) - 1  # 其余的都写成了


def test_a_residue_upgrade_claims_exactly_one_kind_and_records_it() -> None:
    """残差升级那一支要记下认领的 kind，否则残差视图每晚再报一次"可升级"。"""

    brief = KindBrief("整理桌面", 6, 5, ("整理桌面：把散落的文件归到目录里",))
    answer = {"concepts": [{"name": "整理环境", "definition": "一次把工作环境收拾整齐的行为", "role": "behavior", "claims": ["整理桌面"]}], "skipped": []}
    thin, _claims, dropped = assemble_concepts(answer, (brief,), EMPTY, now=NOW, origin=ConceptOrigin.RESIDUE)
    assert thin == () and "只有 6 次" in dropped[0]  # 6 次不够 10：丢掉、退残差，不报错
    definitions, claims, _dropped = assemble_concepts(answer, (KindBrief("整理桌面", 11, 9),), EMPTY, now=NOW, origin=ConceptOrigin.RESIDUE)
    assert definitions[0].source.origin is ConceptOrigin.RESIDUE and definitions[0].source.kind_token == "整理桌面"
    assert claims == {"整理环境": ("整理桌面",)}


def test_a_failed_or_unusable_answer_writes_nothing_and_a_long_tail_costs_no_call() -> None:
    """模型没答成、或答复过不了核对 → 整批一个都不写、留信号；那些 kind 留在残差，下一次再问。"""

    # 形状合格（严格模式的 schema 要求每个键都在）、但过不了核对：行为概念没认领、没规则、也不是谁的上级
    unusable = {
        "name": "写代码",
        "definition": "改代码",
        "role": "behavior",
        "parent": None,
        "claims": [],
        "context": "occurrence",
        "baseline_keys": [],
        "rule": None,
        "grades": [],
        "situation": None,
        "why": "",
    }
    client, provider = recording_client([{"concepts": [unusable], "skipped": []}])
    author = ConceptAuthor(client, clock=lambda: NOW)
    proposal = asyncio.run(author.propose(BRIEFS[:1], EMPTY))
    assert proposal.wrote_nothing and provider.calls == 2  # 结构层纠正重试了一轮，仍不合格
    assert any(signal.startswith("model: ") for signal in proposal.signals)

    # 全是一两次的长尾：凑不出够 10 次的概念，一个调用都不花。
    tail = (KindBrief("查询论文", 2, 2), KindBrief("排查构建问题", 1, 1))
    idle, idle_provider = recording_client([{"concepts": [], "skipped": []}])
    quiet = asyncio.run(ConceptAuthor(idle, clock=lambda: NOW).propose(tail, EMPTY))
    assert idle_provider.calls == 0 and quiet.wrote_nothing and "留在残差" in quiet.signals[0]


def test_an_empty_answer_is_a_legitimate_answer() -> None:
    """"这批一个都不该定义"是合法答复（和触点② 在无关配对上答零条同一个道理）。"""

    client, provider = recording_client([{"concepts": [], "skipped": [{"kind": "查询论文", "why": "太少"}]}])
    proposal = asyncio.run(ConceptAuthor(client, clock=lambda: NOW).propose((KindBrief("查询论文", 4, 4),), EMPTY))
    assert provider.calls == 1 and proposal.wrote_nothing
    assert any("一个都不该定义" in signal for signal in proposal.signals)


def test_existing_concepts_are_not_rewritten() -> None:
    answer = {"concepts": [{"name": "打球", "definition": "参与一场球类运动", "role": "behavior", "claims": ["打球"]}], "skipped": []}
    with pytest.raises(ValueError, match="already exists"):
        assemble_concepts(answer, (KindBrief("打球", 14, 11),), concept_set(), now=NOW)
