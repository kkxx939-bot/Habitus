"""大模型给关系检验的两样建议（语义树新方案 ``13`` ②，第 7 步）：先验与候选调节条件，及其缓存、只读缓存的回放、"靠先验"标签。

模型用按脚本回放的假 Provider（真实模型的对照在 ``语义树数据记录/`` 里跑）。
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from habitus.runtime.scene_advice import RelationAdvice, occasions
from habitus.scene.advice import PriorLevel, Proposal, ProposalBook, settle
from habitus.scene.advice.conditions import (
    MAX_PROPOSALS,
    ConditionProposer,
    assemble_proposals,
    build_condition_request,
)
from habitus.scene.advice.conditions import question_for as condition_question
from habitus.scene.advice.prior import PriorAdvisor, PriorConfig, assemble_ratings, build_prior_request
from habitus.scene.advice.prior import question_for as prior_question
from habitus.scene.advice.store import AdviceStore, AdviceStoreError
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import LaneTests, Status, build_timelines, fold
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.scene.advice_fixtures import answering_client, ratings
from tests.unit.scene.concept_fixtures import base, situation
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    DISCUSS,
    DOCS,
    PUSH,
    RESEARCH,
    TITLES,
    key,
    planted,
)

TRAVEL = situation("出差中", "人在常住地之外过夜")
NIGHT = date(2026, 7, 1)


# --- 先验 ----------------------------------------------------------------------------------------


def test_three_answers_settle_on_the_median_and_a_two_level_split_is_neutral() -> None:
    likely, unsure, unlikely = PriorLevel.LIKELY, PriorLevel.UNSURE, PriorLevel.UNLIKELY
    assert settle([likely, likely, unsure]) is likely
    assert settle([unsure, unlikely, unlikely]) is unlikely
    assert settle([likely, unsure, unlikely]) is unsure  # 差了两档：模型自己都拿不准
    assert settle([likely, likely, unlikely]) is unsure
    assert (likely.weight, unsure.weight, unlikely.weight) == (2.0, 1.0, 0.5)


def test_the_prior_only_sees_names_and_criteria_and_is_asked_three_times_in_different_orders() -> None:
    question = prior_question(CONCEPTS, "session", CODE, [DISCUSS, RESEARCH, PUSH])
    first = build_prior_request(question, question.antecedents).messages[-1].content or ""
    assert "## 后果\n修改代码：修改代码这件事" in first and "- 讨论方案：讨论方案这件事" in first
    for leak in ("次", "概率", "%"):
        assert leak not in first.split("## 前因")[1]  # 不给计数、节律或任何统计量
    client, provider = recording_client(
        [
            ratings({"讨论方案": "likely", "调研": "likely", "提交推送": "unlikely"}),
            ratings({"提交推送": "unlikely", "调研": "unsure", "讨论方案": "likely"}),
            ratings({"调研": "unlikely", "提交推送": "unlikely", "讨论方案": "likely"}),
        ]
    )
    advisor = PriorAdvisor(client, config=PriorConfig(transient_retry_delay_seconds=0.0))
    answer = asyncio.run(advisor.rate(question))
    assert provider.calls == 3
    orders = [
        [line[2:].split("：")[0] for line in prompt.splitlines() if line.startswith("- ")]
        for prompt in provider.prompts
    ]
    assert (
        orders[0] == ["讨论方案", "调研", "提交推送"]
        and orders[1] == list(reversed(orders[0]))
        and orders[2] != orders[0]
    )
    assert dict(answer.levels) == {DISCUSS: PriorLevel.LIKELY, RESEARCH: PriorLevel.UNSURE, PUSH: PriorLevel.UNLIKELY}
    assert any("几遍答得不一样" in signal for signal in answer.signals)


def test_ratings_must_cover_every_antecedent_exactly_once() -> None:
    question = prior_question(CONCEPTS, "session", CODE, [DISCUSS, RESEARCH])
    with pytest.raises(ValueError, match="missing"):
        assemble_ratings(ratings({"讨论方案": "likely"}), question)
    with pytest.raises(ValueError, match="repeats"):
        assemble_ratings({"ratings": [{"concept": "讨论方案", "level": "likely"}] * 2}, question)
    with pytest.raises(ValueError, match="not asked"):
        assemble_ratings(ratings({"讨论方案": "likely", "写文档": "unsure"}), question)


# --- 候选调节条件 --------------------------------------------------------------------------------


def lane_tests(series) -> LaneTests:  # type: ignore[no-untyped-def]
    return LaneTests(build_timelines(series, CONCEPTS, {}, CONFIG)["session"], CONCEPTS, CONFIG, series.cutoff)


def test_the_condition_question_shows_the_antecedent_and_what_came_before_never_what_came_after() -> None:
    series = planted(80, moderated=True)
    tests = lane_tests(series)
    picked = occasions(tests, CONCEPTS, tests.chains(DISCUSS))
    assert 0 < len(picked) <= 6
    question = condition_question(CONCEPTS, "session", DISCUSS, picked, refused=["template=vibes（unknown template）"])
    prompt = build_condition_request(question).messages[-1].content or ""
    assert "## 这一类行为：讨论方案" in prompt and "之前：" in prompt and "已经被拒的提法" in prompt
    # 当天之前的事里只有更早的；讨论方案之后接的修改代码一个字都不出现在"之前"里
    for occasion in picked:
        moment = occasion.moment.split(" ")[1][:5]
        assert all(entry[:5] < moment for entry in occasion.before)


def test_proposals_compile_or_land_on_the_refused_list() -> None:
    concepts = ConceptSet([*CONCEPTS.values(), TRAVEL])
    question = condition_question(concepts, "session", DISCUSS, ())
    parsed = {
        "conditions": [
            {"template": "timeline", "field": None, "value": None, "concept": "调研", "why": "先查过资料才讨论"},
            {"template": "field", "field": "weekend", "value": None, "concept": None, "why": "周末更随意"},
            {"template": "concept", "field": None, "value": None, "concept": "出差中", "why": "出差时节奏不同"},
            {"template": "timeline", "field": None, "value": None, "concept": "讨论方案", "why": "自己"},
            {"template": "field", "field": "place", "value": None, "concept": None, "why": "缺值"},
        ]
    }
    found = assemble_proposals(parsed, question, concepts, night=NIGHT)
    accepted = [item.condition for item in found if item.condition]
    refused = [item.refused for item in found if item.refused]
    assert accepted == [f"timeline:{RESEARCH}", "field:weekend", "concept:出差中"]
    assert len(refused) == 2 and any("antecedent itself" in reason for reason in refused)
    with pytest.raises(ValueError, match=f"at most {MAX_PROPOSALS}"):
        assemble_proposals({"conditions": parsed["conditions"] * 2}, question, concepts, night=NIGHT)


# --- 接线：缓存、只读缓存、提议只用在提出之后 --------------------------------------------------------


def test_priors_are_cached_by_input_and_replays_read_only_the_cache(tmp_path) -> None:
    series = planted(20)
    tests = lane_tests(series)
    store = AdviceStore(tmp_path / "scene")
    client, provider = answering_client()
    advice = RelationAdvice(
        store=store, prior=PriorAdvisor(client, config=PriorConfig(transient_retry_delay_seconds=0.0)), conditions=None
    )
    signals: list[str] = []
    first = asyncio.run(advice.weights(tests, signals))
    asked = provider.calls
    assert asked == 3 * len(TITLES) and first and set(first.values()) == {1.0}
    again = asyncio.run(advice.weights(tests, []))
    assert provider.calls == asked and again == first  # 词表与概念没变：不重问
    replay = RelationAdvice(store=AdviceStore(tmp_path / "other"), prior=advice.prior, conditions=None, cache_only=True)
    notes: list[str] = []
    assert asyncio.run(replay.weights(tests, notes)) == {} and provider.calls == asked
    assert any("缺先验" in note for note in notes)


def test_a_new_concept_only_asks_the_new_pairs(tmp_path) -> None:
    """按（后果, 单个前因）缓存（E9）：加一个概念「提问咨询」，老后果只问这一个新前因、新后果问全部；
    提到「调研」的提示词只有 6 条（新后果 3 遍 + 后果「调研」问新前因 3 遍），整份清单缓存会是 18 条。"""

    series = planted(20)
    store = AdviceStore(tmp_path / "scene")
    client, provider = answering_client()
    advice = RelationAdvice(
        store=store, prior=PriorAdvisor(client, config=PriorConfig(transient_retry_delay_seconds=0.0)), conditions=None
    )
    asyncio.run(advice.weights(lane_tests(series), []))
    before = len(provider.prompts)
    grown = ConceptSet([*CONCEPTS.values(), base("提问咨询")])
    line = build_timelines(series, grown, {}, CONFIG)["session"]
    asyncio.run(advice.weights(LaneTests(line, grown, CONFIG, series.cutoff), []))
    later = provider.prompts[before:]
    assert len(later) == 3 * (len(TITLES) + 1)
    assert sum("调研" in prompt for prompt in later) == 6


def test_conditions_are_asked_once_per_concept_set_and_used_only_from_the_night_they_were_proposed(tmp_path) -> None:
    series = planted(80, moderated=True)
    tests = lane_tests(series)
    store = AdviceStore(tmp_path / "scene")
    client, provider = answering_client()
    advice = RelationAdvice(store=store, prior=None, conditions=ConditionProposer(client))
    night = series.cutoff
    found = asyncio.run(advice.proposals(tests, night, []))
    eligible = [
        name
        for name in (DISCUSS, CODE, RESEARCH, DOCS, PUSH)
        if len(tests.chains(name)) >= CONFIG.thresholds.moderation_min_antecedents
    ]
    assert provider.calls == len(eligible) and set(found) == set(eligible)
    assert found[DISCUSS][0][1] == night
    asyncio.run(advice.proposals(tests, night + timedelta(days=1), []))
    assert provider.calls == len(eligible)  # 概念集没变：不再问
    # 回放比提出更早的一晚：那条提议还不存在
    assert asyncio.run(advice.proposals(tests, night - timedelta(days=1), [])) == {}
    book = store.book("session")
    assert book.asked and all(item.proposed_on == night for item in book.proposals)


def test_the_store_round_trips_and_refuses_tampering(tmp_path) -> None:
    store = AdviceStore(tmp_path / "scene")
    store.write_prior(
        "abc123", lane="session", consequent=CODE, levels={DISCUSS: PriorLevel.LIKELY}, version="v", models=("fake-1",)
    )
    assert store.prior("abc123") == {DISCUSS: PriorLevel.LIKELY} and store.prior("missing") is None
    book = ProposalBook(
        lane="session",
        proposals=(
            Proposal(DISCUSS, NIGHT, "先查过资料", condition=f"timeline:{RESEARCH}"),
            Proposal(DISCUSS, NIGHT, "瞎编", raw={"template": "vibes"}, refused="unknown template"),
        ),
        asked={DISCUSS: "fingerprint"},
    )
    store.write_book(book)
    assert store.book("session") == book
    path = tmp_path / "scene" / "advice" / "conditions" / "session.json"
    path.write_text(path.read_text(encoding="utf-8").replace(":", ": ", 1), encoding="utf-8")
    with pytest.raises(AdviceStoreError, match="canonical"):
        store.book("session")
    with pytest.raises(AdviceStoreError, match="digest"):
        store.prior("../x")


# --- 靠先验 --------------------------------------------------------------------------------------


def test_a_relation_that_passes_only_with_its_prior_is_tagged() -> None:
    """4 个工作日、讨论方案之后一半接修改代码：不加权过不了第 1 道，先验给这一对"很可能"、其余"不太可能"就过了——打"靠先验"。"""

    series = planted(4, seed=5, follow=0.5)
    tests = lane_tests(series)
    weights = {
        item: (2.0 if (item.antecedent, item.consequent) == (DISCUSS, CODE) else 0.5)
        for item in tests.keys()
        if not item.condition
    }
    state, moved = fold(None, tests, weights)
    found = next(item for item in state.relations if item.key == key(DISCUSS, CODE))
    assert found.status is Status.CANDIDATE and found.prior_only
    assert any("靠先验" in item.reason for item in moved if item.key == key(DISCUSS, CODE))
    plain, _moved = fold(None, lane_tests(series))
    assert all(item.key != key(DISCUSS, CODE) or item.status is not Status.CANDIDATE for item in plain.relations)
