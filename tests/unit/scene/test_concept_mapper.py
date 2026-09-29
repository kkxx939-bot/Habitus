"""③ 映射器：旁册要全、机械判据不问模型、材料缺记未决、模型失败不塌整天、留痕；整天映射的标记纪律与续跑。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from habitus.behavior.uri import BehaviorURI
from habitus.scene.concepts import ConceptVectorIndex, GradeMeasure, embedding_text
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from habitus.scene.occurrences.mapper import (
    MAPPER_SYSTEM_PROMPT,
    MAPPER_VERSION,
    ConceptMapper,
    ConceptMapperError,
    MapperConfig,
    OccurrenceFacts,
    TimelineEntry,
    assemble_verdicts,
    map_closed_day,
    mapper_json_schema,
)
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.scene.concept_fixtures import (
    BALL,
    BEDTIME_KEY,
    BREAKFAST,
    DIMENSION,
    EXERCISE,
    SLEEP_LATE,
    TableEmbedder,
    concept_set,
)
from tests.unit.scene.fixtures import DAY1, SUBJECT, Site, at, publish, publish_gap

NOW = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
BASELINE = {BEDTIME_KEY: "23:30"}
TABLE = {
    embedding_text(SLEEP_LATE): (1.0, 0.0, 0.0, 0.0),
    embedding_text(BREAKFAST): (0.0, 1.0, 0.0, 0.0),
    embedding_text(EXERCISE): (0.0, 0.0, 1.0, 0.0),
    embedding_text(BALL): (0.0, 0.0, 0.9, 0.1),
    "就寝\n上床睡觉": (0.9, 0.1, 0.0, 0.0),
    "吃了碗面\n坐下吃了碗面": (0.1, 0.9, 0.0, 0.0),
    "和同事打羽毛球\n打了一小时": (0.0, 0.1, 0.0, 0.9),
}
YES_LATE = {"verdicts": [{"concept": "晚睡", "verdict": "yes"}]}
BREAKFAST_NOT_LATE = {"verdicts": [{"concept": "早餐", "verdict": "yes"}, {"concept": "晚睡", "verdict": "no"}]}
LATE_NOT_BREAKFAST = {"verdicts": [{"concept": "早餐", "verdict": "no"}, {"concept": "晚睡", "verdict": "yes"}]}


def index(*definitions) -> ConceptVectorIndex:
    built = ConceptVectorIndex("fake-embed", DIMENSION)
    for definition in definitions or (SLEEP_LATE, BREAKFAST, EXERCISE, BALL):
        built = built.with_vector(definition, TABLE[embedding_text(definition)])
    return built


def mapper_for(bodies, *, vectors: ConceptVectorIndex | None = None, config: MapperConfig | None = None, **kwargs):
    client, provider = recording_client(list(bodies))
    embedder = TableEmbedder(TABLE)
    mapper = ConceptMapper(
        client,
        embedder,
        concept_set(),
        vectors if vectors is not None else index(),
        config=config or MapperConfig(recall_limit=2, transient_retry_delay_seconds=0.0),
        subject=SUBJECT,
        clock=lambda: NOW,
        **kwargs,
    )
    return mapper, client, provider, embedder


def document_for(site: Site, uri: str):
    return site.behavior_tree.read(BehaviorURI.parse(uri).to_address())


def timeline_for(site: Site, *uris: str) -> tuple[TimelineEntry, ...]:
    return tuple(TimelineEntry.from_document(document_for(site, uri)) for uri in uris)


def gaps_for(site: Site, day) -> tuple[TimelineEntry, ...]:
    from habitus.behavior.model import BehaviorKind

    return tuple(TimelineEntry.from_gap(document) for document in site.behavior_tree.read_day(BehaviorKind.GAP, day))


def test_a_mapper_refuses_a_sidecar_that_does_not_cover_every_leaf_behaviour_concept() -> None:
    """旁册空着就跑，会得到一整天"零命中"并盖上完成标记；所以在构造时拦。"""

    with pytest.raises(ConceptMapperError, match="missing or stale"):
        mapper_for([], vectors=ConceptVectorIndex("fake-embed", DIMENSION))
    with pytest.raises(ConceptMapperError, match="早餐"):
        mapper_for([], vectors=index(SLEEP_LATE, BALL))
    mapper, *_ = mapper_for([], vectors=index(SLEEP_LATE, BALL, BREAKFAST))  # 「运动」有子概念，不参与，缺它没关系
    assert mapper.version == f"{MAPPER_VERSION}+emb:fake-embed+llm:fake-1+concepts:{concept_set().fingerprint}"


def test_a_mechanical_rule_runs_first_and_the_model_only_answers_the_semantic_gate(tmp_path) -> None:
    """就寝 02:10、常态 23:30 → 偏移 160 分钟，规则通过 → 只问模型"这是不是入睡"→ 「晚睡·轻」，档是算法定的。

    「早餐」要当天时间线、没给 → 未决，不进 enum。22:00 的就寝规则不通过 → 一次都不问模型。
    """

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 2, 10, summary="上床睡觉", lasts_minutes=30)
    mapper, client, provider, embedder = mapper_for([YES_LATE])

    record = asyncio.run(mapper.map(document_for(site, uri), situation_hits=(ConceptHit("出差中"),), baseline=BASELINE))

    assert record.hits == (ConceptHit("晚睡", "轻"),)
    assert record.unresolved == ("早餐",) and "unresolved: 早餐 需要当天时间线" in record.signals
    assert record.situation_hits == (ConceptHit("出差中"),) and record.baseline_snapshot == BASELINE
    assert record.kind_token == "就寝" and record.last_observed_at == at(DAY1, 2, 40)
    assert record.mapper == mapper.version and record.mapped_at == NOW
    assert client.schemas[-1]["properties"]["verdicts"]["items"]["properties"]["concept"]["enum"] == ["晚睡"]
    assert provider.calls == 1 and embedder.queries == ["就寝\n上床睡觉"]
    # C-8：规则已经把数值那一半判成立了，渲染时说清楚，模型不许再自己算常态偏移。
    assert "- 晚睡：入睡时刻晚于常态两小时以上 ← 数值条件已由算法判定成立（" in provider.prompts[-1]
    assert "这一条实际是 +160）：只判这条行为是不是「晚睡」说的那件事" in provider.prompts[-1]
    assert f"- {BEDTIME_KEY}：23:30" in provider.prompts[-1]

    early = publish(site.behavior_tree, DAY1, "就寝", 22, 0, summary="上床睡觉")
    record = asyncio.run(mapper.map(document_for(site, early), baseline=BASELINE))
    assert record.hits == () and record.unresolved == ("早餐",) and provider.calls == 1  # 规则说不是，模型没被问
    # 常态没给：机械判据判不了，是未决，不是 false；同样不问模型。
    record = asyncio.run(mapper.map(document_for(site, uri)))
    assert record.hits == () and set(record.unresolved) == {"晚睡", "早餐"} and provider.calls == 1
    assert f"unresolved: 晚睡 缺常态 {BEDTIME_KEY}" in record.signals


def test_a_day_context_concept_is_asked_with_the_day_timeline(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    wake = publish(site.behavior_tree, DAY1, "起床", 6, 40, summary="起床下楼")
    meal = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, summary="坐下吃了碗面", lasts_minutes=20, kind="吃饭")
    mapper, client, provider, _ = mapper_for([BREAKFAST_NOT_LATE])

    record = asyncio.run(mapper.map(document_for(site, meal), baseline=BASELINE, timeline=timeline_for(site, wake, meal)))

    assert record.hits == (ConceptHit("早餐"),) and record.unresolved == ()
    # 07:00 比常态就寝晚 7.5 小时，「晚睡」的规则通过了——它是不是入睡只有模型答得了，所以也在 enum 里。
    assert client.schemas[-1]["properties"]["verdicts"]["items"]["properties"]["concept"]["enum"] == ["早餐", "晚睡"]
    prompt = provider.prompts[-1]
    assert "## 当天时间线（按时刻；标【未观测】/【没读懂】的是观测空白" in prompt
    assert "- 06:40–06:50 起床（起床下楼）" in prompt and "- 07:00–07:20 吃了碗面（坐下吃了碗面）  ← 这条" in prompt
    assert "- 早餐：起床后两小时内的第一次进食" in prompt and "目标：（说不出目标）" in prompt
    assert "同在：（无）" in prompt and "地点：厨房" in prompt


def test_the_timeline_lists_observation_gaps_and_unknown_becomes_unresolved(tmp_path) -> None:
    """用户 09-27「没看到就不算」：时间线要列空白段；判据引用的事件落在空白里 → 模型答 unknown → 记未决。

    不列空白段的话时间线看上去连续，模型按"里面没有就是没发生"把这顿早餐答成 no，而 no 在账本里是反面证据
    （七d-2 的假负样本从映射口进来）。unknown 不是第三种结果，落在现成的 ``unresolved`` 里：不进分母不进分子。
    """

    site = Site(tmp_path, now=NOW)
    meal = publish(site.behavior_tree, DAY1, "吃了碗面", 9, 30, summary="坐下吃了碗面", lasts_minutes=20, kind="吃饭")
    publish_gap(site.behavior_tree, DAY1, (7, 0), (9, 0))  # 起床那一段没在看
    unknown = {"verdicts": [{"concept": "早餐", "verdict": "unknown"}, {"concept": "晚睡", "verdict": "no"}]}
    mapper, _client, provider, _ = mapper_for([unknown])

    timeline = timeline_for(site, meal) + gaps_for(site, DAY1)
    record = asyncio.run(mapper.map(document_for(site, meal), baseline=BASELINE, timeline=timeline))

    prompt = provider.prompts[-1]
    assert "- 07:00–09:00 【未观测】这段时间没有可用的观测，这段里发生过什么不知道" in prompt
    assert "答 unknown（看不到，不是没发生）" in MAPPER_SYSTEM_PROMPT
    # 「早餐」既没命中也没落空：进未决。
    assert record.hits == () and record.unresolved == ("早餐",)
    assert "unresolved: 早餐 模型答看不到（材料里判不了）" in record.signals


def test_mapping_a_closed_day_puts_that_days_gaps_on_the_timeline(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "吃了碗面", 9, 30, summary="坐下吃了碗面", kind="吃饭")
    publish_gap(site.behavior_tree, DAY1, (7, 0), (9, 0), kind="没读懂")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider, _ = mapper_for([BREAKFAST_NOT_LATE])
    asyncio.run(map_closed_day(site.behavior_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE))
    assert "- 07:00–09:00 【没读懂】" in provider.prompts[-1]


def test_previous_hits_join_the_recall_with_a_cap_and_a_flip_leaves_a_signal(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "和同事打羽毛球", 19, 0, summary="打了一小时", kind="打羽毛球", lasts_minutes=60)
    body = {"verdicts": [{"concept": "打球", "verdict": "no"}, {"concept": "早餐", "verdict": "no"}]}
    mapper, client, provider, _ = mapper_for(
        [body],
        previous_hits=lambda kind: ("打球", "运动", "出差中", "不存在的") if kind == "打羽毛球" else (),
        config=MapperConfig(recall_limit=1, previous_hits_limit=1, transient_retry_delay_seconds=0.0),
    )
    record = asyncio.run(mapper.map(document_for(site, uri), baseline=BASELINE, timeline=timeline_for(site, uri)))
    # 召回：向量 top-1（早餐）∪ 以前命中过的（只留叶子行为概念、按闸截断 → 打球）。
    assert client.schemas[-1]["properties"]["verdicts"]["items"]["properties"]["concept"]["enum"] == ["打球", "早餐"]
    assert record.hits == ()
    assert "flip: kind 打羽毛球 以前命中过 打球，这次没有" in record.signals
    assert provider.calls == 1


def test_a_model_failure_leaves_the_candidates_unresolved_instead_of_raising(tmp_path) -> None:
    """结构层两轮都没救回来 → 这些候选未决 + 信号；``model_client`` 的异常不出映射器。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, summary="坐下吃了碗面", kind="吃饭")
    mapper, _client, provider, _ = mapper_for([{"verdicts": []}])
    record = asyncio.run(mapper.map(document_for(site, uri), baseline=BASELINE, timeline=timeline_for(site, uri)))
    assert record.hits == () and set(record.unresolved) == {"早餐", "晚睡"}
    assert any(signal.startswith("model: ModelStructuredOutputError") for signal in record.signals)
    assert provider.calls == 2


def test_an_incomplete_answer_is_corrected_by_the_structured_layer_and_leaves_a_note(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, summary="坐下吃了碗面", kind="吃饭")
    mapper, _client, provider, _ = mapper_for([{"verdicts": [{"concept": "晚睡", "verdict": "no"}]}, BREAKFAST_NOT_LATE])
    record = asyncio.run(mapper.map(document_for(site, uri), baseline=BASELINE, timeline=timeline_for(site, uri)))
    assert provider.calls == 2 and record.hits == (ConceptHit("早餐"),)
    assert "structured: answered on attempt 2" in record.signals

    candidates = (BALL, BREAKFAST)
    assert assemble_verdicts({"verdicts": [{"concept": "打球", "verdict": "yes"}, {"concept": "早餐", "verdict": "no"}]}, candidates) == (("打球",), ())
    # unknown 不是"没命中"：它单独一列，映射器把它记进现成的未决（用户 09-27"没看到就不算"）。
    unknown = {"verdicts": [{"concept": "打球", "verdict": "unknown"}, {"concept": "早餐", "verdict": "no"}]}
    assert assemble_verdicts(unknown, candidates) == ((), ("打球",))
    with pytest.raises(ValueError, match="missing"):
        assemble_verdicts({"verdicts": [{"concept": "打球", "verdict": "yes"}]}, candidates)
    with pytest.raises(ValueError, match="repeats"):
        assemble_verdicts({"verdicts": [{"concept": "打球", "verdict": "yes"}, {"concept": "打球", "verdict": "no"}, {"concept": "早餐", "verdict": "no"}]}, candidates)
    with pytest.raises(ValueError, match="not a candidate"):
        assemble_verdicts({"verdicts": [{"concept": "晚睡", "verdict": "yes"}, {"concept": "早餐", "verdict": "no"}]}, candidates)
    with pytest.raises(ValueError, match="verdict must be one of"):
        assemble_verdicts({"verdicts": [{"concept": "打球", "verdict": "maybe"}, {"concept": "早餐", "verdict": "no"}]}, candidates)
    with pytest.raises(ConceptMapperError):
        mapper_json_schema(())


def test_situation_hits_must_be_situation_concepts_with_defined_grades(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 2, 10)
    mapper, *_ = mapper_for([])
    document = document_for(site, uri)
    with pytest.raises(ConceptMapperError, match="not a situation concept"):
        asyncio.run(mapper.map(document, situation_hits=(ConceptHit("早餐"),)))
    with pytest.raises(ConceptMapperError, match="not a situation concept"):
        asyncio.run(mapper.map(document, situation_hits=(ConceptHit("在家"),)))
    with pytest.raises(ConceptMapperError, match="no grade"):
        asyncio.run(mapper.map(document, situation_hits=(ConceptHit("出差中", "很远"),)))


def test_facts_come_from_the_document_and_measures_are_local_time(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 23, 50, summary="上床", lasts_minutes=25, goal="睡觉")
    facts = OccurrenceFacts.from_document(document_for(site, uri))
    assert facts.measures()[GradeMeasure.START_MINUTE_OF_DAY] == 23 * 60 + 50
    assert facts.measures()[GradeMeasure.DURATION_MINUTES] == 25
    assert facts.embedding_query() == "就寝\n上床" and facts.goal == "睡觉" and facts.steps == ()
    assert "目标：睡觉" in facts.render(subject=SUBJECT)


def test_mapping_a_closed_day_withdraws_the_marker_resumes_and_marks_the_day(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "就寝", 2, 10, summary="上床睡觉")
    publish(site.behavior_tree, DAY1, "起床", 6, 40, summary="起床下楼")
    publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, summary="坐下吃了碗面", kind="吃饭")
    publish(site.behavior_tree, DAY1, "吃了碗面-2", 7, 0, kind="吃饭", original_name="吃了碗面", summary="坐下吃了碗面")  # 撞车消歧重复
    store = ConceptHitStore(tmp_path / "scene")
    # 就寝 02:10：召回 早餐/晚睡，规则通过 → 问两个；起床（查表外，召回到 打球/早餐）答两个 false；吃面：早餐 true、晚睡 false。
    bodies = [LATE_NOT_BREAKFAST, {"verdicts": [{"concept": "打球", "verdict": "no"}, {"concept": "早餐", "verdict": "no"}]}, BREAKFAST_NOT_LATE]
    mapper, _client, provider, _ = mapper_for(bodies)

    report = asyncio.run(
        map_closed_day(
            site.behavior_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (ConceptHit("周末"),), baseline_for=lambda _d: BASELINE
        )
    )
    assert (report.mapped, report.resumed, report.duplicates_skipped, report.stale_removed, report.unresolved) == (3, 0, 1, 0, 0)
    records = store.read_day(DAY1)
    assert [record.hits for record in records] == [(ConceptHit("晚睡", "轻"),), (), (ConceptHit("早餐"),)]
    assert all(record.situation_hits == (ConceptHit("周末"),) for record in records)
    marker = store.read_marker(DAY1)
    assert marker is not None and marker.concepts == {"晚睡", "早餐", "周末"} and store.days_done(mapper=mapper.version) == {DAY1}
    assert provider.calls == 3

    # 续跑：同口径、已判成的记录不重问模型；标记先撤再落。
    report = asyncio.run(
        map_closed_day(site.behavior_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE)
    )
    assert (report.mapped, report.resumed) == (0, 3) and provider.calls == 3
    assert store.days_done() == {DAY1}


def test_resume_reasks_only_when_the_inputs_changed(tmp_path) -> None:
    """续跑判据是"输入没变"：只因缺常态而未决、常态表又没变 → 不重问；常态补上了 → 重问；模型上次没答成 → 重问。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "就寝", 2, 10, summary="上床睡觉")
    store = ConceptHitStore(tmp_path / "scene")
    only_breakfast = {"verdicts": [{"concept": "早餐", "verdict": "no"}]}
    mapper, _client, provider, _ = mapper_for([only_breakfast, LATE_NOT_BREAKFAST])
    # 第一夜没有常态：「晚睡」未决（缺常态），只问了「早餐」一次。
    report = asyncio.run(map_closed_day(site.behavior_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: {}))
    assert (report.mapped, report.unresolved) == (1, 1) and provider.calls == 1
    assert store.read_day(DAY1)[0].unresolved == ("晚睡",)
    # 第二夜常态还是没有 → 输入没变，不重问。
    report = asyncio.run(map_closed_day(site.behavior_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: {}))
    assert (report.mapped, report.resumed) == (0, 1) and provider.calls == 1
    # 第三夜常态到了 → 材料变了，重问，判成。
    report = asyncio.run(map_closed_day(site.behavior_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE))
    assert (report.mapped, report.unresolved) == (1, 0) and provider.calls == 2
    # 模型这一次没答成的记录：下一夜重问。
    failing, *_ = mapper_for([{"verdicts": []}, {"verdicts": []}])  # 结构层两轮都不成形
    store.retain_only(DAY1, frozenset())
    report = asyncio.run(map_closed_day(site.behavior_tree, store, failing, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE))
    assert report.unresolved == 1 and any(signal.startswith("model:") for signal in store.read_day(DAY1)[0].signals)
    recovered, _c, recovered_provider, _e = mapper_for([LATE_NOT_BREAKFAST])
    report = asyncio.run(map_closed_day(site.behavior_tree, store, recovered, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE))
    assert (report.mapped, report.resumed, report.unresolved) == (1, 0, 0) and recovered_provider.calls == 1


def test_mapping_refuses_to_wipe_a_day_the_behaviour_tree_cannot_see(tmp_path) -> None:
    """行为树这一天比盘上少（读不到、或根指错到别的树）：多半是根指错或封口日算错，不能静默清掉多出来的再盖章。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 2, 10, summary="上床睡觉")
    publish(site.behavior_tree, DAY1, "起床", 6, 40, summary="起床下楼")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, *_ = mapper_for([LATE_NOT_BREAKFAST, {"verdicts": [{"concept": "打球", "verdict": "no"}, {"concept": "早餐", "verdict": "no"}]}, BREAKFAST_NOT_LATE])
    asyncio.run(map_closed_day(site.behavior_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE))
    assert len(store.read_day(DAY1)) == 2
    empty_tree = Site(tmp_path / "empty", now=NOW).behavior_tree
    with pytest.raises(ConceptMapperError, match="force=True"):
        asyncio.run(map_closed_day(empty_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE))
    other_tree = Site(tmp_path / "elsewhere", now=NOW).behavior_tree
    publish(other_tree, DAY1, "吃了碗面", 7, 0, summary="坐下吃了碗面", kind="吃饭")  # 根指错到只有 1 条的树：盘上 2 条比树上多，照样拒
    with pytest.raises(ConceptMapperError, match="holds 2"):
        asyncio.run(map_closed_day(other_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE))
    report = asyncio.run(
        map_closed_day(other_tree, store, mapper, DAY1, now=NOW, situation_for=lambda _d: (), baseline_for=lambda _d: BASELINE, force=True)
    )
    assert (report.mapped, report.stale_removed) == (1, 2) and len(store.read_day(DAY1)) == 1
    assert BehaviorURI.parse(uri).to_address().identity_name  # 只是让 uri 有用处
