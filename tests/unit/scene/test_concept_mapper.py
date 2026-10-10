"""③ 映射器（裁定 20）：基础概念从类编号现读、不调模型；候选只是挂在这个类上的细分概念；数值区别算法判、
语义区别按声明的材料问模型；材料缺记未决、模型失败不塌整天；「待定」只记情境、「非事件」不映射；整天映射的标记纪律与续跑。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.uri import BehaviorURI
from habitus.model_client import ModelTransportError
from habitus.scene.concepts import ConceptSet, ContextScope, GradeMeasure
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from habitus.scene.occurrences.mapper import (
    MAPPER_SYSTEM_PROMPT,
    MAPPER_VERSION,
    ConceptMapper,
    ConceptMapperError,
    MapperConfig,
    MappingMaterial,
    OccurrenceFacts,
    TimelineEntry,
    assemble_verdicts,
    map_closed_day,
    mapper_json_schema,
)
from habitus.scene.occurrences.model import UnresolvedReason
from habitus.series.reader import read_series
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import (
    ALL_CONCEPTS,
    BEDTIME,
    BEDTIME_KEY,
    EAT_ON_WAKING,
    REWORK,
    SLEEP_LATE,
    cid,
    concept_set,
    refinement,
)
from tests.unit.scene.fixtures import DAY1, SUBJECT, Site, at, publish, publish_gap

NOW = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
BASELINE = {BEDTIME_KEY: "23:30"}
YES_EAT = {"verdicts": [{"concept": "起床就吃", "verdict": "yes"}]}
NO_EAT = {"verdicts": [{"concept": "起床就吃", "verdict": "no"}]}
YES_REWORK = {"verdicts": [{"concept": "返工", "verdict": "yes"}]}
PENDING = "s-待定"
NOT_EVENT = "s-非事件"


class LaneMapper(ConceptMapper):
    """单条映射的测试默认在会话 lane（与概念夹具同一条）；整天映射由 ``map_closed_day`` 按事件序列给 lane。"""

    async def map(self, document, *, lane: str = "session", **kwargs):  # type: ignore[override]
        return await super().map(document, lane=lane, **kwargs)


def mapper_for(bodies, *, concepts: ConceptSet | None = None, config: MapperConfig | None = None):
    client, provider = recording_client(list(bodies))
    mapper = LaneMapper(
        client,
        concepts or concept_set(),
        config=config or MapperConfig(transient_retry_delay_seconds=0.0),
        subject=SUBJECT,
        clock=lambda: NOW,
    )
    return mapper, client, provider


def map_day(
    site: Site, store: ConceptHitStore, mapper: ConceptMapper, day=DAY1, *, baseline=None, situations=(), force=False
):
    return asyncio.run(
        map_closed_day(
            site.behavior_tree,
            store,
            mapper,
            day,
            now=NOW,
            series=read_series(site.behavior_tree, cutoff=day + timedelta(days=1)),
            situation_for=lambda _d: situations,
            baseline_for=lambda _d: BASELINE if baseline is None else baseline,
            force=force,
        )
    )


def document_for(site: Site, uri: str):
    return site.behavior_tree.read(BehaviorURI.parse(uri).to_address())


def everything(site: Site):
    return read_series(site.behavior_tree, cutoff=DAY1 + timedelta(days=60))


def timeline_for(site: Site, *uris: str) -> tuple[TimelineEntry, ...]:
    records = {record.uri: record for record in everything(site).records}
    return tuple(TimelineEntry.from_record(records[uri]) for uri in uris)


def gaps_for(site: Site, day) -> tuple[TimelineEntry, ...]:
    return tuple(TimelineEntry.from_gap(gap) for gap in everything(site).gaps_on(day))


def enum_of(client) -> list[str]:
    return client.schemas[-1]["properties"]["verdicts"]["items"]["properties"]["concept"]["enum"]


def test_the_version_follows_the_judged_concepts_not_the_class_names() -> None:
    """口径 = 提示词 + schema + 判定模型 + 要判的概念的指纹。类改名（基础概念的显示名）不必重映射；改细分概念的区别要。"""

    mapper, *_ = mapper_for([])
    assert mapper.version_for(kind_id("早餐")).startswith(f"{MAPPER_VERSION}+llm:fake-1+candidates:")
    renamed = ConceptSet(replace(item, title="睡觉") if item is BEDTIME else item for item in ALL_CONCEPTS)
    assert renamed.fingerprint == concept_set().fingerprint
    revised = ConceptSet(
        replace(item, definition="比常态晚三小时以上") if item is SLEEP_LATE else item for item in ALL_CONCEPTS
    )
    assert revised.fingerprint != concept_set().fingerprint


def test_a_class_without_refinements_is_read_from_its_id_and_never_asks_the_model(tmp_path) -> None:
    """「打球」这个类上没挂细分概念：候选为空，一次模型都不调；它的基础概念从编号现读、不存成命中。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "和同事打羽毛球", 19, 0, kind="打球", lasts_minutes=60)
    mapper, _client, provider = mapper_for([])

    record = asyncio.run(mapper.map(document_for(site, uri)))

    assert provider.calls == 0 and record.hits == () and not record.unresolved
    assert record.classified and record.kind_token == cid("打球")
    assert dict(record.graded_hits) == {cid("打球"): None}


def test_a_numeric_refinement_is_judged_by_the_algorithm_alone(tmp_path) -> None:
    """就寝 02:10、常态 23:30 → 偏移 160 分钟 → 「晚睡·轻」。这条已经是就寝，"是不是入睡"不用再问，模型一次都不调。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "上床", 2, 10, kind="就寝", summary="上床睡觉", lasts_minutes=30)
    mapper, _client, provider = mapper_for([])

    record = asyncio.run(
        mapper.map(
            document_for(site, uri), situation_hits=(ConceptHit("出差中"),), material=MappingMaterial(baseline=BASELINE)
        )
    )

    assert record.hits == (ConceptHit("晚睡", "轻"),) and not record.unresolved
    assert record.lane == "session" and record.recent_digest is None
    assert dict(record.graded_hits) == {cid("就寝"): None, "晚睡": "轻"}
    assert record.situation_hits == (ConceptHit("出差中"),) and record.baseline_snapshot == BASELINE
    assert (
        record.last_observed_at == at(DAY1, 2, 40)
        and record.mapper == mapper.version_for(record.kind_token)
        and record.mapped_at == NOW
    )
    assert provider.calls == 0

    early = publish(site.behavior_tree, DAY1, "上床", 22, 0, kind="就寝")
    record = asyncio.run(mapper.map(document_for(site, early), material=MappingMaterial(baseline=BASELINE)))
    assert record.hits == () and not record.unresolved
    # 常态没给：数值判不了，是未决，不是 false。
    record = asyncio.run(mapper.map(document_for(site, uri)))
    assert record.hits == () and dict(record.unresolved) == {"晚睡": UnresolvedReason.BASELINE_MISSING}
    assert f"unresolved: 晚睡 缺常态 {BEDTIME_KEY}" in record.signals
    assert provider.calls == 0


def test_a_semantic_refinement_is_asked_with_the_source_class_and_the_day_timeline(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    wake = publish(site.behavior_tree, DAY1, "起床下楼", 6, 40, kind="起床", summary="起床下楼")
    meal = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面", lasts_minutes=20)
    mapper, client, provider = mapper_for([YES_EAT])

    record = asyncio.run(
        mapper.map(document_for(site, meal), material=MappingMaterial(timeline=timeline_for(site, wake, meal)))
    )

    assert record.hits == (ConceptHit("起床就吃"),) and not record.unresolved
    assert enum_of(client) == ["起床就吃"]  # 只有挂在「早餐」上的细分概念
    prompt = provider.prompts[-1]
    assert "## 这条行为（已归为「早餐」）" in prompt
    assert "## 当天时间线（按时刻；标【未观测】/【没读懂】的是观测空白" in prompt
    assert "- 06:40–06:50 起床下楼（起床下楼）" in prompt and "- 07:00–07:20 吃了碗面（坐下吃了碗面）  ← 这条" in prompt
    assert f"- 起床就吃：{EAT_ON_WAKING.definition}" in prompt and "目标：（说不出目标）" in prompt
    assert "同在：（无）" in prompt and "地点：厨房" in prompt
    # 没给时间线 → 未决，不问模型。
    record = asyncio.run(mapper.map(document_for(site, meal)))
    assert dict(record.unresolved) == {"起床就吃": UnresolvedReason.TIMELINE_MISSING}
    assert "unresolved: 起床就吃 需要当天时间线" in record.signals
    assert provider.calls == 2  # 判两次


def test_each_record_is_judged_twice_and_a_disagreement_counts_on_neither_side(tmp_path) -> None:
    """同一条问两遍、第二遍候选倒过来排：一遍 yes 一遍 no → 未决（两次答得不一样），不算命中也不算没有。"""

    site = Site(tmp_path, now=NOW)
    wake = publish(site.behavior_tree, DAY1, "起床下楼", 6, 40, kind="起床", summary="起床下楼")
    meal = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面", lasts_minutes=20)
    mapper, _client, provider = mapper_for([YES_EAT, NO_EAT])
    record = asyncio.run(
        mapper.map(document_for(site, meal), material=MappingMaterial(timeline=timeline_for(site, wake, meal)))
    )
    assert record.hits == () and dict(record.unresolved) == {"起床就吃": UnresolvedReason.INCONSISTENT}
    assert provider.calls == 2 and any("两次答得不一样（yes / no）" in signal for signal in record.signals)
    assert "judged-twice" in mapper.version_for(kind_id("早餐"))


def test_the_second_ask_lists_the_candidates_in_reverse(tmp_path) -> None:
    concepts = ConceptSet([*ALL_CONCEPTS, refinement("早起吃", "七点前吃的", "早餐", context=ContextScope.OCCURRENCE)])
    site = Site(tmp_path, now=NOW)
    meal = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面")
    both = {"verdicts": [{"concept": "起床就吃", "verdict": "no"}, {"concept": "早起吃", "verdict": "yes"}]}
    mapper, _client, provider = mapper_for([both], concepts=concepts)
    wake = publish(site.behavior_tree, DAY1, "起床下楼", 6, 40, kind="起床", summary="起床下楼")
    record = asyncio.run(
        mapper.map(document_for(site, meal), material=MappingMaterial(timeline=timeline_for(site, wake, meal)))
    )
    first, second = provider.prompts[-2:]
    order = lambda prompt: sorted(("起床就吃", "早起吃"), key=lambda name: prompt.index(f"- {name}："))  # noqa: E731
    assert order(second) == list(reversed(order(first)))
    assert record.hits == (ConceptHit("早起吃"),)


def test_the_timeline_lists_observation_gaps_and_unknown_becomes_unresolved(tmp_path) -> None:
    """用户 09-27「没看到就不算」：时间线要列空白段；区别引用的事件落在空白里 → 模型答 unknown → 记未决。"""

    site = Site(tmp_path, now=NOW)
    meal = publish(site.behavior_tree, DAY1, "吃了碗面", 9, 30, kind="早餐", summary="坐下吃了碗面", lasts_minutes=20)
    publish_gap(site.behavior_tree, DAY1, (7, 0), (9, 0))  # 起床那一段没在看
    mapper, _client, provider = mapper_for([{"verdicts": [{"concept": "起床就吃", "verdict": "unknown"}]}])

    timeline = timeline_for(site, meal) + gaps_for(site, DAY1)
    record = asyncio.run(mapper.map(document_for(site, meal), material=MappingMaterial(timeline=timeline)))

    assert "- 07:00–09:00 【未观测】这段时间没有可用的观测，这段里发生过什么不知道" in provider.prompts[-1]
    assert "答 unknown（看不到，不是没发生）" in MAPPER_SYSTEM_PROMPT
    assert record.hits == () and dict(record.unresolved) == {"起床就吃": UnresolvedReason.MODEL_UNSEEN}
    assert "unresolved: 起床就吃 模型答看不到（材料里判不了）" in record.signals


def test_mapping_a_closed_day_puts_that_days_gaps_on_the_timeline(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "吃了碗面", 9, 30, kind="早餐", summary="坐下吃了碗面")
    publish_gap(site.behavior_tree, DAY1, (7, 0), (9, 0), kind="没读懂")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([NO_EAT])
    map_day(site, store, mapper)
    assert "- 07:00–09:00 【没读懂】" in provider.prompts[-1]


def test_a_recent_refinement_reads_the_previous_days_of_the_same_class(tmp_path) -> None:
    """「返工」要看近几天「修改代码」的记录：整天映射时往前读 ``recent_days`` 天、只取同一个类；别的类、窗外的不进来。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1 - timedelta(days=1), "改记忆召回", 15, 0, kind="修改代码", summary="改召回排序")
    publish(site.behavior_tree, DAY1 - timedelta(days=2), "查资料", 10, 0, kind="调研", summary="看论文")
    publish(site.behavior_tree, DAY1 - timedelta(days=5), "改很久以前的代码", 10, 0, kind="修改代码", summary="窗外")
    publish(site.behavior_tree, DAY1, "再改记忆召回", 9, 0, kind="修改代码", summary="又改召回排序")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, client, provider = mapper_for(
        [YES_REWORK], config=MapperConfig(recent_days=3, transient_retry_delay_seconds=0.0)
    )

    map_day(site, store, mapper)

    (record,) = store.read_day(DAY1)
    assert record.hits == (ConceptHit("返工"),) and enum_of(client) == ["返工"]
    prompt = provider.prompts[-1]
    assert "## 近几天「修改代码」的记录" in prompt and "改记忆召回（改召回排序）" in prompt
    assert "查资料" not in prompt and "改很久以前的代码" not in prompt
    assert f"- 返工：{REWORK.definition}" in prompt


def test_a_recent_refinement_without_that_material_stays_unresolved(tmp_path) -> None:
    """近几天记录"没给"（``recent=None``）与"给了、这几天没有"（``()``）是两回事：前者未决，后者照问。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "改记忆召回", 9, 0, kind="修改代码")
    mapper, _client, provider = mapper_for([{"verdicts": [{"concept": "返工", "verdict": "no"}]}])
    record = asyncio.run(mapper.map(document_for(site, uri)))
    assert dict(record.unresolved) == {"返工": UnresolvedReason.RECENT_MISSING}
    assert "unresolved: 返工 需要近几天同类记录" in record.signals
    assert provider.calls == 0
    record = asyncio.run(mapper.map(document_for(site, uri), material=MappingMaterial(recent=())))
    assert record.hits == () and not record.unresolved and provider.calls == 2
    assert "（这几天没有）" in provider.prompts[-1]


def test_a_model_failure_leaves_the_candidates_unresolved_instead_of_raising(tmp_path) -> None:
    """结构层两轮都没救回来 → 这些候选未决 + 信号；``model_client`` 的异常不出映射器。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐")
    mapper, _client, provider = mapper_for([{"verdicts": []}])
    record = asyncio.run(
        mapper.map(document_for(site, uri), material=MappingMaterial(timeline=timeline_for(site, uri)))
    )
    assert record.hits == () and dict(record.unresolved) == {"起床就吃": UnresolvedReason.MODEL_FAILED}
    # 失败信号由共用的调模型那一段（``scene.llm``）给，说得出是哪一层没过。
    assert any(signal.startswith("model: ModelStructuredOutputError") for signal in record.signals)
    assert provider.calls == 2


def test_an_incomplete_answer_is_corrected_by_the_structured_layer_and_leaves_a_note(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐")
    mapper, _client, provider = mapper_for([{"verdicts": []}, YES_EAT])
    record = asyncio.run(
        mapper.map(document_for(site, uri), material=MappingMaterial(timeline=timeline_for(site, uri)))
    )
    assert provider.calls == 3  # 第一次被结构层纠正一轮，第二次一次答对 and record.hits == (ConceptHit("起床就吃"),)
    assert "structured: answered on attempt 2" in record.signals

    candidates = (EAT_ON_WAKING, REWORK)
    assert assemble_verdicts(
        {"verdicts": [{"concept": "起床就吃", "verdict": "yes"}, {"concept": "返工", "verdict": "no"}]}, candidates
    ) == (("起床就吃",), ())
    # unknown 不是"没命中"：它单独一列，映射器把它记进现成的未决（用户 09-27"没看到就不算"）。
    unknown = {"verdicts": [{"concept": "起床就吃", "verdict": "unknown"}, {"concept": "返工", "verdict": "no"}]}
    assert assemble_verdicts(unknown, candidates) == ((), ("起床就吃",))
    with pytest.raises(ValueError, match="missing"):
        assemble_verdicts({"verdicts": [{"concept": "起床就吃", "verdict": "yes"}]}, candidates)
    with pytest.raises(ValueError, match="repeats"):
        assemble_verdicts(
            {
                "verdicts": [
                    {"concept": "起床就吃", "verdict": "yes"},
                    {"concept": "起床就吃", "verdict": "no"},
                    {"concept": "返工", "verdict": "no"},
                ]
            },
            candidates,
        )
    with pytest.raises(ValueError, match="not a candidate"):
        assemble_verdicts(
            {"verdicts": [{"concept": "晚睡", "verdict": "yes"}, {"concept": "返工", "verdict": "no"}]}, candidates
        )
    with pytest.raises(ValueError, match="verdict must be one of"):
        assemble_verdicts(
            {"verdicts": [{"concept": "起床就吃", "verdict": "maybe"}, {"concept": "返工", "verdict": "no"}]},
            candidates,
        )
    with pytest.raises(ConceptMapperError):
        mapper_json_schema(())


def test_a_pending_occurrence_only_carries_situations(tmp_path) -> None:
    """「待定」没有类：挂不上任何细分概念，也没有基础概念；只记那一刻的情境。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "讨论提示词", 15, 0, kind=PENDING)
    mapper, _client, provider = mapper_for([])
    record = asyncio.run(mapper.map(document_for(site, uri), classified=False, situation_hits=(ConceptHit("周末"),)))
    assert not record.classified and record.kind_token == PENDING and record.base_concept is None
    assert record.hits == () and dict(record.graded_hits) == {} and record.situation_hits == (ConceptHit("周末"),)
    assert provider.calls == 0


def test_situation_hits_must_be_situation_concepts_with_defined_grades(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "上床", 2, 10, kind="就寝")
    mapper, *_ = mapper_for([])
    document = document_for(site, uri)
    with pytest.raises(ConceptMapperError, match="not a situation concept"):
        asyncio.run(mapper.map(document, situation_hits=(ConceptHit("晚睡"),)))
    with pytest.raises(ConceptMapperError, match="not a situation concept"):
        asyncio.run(mapper.map(document, situation_hits=(ConceptHit("在家"),)))
    with pytest.raises(ConceptMapperError, match="no grade"):
        asyncio.run(mapper.map(document, situation_hits=(ConceptHit("出差中", "很远"),)))


def test_facts_come_from_the_document_and_measures_are_local_time(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "上床", 23, 50, kind="就寝", summary="上床", lasts_minutes=25, goal="睡觉")
    facts = OccurrenceFacts.from_document(document_for(site, uri))
    assert facts.measures()[GradeMeasure.START_MINUTE_OF_DAY] == 23 * 60 + 50
    assert facts.measures()[GradeMeasure.DURATION_MINUTES] == 25
    assert facts.kind_token == cid("就寝") and facts.goal == "睡觉" and facts.steps == ()
    assert "目标：睡觉" in facts.render(subject=SUBJECT)


def test_mapping_a_closed_day_skips_duplicates_and_non_events_resumes_and_marks_the_day(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "上床", 2, 10, kind="就寝", summary="上床睡觉")
    publish(site.behavior_tree, DAY1, "起床下楼", 6, 40, kind="起床")
    publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面")
    publish(
        site.behavior_tree, DAY1, "吃了碗面-2", 7, 0, kind="早餐", original_name="吃了碗面", summary="坐下吃了碗面"
    )  # 撞车消歧重复
    publish(site.behavior_tree, DAY1, "讨论提示词", 15, 0, kind=PENDING)
    publish(site.behavior_tree, DAY1, "看了眼手机", 15, 30, kind=NOT_EVENT)
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([YES_EAT])  # 只有吃面那条要问模型

    report = map_day(site, store, mapper, situations=(ConceptHit("周末"),))

    assert (report.mapped, report.resumed, report.duplicates_skipped, report.not_events_skipped) == (4, 0, 1, 1)
    assert (report.stale_removed, report.unresolved) == (0, 0)
    records = {record.address.name: record for record in store.read_day(DAY1)}
    assert set(records) == {"上床", "起床下楼", "吃了碗面", "讨论提示词"}  # 非事件没有记录
    assert records["上床"].hits == (ConceptHit("晚睡", "轻"),) and records["吃了碗面"].hits == (ConceptHit("起床就吃"),)
    assert not records["讨论提示词"].classified and records["起床下楼"].hits == ()
    assert all(record.situation_hits == (ConceptHit("周末"),) for record in records.values())
    marker = store.read_marker(DAY1)
    assert marker is not None and marker.records == 4
    # 标记汇总的是记录上读得出的概念：类编号（基础概念，不管概念集里有没有）+ 细分 + 情境。
    assert marker.concepts == {cid("就寝"), kind_id("起床"), cid("早餐"), "晚睡", "起床就吃", "周末"}
    # 完成标记按这一天自己的口径：那天出现的类各自的口径 + 情境（E9）
    assert marker.mapper == mapper.day_version(record.kind_token for record in records.values())
    assert store.days_done() == {DAY1} and provider.calls == 2

    # 续跑：同口径、编号没变、输入没变的记录不重问模型。
    report = map_day(site, store, mapper, situations=(ConceptHit("周末"),))
    assert (report.mapped, report.resumed) == (0, 4) and provider.calls == 2
    # 只有情境变了（这一晚不是周末了）：情境就地重算，不问模型
    report = map_day(site, store, mapper)
    assert (report.mapped, report.resumed, report.refreshed) == (0, 0, 4) and provider.calls == 2
    assert all(record.situation_hits == () for record in store.read_day(DAY1))


def test_a_kind_token_rewritten_by_the_vocabulary_forces_a_rejudge(tmp_path) -> None:
    """词表迁移改了树上这一条的编号：候选是挂在类上的细分概念，类换了候选就换了——盘上的旧记录一律重判，报成"变了"。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([NO_EAT])
    map_day(site, store, mapper)
    (old,) = store.read_day(DAY1)
    # 模拟迁移之前的样子：盘上那条记的还是「待定」。
    store.write(replace(old, kind_token=PENDING, classified=False))

    report = map_day(site, store, mapper)

    assert (report.mapped, report.resumed, report.rewritten) == (1, 0, 1) and report.changed
    (record,) = store.read_day(DAY1)
    assert record.kind_token == cid("早餐") and record.classified and provider.calls == 4


def test_resume_reasks_only_when_the_inputs_changed(tmp_path) -> None:
    """续跑判据是"输入没变"：只因缺常态而未决、常态表又没变 → 不重判；常态补上了 → 重判；模型上次没答成 → 重问。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "上床", 2, 10, kind="就寝", summary="上床睡觉")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([])
    # 第一夜没有常态：「晚睡」未决（缺常态）。
    report = map_day(site, store, mapper, baseline={})
    assert (report.mapped, report.unresolved) == (1, 1)
    assert dict(store.read_day(DAY1)[0].unresolved) == {"晚睡": UnresolvedReason.BASELINE_MISSING}
    # 第二夜常态还是没有 → 输入没变，不重判。
    report = map_day(site, store, mapper, baseline={})
    assert (report.mapped, report.resumed) == (0, 1)
    # 第三夜常态到了 → 材料变了，重判，判成；数值区别从头到尾不调模型。
    report = map_day(site, store, mapper)
    assert (report.mapped, report.unresolved) == (1, 0) and provider.calls == 0

    # 模型这一次没答成的记录：下一夜重问。
    meal_site = Site(tmp_path / "meal", now=NOW)
    publish(meal_site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐")
    meal_store = ConceptHitStore(tmp_path / "meal" / "scene")
    failing, *_ = mapper_for([{"verdicts": []}, {"verdicts": []}])  # 结构层两轮都不成形
    report = map_day(meal_site, meal_store, failing)
    assert report.unresolved == 1 and any(
        signal.startswith("model:") for signal in meal_store.read_day(DAY1)[0].signals
    )
    recovered, _c, recovered_provider = mapper_for([YES_EAT])
    report = map_day(meal_site, meal_store, recovered)
    assert (report.mapped, report.resumed, report.unresolved) == (1, 0, 0) and recovered_provider.calls == 2


def test_mapping_refuses_to_wipe_a_day_the_behaviour_tree_cannot_see(tmp_path) -> None:
    """行为树这一天比盘上少（读不到、或根指错到别的树）：多半是根指错或封口日算错，不能静默清掉多出来的再盖章。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "上床", 2, 10, kind="就寝")
    publish(site.behavior_tree, DAY1, "起床下楼", 6, 40, kind="起床")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, *_ = mapper_for([])
    map_day(site, store, mapper)
    assert len(store.read_day(DAY1)) == 2
    empty = Site(tmp_path / "empty", now=NOW)
    with pytest.raises(ConceptMapperError, match="force=True"):
        map_day(empty, store, mapper)
    other = Site(tmp_path / "elsewhere", now=NOW)
    publish(other.behavior_tree, DAY1, "打了球", 19, 0, kind="打球")  # 根指错到只有 1 条的树：盘上 2 条比树上多，照样拒
    with pytest.raises(ConceptMapperError, match="holds 2"):
        map_day(other, store, mapper)
    report = map_day(other, store, mapper, force=True)
    assert (report.mapped, report.stale_removed) == (1, 2) and len(store.read_day(DAY1)) == 1


def test_the_day_timeline_only_carries_the_same_lane_and_gaps(tmp_path) -> None:
    """两条 lane 相互独立（裁定 21-2）：判会话 lane 的一条时，时间线里没有物理 lane 的记录；观测空白没有 lane，两边都放。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "起床下楼", 6, 40, kind="起床", summary="起床下楼")
    publish(site.behavior_tree, DAY1, "出门散步", 6, 50, kind=kind_id("散步", Lane.PHYSICAL), summary="在小区里走")
    publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面")
    publish_gap(site.behavior_tree, DAY1, (5, 0), (6, 30))
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([YES_EAT])

    map_day(site, store, mapper)

    prompt = provider.prompts[-1]
    assert "起床下楼" in prompt and "【未观测】" in prompt
    assert "出门散步" not in prompt
    records = {record.address.name: record for record in store.read_day(DAY1)}
    assert records["出门散步"].lane == "physical" and records["吃了碗面"].lane == "session"
    assert provider.calls == 2  # 物理 lane 那条的类上没挂细分概念，不问模型


def test_the_day_timeline_stops_at_the_record_being_judged(tmp_path) -> None:
    """时间线截在这一条最后所见（裁定 27 第 10 条，E6）：上午那条看不到下午的事——细分标签不能取决于之后发生了什么。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "起床下楼", 6, 40, kind="起床", summary="起床下楼")
    publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面")
    publish(site.behavior_tree, DAY1, "下午又吃了一顿", 15, 0, kind="起床", summary="傍晚补了一顿")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([YES_EAT])

    map_day(site, store, mapper)

    prompt = provider.prompts[-1]
    assert "起床下楼" in prompt and "吃了碗面" in prompt
    assert "下午又吃了一顿" not in prompt


def test_the_baseline_shown_to_the_model_uses_class_names_not_ids(tmp_path) -> None:
    """常态表给模型看时，键里的概念显示类名（裁定 21-1）；盘上记录照旧存原键。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐")
    mapper, _client, provider = mapper_for([YES_EAT])
    record = asyncio.run(
        mapper.map(
            document_for(site, uri), material=MappingMaterial(baseline=BASELINE, timeline=timeline_for(site, uri))
        )
    )
    prompt = provider.prompts[-1]
    assert "- 就寝的近期常态时刻：23:30" in prompt
    assert cid("就寝") not in prompt
    assert dict(record.baseline_snapshot) == BASELINE


def test_resume_does_not_reask_an_unseen_answer_when_the_material_is_the_same(tmp_path) -> None:
    """模型答"看不到"：材料没变，再问一遍答案不会变——续跑不重问。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "吃了碗面", 9, 30, kind="早餐")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([{"verdicts": [{"concept": "起床就吃", "verdict": "unknown"}]}])
    map_day(site, store, mapper)
    report = map_day(site, store, mapper)
    assert (report.mapped, report.resumed) == (0, 1) and provider.calls == 2


@pytest.mark.parametrize(
    ("material", "reason"),
    [
        ("timeline", UnresolvedReason.TIMELINE_MISSING),
        ("recent", UnresolvedReason.RECENT_MISSING),
    ],
)
def test_resume_reasks_when_material_was_missing(tmp_path, material: str, reason: UnresolvedReason) -> None:
    """当时没给到材料（时间线 / 近几天记录）而未决的：整天映射时材料有了，要重问。"""

    site = Site(tmp_path, now=NOW)
    kind, answer = ("早餐", YES_EAT) if material == "timeline" else ("修改代码", YES_REWORK)
    uri = publish(site.behavior_tree, DAY1, "这一条", 9, 0, kind=kind)
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([answer])
    old = asyncio.run(mapper.map(document_for(site, uri)))  # 不给材料：未决
    assert dict(old.unresolved) == {next(iter(old.unresolved)): reason}
    store.write(old)

    report = map_day(site, store, mapper)

    assert (report.mapped, report.resumed, report.rewritten) == (1, 0, 1) and provider.calls == 2
    (record,) = store.read_day(DAY1)
    assert not record.unresolved and record.hits


def test_a_change_in_the_previous_days_records_forces_a_rejudge(tmp_path) -> None:
    """ "近几天同类记录"换了（前几天多了一条修改代码，或别的日子被迁移改了编号）：要这份材料的细分概念重判。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1 - timedelta(days=1), "改记忆召回", 15, 0, kind="修改代码")
    publish(site.behavior_tree, DAY1, "再改记忆召回", 9, 0, kind="修改代码")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, provider = mapper_for([YES_REWORK, YES_REWORK])
    map_day(site, store, mapper)
    (first,) = store.read_day(DAY1)
    assert first.recent_digest is not None

    report = map_day(site, store, mapper)
    assert (report.mapped, report.resumed) == (0, 1) and provider.calls == 2  # 材料没变：不重问

    publish(site.behavior_tree, DAY1 - timedelta(days=2), "又一处改动", 11, 0, kind="修改代码")
    report = map_day(site, store, mapper)

    assert (report.mapped, report.rewritten) == (1, 1) and report.changed and provider.calls == 4
    (second,) = store.read_day(DAY1)
    assert second.recent_digest != first.recent_digest


def test_a_day_with_a_model_failure_is_not_stamped_and_is_asked_again(tmp_path) -> None:
    """模型这一次没答成：这一天不盖完成章（盖了就不会再被访问，"没答成"就永远是"看不到"）；模型回来之后再映射一次，盖上（E10）。"""

    site = Site(tmp_path, now=NOW)
    publish(site.behavior_tree, DAY1, "吃了碗面", 7, 0, kind="早餐", summary="坐下吃了碗面")
    store = ConceptHitStore(tmp_path / "scene")
    mapper, _client, _provider = mapper_for([ModelTransportError("down")])
    report = map_day(site, store, mapper)
    assert report.model_failed > 0 and store.read_marker(DAY1) is None
    mapper, _client, provider = mapper_for([YES_EAT])
    report = map_day(site, store, mapper)
    assert report.model_failed == 0 and store.read_marker(DAY1) is not None and provider.calls > 0


def test_a_new_refinement_only_reopens_the_days_its_class_appears_on() -> None:
    """给「修改代码」加一个细分概念：只有出现修改代码的日子口径变了要回填；只有早餐的那天照旧算完成（E9）。"""

    before, *_ = mapper_for([], concepts=ConceptSet(item for item in ALL_CONCEPTS if item is not REWORK))
    after, *_ = mapper_for([], concepts=ConceptSet(ALL_CONCEPTS))
    breakfast_only = [kind_id("早餐")]
    with_coding = [kind_id("早餐"), kind_id("修改代码")]
    assert before.day_version(breakfast_only) == after.day_version(breakfast_only)
    assert before.day_version(with_coding) != after.day_version(with_coding)
