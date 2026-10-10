"""夜批：同步词表 → 迁移重映射 → 映射，在真链子上跑一遍。

词表口用一个假的 ``ClassCatalog``：类与变更流由测试直接给（真实的变更日志读法在 ``test_scene_vocabulary`` 里测）。
行为树上的编号用 ``kind_id(类名)``，与概念夹具里基础概念的身份同一套。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import timedelta

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.model_client import ChatClient, ModelResponse
from habitus.model_client.structured import StructuredChatClient
from habitus.runtime.scene_night import SceneNightConfig, SceneNightlyRun
from habitus.runtime.scene_rhythms import class_curves
from habitus.scene.calendar import NominalCalendar
from habitus.scene.concepts import ConceptSet, ConceptStore, ContextScope
from habitus.scene.concepts.catalog import CatalogChanges, CatalogClass, MovedOccurrence
from habitus.scene.occurrences import ConceptHitStore
from habitus.scene.occurrences.mapper import ConceptMapper, MapperConfig
from habitus.scene.relations import RelationConfig
from habitus.scene.relations.store import RelationStore
from habitus.series.reader import read_series
from tests.unit.foresight.scripted_model import ScriptedProvider, model_config
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.prediction_tree_fixtures import curve, tree
from tests.unit.scene.concept_fixtures import (
    BEDTIME_KEY,
    EAT_ON_WAKING,
    SLEEP_LATE,
    TRAVELLING,
    WEEKEND,
    group,
    refinement,
)
from tests.unit.scene.fixtures import CST, DAY1, SUBJECT, Site, at, publish
from tests.unit.scene.hit_fixtures import MAPPER, record

TONIGHT = DAY1 + timedelta(days=3)  # 2026-08-18，周二
NOW = at(TONIGHT + timedelta(days=1), 3, 0)
SLEEP, BREAKFAST, MEAL, BRUNCH = (kind_id(name) for name in ("就寝", "早餐", "吃饭", "早午餐"))
#: 词表里的类（编号 → 类名、判据）。「吃饭」「早午餐」只在合并 / 拆分的测试里出现。
CLASSES = {SLEEP: ("就寝", "上床睡觉"), BREAKFAST: ("早餐", "早上的第一顿饭"), MEAL: ("吃饭", "坐下吃一顿饭")}
#: 夜批开跑前盘上已有的、模型写的概念（基础概念由同步词表生成）。
AUTHORED = (SLEEP_LATE, EAT_ON_WAKING, TRAVELLING, WEEKEND)


@dataclass
class FakeCatalog:
    """按 ``ClassCatalog`` 协议的假词表口：``versions`` 是每一版的变更，按版本号从 1 起排。"""

    active: dict[str, tuple[str, str]] = field(default_factory=lambda: dict(CLASSES))
    retired: dict[str, tuple[str, str]] = field(default_factory=dict)
    versions: list[CatalogChanges] = field(default_factory=list)

    def classes(self) -> tuple[CatalogClass, ...]:
        found = [
            CatalogClass(class_id, name, criterion, "session", True)
            for class_id, (name, criterion) in self.active.items()
        ]
        found += [
            CatalogClass(class_id, name, criterion, "session", False)
            for class_id, (name, criterion) in self.retired.items()
        ]
        return tuple(found)

    def version(self) -> int:
        return len(self.versions) + 1

    def changes_since(self, version: int) -> CatalogChanges:
        """第 1 版是起步（全部类），之后每一版是 ``versions`` 里的一项；合起来给。"""

        later = self.versions[max(0, version - 1) :]
        return CatalogChanges(
            since=version,
            version=self.version(),
            split_from={key: value for change in later for key, value in change.split_from.items()},
            merged_into={key: value for change in later for key, value in change.merged_into.items()},
            moved=tuple(item for change in later for item in change.moved),
        )

    def split(self, source: str, new: str, title: str) -> None:
        self.active[new] = (title, f"{title}这件事")
        self.versions.append(CatalogChanges(since=0, version=0, split_from={new: source}))

    def merge(self, source: str, target: str) -> None:
        self.retired[source] = self.active.pop(source)
        self.versions.append(CatalogChanges(since=0, version=0, merged_into={source: target}))

    def move(self, uri: str, source: str, target: str) -> None:
        self.versions.append(CatalogChanges(since=0, version=0, moved=(MovedOccurrence(uri, source, target),)))


class AnswerByCandidates(ScriptedProvider):
    """假模型答映射（只有语义区别会问到，晚睡是规则判的）：``yes`` 里写 ``行为名/概念名``，其余一律 ``no``；计 ``calls``、记 ``prompts``。"""

    def __init__(self, yes: frozenset[str]) -> None:
        super().__init__([])
        self.yes = yes

    async def complete_async(self, request):  # type: ignore[no-untyped-def, override]
        prompt = request.request.messages[-1].content or ""
        self.prompts.append(prompt)
        self.calls += 1
        behaviour = next(line.removeprefix("行为：") for line in prompt.splitlines() if line.startswith("行为："))
        section = prompt.split("## 候选细分概念")[-1]
        names = [line.split("：", 1)[0].removeprefix("- ") for line in section.splitlines() if line.startswith("- ")]
        body = {
            "verdicts": [
                {"concept": name, "verdict": "yes" if f"{behaviour}/{name}" in self.yes else "no"} for name in names
            ]
        }
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )


def site_with_three_bedtimes(tmp_path) -> tuple[Site, ConceptHitStore]:
    """前三天各一条就寝（够常态的三个样本），直接写进命中盘——历史不必再调模型。"""

    site = Site(tmp_path, now=NOW)
    hits = ConceptHitStore(tmp_path / "scene")
    for offset, (hour, minute) in enumerate(((23, 30), (23, 40), (23, 20))):
        day = DAY1 + timedelta(days=offset)
        hits.write(record(day, "睡觉", hour, minute, "就寝"))
        hits.complete_day(day, records=1, completed_at=at(day, 23, 59), mapper=MAPPER)
    return site, hits


def nightly(
    tmp_path,
    hits: ConceptHitStore,
    site: Site,
    *,
    yes: frozenset[str] = frozenset(),
    catalog: FakeCatalog | None = None,
) -> tuple[SceneNightlyRun, AnswerByCandidates]:
    concepts = ConceptStore(tmp_path / "scene")
    for definition in AUTHORED:
        if not concepts.exists(definition.name):
            concepts.write(definition)
    provider = AnswerByCandidates(yes)
    client = StructuredChatClient(ChatClient(model_config(), provider), validation_retries=1)

    def mapper_for(concept_set: ConceptSet) -> ConceptMapper:
        return ConceptMapper(
            client,
            concept_set,
            config=MapperConfig(transient_retry_delay_seconds=0.0),
            subject=SUBJECT,
            clock=lambda: NOW,
        )

    run = SceneNightlyRun(
        behavior_tree=site.behavior_tree,
        catalog=catalog or FakeCatalog(),
        concepts=concepts,
        hits=hits,
        calendar=NominalCalendar(),
        timezone=CST,
        mapper_for=mapper_for,
        relations=RelationStore(tmp_path / "scene"),
        relation_config=RelationConfig(slot_minutes=15, transition_window_slots=3, recurrence_window_days=90),
        subject=SUBJECT,
    )
    return run, provider


def prediction_tree():
    """早餐每天 07:00–08:30 有个峰；就寝夜里有个峰（曲线按类编号存）。"""

    curves = {}
    for weekday in range(7):
        curves[(weekday, BREAKFAST)] = curve({14: 0.2, 15: 0.5, 16: 0.1})
        curves[(weekday, SLEEP)] = curve({46: 0.4, 47: 0.5})
    return tree(curves)


def night(run: SceneNightlyRun, day, now):  # type: ignore[no-untyped-def]
    """跑一晚：序列按"夜批那天"截（``now`` 的日子），与预测树同一份。"""

    series = read_series(run.behavior_tree, cutoff=now.date())
    assert day < series.cutoff  # 第 N 晚映射 N 之前封口的日子
    return asyncio.run(run.run(series=series, tree=prediction_tree(), now=now))


def publish_tonight(site: Site, day=TONIGHT) -> tuple[str, str]:
    """就寝 02:10（比常态 23:30 晚 160 分钟 → 晚睡）+ 07:30 一碗面（早餐这个类）。"""

    bed = publish(site.behavior_tree, day, "就寝", 2, 10, summary="上床睡觉", kind="就寝", lasts_minutes=30)
    meal = publish(site.behavior_tree, day, "吃了碗面", 7, 30, summary="坐下吃了碗面", kind="早餐")
    return bed, meal


def test_the_steps_run_in_order_on_a_real_chain(tmp_path) -> None:
    """就寝 02:10 → 规则判出晚睡（不问模型）；07:30 那碗面是早餐这个类，「起床就吃」要当天时间线、问模型。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, provider = nightly(tmp_path, hits, site, yes=frozenset({"吃了碗面/起床就吃"}))

    report = night(run, TONIGHT, NOW)
    # 只有「起床就吃」（语义区别、要当天时间线）问了模型；晚睡是规则判的
    assert report.mapped == 2 and report.unresolved == 0 and provider.calls == 2  # 「起床就吃」判两次
    assert "起床就吃" in provider.prompts[0] and "晚睡" not in provider.prompts[0].split("## 候选细分概念")[-1]
    assert report.concepts == 3 + len(AUTHORED)  # 三个类的基础概念 + 盘上写好的四个
    records = {record.kind_token: record for record in run.hits.read_day(TONIGHT)}
    assert {hit.concept for hit in records[SLEEP].hits} == {"晚睡"}
    assert {hit.concept for hit in records[BREAKFAST].hits} == {"起床就吃"}
    # 映射时 lane 由词表口给（``catalog.lane_of``），记进命中记录
    assert {record.lane for record in records.values()} == {"session"}
    assert "映射 1 天 2 条" in report.summary()
    # B4：关系表这一晚落了盘（数据只有几天：全是样本不够，没有迁移）
    assert run.relations.nights("session") == (NOW.date(),) and report.transitions == 0
    assert report.relations["session"]["candidate"] == 0 and report.relations["session"]["sparse"] > 0


def test_the_first_night_generates_a_base_concept_per_class_and_records_the_synced_version(tmp_path) -> None:
    """同步词表：每个类一个基础概念（身份 = 类编号，显示名 = 类名），停用的类也有（标停用）；同步到的版本记下。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    assert run.concepts.synced_version() == 0
    report = night(run, TONIGHT, NOW)
    concepts = run.concepts.read_all()
    assert {concepts[class_id].label for class_id in CLASSES} == {"就寝", "早餐", "吃饭"}
    assert concepts.base_for(SLEEP) == SLEEP and not concepts[SLEEP].retired
    assert run.concepts.synced_version() == catalog.version() == 1
    assert sum("已同步" in signal for signal in report.signals) == 3
    # 第二夜没有变更：基础概念一个都不重写
    again = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert not any("已同步" in signal for signal in again.signals)
    # 类改了名：基础概念跟着改显示名，身份（编号）不变
    catalog.active[MEAL] = ("正餐", "坐下吃一顿饭")
    renamed = night(run, TONIGHT + timedelta(days=2), NOW + timedelta(days=2))
    assert run.concepts.read_all()[MEAL].label == "正餐" and any("「正餐」" in signal for signal in renamed.signals)


def test_a_moved_occurrence_on_an_earlier_day_remaps_that_day(tmp_path) -> None:
    """迁移把 TONIGHT 那碗面从「早餐」改到「吃饭」：那一天重映射（编号变了就重判）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    _bed, meal = publish_tonight(site)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    night(run, TONIGHT, NOW)

    address = BehaviorURI.parse(meal).to_address()
    BehaviorDocumentWriter(site.behavior_tree, ProcessLocalLockStore(), clock=lambda: NOW).restamp_kind_token(
        address, MEAL
    )
    catalog.move(meal, BREAKFAST, MEAL)
    report = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert any(signal.startswith(f"vocabulary: {TONIGHT} 受迁移影响 → 重判 1 条") for signal in report.signals)
    assert run.hits.read(address).kind_token == MEAL
    assert report.remapped_days == 1


def test_a_missing_baseline_is_reported_and_the_candidate_goes_unresolved(tmp_path) -> None:
    """常态样本不够 → 那个键不给 → 引用它的「晚睡」记未决（不是"没命中"），而且夜批把这件事报出来。"""

    site = Site(tmp_path, now=NOW)
    hits = ConceptHitStore(tmp_path / "scene")  # 一条历史都没有
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="就寝", lasts_minutes=30)
    run, _provider = nightly(tmp_path, hits, site)

    report = night(run, TONIGHT, NOW)
    assert report.mapped == 1 and report.unresolved == 1
    assert any(f"{BEDTIME_KEY} 样本不够" in signal for signal in report.signals)


def test_a_move_also_redoes_later_days_and_the_days_that_read_it_as_recent_records(tmp_path) -> None:
    """迁移改了 TONIGHT 那碗面的编号：
    - 夜批补跑更早的日子时，已经映射过的更晚的日子也重做（不只看"今天之前"）；
    - 有要"近几天同类记录"的细分概念时，之后几天的材料变了 → 那几天里那个类的记录重判（材料摘要变了）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    _bed, meal = publish_tonight(site)
    later = TONIGHT + timedelta(days=1)
    publish(site.behavior_tree, later, "又吃了碗面", 7, 40, summary="坐下又吃了碗面", kind="早餐")
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    run.concepts.write(refinement("连着吃面", "前几天也吃过同样的早餐", "早餐", context=ContextScope.RECENT))
    night(run, TONIGHT, NOW)
    night(run, later, NOW + timedelta(days=1))
    before = run.hits.read(BehaviorURI.parse(next(iter_records(run, later))).to_address()).recent_digest

    address = BehaviorURI.parse(meal).to_address()
    BehaviorDocumentWriter(site.behavior_tree, ProcessLocalLockStore(), clock=lambda: NOW).restamp_kind_token(
        address, MEAL
    )
    catalog.move(meal, BREAKFAST, MEAL)
    # 补跑更早的一天（第一条历史之前那天，还没映射过）：TONIGHT 与它之后的 later 都已映射过，都在迁移影响范围里
    earlier = DAY1 - timedelta(days=1)
    report = night(run, earlier, NOW + timedelta(days=2))
    assert any(signal.startswith(f"vocabulary: {TONIGHT} 受迁移影响 → 重判 1 条") for signal in report.signals)
    assert any(signal.startswith(f"vocabulary: {later} 受迁移影响 → 重判 1 条") for signal in report.signals)
    after = run.hits.read(BehaviorURI.parse(next(iter_records(run, later))).to_address()).recent_digest
    assert before is not None and after is not None and before != after


def iter_records(run: SceneNightlyRun, day):  # type: ignore[no-untyped-def]
    """那一天早餐这个类的记录地址。"""

    return (record.occurrence_uri for record in run.hits.read_day(day) if record.kind_token == BREAKFAST)


def test_a_split_rewrites_group_members_and_the_bridge_follows(tmp_path) -> None:
    """汇总概念「用餐」= 早餐 + 吃饭：从早餐拆出早午餐 → 同步时成员改写成 早餐 + 吃饭 + 早午餐，节律的桥与命中读同一份成员
    （第二轮评审 C2）；早餐并进吃饭 → 成员里的早餐换成吃饭（去重）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    run.concepts.write(group("用餐", "坐下吃东西", "早餐", "吃饭"))
    night(run, TONIGHT, NOW)
    catalog.split(BREAKFAST, BRUNCH, "早午餐")
    night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    concepts = run.concepts.read_all()
    assert concepts["用餐"].classes == (BREAKFAST, MEAL, BRUNCH)
    assert class_curves(concepts)["用餐"] == (BREAKFAST, MEAL, BRUNCH)
    assert "用餐" in concepts.ancestors(BRUNCH)  # 命中聚合也认新成员

    catalog.merge(BREAKFAST, MEAL)
    night(run, TONIGHT + timedelta(days=2), NOW + timedelta(days=2))
    assert run.concepts.read_all()["用餐"].classes == (MEAL, BRUNCH)


def test_with_model_advice_the_night_asks_priors_once_and_reads_them_from_the_cache_after(tmp_path) -> None:
    """B4 接上先验：第一晚每个后果乱序问三遍并缓存；第二晚词表与概念没变，一次都不再问。"""

    from habitus.runtime.scene_advice import RelationAdvice
    from habitus.scene.advice.prior import PriorAdvisor, PriorConfig
    from habitus.scene.advice.store import AdviceStore
    from tests.unit.scene.advice_fixtures import answering_client

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    client, advisor_model = answering_client()
    run.advice = RelationAdvice(
        store=AdviceStore(tmp_path / "scene"),
        prior=PriorAdvisor(client, config=PriorConfig(transient_retry_delay_seconds=0.0)),
        conditions=None,
    )
    report = night(run, TONIGHT, NOW)
    first = advisor_model.calls
    assert first > 0 and first % 3 == 0 and any(signal.startswith("prior:") for signal in report.signals)
    assert list((tmp_path / "scene" / "advice" / "priors").glob("*.json"))
    night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert advisor_model.calls == first


def three_days_of_meals_and_sleep(site: Site) -> list:  # type: ignore[type-arg]
    days = [TONIGHT - timedelta(days=offset) for offset in (2, 1, 0)]
    for day in days:
        publish_tonight(site, day)
    return days


def test_mapping_works_through_a_backlog_newest_first_within_a_nightly_budget(tmp_path) -> None:
    """一晚最多映射今晚那天 + 预算那么多天，最新的先；没轮到的报出来，下一晚接着补。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    days = three_days_of_meals_and_sleep(site)
    run, _provider = nightly(tmp_path, hits, site)
    run.config = SceneNightConfig(mapping_days_per_night=1)
    first = night(run, TONIGHT, NOW)
    assert first.mapped_days == (days[2], days[1]) and first.backlog == 1
    assert any(signal.startswith("mapping: 还有 1 天") for signal in first.signals)
    second = night(run, TONIGHT, NOW + timedelta(days=1))
    assert second.mapped_days == (days[0],) and second.backlog == 0


def test_a_new_refinement_backfills_only_the_records_of_its_class(tmp_path) -> None:
    """新写了一个挂在「就寝」上的语义细分概念：已映射的日子都要回填，但只有就寝那几条问模型，早餐那几条照常续跑。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    days = three_days_of_meals_and_sleep(site)
    run, provider = nightly(tmp_path, hits, site)
    night(run, TONIGHT, NOW)
    asked = provider.calls
    run.concepts.write(refinement("睡前看手机", "上床之前还在看手机", "就寝"))
    report = night(run, TONIGHT, NOW + timedelta(days=1))
    assert set(report.mapped_days) == set(days) and report.backlog == 0
    prompts = provider.prompts[asked:]
    assert provider.calls - asked == 2 * len(days)  # 每天一条就寝、判两次
    assert all("行为：就寝" in prompt and "睡前看手机" in prompt for prompt in prompts)
    assert report.resumed == len(days)  # 早餐那几条候选没变：续跑


def test_concepts_are_authored_when_the_class_list_changes_and_once_a_week_from_monday(tmp_path) -> None:
    from habitus.runtime.scene_authoring import ConceptAuthoring
    from habitus.scene.concepts.author import AuthorConfig, ConceptAuthor
    from tests.unit.foresight.scripted_model import recording_client

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    body = {
        "concepts": [
            {
                "name": "吃得早",
                "definition": "七点以前吃的早餐",
                "role": "behavior",
                "kind": "refinement",
                "classes": ["早餐"],
                "context": "occurrence",
                "baseline_keys": [],
                "rule": None,
                "grades": [],
                "situation": None,
                "why": "提醒句一样，只是时间早",
            }
        ]
    }
    client, author_model = recording_client([body])
    run.authoring = ConceptAuthoring(
        ConceptAuthor(client, config=AuthorConfig(transient_retry_delay_seconds=0.0), clock=lambda: NOW)
    )
    report = night(run, TONIGHT, NOW)
    assert author_model.calls == 1 and "吃得早" in run.concepts.read_all()
    assert any("新概念「吃得早」" in signal for signal in report.signals)
    assert "例：吃了碗面：坐下吃了碗面" in author_model.prompts[0]  # 真例子来自序列
    assert NOW.date().weekday() == 2  # 第一晚是周三
    night(run, TONIGHT, NOW + timedelta(days=1))
    assert author_model.calls == 1  # 周四：类清单没变、这一周写过了，不再问
    catalog.active[MEAL] = ("正餐", "坐下吃一顿饭")
    report = night(run, TONIGHT, NOW + timedelta(days=2))
    assert author_model.calls == 2 and any("类清单变了" in signal for signal in report.signals)  # 改了名：当晚问
    report = night(run, TONIGHT, NOW + timedelta(days=5))
    assert author_model.calls == 3 and any(
        "这一周还没写过" in signal for signal in report.signals
    )  # 下一个周一：每周一次
    night(run, TONIGHT, NOW + timedelta(days=6))
    assert author_model.calls == 3  # 周二：这一周已写过


def test_a_day_that_gets_a_late_record_after_its_stamp_is_mapped_again(tmp_path) -> None:
    """那天盖完完成章之后，归约又补发进来一条：标记的条数与序列对不上，下一晚重做这一天，补发的那条有命中、旧的续跑（E1）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    night(run, TONIGHT, NOW)
    marker = hits.read_marker(TONIGHT)
    assert marker is not None and marker.records == 2
    publish(site.behavior_tree, TONIGHT, "又吃了点", 23, 50, summary="夜宵", kind="早餐")
    report = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert TONIGHT in report.mapped_days and report.resumed == 2 and report.mapped == 1
    marker = hits.read_marker(TONIGHT)
    assert marker is not None and marker.records == 3 and len(hits.read_day(TONIGHT)) == 3


def test_a_corrupt_latest_relation_file_does_not_stall_the_nights(tmp_path) -> None:
    """最近一晚的关系文件坏了：下一晚接在更早一个读得了的晚上之后折叠、写出新的一晚，并报出来——不让夜批与预测层互相卡死（E10）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    night(run, TONIGHT, NOW)
    store = run.relations
    (lane,) = store.lanes()
    latest = store.latest_night(lane)
    assert latest is not None
    (tmp_path / "scene" / "relations" / lane / f"{latest.isoformat()}.json").write_text("{", encoding="utf-8")
    report = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert any("读不了" in signal for signal in report.signals)
    assert store.latest_night(lane) == latest + timedelta(days=1)
    store.read(lane, latest + timedelta(days=1))
