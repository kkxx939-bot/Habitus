"""夜批：同步词表 → 迁移重映射 → 映射，在真链子上跑一遍。

词表口用一个假的 ``ClassCatalog``：类与变更流由测试直接给（真实的变更日志读法在 ``test_scene_vocabulary`` 里测）。
行为树上的编号用 ``kind_id(类名)``，与概念夹具里基础概念的身份同一套。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import timedelta

from habitus.behavior.tree.writer import BehaviorDocumentWriter
from habitus.behavior.uri import BehaviorURI
from habitus.conditions import LocalCalendar
from habitus.model_client import ChatClient, ModelResponse
from habitus.model_client.structured import StructuredChatClient
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.scene.concepts import ConceptSet, ContextScope
from habitus.scene.concepts.calendar_situations import calendar_concepts
from habitus.scene.concepts.catalog import CatalogChanges, CatalogClass, MovedOccurrence
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences import ConceptHitStore
from habitus.scene.occurrences.mapper import ConceptMapper, ConceptMapperError, MapperConfig
from habitus.scene.occurrences.model import UnresolvedReason
from habitus.scene.relations import RelationConfig
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.series.reader import read_series
from tests.lock_store import ProcessLocalLockStore
from tests.unit.foresight.scripted_model import ScriptedProvider, model_config
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import (
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
        calendar=LocalCalendar(),
        timezone=CST,
        mapper_for=mapper_for,
        relations=RelationStore(tmp_path / "scene"),
        relation_config=RelationConfig(
            slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2
        ),
        subject=SUBJECT,
        rulers=RULERS,
    )
    return run, provider


RULERS = RhythmRulers(
    slot_minutes=15,
    merge_window_seconds=2700,
    pool_half_width=3,
    shrink_slot_to_pool=5.0,
    shrink_pool_to_all_day=5.0,
    min_peak_mass=0.10,
    recent=35,
)


@dataclass(frozen=True)
class Night:
    """一晚的读数：换代的报告（其余字段从它取）+ 这一晚队列按条映射的读数。"""

    report: NightReport
    mapped_days: tuple
    mapped: int
    unresolved: int
    model_failed: int
    remapped_days: int
    signals: tuple

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self.report, name)


async def queue_then_generate(run: SceneNightlyRun, series, now):  # type: ignore[no-untyped-def]
    """一晚 = 模型队列的各步 + 换代的语义树一步（收口删了离线一次做完的 ``SceneNightlyRun.run``）：

    1. 同步词表（含迁移牵到的重映射）——线上概念由上一代换代写、队列只重映射；测试里一晚没有"上一代"，这里一起做；
    2. 写概念（待生效，这一代并入）；
    3. 按条映射（已封口的日子全部算到了余量）；一条写不出只跳过它（那一天不盖章）；
    4. 先验对着"下一次换代之后的概念集"问、候选条件对着这一代的概念集问（提议记成这一代的日子）；
    5. ``generate``：同步 → 封口 → 并入待生效的概念 → 节律 → 折叠。

    返回 ``Night``：``generate`` 的报告，加上这一晚队列映射的读数（换代的报告里没有映射），信号把队列那几步的接在前面。"""

    from habitus.scene.occurrences.mapper import ConceptMapperError
    from habitus.scene.occurrences.model import UnresolvedReason

    signals: list[str] = []
    synced = await run.syncing.run(series, now, signals, remap=True)
    if run.authoring is not None:
        await run.author(series, signals)
    concepts = run.concepts.read_all()
    mapper = run.mapper_for(concepts)
    plan = await asyncio.to_thread(
        run.mapping.plan, series, concepts, mapper, sealed_before=series.cutoff, ready=lambda _entry: True
    )
    days: set = set()  # type: ignore[type-arg]
    mapped = unresolved = model_failed = 0
    for item in plan:
        try:
            hits = await run.mapping.map_one(item)
        except ConceptMapperError as exc:
            signals.append(f"mapping: {item.entry.day} 写不出（{exc}）")
            continue
        days.add(item.entry.day)
        mapped += 1
        unresolved += len(hits.unresolved)
        model_failed += sum(reason is UnresolvedReason.MODEL_FAILED for reason in hits.unresolved.values())
    if run.advice is not None:
        prospect, settled = await run.prospect(series, now)
        lanes = sorted({lane for name in prospect.behaviors() if (lane := prospect.lane_of(name)) is not None})
        for lane in lanes:
            await run.advice.ask_priors(prospect, lane, signals, settled)
        lines = await run.relation_stage.timelines(series, concepts)
        for lane, line in sorted(lines.items()):
            await run.relation_stage.ask_conditions(lane, line, series, concepts, signals, proposed_on=series.cutoff)
    report = await run.generate(series=series, now=now)
    return Night(
        report=report,
        mapped_days=tuple(sorted(days, reverse=True)),
        mapped=mapped,
        unresolved=unresolved,
        model_failed=model_failed,
        remapped_days=synced.remapped,
        signals=tuple(dict.fromkeys((*signals, *report.signals))),  # 同步词表两边都走：同一条信号只留一次
    )


def night(run: SceneNightlyRun, day, now):  # type: ignore[no-untyped-def]
    """跑一晚：序列按"换代那天"截（``now`` 的日子）。"""

    series = read_series(run.behavior_tree, cutoff=now.date())
    assert day < series.cutoff  # 第 N 晚映射 N 之前封口的日子
    return asyncio.run(queue_then_generate(run, series, now))


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
    # 日历情境只长数据里出现过的取值：序列里只有今晚那一天（周二、18 号，日型没数据答不出）——周二、18号两个（裁定 59 第 7 条）
    calendar = {item.name for item in calendar_concepts(now=NOW, observed=run.calendar.of_day(TONIGHT))} - {
        item.name for item in AUTHORED
    }
    assert calendar == {"周二", "18号"} and report.concepts == 3 + len(AUTHORED) + len(calendar)
    records = {record.kind_token: record for record in run.hits.read_day(TONIGHT)}
    assert {hit.concept for hit in records[SLEEP].hits} == {"晚睡"}
    assert {hit.concept for hit in records[BREAKFAST].hits} == {"起床就吃"}
    # 映射时 lane 由词表口给（``catalog.lane_of``），记进命中记录
    assert {record.lane for record in records.values()} == {"session"}
    assert "封口 1 天" in report.summary()
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
    """常态样本不够 → 那个键不给 → 引用它的「晚睡」记未决（不是"没命中"），未决原因记在命中记录上。"""

    site = Site(tmp_path, now=NOW)
    hits = ConceptHitStore(tmp_path / "scene")  # 一条历史都没有
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="就寝", lasts_minutes=30)
    run, _provider = nightly(tmp_path, hits, site)

    report = night(run, TONIGHT, NOW)
    assert report.mapped == 1 and report.unresolved == 1
    # 收口改过：原来离线一次做完的按天映射还把"样本不够"记成一条信号；按条映射（模型队列）不记，未决原因在命中记录里
    (found,) = run.hits.read_day(TONIGHT)
    assert set(found.unresolved.values()) == {UnresolvedReason.BASELINE_MISSING}


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


def three_days_of_meals_and_sleep(site: Site, hits: ConceptHitStore) -> list:  # type: ignore[type-arg]
    """连着三天各一条就寝、一碗面。其中两天与夹具里"直接写进命中盘"的合成历史重叠：那两天撤掉合成的命中与完成章（收口改过：
    原来离线一次做完的按天映射先清掉树上没有的命中再映射；按条映射不清，封口见到"盘上比树上多"按设计拒绝盖章）。"""

    days = [TONIGHT - timedelta(days=offset) for offset in (2, 1, 0)]
    for day in days:
        publish_tonight(site, day)
        if day in hits.days_done():
            hits.retain_only(day, frozenset())
    return days


def test_mapping_works_through_the_whole_backlog_without_a_nightly_budget(tmp_path) -> None:
    """映射按槽（阶段 2 第三小步）：一晚 14 天的预算去掉了——还没映射的日子这一晚都做完、都盖章，不剩；第二晚没有可映射的。
    原来这条守的是"一晚最多映射今晚那天 + 预算那么多天，没轮到的下一晚接着补"。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    days = three_days_of_meals_and_sleep(site, hits)
    run, _provider = nightly(tmp_path, hits, site)
    first = night(run, TONIGHT, NOW)
    assert first.mapped_days == (days[2], days[1], days[0])
    assert not any(signal.startswith("mapping: 还有") for signal in first.signals)
    assert set(days) <= run.hits.days_done()
    second = night(run, TONIGHT, NOW + timedelta(days=1))
    assert second.mapped_days == ()


def test_a_new_refinement_backfills_only_the_records_of_its_class(tmp_path) -> None:
    """新写了一个挂在「就寝」上的语义细分概念：已映射的日子都要回填，但只有就寝那几条问模型，早餐那几条照常续跑。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    days = three_days_of_meals_and_sleep(site, hits)
    run, provider = nightly(tmp_path, hits, site)
    night(run, TONIGHT, NOW)
    asked = provider.calls
    run.concepts.write(refinement("睡前看手机", "上床之前还在看手机", "就寝"))
    report = night(run, TONIGHT, NOW + timedelta(days=1))
    assert set(report.mapped_days) == set(days)
    prompts = provider.prompts[asked:]
    assert provider.calls - asked == 2 * len(days)  # 每天一条就寝、判两次
    assert all("行为：就寝" in prompt and "睡前看手机" in prompt for prompt in prompts)
    assert report.mapped == len(days)  # 只有就寝那几条重判；早餐那几条的口径没变、作数，不进按条映射的计划（原来按天映射时记"续跑"）


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
    assert TONIGHT in report.mapped_days and report.mapped == 1  # 只映射补发的那一条；旧的两条作数（原来按天映射时记"续跑 2"）
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


def test_rhythm_anchors_come_from_last_nights_established_calendar_relations(tmp_path) -> None:
    """节律那一行的"锚在周二"读上一晚已经成立的日历关系，只留最具体的那一族（裁定 59）。"""

    from habitus.scene.relations import RelationKey, Segment
    from habitus.scene.relations.state import LaneState, Relation, Status

    site, hits = site_with_three_bedtimes(tmp_path)
    run, _provider = nightly(tmp_path, hits, site)
    concepts = ConceptSet((*calendar_concepts(now=NOW),))
    earlier = TONIGHT - timedelta(days=1)
    held = (
        Relation(
            RelationKey("session", "周二", MEAL, Segment.SITUATION), Status.ESTABLISHED, True, established_on=earlier
        ),
        Relation(
            RelationKey("session", "工作日", MEAL, Segment.SITUATION), Status.ESTABLISHED, True, established_on=earlier
        ),
        Relation(
            RelationKey("session", "周三", SLEEP, Segment.SITUATION), Status.CANDIDATE, True, discovered_on=earlier
        ),
    )
    run.relations.write(LaneState(lane="session", night=earlier, family=3, relations=held), ())
    assert run._anchors(concepts, TONIGHT) == {
        MEAL: ("周二", "工作日")
    }  # 候选不算；周几与日型两条都留（检验周几时日型固定住了）


def test_rerunning_the_same_night_reverts_the_relations_a_split_touched_again(tmp_path) -> None:
    """词表拆出新类之后那一晚，涉及被拆类的关系退回候选（裁定 59）。同一晚重跑（同日重启、封口滞后时晚序号不变）照样退回：
    改了哪些类按上一晚关系状态记的词表版本算，不按概念集的同步进度（第一趟已推进，第二趟就看不到变化了；2026-10-18 四方评审）。"""

    from habitus.scene.relations import RelationKey, Segment
    from habitus.scene.relations.state import LaneState, Relation, Status

    site, hits = site_with_three_bedtimes(tmp_path)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    publish_tonight(site)
    earlier = TONIGHT - timedelta(days=1)
    held = (
        Relation(
            RelationKey("session", SLEEP, BREAKFAST, Segment.NEXT_DAY), Status.ESTABLISHED, True, established_on=earlier
        ),
    )
    run.relations.write(LaneState(lane="session", night=earlier, family=1, relations=held, vocabulary=1), ())
    catalog.split(BREAKFAST, BRUNCH, "早午餐")
    for _ in range(2):
        report = night(run, TONIGHT, NOW)
        stored = run.relations.read("session", report.night)
        assert stored.vocabulary == catalog.version() == 2
        assert all(item.status is not Status.ESTABLISHED for item in stored.relations if item.key == held[0].key)
        reverted = [
            item
            for item in run.relations.transitions("session")
            if item.night == report.night and item.key == held[0].key
        ]
        assert reverted and reverted[0].before is Status.ESTABLISHED and reverted[0].after is Status.CANDIDATE


def test_weekday_sets_grow_only_from_raising_relations(tmp_path) -> None:
    """周几集合只从抬高的周几关系合成：工作日天天做的事，周六、周日是压低的，不能和周一、周五并成一个集合再检验
    （2026-10-18 四方评审）。"""

    from habitus.scene.relations import RelationKey, Segment
    from habitus.scene.relations.state import LaneState, Relation, Status

    site, hits = site_with_three_bedtimes(tmp_path)
    run, _provider = nightly(tmp_path, hits, site)
    concepts = ConceptSet((*calendar_concepts(now=NOW),))
    for name in concepts:
        run.concepts.write(concepts[name])

    def held(antecedent: str, upward: bool) -> Relation:
        return Relation(
            RelationKey("session", antecedent, MEAL, Segment.SITUATION), Status.CANDIDATE, upward, discovered_on=TONIGHT
        )

    state = LaneState(
        "session", TONIGHT, 1, (held("周一", True), held("周五", True), held("周六", False), held("周日", False))
    )
    signals: list[str] = []
    run.relation_stage._grow_value_sets(run.concepts.read_all(), state, NOW, signals)
    assert "周一、周五" in run.concepts.read_all() and "周一、周五、周六、周日" not in run.concepts.read_all()
    assert "周六、周日" not in run.concepts.read_all()


def test_the_night_grows_value_sets_from_the_relations_it_just_folded(tmp_path, monkeypatch) -> None:
    """B4 折叠落盘之后，这一晚就从刚折出的关系里补取值集合（下一晚起一起检验），不等别处去调。"""

    import habitus.runtime.scene_night_relations as relations_module
    from habitus.scene.relations import RelationKey, Segment
    from habitus.scene.relations.state import LaneState, Relation, Status

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    concepts = ConceptSet((*calendar_concepts(now=NOW),))
    for name in concepts:
        run.concepts.write(concepts[name])

    def held(antecedent: str) -> Relation:
        return Relation(
            RelationKey("session", antecedent, MEAL, Segment.SITUATION), Status.CANDIDATE, True, discovered_on=TONIGHT
        )

    def folded(previous, tests, weights, *, watching):  # type: ignore[no-untyped-def]
        return LaneState("session", tests.cutoff, 1, (held("周一"), held("周五"))), ()

    monkeypatch.setattr(relations_module, "fold", folded)
    report = night(run, TONIGHT, NOW)

    assert "周一、周五" in run.concepts.read_all()
    assert any("补上日历取值集合「周一、周五」" in signal for signal in report.signals)


def test_a_relation_table_that_cannot_be_written_fails_only_that_lane(tmp_path, monkeypatch) -> None:
    """关系表写不出（超限）：这一晚照常走完、留信号，预测层读上一代；不让整晚停（26 号清单 2.5.2）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)

    def oversized(_state, _transitions):
        raise RelationStoreError("relation night exceeds its size bound")

    monkeypatch.setattr(run.relations, "write", oversized)
    report = night(run, TONIGHT, NOW)

    assert any(signal.startswith("relations: session 这一晚的关系表写不出") for signal in report.signals)
    assert "session" not in report.relations


def test_a_class_named_like_a_concept_in_use_neither_stalls_the_nights_nor_leaves_a_file(tmp_path) -> None:
    """词表合法、概念集不自洽的三种撞名：并掉的「吃饭」又长出一个「吃饭」（停用的不算撞，两个都留着）；拆出的类叫「晚睡」
    撞了模型写的细分概念；类叫「周二」撞了日历情境。后两种那份概念不写盘、报出来，每一晚照常跑完、关系表照常落盘。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    night(run, TONIGHT, NOW)

    catalog.merge(MEAL, BREAKFAST)
    night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    reborn = kind_id("又长出来的吃饭")
    catalog.active[reborn] = ("吃饭", "坐下吃一顿饭")
    late = kind_id("拆出来的晚睡")
    catalog.split(SLEEP, late, "晚睡")
    tuesday = kind_id("周二例会")
    catalog.active[tuesday] = ("周二", "周二的例会")
    for offset in (2, 3):
        report = night(run, TONIGHT + timedelta(days=offset), NOW + timedelta(days=offset))
        assert run.relations.nights("session")[-1] == (NOW + timedelta(days=offset)).date()
        refused = [signal for signal in report.signals if "没同步" in signal]
        assert len(refused) == 2
        assert any("「晚睡」" in signal and "both show as" in signal for signal in refused)
        assert any("「周二」" in signal for signal in refused)
    concepts = run.concepts.read_all()
    assert concepts[reborn].label == "吃饭" and not concepts[reborn].retired and concepts[MEAL].retired
    assert late not in concepts and tuesday not in concepts
    on_disk = {path.name for path in (tmp_path / "scene" / "concepts").glob("*.md")}
    assert f"{late}.md" not in on_disk and f"{tuesday}.md" not in on_disk


class AnswerWithAWrongName(AnswerByCandidates):
    """假模型每次都答一个不在选项里的名字：结构层两轮都过不去，报错里带着 schema 片段（比单条信号的上限长）。"""

    async def complete_async(self, request):  # type: ignore[no-untyped-def, override]
        self.calls += 1
        body = {"verdicts": [{"concept": "起床就吃（早餐）", "verdict": "yes"}]}
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )


def test_a_mapping_the_model_never_answers_leaves_its_day_unsealed_and_the_night_finishes(tmp_path) -> None:
    """模型两轮都没答成、失败信号超长：那条记未决（模型没答成）、那天不盖章，这一晚照样跑完 B4；下一晚那条再映射。
    收口改过：原来那天进离线路径自己的按天重试账（``scene-mapper``，已删）；队列里一项没成按队列的重试账退避，见验收 A1-7。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, provider = nightly(tmp_path, hits, site)
    provider.complete_async = AnswerWithAWrongName(frozenset()).complete_async.__get__(provider)  # type: ignore[method-assign]
    report = night(run, TONIGHT, NOW)
    assert report.model_failed == 1 and run.relations.nights("session") == (NOW.date(),)
    assert TONIGHT not in run.hits.days_done()
    later = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert TONIGHT in later.mapped_days and later.model_failed == 1 and TONIGHT not in run.hits.days_done()


def test_a_record_that_cannot_be_written_fails_only_its_day(tmp_path, monkeypatch) -> None:
    """映射器写不出一条记录（``ConceptMapperError``，比如常态表超了记录上限）：那天不盖章，这一晚其余照常。收口改过：
    原来那天还进离线路径的按天重试账、记一条信号（都已删）；队列里一项抛了按队列的重试账退避，见验收 A1-7。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)

    async def refuse(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ConceptMapperError("baseline_snapshot carries at most 32 entries")

    monkeypatch.setattr(ConceptMapper, "map", refuse)
    report = night(run, TONIGHT, NOW)
    assert run.relations.nights("session") == (NOW.date(),) and TONIGHT not in run.hits.days_done()
    assert report.mapped_days == ()  # 那条写不出，这一晚没有映射成的


def _yesterday_breakfast_night(tmp_path, *, both: bool):  # type: ignore[no-untyped-def]
    """「昨天吃过早餐」（派生情境，看前一天的命中）：第 10、11 天各一条早餐、第 10 天还有一条就寝；``both`` 为假时第 10 天
    那条早餐先不写。"""

    from habitus.scene.concepts.situation import SituationBasis, SituationRule
    from habitus.scene.concepts.sync import base_concept
    from tests.unit.scene.concept_fixtures import situation
    from tests.unit.scene.fixtures import Site, at

    first, second = DAY1 + timedelta(days=10), DAY1 + timedelta(days=11)
    site = Site(tmp_path, now=at(second + timedelta(days=3), 3, 0))
    hits = ConceptHitStore(tmp_path / "scene")
    early = publish(site.behavior_tree, first, "吃早饭", 8, 0, kind="早餐") if both else None
    publish(site.behavior_tree, first, "上床", 23, 0, kind="就寝")
    late = publish(site.behavior_tree, second, "吃早饭", 7, 30, kind="早餐")
    run, _provider = nightly(tmp_path, hits, site)
    run.concepts.write(base_concept(CatalogClass(BREAKFAST, "早餐", "早上的第一顿饭", "session", True), now=NOW))
    rule = SituationRule(SituationBasis.YESTERDAY, concept=BREAKFAST)
    run.concepts.write(situation("昨天吃过早餐", "前一天吃过早餐", rule=rule))
    return site, run, (first, second), (early, late)


def _yesterday_hit(run: SceneNightlyRun, day, uri) -> bool:  # type: ignore[no-untyped-def]
    record = next(item for item in run.hits.read_day(day) if item.occurrence_uri == uri)
    return "昨天吃过早餐" in {hit.concept for hit in record.situation_hits}


def test_days_backfilled_in_one_night_are_mapped_oldest_first_so_history_situations_see_them(tmp_path) -> None:
    """同一晚回填相邻两天：挑日子仍是最新的先，映射按先后——第 11 天那条早餐算「昨天吃过早餐」时第 10 天已经映射好。"""

    _site, run, (first, second), (_early, late) = _yesterday_breakfast_night(tmp_path, both=True)
    report = night(run, second, at(second + timedelta(days=1), 3, 0))
    assert set(report.mapped_days) == {first, second}
    assert _yesterday_hit(run, second, late)


def test_a_day_mapped_later_revisits_the_days_that_read_it_as_history(tmp_path) -> None:
    """第 10、11 天先映射完（那时第 10 天只有就寝，第 11 天「昨天吃过早餐」判过不成立）；之后第 10 天补进早餐、那一晚只重映射
    第 10 天——读它的第 11 天回访一遍，只就地重算情境、不问模型，「昨天吃过早餐」补上。"""

    site, run, (first, second), (_early, late) = _yesterday_breakfast_night(tmp_path, both=False)
    night(run, second, at(second + timedelta(days=1), 3, 0))
    assert not _yesterday_hit(run, second, late)
    publish(site.behavior_tree, first, "吃早饭", 8, 0, kind="早餐")
    report = night(run, second + timedelta(days=1), at(second + timedelta(days=2), 3, 0))
    assert report.mapped_days == (first,)
    assert _yesterday_hit(run, second, late)


def test_the_heavy_steps_run_off_the_event_loop(tmp_path, monkeypatch) -> None:
    """夜批跑在本地服务的事件循环上：读整支概念、同步词表、节律、时间线、检验与折叠、机会表的材料都在线程里算，
    同一个循环上的心跳每一步都不停（每一步各拖 0.3 秒，心跳最长的间隔也远小于它）。"""

    import threading
    import time

    import habitus.runtime.scene_night as night_module
    import habitus.runtime.scene_night_relations as relations_module
    import habitus.runtime.scene_night_sync as sync_module

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    on_loop: list[str] = []
    loop_thread: list[int] = []

    def slowed(name: str, real):  # type: ignore[no-untyped-def]
        def slow(*args, **kwargs):  # type: ignore[no-untyped-def]
            if threading.get_ident() == loop_thread[0]:
                on_loop.append(name)
            # 用 sleep 模拟一步重活（不占 GIL），一步 1 s：放在事件循环线程上跑（或循环同步等它），心跳至少断开 1 s。
            # 共享 CI 虚拟机偶尔整体停顿 0.26–0.38 s（windows-latest 3.12/3.13、ubuntu-latest 3.13 实测），阈值取 0.5 s，
            # 两边各留一倍余量；原先忙循环 0.3 s、阈值 0.2 s 时两者挨得太近
            time.sleep(1.0)
            return real(*args, **kwargs)

        return slow

    # 同步词表那一段（计划、覆盖）拆到了 ``scene_night_sync``（实时化阶段 2）；节律仍在 ``scene_night``
    for module, name in ((sync_module, "plan_sync"), (night_module, "rhythms_of"), (sync_module, "coverage_by_lane")):
        monkeypatch.setattr(module, name, slowed(name, getattr(module, name)))
    # 时间线与折叠在 B4 那一段（``scene_night_relations``）
    for name in ("build_timelines", "fold"):
        monkeypatch.setattr(relations_module, name, slowed(name, getattr(relations_module, name)))
    monkeypatch.setattr(run.concepts, "read_all", slowed("read_all", run.concepts.read_all))

    async def main() -> float:
        loop_thread.append(threading.get_ident())
        beats: list[float] = []

        async def heart() -> None:
            while True:
                beats.append(time.perf_counter())
                await asyncio.sleep(0.02)

        task = asyncio.create_task(heart())
        await asyncio.sleep(0.05)
        await run.generate(series=read_series(run.behavior_tree, cutoff=NOW.date()), now=NOW)  # 收口：离线 ``run`` 删了，守换代那一步
        await asyncio.sleep(0.05)
        task.cancel()
        return max(later - earlier for earlier, later in zip(beats, beats[1:], strict=False))

    longest = asyncio.run(main())
    assert on_loop == [] and longest < 0.5


def test_the_coverage_is_computed_once_a_night(tmp_path, monkeypatch) -> None:
    """各 lane 的观测覆盖只取决于序列：换代一次算一次，节律与机会表的材料都用这一份。"""

    import habitus.runtime.scene_night as night_module
    import habitus.runtime.scene_night_sync as sync_module
    import habitus.scene.concepts.rhythm as rhythm_module
    from habitus.scene.concepts.rhythm import coverage_of

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    calls: list[str] = []
    real_by_lane = night_module.coverage_by_lane

    def by_lane(series, *args, **kwargs):  # type: ignore[no-untyped-def]
        # 映射按槽之后，没有那一代节律文件的日子按"截止到那一天的序列"现算节律（另一份序列，各算各的覆盖）：只数这一晚这一份的
        if series.cutoff == TONIGHT + timedelta(days=1):
            calls.append("lanes")
        return real_by_lane(series, *args, **kwargs)

    def single(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("one lane")
        return coverage_of(*args, **kwargs)

    # 覆盖在同步词表那一段算（拆到了 ``scene_night_sync``）；``scene_night`` 里只剩同步没做成时的退路，两处都数
    monkeypatch.setattr(night_module, "coverage_by_lane", by_lane)
    monkeypatch.setattr(sync_module, "coverage_by_lane", by_lane)
    monkeypatch.setattr(rhythm_module, "coverage_of", single)
    # 收口：离线 ``run`` 删了，数换代那一步（模型队列的同步词表另算自己的一份，不在这一晚的换代里）
    report = asyncio.run(run.generate(series=read_series(run.behavior_tree, cutoff=NOW.date()), now=NOW))
    assert calls == ["lanes"] and report.materials is not None and report.materials.coverage


def test_a_proposal_book_that_cannot_be_written_or_read_does_not_stall_the_night(tmp_path, monkeypatch) -> None:
    """提议簿写不出（超了单份上限）：这一晚用盘上那份，关系表照常落盘；读不了：这条 lane 这一晚沿用上一晚，整晚照样跑完。"""

    from datetime import date

    from habitus.runtime.scene_advice import RelationAdvice
    from habitus.scene.advice.model import Proposal, ProposalBook
    from habitus.scene.advice.store import AdviceStore, AdviceStoreError

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    store = AdviceStore(tmp_path / "scene")
    gone = Proposal(antecedent="s-k999999", proposed_on=date(2026, 8, 1), why="前因已删", refused="unknown template")
    store.write_book(ProposalBook(lane="session", proposals=(gone,), asked={}))
    run.advice = RelationAdvice(store=store, prior=None, conditions=None)

    def too_big(book):  # type: ignore[no-untyped-def]
        raise AdviceStoreError("scene record exceeds its byte budget")

    monkeypatch.setattr(store, "write_book", too_big)
    report = night(run, TONIGHT, NOW)
    assert run.relations.nights("session") == (NOW.date(),)
    assert any("提议簿写不出" in signal for signal in report.signals)

    def unreadable(lane):  # type: ignore[no-untyped-def]
        raise AdviceStoreError("proposal book is corrupt")

    monkeypatch.setattr(store, "book", unreadable)
    later = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert run.relations.nights("session") == (NOW.date(),)
    assert any("提议簿读不了" in signal for signal in later.signals) and later.materials is not None


def test_a_move_on_a_day_not_yet_in_this_series_leaves_its_mapping_alone(tmp_path) -> None:
    """补跑更早的一晚时，迁移改到的那天还不在这一份序列里（没封口）：不在这一晚重映射——拿这份序列去映射它会把那天
    盘上的命中当成"树上没有"全清掉；等它封口那一晚照常映射。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    _bed, meal = publish_tonight(site)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    night(run, TONIGHT, NOW)
    before = len(run.hits.read_day(TONIGHT))
    address = BehaviorURI.parse(meal).to_address()
    BehaviorDocumentWriter(site.behavior_tree, ProcessLocalLockStore(), clock=lambda: NOW).restamp_kind_token(
        address, MEAL
    )
    catalog.move(meal, BREAKFAST, MEAL)
    report = night(run, TONIGHT - timedelta(days=1), at(TONIGHT, 3, 0))
    assert report.remapped_days == 0 and before == 2 and len(run.hits.read_day(TONIGHT)) == before
