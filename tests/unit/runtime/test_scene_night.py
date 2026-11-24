"""换代的语义树一步：同步词表 → 封口（命中记录由代码直接生成）→ 关系，在真链子上跑一遍。

词表口用一个假的 ``ClassCatalog``：在用的词与修订号由测试直接给（真实的在用状态读法在 ``test_scene_vocabulary`` 里测）。
行为树上的编号用 ``kind_id(类名)``，与概念夹具里行为概念的身份同一套。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from habitus.conditions import LocalCalendar
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.calendar_situations import calendar_concepts
from habitus.scene.concepts.catalog import CatalogWord, CatalogWords
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences import ConceptHitStore, ConceptHitStoreError
from habitus.scene.occurrences.daily import HITS_VERSION
from habitus.scene.relations import RelationConfig
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.series.reader import read_series
from habitus.temporal import Rulers
from tests.unit.foresight.scripted_model import ScriptedProvider
from tests.unit.kind_ids import kind_id
from tests.unit.scene.fixtures import DAY1, Site, at, publish
from tests.unit.scene.hit_fixtures import record

TONIGHT = DAY1 + timedelta(days=3)  # 2026-08-18，周二
NOW = at(TONIGHT + timedelta(days=1), 3, 0)
SLEEP, BREAKFAST, MEAL, BRUNCH = (kind_id(name) for name in ("就寝", "早餐", "吃饭", "早午餐"))
#: 在用的词（编号 → 名字、定义）。
CLASSES = {SLEEP: ("就寝", "上床睡觉"), BREAKFAST: ("早餐", "早上的第一顿饭"), MEAL: ("吃饭", "坐下吃一顿饭")}


@dataclass
class FakeCatalog:
    """按 ``ClassCatalog`` 协议的假词表口：``active`` 是在用的词（编号 → 名字、定义），``revision`` 是在用状态的修订号。"""

    active: dict[str, tuple[str, str]] = field(default_factory=lambda: dict(CLASSES))
    revision: int = 1

    def current(self) -> CatalogWords:
        return CatalogWords(
            self.revision,
            tuple(
                CatalogWord(word_id, name, definition, "session") for word_id, (name, definition) in self.active.items()
            ),
        )


def site_with_three_bedtimes(tmp_path) -> tuple[Site, ConceptHitStore]:
    """前三天各一条就寝，直接写进命中盘、盖好章（行为树上没有这几天：它们不在序列里，封口不碰它们）。"""

    site = Site(tmp_path, now=NOW)
    hits = ConceptHitStore(tmp_path / "scene", lane="session")
    for offset, (hour, minute) in enumerate(((23, 30), (23, 40), (23, 20))):
        day = DAY1 + timedelta(days=offset)
        hits.write(record(day, "睡觉", hour, minute, "就寝"))
        hits.complete_day(day, records=1, completed_at=at(day, 23, 59), hits_version=HITS_VERSION, digest="history")
    return site, hits


def nightly(
    tmp_path, hits: ConceptHitStore, site: Site, *, catalog: FakeCatalog | None = None
) -> tuple[SceneNightlyRun, ScriptedProvider]:
    """一支语义树；第二个返回值是一个没有剧本的假模型（语义树这一步不调模型；要先验的测试自己接 ``advice``）。"""

    provider = ScriptedProvider([])
    run = SceneNightlyRun(
        behavior_tree=site.behavior_tree,
        catalog=catalog or FakeCatalog(),
        concepts=ConceptStore(tmp_path / "scene"),
        hits=hits,
        calendar=LocalCalendar(),
        relations=RelationStore(tmp_path / "scene"),
        relation_config=RelationConfig(
            slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2
        ),
        rulers=RULERS,
    )
    return run, provider


RULERS = Rulers(
    slot_minutes=15,
    merge_window_seconds=2700,
    pool_half_width=3,
    shrink_slot_to_pool=5.0,
    shrink_pool_to_all_day=5.0,
    recent=35,
)


@dataclass(frozen=True)
class Night:
    """一晚的读数：换代的报告（其余字段从它取）+ 队列那几步的信号接在前面。"""

    report: NightReport
    signals: tuple

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self.report, name)


async def queue_then_generate(run: SceneNightlyRun, series, now):  # type: ignore[no-untyped-def]
    """一晚 = 模型队列的补问 + 换代的语义树一步：先验对着"概念集 + 同步词表将要写的"问；``generate``：同步 → 封口 → 折叠。"""

    signals: list[str] = []
    if run.advice is not None:
        plan = await run.syncing.plan(series, now)
        settled = run.concepts.read_all()
        merged = {**dict(settled.items()), **{item.identity: item for item in plan.write}}
        prospect = ConceptSet(merged[identity] for identity in sorted(merged))
        lanes = sorted({lane for name in prospect.behaviors() if (lane := prospect.lane_of(name)) is not None})
        for lane in lanes:
            await run.advice.ask_priors(prospect, lane, signals, frozenset(settled))
    report = await run.generate(series=series, words=run.syncing.catalog.current(), now=now)
    return Night(report=report, signals=tuple(dict.fromkeys((*signals, *report.signals))))


def night(run: SceneNightlyRun, day, now):  # type: ignore[no-untyped-def]
    """跑一晚：序列按"换代那天"截（``now`` 的日子）。"""

    series = read_series(run.behavior_tree, cutoff=now.date())
    assert day < series.cutoff  # 第 N 晚封 N 之前封口的日子
    return asyncio.run(queue_then_generate(run, series, now))


def publish_tonight(site: Site, day=TONIGHT) -> tuple[str, str]:
    """就寝 02:10 + 07:30 一碗面（早餐这个类）。"""

    bed = publish(site.behavior_tree, day, "就寝", 2, 10, summary="上床睡觉", kind="就寝", lasts_minutes=30)
    meal = publish(site.behavior_tree, day, "吃了碗面", 7, 30, summary="坐下吃了碗面", kind="早餐")
    return bed, meal


def test_the_steps_run_in_order_on_a_real_chain(tmp_path) -> None:
    """同步词表 → 封口：今晚那一天的两条记录各有一份命中记录（类编号、lane、那一天的日历情境），盖章；对话模型一次都不调。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    bed, meal = publish_tonight(site)
    run, provider = nightly(tmp_path, hits, site)

    report = night(run, TONIGHT, NOW)
    assert provider.calls == 0
    # 日历情境只长数据里出现过的取值：序列里只有今晚那一天（周二、18 号，日型没数据答不出）——周二、18号两个（裁定 59 第 7 条）
    calendar = {item.name for item in calendar_concepts(now=NOW, observed=run.calendar.of_day(TONIGHT))}
    assert calendar == {"周二", "18号"} and report.concepts == 3 + len(calendar)
    records = {record.occurrence_uri: record for record in run.hits.read_day(TONIGHT)}
    assert {uri: item.kind_token for uri, item in records.items()} == {bed: SLEEP, meal: BREAKFAST}
    assert {item.lane for item in records.values()} == {"session"}
    assert all({hit.concept for hit in item.situation_hits} == {"周二", "18号"} for item in records.values())
    assert TONIGHT in run.hits.days_done() and report.sealed_days == 1 and "封口 1 天" in report.summary()
    # B4：关系表这一晚落了盘（数据只有几天：全是样本不够，没有迁移）
    assert run.relations.nights("session") == (NOW.date(),) and report.transitions == 0
    assert report.relations["session"]["candidate"] == 0 and report.relations["session"]["sparse"] > 0


def test_the_first_night_generates_a_base_concept_per_class_and_records_the_synced_version(tmp_path) -> None:
    """同步词表：每个在用的词一个行为概念（身份 = 词编号，显示名 = 词名）；同步到的版本（在用状态的修订号）记下。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    catalog = FakeCatalog()
    run, _provider = nightly(tmp_path, hits, site, catalog=catalog)
    assert run.concepts.synced_version() == 0
    report = night(run, TONIGHT, NOW)
    concepts = run.concepts.read_all()
    assert {concepts[class_id].label for class_id in CLASSES} == {"就寝", "早餐", "吃饭"}
    assert concepts.base_for(SLEEP) == SLEEP and not concepts[SLEEP].retired
    assert run.concepts.synced_version() == catalog.current().version == 1
    assert sum("已同步" in signal for signal in report.signals) == 3
    # 第二夜没有变更：行为概念一个都不重写
    again = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert not any("已同步" in signal for signal in again.signals)
    # 类改了名：行为概念跟着改显示名，身份（编号）不变
    catalog.active[MEAL] = ("正餐", "坐下吃一顿饭")
    renamed = night(run, TONIGHT + timedelta(days=2), NOW + timedelta(days=2))
    assert run.concepts.read_all()[MEAL].label == "正餐" and any("「正餐」" in signal for signal in renamed.signals)


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
        prior=PriorAdvisor(client, config=PriorConfig()),
    )
    report = night(run, TONIGHT, NOW)
    first = advisor_model.calls
    assert first > 0 and first % 3 == 0 and any(signal.startswith("prior:") for signal in report.signals)
    # 第一晚盘上还没有概念：每一对都是"同一代新来的"，归到按前因那种问法（``scene.advice.pairing``）
    assert list((tmp_path / "scene" / "advice").glob("priors*/*.json"))
    night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert advisor_model.calls == first


def three_days_of_meals_and_sleep(site: Site, hits: ConceptHitStore) -> list:  # type: ignore[type-arg]
    """连着三天各一条就寝、一碗面。其中两天与夹具里"直接写进命中盘"的合成历史重叠：那两天撤掉合成的命中与完成章。"""

    days = [TONIGHT - timedelta(days=offset) for offset in (2, 1, 0)]
    for day in days:
        publish_tonight(site, day)
        if day in hits.days_done():
            hits.retain_only(day, frozenset())
    return days


def test_one_generation_seals_the_whole_backlog(tmp_path) -> None:
    """还没封好的已封口日子一次换代全部封好、盖章；第二次换代没有可封的。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    days = three_days_of_meals_and_sleep(site, hits)
    run, _provider = nightly(tmp_path, hits, site)
    first = night(run, TONIGHT, NOW)
    assert set(days) <= run.hits.days_done() and first.sealed_days == len(days)
    second = night(run, TONIGHT, NOW + timedelta(days=1))
    assert second.sealed_days == 0 and second.seal_rewritten == 0


def test_a_day_that_gets_a_late_record_after_its_stamp_is_sealed_again(tmp_path) -> None:
    """那天盖完完成章之后，归约又补发进来一条：标记的条数与序列对不上，下一代重封这一天，补发的那条有命中记录、旧的两条
    不重写（E1）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    night(run, TONIGHT, NOW)
    marker = hits.read_marker(TONIGHT)
    assert marker is not None and marker.records == 2
    publish(site.behavior_tree, TONIGHT, "又吃了点", 23, 50, summary="夜宵", kind="早餐")
    report = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert report.sealed_days == 1 and report.seal_rewritten == 1  # 只写补发的那一条；旧的两条不变
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


def test_a_day_that_cannot_be_sealed_fails_only_that_day(tmp_path, monkeypatch) -> None:
    """一天的命中记录写不出：那天不盖章、留信号，这一晚其余照常（关系表照常落盘）；下一代再封。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    real = hits.write

    def refuse(record):  # type: ignore[no-untyped-def]
        raise ValueError("concept hits record exceeds its byte budget")

    monkeypatch.setattr(hits, "write", refuse)
    report = night(run, TONIGHT, NOW)
    assert run.relations.nights("session") == (NOW.date(),) and TONIGHT not in run.hits.days_done()
    assert report.unsealed_days == 1 and any(
        signal.startswith(f"seal: {TONIGHT} 封不了口") for signal in report.signals
    )
    monkeypatch.setattr(hits, "write", real)
    later = night(run, TONIGHT + timedelta(days=1), NOW + timedelta(days=1))
    assert TONIGHT in run.hits.days_done() and later.sealed_days == 1


def test_a_day_whose_hits_on_disk_cannot_be_read_fails_the_seal_step(tmp_path) -> None:
    """读盘一侧出错不收成"下一代再封"（2026-11-14 收口修补）：盘上一份命中记录读不了，它不会自己好——封口这一步失败，
    严格跑法往上抛、尽力而为的记进 ``failures`` 让换代降级，而不是只留一条信号、那一天每一代都封不了口却算这一步做成。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    night(run, TONIGHT, NOW)
    assert TONIGHT in run.hits.days_done()
    # 盘上一份命中记录多了个字段（读不了），完成章撤掉让这一天重封
    record = next((tmp_path / "scene" / "occurrences").rglob("*.md"))
    record.write_text(
        record.read_text(encoding="utf-8").replace('"classified":true', '"classified":true,"x":1'),
        encoding="utf-8",
        newline="",
    )
    run.hits.discard_completion(TONIGHT)
    series = read_series(run.behavior_tree, cutoff=NOW.date())

    with pytest.raises(ConceptHitStoreError, match="is corrupt"):
        asyncio.run(run.generate(series=series, words=run.syncing.catalog.current(), now=NOW))

    failures: list[tuple[str, BaseException]] = []
    report = asyncio.run(run.generate(series=series, words=run.syncing.catalog.current(), now=NOW, failures=failures))
    assert [(step, type(exc)) for step, exc in failures] == [("seal", ConceptHitStoreError)]
    assert report.unsealed_days == 0 and not any("封不了口" in signal for signal in report.signals)
    assert TONIGHT not in run.hits.days_done()


def test_the_heavy_steps_run_off_the_event_loop(tmp_path, monkeypatch) -> None:
    """夜批跑在本地服务的事件循环上：读整支概念、同步词表、时间线、检验与折叠、机会表的材料都在线程里算，
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

    # 同步词表那一段（计划、覆盖）拆到了 ``scene_night_sync``（实时化阶段 2）；机会表的材料（复现）在 ``scene_night``
    for module, name in (
        (sync_module, "plan_sync"),
        (night_module, "recurrences_of"),
        (sync_module, "coverage_by_lane"),
    ):
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
        await run.generate(
            series=read_series(run.behavior_tree, cutoff=NOW.date()), words=run.syncing.catalog.current(), now=NOW
        )  # 收口：离线 ``run`` 删了，守换代那一步
        await asyncio.sleep(0.05)
        task.cancel()
        return max(later - earlier for earlier, later in zip(beats, beats[1:], strict=False))

    longest = asyncio.run(main())
    assert on_loop == [] and longest < 0.5


def test_the_coverage_is_computed_once_a_night(tmp_path, monkeypatch) -> None:
    """各 lane 的观测覆盖只取决于序列：换代一次算一次，机会表的材料用这一份。"""

    import habitus.runtime.scene_night_run as run_module
    import habitus.runtime.scene_night_sync as sync_module

    site, hits = site_with_three_bedtimes(tmp_path)
    publish_tonight(site)
    run, _provider = nightly(tmp_path, hits, site)
    calls: list[str] = []
    real_by_lane = sync_module.coverage_by_lane

    def by_lane(series, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("lanes")
        return real_by_lane(series, *args, **kwargs)

    # 覆盖在同步词表那一段算（``scene_night_sync``）；``scene_night_run`` 里只剩同步没做成时的退路，两处都数
    monkeypatch.setattr(run_module, "coverage_by_lane", by_lane)
    monkeypatch.setattr(sync_module, "coverage_by_lane", by_lane)
    report = asyncio.run(
        run.generate(
            series=read_series(run.behavior_tree, cutoff=NOW.date()), words=run.syncing.catalog.current(), now=NOW
        )
    )
    assert calls == ["lanes"] and report.materials is not None and report.materials.coverage
