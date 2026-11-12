"""映射按槽（实时化阶段 2 第三小步；``立项-B.md`` 第 4 节里单测守得住的那几行）。

- B-1：一条记录所在的槽过后留够余量就映射，不等那一天封口；"过了几槽"按预测拍水位算（没接按时钟）；它那一天的那一代做完了才映；
- B-4：同一条记录在三个时刻映射（槽刚过余量、当天深夜、次日封口后），提示词与命中相同；
- B-5：余量设成 0，之后才落树的记录让当天时间线变了，受影响的那一条被重判，结果与一次做完相同；
- B-6：声明常态键的细分概念，当天的记录映射时常态不含当天，实时与批量相同；
- B-7：类是最近一代之后才变的、且有细分概念要近几天同类记录：这份材料按"没给"，记 ``RECENT_MISSING``，下一代补判后有结果；
- B-8：换代时昨天有一条记录还没映射：昨天不盖章、关系照判，队列补上后下一代盖章；
- B-9：换代的语义树一步不调模型，封口有读数；
- 对抗：命中记录口径不对时换代不盖章；节律文件坏了明确报错。

B-2、B-3 是回放（报告里）；B-10 在 ``tests/unit/config/test_config_loading.py``。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from habitus.runtime.generation_marks import GenerationMark, GenerationMarks
from habitus.runtime.generation_summary import scene_summary
from habitus.runtime.generation_watermark import TickWatermark
from habitus.runtime.model_queue import MAPPING, QueueSettings
from habitus.runtime.scene_rhythms import RhythmFileError, RhythmFiles
from habitus.scene.concepts.catalog import CatalogChanges, MovedOccurrence
from habitus.scene.occurrences import ConceptHitStore
from habitus.scene.occurrences.daily import mappable_from
from habitus.scene.occurrences.document import decode, encode
from habitus.scene.occurrences.model import UnresolvedReason
from habitus.series.reader import read_series
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_model_queue import outcomes, queue_for
from tests.unit.runtime.test_scene_night import NOW, SLEEP, TONIGHT, FakeCatalog, nightly, publish_tonight
from tests.unit.scene.concept_fixtures import CODING, REWORK
from tests.unit.scene.fixtures import DAY1, Site, at, publish
from tests.unit.scene.hit_fixtures import MAPPER, record

MEAL_READY = at(TONIGHT, 8, 15)  # 07:30 那碗面最后所见 07:40：所在的槽到 07:45，再留 2 槽


def scene_at(root, site: Site, **options):  # type: ignore[no-untyped-def]
    """同一棵行为树上的一份语义树（自己的命中仓，前三天的就寝已在盘上）。"""

    hits = ConceptHitStore(root / "scene")
    for offset, (hour, minute) in enumerate(((23, 30), (23, 40), (23, 20))):
        day = DAY1 + timedelta(days=offset)
        hits.write(record(day, "睡觉", hour, minute, "就寝"))
        hits.complete_day(day, records=1, completed_at=at(day, 23, 59), mapper=MAPPER)
    scene, provider = nightly(root, hits, site, **options)
    # 词表先同步好（基础概念落盘）：线上一条记录要它那一天的那一代做完才映射，那一代的语义树一步已经同步过
    series = read_series(site.behavior_tree, cutoff=TONIGHT + timedelta(days=2))
    asyncio.run(scene.syncing.run(series, NOW, [], remap=False))
    return scene, provider


def world(tmp_path, **options):  # type: ignore[no-untyped-def]
    site = Site(tmp_path, now=NOW)
    bed, meal = publish_tonight(site)
    scene, provider = scene_at(tmp_path, site, **options)
    return site, scene, provider, bed, meal


def hit(scene, day, uri):  # type: ignore[no-untyped-def]
    return next(item for item in scene.hits.read_day(day) if item.occurrence_uri == uri)


def stripped(found):  # type: ignore[no-untyped-def]
    """一条命中记录，``mapped_at`` 抹平（什么时候映射不算内容）。"""

    return encode(replace(found, mapped_at=NOW))


# ── B-1：所在的槽过后留够余量就映射 ─────────────────────────────────────


def test_mappable_from_is_the_end_of_the_records_slot_plus_the_margin() -> None:
    assert mappable_from(at(TONIGHT, 7, 40), slot_minutes=15, margin_slots=2) == MEAL_READY
    assert mappable_from(at(TONIGHT, 7, 45), slot_minutes=15, margin_slots=2) == at(TONIGHT, 8, 30)
    assert mappable_from(at(TONIGHT, 7, 40), slot_minutes=15, margin_slots=0) == at(TONIGHT, 7, 45)
    assert mappable_from(at(TONIGHT, 23, 50), slot_minutes=15, margin_slots=2) == at(TONIGHT + timedelta(days=1), 0, 30)


def test_b_1_a_record_is_mapped_once_its_slot_and_the_margin_have_passed_without_waiting_for_the_day_to_seal(
    tmp_path,
) -> None:
    """当天 08:14：就寝那条（02:10–02:40）早过了余量，映射了；07:30 那碗面还差一分钟，不算欠着。08:15 一到就映射、问了模型。
    这一天没封口、队列也不盖章，情境留空。余量不生效（等封口才映射、或不等余量）这里会变红。"""

    site, scene, provider, bed, meal = world(tmp_path)
    clock = [MEAL_READY - timedelta(minutes=1)]
    queue = queue_for(tmp_path, scene, site, clock=lambda: clock[0], sealed=lambda: None)
    report = asyncio.run(queue.drain())
    assert [item.occurrence_uri for item in scene.hits.read_day(TONIGHT)] == [bed]
    assert report.clear and report.owed[MAPPING] == 0 and provider.calls == 0
    clock[0] = MEAL_READY
    report = asyncio.run(queue.drain())
    assert outcomes(report) == [(MAPPING, "done")] and provider.calls == 2  # 「起床就吃」问模型、判两次
    found = hit(scene, TONIGHT, meal)
    assert found.situation_hits == () and found.situations_checked == ()
    assert TONIGHT not in scene.hits.days_done()


def test_the_margin_is_counted_on_the_tick_watermark_and_the_records_generation_must_be_done(tmp_path) -> None:
    """接了预测层："过了几槽"按预测拍水位算（墙钟到了、水位没到不算）；它那一天的那一代没做完（没有完成标记）也不映射——
    历史的那部分材料要那一代冻住的。"""

    site, scene, provider, _bed, meal = world(tmp_path)
    ticked = [MEAL_READY - timedelta(minutes=15)]
    marks = GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path)
    queue = queue_for(
        tmp_path,
        scene,
        site,
        clock=lambda: at(TONIGHT, 12, 0),
        sealed=lambda: None,
        marks=marks,
        watermark=lambda: TickWatermark(ticked_at=ticked[0]),
    )
    assert asyncio.run(queue.drain()).done == ()  # 今天这一代还没有：一条都不映射
    marks.write(GenerationMark(day=TONIGHT, finished_at=at(TONIGHT, 1, 0)))
    asyncio.run(queue.drain())
    assert meal not in {item.occurrence_uri for item in scene.hits.read_day(TONIGHT)} and provider.calls == 0
    ticked[0] = MEAL_READY
    asyncio.run(queue.drain())
    assert hit(scene, TONIGHT, meal).unresolved == {} and provider.calls == 2


# ── B-4：三个时刻映射，提示词与命中相同 ─────────────────────────────────


def test_b_4_a_record_mapped_at_three_moments_gets_the_same_prompt_and_the_same_hits(tmp_path) -> None:
    """07:30 那碗面在三个时刻各映射一遍（三份各自的命中仓、同一棵行为树）：槽刚过余量（08:15，这一天只有两条记录）；当天
    23:50（之后又多了三条记录）；次日封口之后。问模型的提示词逐字相同，命中记录除 ``mapped_at`` 外逐字节相同。材料取映射
    那一刻的全天时间线、或节律取含当天的那一份，这里会变红。"""

    site = Site(tmp_path, now=NOW)
    bed, meal = publish_tonight(site)
    yes = frozenset({"吃了碗面/起床就吃"})
    first, first_model = scene_at(tmp_path / "first", site, yes=yes)
    asyncio.run(queue_for(tmp_path / "first", first, site, clock=lambda: MEAL_READY, sealed=lambda: None).drain())

    publish(site.behavior_tree, TONIGHT, "午饭", 12, 0, summary="吃了午饭", kind="早餐")
    publish(site.behavior_tree, TONIGHT, "加餐", 21, 0, summary="吃了点东西", kind="早餐")
    publish(site.behavior_tree, TONIGHT, "睡了", 23, 20, summary="上床睡觉", kind="就寝")
    late, late_model = scene_at(tmp_path / "late", site, yes=yes)
    asyncio.run(
        queue_for(tmp_path / "late", late, site, clock=lambda: at(TONIGHT, 23, 50), sealed=lambda: None).drain()
    )

    tomorrow = TONIGHT + timedelta(days=1)
    publish(site.behavior_tree, tomorrow, "吃了碗粉", 7, 10, summary="坐下吃了碗粉", kind="早餐")
    after, after_model = scene_at(tmp_path / "after", site, yes=yes)
    asyncio.run(queue_for(tmp_path / "after", after, site, clock=lambda: NOW, sealed=lambda: tomorrow).drain())

    def asked(provider):  # type: ignore[no-untyped-def]
        return sorted(prompt for prompt in provider.prompts if "行为：吃了碗面" in prompt)

    assert len(asked(first_model)) == 2 and asked(first_model) == asked(late_model) == asked(after_model)
    for uri in (bed, meal):
        records = [stripped(hit(scene, TONIGHT, uri)) for scene in (first, late, after)]
        assert records[0] == records[1] == records[2]
    assert [item.concept for item in hit(first, TONIGHT, meal).hits] == ["起床就吃"]


# ── B-5：余量设成 0 ─────────────────────────────────────────────────────


def test_b_5_with_no_margin_a_record_landing_late_gets_the_affected_one_rejudged(tmp_path) -> None:
    """余量 0：07:45 就映射了那碗面。之后才落树的一条 07:10 的记录（开始早于它的最后所见）让它的当天时间线变了——按现有的
    续跑判据重判这一条（多 2 次模型调用），重判后的命中与"等都落树了再映射"相同。"""

    site, scene, provider, _bed, meal = world(tmp_path)
    clock = [at(TONIGHT, 7, 45)]
    queue = queue_for(
        tmp_path,
        scene,
        site,
        clock=lambda: clock[0],
        sealed=lambda: None,
        settings=QueueSettings(mapping_margin_slots=0),
    )
    asyncio.run(queue.drain())
    early = hit(scene, TONIGHT, meal)
    assert provider.calls == 2
    publish(site.behavior_tree, TONIGHT, "起床", 7, 10, summary="起床洗漱", kind="早餐")
    clock[0] = at(TONIGHT, 8, 0)
    report = asyncio.run(queue.drain())
    rejudged = [item for item in report.done if item.detail.get("rejudged")]
    assert len(rejudged) == 1 and provider.calls - 2 - 2 == 2  # 新落树的那条自己 2 次 + 重判那碗面多出的 2 次
    late = hit(scene, TONIGHT, meal)
    assert late.timeline_digest != early.timeline_digest

    patient, _model = scene_at(tmp_path / "patient", site)
    asyncio.run(
        queue_for(tmp_path / "patient", patient, site, clock=lambda: at(TONIGHT, 9, 0), sealed=lambda: None).drain()
    )
    assert stripped(late) == stripped(hit(patient, TONIGHT, meal))


# ── B-6：常态不含当天 ───────────────────────────────────────────────────


def test_b_6_the_baseline_of_todays_record_leaves_today_out_and_is_the_same_live_and_in_batch(tmp_path) -> None:
    """「晚睡」声明了常态键（就寝的近期常态开始时刻）。今天 02:10 那条就寝当天映射时：常态只由前三天算（23:30），节律按
    截止到今天的序列算（不含今天这一条）；封口之后再映射（批量）用的是同一份——常态值、命中都相同。节律取映射那一刻的序列
    （含当天）时这里会变红。"""

    site, live, _provider, bed, _meal = world(tmp_path)
    seen: list[object] = []
    compute = live._rhythms

    def spied(series, concepts, coverage):  # type: ignore[no-untyped-def]
        seen.append(series.cutoff)
        return compute(series, concepts, coverage)

    live._rhythms = spied  # type: ignore[method-assign]
    asyncio.run(queue_for(tmp_path, live, site, clock=lambda: at(TONIGHT, 9, 0), sealed=lambda: None).drain())
    found = hit(live, TONIGHT, bed)
    assert [item.concept for item in found.hits] == ["晚睡"] and found.baseline_snapshot  # 比常态晚 160 分钟
    assert seen and all(cutoff <= TONIGHT for cutoff in seen)  # 节律没看今天

    batch, _model = scene_at(tmp_path / "batch", site)
    seen.clear()
    batch._rhythms = spied  # type: ignore[method-assign]
    tomorrow = TONIGHT + timedelta(days=1)
    asyncio.run(queue_for(tmp_path / "batch", batch, site, clock=lambda: NOW, sealed=lambda: tomorrow).drain())
    assert stripped(hit(batch, TONIGHT, bed)) == stripped(found)
    assert seen and all(cutoff <= TONIGHT for cutoff in seen)  # 封口后补映射：仍按截止到那一天的序列


def test_the_rhythms_a_generation_froze_are_what_later_mappings_of_that_day_read(tmp_path) -> None:
    """换代把这一代的节律落盘一份；之后映射那一天的记录读它，不再现算（文件在就不看后来的序列）。"""

    site, scene, _provider, _bed, _meal = world(tmp_path)
    scene.rhythm_files = RhythmFiles(tmp_path / "scene" / "generation" / "rhythms", artifact_root=tmp_path)
    series = read_series(site.behavior_tree, cutoff=TONIGHT)
    asyncio.run(scene.generate(series=series, now=at(TONIGHT, 1, 0)))
    frozen = scene.rhythm_files.read(TONIGHT)
    assert frozen is not None

    def no_compute(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the frozen rhythms must be read, not recomputed")

    scene._rhythms = no_compute  # type: ignore[method-assign]
    concepts = scene.concepts.read_all()
    live = read_series(site.behavior_tree, cutoff=TONIGHT + timedelta(days=1))
    assert dict(scene.rhythms_for(TONIGHT, concepts, live)) == dict(frozen)


# ── B-7：类是最近一代之后才变的，近几天同类记录按"没给" ─────────────────


def test_b_7_a_class_changed_after_the_latest_generation_waits_for_the_next_one_before_recent_records_are_given(
    tmp_path,
) -> None:
    """「返工」要近几天「修改代码」的记录。今天这条修改代码映射时，词表在最近一代之后动过这个类（把前天一条记录归了进来）：
    这份材料按"没给"，「返工」记未决 ``RECENT_MISSING``、不问模型；之后队列不反复重映射它，换代也不给这一天盖章。下一代
    做完（冻住的词表版本追上了）：补判，问了模型，有结果。"""

    code = kind_id("修改代码")
    catalog = FakeCatalog()
    catalog.active[code] = ("修改代码", "改动代码文件")
    site = Site(tmp_path, now=NOW)
    earlier = publish(
        site.behavior_tree, TONIGHT - timedelta(days=1), "改登录页", 15, 0, summary="改了登录页", kind="修改代码"
    )
    today = publish(site.behavior_tree, TONIGHT, "又改登录页", 10, 0, summary="又改了登录页", kind="修改代码")
    scene, provider = scene_at(tmp_path, site, catalog=catalog, yes=frozenset({"又改登录页/返工"}))
    for definition in (CODING, REWORK):
        scene.concepts.write(definition)
    frozen = [catalog.version()]
    scene.frozen_vocabulary = lambda: frozen[0]
    catalog.versions.append(CatalogChanges(since=0, version=0, moved=(MovedOccurrence(earlier, "待定", code),)))
    queue = queue_for(tmp_path, scene, site, clock=lambda: at(TONIGHT, 12, 0), sealed=lambda: None)

    asyncio.run(queue.drain())
    waiting = hit(scene, TONIGHT, today)
    assert waiting.unresolved == {"返工": UnresolvedReason.RECENT_MISSING} and waiting.recent_digest is None
    asked = provider.calls
    assert not [prompt for prompt in provider.prompts if "又改登录页" in prompt]
    assert asyncio.run(queue.drain()).done == ()  # 只差这份材料：不反复重映射
    series = read_series(site.behavior_tree, cutoff=TONIGHT + timedelta(days=1))
    report = asyncio.run(scene.generate(series=series, now=NOW))
    assert TONIGHT not in scene.hits.days_done() and report.unsealed_records == 1  # 还不作数：不盖章

    frozen[0] = catalog.version()  # 下一代做完了
    asyncio.run(queue.drain())
    judged = hit(scene, TONIGHT, today)
    assert provider.calls > asked and len([prompt for prompt in provider.prompts if "行为：又改登录页" in prompt]) == 2
    assert judged.unresolved == {} and judged.recent_digest is not None
    assert [item.concept for item in judged.hits] == ["返工"]
    asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1)))
    assert TONIGHT in scene.hits.days_done()


# ── B-8、B-9：换代封口 ──────────────────────────────────────────────────


def test_b_8_a_day_with_an_unmapped_record_is_not_sealed_and_is_sealed_by_the_next_generation(tmp_path) -> None:
    """换代时昨天有一条记录还没映射（07:30 那碗面）：昨天不盖章（对细分概念算看不到）、报出条数，关系照判、这一步不调模型
    （B-9）；队列补上之后的下一代：重算情境、核条数、盖章，封口有读数。情境改回队列里算、或换代不核条数就盖章，这里会变红。"""

    site, scene, provider, bed, meal = world(tmp_path)
    asyncio.run(queue_for(tmp_path, scene, site, clock=lambda: at(TONIGHT, 4, 0), sealed=lambda: None).drain())
    assert [item.occurrence_uri for item in scene.hits.read_day(TONIGHT)] == [bed]
    series = read_series(site.behavior_tree, cutoff=TONIGHT + timedelta(days=1))
    asked = provider.calls

    first = asyncio.run(scene.generate(series=series, now=NOW))
    assert TONIGHT not in scene.hits.days_done()
    assert (first.sealed_days, first.unsealed_days, first.unsealed_records) == (0, 1, 1)
    assert first.relations_held is None and first.relations and provider.calls == asked  # 关系照判；换代不调模型
    assert hit(scene, TONIGHT, bed).situations_checked == ()  # 没封口：情境还空着

    tomorrow = TONIGHT + timedelta(days=1)
    asyncio.run(queue_for(tmp_path, scene, site, clock=lambda: NOW, sealed=lambda: tomorrow).drain())
    assert TONIGHT not in scene.hits.days_done()  # 队列只映射，不盖章
    asked = provider.calls
    second = asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1)))
    assert TONIGHT in scene.hits.days_done() and provider.calls == asked
    assert (second.sealed_days, second.unsealed_days, second.seal_rewritten) == (1, 0, 2)
    assert all(hit(scene, TONIGHT, uri).situations_checked for uri in (bed, meal))  # 情境是封口时统一算的
    summary = scene_summary(second)
    assert summary["sealed_days"] == 1 and isinstance(summary["seal_seconds"], float)


def test_a_hit_record_of_another_mapper_keeps_the_generation_from_sealing_the_day(tmp_path) -> None:
    """对抗：昨天有一条命中记录的口径不是现在的口径——换代不盖章（不拿它凑数）；队列认出来重判，下一代盖章。"""

    site, scene, _provider, bed, _meal = world(tmp_path)
    tomorrow = TONIGHT + timedelta(days=1)
    queue = queue_for(tmp_path, scene, site, clock=lambda: NOW, sealed=lambda: tomorrow)
    asyncio.run(queue.drain())
    scene.hits.write(replace(hit(scene, TONIGHT, bed), mapper="some-older-mapper"))
    series = read_series(site.behavior_tree, cutoff=tomorrow)
    report = asyncio.run(scene.generate(series=series, now=NOW))
    assert TONIGHT not in scene.hits.days_done() and report.unsealed_records == 1
    redone = asyncio.run(queue.drain())
    assert [item.detail.get("rejudged") for item in redone.done if item.category == MAPPING] == [1]
    asyncio.run(scene.generate(series=series, now=NOW))
    assert TONIGHT in scene.hits.days_done() and hit(scene, TONIGHT, bed).mapper != "some-older-mapper"


def test_a_corrupt_rhythm_file_is_an_error_not_a_silent_recompute(tmp_path) -> None:
    """对抗：这一代的节律文件坏了——读它的那一边明确报错（队列记在这一类的读盘错误上），不悄悄换成现算的另一份。"""

    site, scene, _provider, _bed, _meal = world(tmp_path)
    files = RhythmFiles(tmp_path / "scene" / "generation" / "rhythms", artifact_root=tmp_path)
    scene.rhythm_files = files
    files.root.mkdir(parents=True)
    files.path(TONIGHT).write_text('{"schema": "generation_rhythms_v1", "day": "2026-08-18", "rhythms": 3}')
    with pytest.raises(RhythmFileError):
        files.read(TONIGHT)
    queue = queue_for(tmp_path, scene, site, clock=lambda: at(TONIGHT, 9, 0), sealed=lambda: None)
    with pytest.raises(RhythmFileError):
        asyncio.run(queue.drain())
    assert queue.state.errors == {MAPPING: "RhythmFileError"} and scene.hits.read_day(TONIGHT) == ()


def test_rhythm_files_round_trip_the_peaks_and_are_pruned_with_the_marks(tmp_path) -> None:
    from habitus.scene.concepts.rhythm import Rhythm, RhythmPeak

    files = RhythmFiles(tmp_path / "rhythms", artifact_root=tmp_path)
    rhythm = Rhythm(
        concept=SLEEP, peaks=(RhythmPeak(1, 1380, 1440, 0.9),), days_per_occurrence=1.0, occurrence_day_share=1.0
    )
    files.write(TONIGHT, {SLEEP: rhythm, kind_id("早餐"): Rhythm(concept=kind_id("早餐"))})
    read = files.read(TONIGHT)
    assert read is not None and set(read) == {SLEEP} and read[SLEEP].peaks == rhythm.peaks  # 只存带峰的
    assert files.read(TONIGHT - timedelta(days=1)) is None
    assert files.prune(TONIGHT + timedelta(days=400), 366) == 1 and files.read(TONIGHT) is None


def test_decoding_keeps_the_hit_documents_the_seal_rewrites_canonical(tmp_path) -> None:
    """封口写回情境走的是命中记录的规范编码：读回再编码逐字节相同。"""

    site, scene, _provider, bed, _meal = world(tmp_path)
    tomorrow = TONIGHT + timedelta(days=1)
    asyncio.run(queue_for(tmp_path, scene, site, clock=lambda: NOW, sealed=lambda: tomorrow).drain())
    asyncio.run(scene.generate(series=read_series(site.behavior_tree, cutoff=tomorrow), now=NOW))
    path = scene.hits.path_for(hit(scene, TONIGHT, bed).address)
    text = path.read_text(encoding="utf-8")
    assert encode(decode(text)) == text and isinstance(hit(scene, TONIGHT, bed).mapped_at, datetime)
