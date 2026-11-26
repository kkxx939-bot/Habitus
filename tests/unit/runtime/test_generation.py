"""换代（实时化方案第 3 节，阶段 1）：取代夜线的五段 Job。迁自 ``test_nightly.py`` / ``test_nightly_line.py`` 里仍须成立的不变式
（先后、只读一次序列、结算在语义树之前、重试 → 结算、本地日、同一棵树），加上换代自己的：触发要连续定稿到
昨天与预测拍水位过了余量、必须成 / 尽力而为、各个退路的计数、完成标记的三值。

实时化阶段 2（A1）起换代里没有模型调用：探模型、词表的每晚新增与定期拆改、语义树里调模型的那几段归模型队列
（``test_model_queue.py``），重试收尾只丢弃不试，关系沿用上一代的条件来自"模型队列是不是在整体退避"。这里原来守着
"换代里调模型"的那几条按新行为改了（各条的文档串里写了改了什么）。"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.behavior.retry import RetryReport
from habitus.model_client import ModelUnreachable
from habitus.runtime.foresight_days import DayReport
from habitus.runtime.foresight_forming import Formed
from habitus.runtime.foresight_nightly import LaneOpportunitySummary, OpportunityReport
from habitus.runtime.generation import Generation
from habitus.runtime.generation_marks import GenerationMark, GenerationMarkError, GenerationMarks
from habitus.runtime.generation_result import (
    ACTIVE,
    CALENDAR,
    DAYS,
    DEGRADED,
    DONE,
    FAILED,
    FALLBACK_CALENDAR,
    FALLBACK_RELATIONS,
    FALLBACK_SETTLEMENT,
    FALLBACK_UNSTAMPED,
    HABITS,
    MARK,
    NO_RECORDS,
    OPPORTUNITIES,
    RELATION_CARDS,
    RETRY,
    SCENE,
    SERIES,
    SETTLE,
    SKIPPED,
    GenerationResult,
)
from habitus.runtime.generation_vocabulary import Counted
from habitus.runtime.generation_watermark import TickWatermark
from habitus.scene.concepts.catalog import ClassCatalog
from tests.unit.runtime.test_scene_night import NOW, nightly, publish_tonight, site_with_three_bedtimes

SHANGHAI = ZoneInfo("Asia/Shanghai")
TODAY = NOW.astimezone(SHANGHAI).date()


def sealed_through(last):  # type: ignore[no-untyped-def]
    """归约说"到 ``last`` 为止都封口了"（往前连续 30 天）。"""

    return lambda: tuple(last - timedelta(days=offset) for offset in range(30))


class StubVocabulary:
    """ "数在用的词"那一步的替身：在用的词就是语义树那份假词表口给的（这里不数，数法在 ``test_kinds_vocabulary``）。"""

    def __init__(self, catalog: ClassCatalog) -> None:
        self.catalog = catalog
        self.counts = 0

    def revision(self) -> int:
        return self.catalog.current().version

    def count(self, series, today):  # type: ignore[no-untyped-def]
        self.counts += 1
        return Counted(self.catalog.current(), added=0, waiting=0, deleted=0, exited=0, candidates=0)


def assembled(tmp_path, *, clock=lambda: NOW, closed_days=None, vocabulary=None, **options):  # type: ignore[no-untyped-def]
    site, hits = site_with_three_bedtimes(tmp_path)
    scene, _provider = nightly(tmp_path, hits, site)
    return site, Generation(
        behavior_tree=site.behavior_tree,
        scene=scene,
        zone=SHANGHAI,
        closed_days=closed_days if closed_days is not None else sealed_through(TODAY - timedelta(days=1)),
        vocabulary=vocabulary if vocabulary is not None else StubVocabulary(scene.syncing.catalog),
        marks=GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path),
        clock=clock,
        **options,
    )


def assembled_with(tmp_path, site, scene, **options) -> Generation:  # type: ignore[no-untyped-def]
    """拿已经建好的那棵树与语义树装一个换代（预测层的测试用：机会表、结算由调用方给）。"""

    values = dict(
        closed_days=sealed_through(TODAY - timedelta(days=1)),
        vocabulary=StubVocabulary(scene.syncing.catalog),
        marks=GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path),
        clock=lambda: NOW,
    )
    values.update(options)
    return Generation(behavior_tree=site.behavior_tree, scene=scene, zone=SHANGHAI, **values)  # type: ignore[arg-type]


def generate(run: Generation, day: date | None = None) -> GenerationResult:
    return asyncio.run(run.run(day if day is not None else run.today()))


class StubRetry:
    """重试收尾的替身：按顺序交出 ``reports``（最后一份之后重复它）；``order`` 记每次调到的 ``through``。A1 起换代调的是
    不试的 ``close``（同步、不调模型），不再是 ``run_once``。"""

    branches = ()

    def __init__(self, *reports, order=None, label="重试") -> None:  # type: ignore[no-untyped-def]
        self.reports = list(reports) or [RetryReport()]
        self.order = order if order is not None else []
        self.label = label
        self.cleaned = 0

    def clean_leftovers(self) -> int:
        self.cleaned += 1
        return 0

    def close(self, *, through):  # type: ignore[no-untyped-def]
        report = self.reports[0] if len(self.reports) == 1 else self.reports.pop(0)
        if isinstance(report, BaseException):
            raise report
        self.order.append(self.label.format(through=through))
        return report


class StubSettlement:
    ledger = None
    opportunities = None
    lane = "session"

    def __init__(self, order=None, fails=None) -> None:  # type: ignore[no-untyped-def]
        self.order = order if order is not None else []
        self.fails = fails

    def run_once(self):  # type: ignore[no-untyped-def]
        from habitus.runtime.foresight_settlement import SettlementReport

        self.order.append("结算")
        if self.fails is not None and self.fails():
            raise OSError("ledger")
        return SettlementReport(0, {}, 0, 0, ())


def wired(tmp_path, order: list[str], *, retried=None, **options):  # type: ignore[no-untyped-def]
    """真实语义树，其余各步换成只记先后的替身（挂在构造之后，绕开类型检查）。"""

    site, run = assembled(tmp_path, **options)
    publish_tonight(site)

    class Foresight:
        def run(self, *, night, **_materials):  # type: ignore[no-untyped-def]
            order.append("机会表")
            return OpportunityReport(night)

    class Days:
        async def run_once(self):  # type: ignore[no-untyped-def]
            order.append("日子")
            return DayReport(datetime.now().astimezone(), Formed())

    scene_run = run.scene.generate

    async def scene_then(**kwargs):  # type: ignore[no-untyped-def]
        order.append("语义树")
        return await scene_run(**kwargs)

    run.scene.generate = scene_then  # type: ignore[method-assign]
    run.retry = StubRetry(*(() if retried is None else (retried,)), order=order)  # type: ignore[assignment]
    run.settlement = StubSettlement(order)  # type: ignore[assignment]
    run.foresight = Foresight()  # type: ignore[assignment]
    run.days = Days()  # type: ignore[assignment]
    return run


#: A1：探模型、词表两步不在换代里了；词表方案 2026-11-14：续做迁移那一步没有了。
ALL = ["重试", "结算", "语义树", "机会表", "日子"]


def outcomes(result: GenerationResult) -> dict[str, str]:
    return {item.step: item.outcome for item in result.steps}


def test_a_generation_runs_its_steps_in_the_ruled_order_and_writes_its_mark(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    result = generate(run)
    assert order == ALL
    assert [item.step for item in result.steps] == [
        CALENDAR,
        RETRY,
        SETTLE,
        SERIES,
        ACTIVE,
        SCENE,
        OPPORTUNITIES,
        DAYS,
        HABITS,
        RELATION_CARDS,
        MARK,
    ]
    assert result.complete and run.marks.finished(TODAY) is True
    mark = run.marks.read(TODAY)
    assert mark is not None and mark.series_cutoff == TODAY and mark.vocabulary_version == 1


def test_one_read_feeds_the_semantic_layer_and_is_done_off_the_event_loop(tmp_path, monkeypatch) -> None:
    site, run = assembled(tmp_path)
    publish_tonight(site)
    reads: list[object] = []
    threads: list[bool] = []
    original = run.series.read

    def counting(*, cutoff):  # type: ignore[no-untyped-def]
        reads.append(cutoff)
        threads.append(threading.current_thread() is threading.main_thread())
        return original(cutoff=cutoff)

    monkeypatch.setattr(run.series, "read", counting)
    result = generate(run)
    assert reads == [TODAY] and threads == [False]
    scene = result.step(SCENE)
    assert scene is not None and scene.outcome == DONE
    # 换代的封口直接生成命中记录、盖章：今晚封口的那天两条记录都有命中记录，没有"还没封好的记录"
    assert "mapped_days" not in scene.report and result.fallbacks[FALLBACK_UNSTAMPED] == 0


def test_nothing_to_build_on_an_empty_tree_still_writes_the_mark(tmp_path) -> None:
    _site, run = assembled(tmp_path)
    result = generate(run)
    assert result.nothing == NO_RECORDS and result.complete
    assert [item.step for item in result.steps if item.step in (SCENE, OPPORTUNITIES, DAYS)] == []


def test_today_is_the_subjects_local_day(tmp_path) -> None:
    _site, run = assembled(tmp_path, clock=lambda: datetime(2026, 8, 17, 16, 30, tzinfo=UTC))
    assert str(run.today()) == "2026-08-18"


def test_the_scene_run_must_read_the_same_tree(tmp_path) -> None:
    site, _run = assembled(tmp_path)
    _other_site, other = assembled(tmp_path / "other")
    with pytest.raises(ValueError, match="same behaviour tree"):
        Generation(
            behavior_tree=site.behavior_tree,
            scene=other.scene,
            zone=SHANGHAI,
            closed_days=tuple,
            vocabulary=other.vocabulary,
            marks=other.marks,
        )


def test_a_generation_waits_for_contiguous_closed_days_through_yesterday(tmp_path) -> None:
    """缺陷 4 / E6：只有定稿日从最早一天起连续到昨天才换代；较早一天还没定稿、较晚一天已定稿是不连续，报出卡住的那一天。"""

    closed: list[date] = []
    _site, run = assembled(tmp_path, closed_days=lambda: tuple(closed))
    yesterday = TODAY - timedelta(days=1)
    assert run.ready().day is None  # 一天都还没定稿
    closed.extend([yesterday - timedelta(days=3), yesterday - timedelta(days=1)])
    ready = run.ready()
    assert ready.day is None and ready.stuck == yesterday - timedelta(days=2)  # 中间那天没定稿
    closed.append(yesterday)
    assert run.ready().stuck == yesterday - timedelta(days=2)  # 昨天定稿了也不越过它
    closed.append(yesterday - timedelta(days=2))
    assert run.ready().day == TODAY
    generate(run)
    assert run.ready().day is None  # 完成标记在：同一代不再做


def test_a_generation_waits_until_the_tick_watermark_passes_the_margin(tmp_path) -> None:
    """用户 11-10「按槽留余量」：预测拍水位过了 max(今天零点, 最后一段跨过零点的「未观测」结尾) + 余量才换代；今天白天合盖
    写的空白（不跨零点）不让等待重来（用户 2026-11-16）。"""

    mark = [TickWatermark()]
    _site, run = assembled(tmp_path, watermark=lambda: mark[0], margin=timedelta(minutes=45))
    midnight = datetime.combine(TODAY, datetime.min.time(), tzinfo=SHANGHAI)
    assert run.ready().day is None  # 还没有水位
    mark[0] = TickWatermark(ticked_at=midnight + timedelta(minutes=30))
    assert run.ready().day is None
    mark[0] = TickWatermark(ticked_at=midnight + timedelta(minutes=45))
    assert run.ready().day == TODAY
    woke = midnight + timedelta(hours=1, minutes=30)
    mark[0] = TickWatermark(ticked_at=midnight + timedelta(hours=2), unwatched_until=woke, overnight_until=woke)
    assert run.ready().day is None  # 跨夜的空白刚写下：再等余量
    mark[0] = TickWatermark(
        ticked_at=midnight + timedelta(hours=2, minutes=15), unwatched_until=woke, overnight_until=woke
    )
    assert run.ready().day == TODAY
    daytime = midnight + timedelta(hours=2, minutes=10)
    mark[0] = TickWatermark(
        ticked_at=midnight + timedelta(hours=2, minutes=15), unwatched_until=daytime, overnight_until=woke
    )
    assert run.ready().day == TODAY  # 白天合盖的空白不跨零点：不再等


def test_the_calendar_is_refreshed_first_and_a_failing_refresh_is_a_fallback(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    run.refresh_calendar = lambda: order.append("日历")
    generate(run)
    assert order[:2] == ["日历", "重试"]

    def broken() -> None:
        raise TypeError("unhashable type: 'list'")

    order.clear()
    run = wired(tmp_path / "b", order)
    run.refresh_calendar = broken
    result = generate(run)
    assert outcomes(result)[CALENDAR] == DEGRADED and result.fallbacks[FALLBACK_CALENDAR] == 1
    assert result.complete and order == ALL


def test_the_retry_runs_before_the_settlement_and_an_unfinished_retry_skips_the_settlement(tmp_path) -> None:
    """裁定 68 第 3 条：重试收尾处理昨天及更早的，结算在它之后；重试收尾没完成（抛了）这一代跳过结算、记"结算留到下次"，
    其余照做（用户 11-10）。A1：重试收尾不试、不调模型，报告里的 ``unavailable`` 是"到封口还没补回、丢弃了"的项数——不再是
    "模型调不通"，不跳过结算。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    run.retry = StubRetry(RetryReport(taken=1, recovered=1), order=order, label="重试到 {through}")  # type: ignore[assignment]
    generate(run)
    assert order[:2] == [f"重试到 {TODAY - timedelta(days=1)}", "结算"]
    order.clear()
    run = wired(tmp_path / "dropped", order)
    run.retry = StubRetry(RetryReport(taken=2, unavailable=1), order=order)  # type: ignore[assignment]
    result = generate(run)
    assert "结算" in order and outcomes(result)[RETRY] == DONE and result.fallbacks[FALLBACK_SETTLEMENT] == 0
    order.clear()
    run = wired(tmp_path / "broken", order)
    run.retry = StubRetry(OSError("disk"), order=order)  # type: ignore[assignment]
    result = generate(run)
    assert "结算" not in order and result.complete
    assert outcomes(result)[RETRY] == DEGRADED and outcomes(result)[SETTLE] == SKIPPED
    assert result.fallbacks[FALLBACK_SETTLEMENT] == 1


def test_a_failing_settlement_is_left_to_the_next_generation(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    run.settlement = StubSettlement(order, fails=lambda: True)  # type: ignore[assignment]
    result = generate(run)
    assert outcomes(result)[SETTLE] == DEGRADED and result.fallbacks[FALLBACK_SETTLEMENT] == 1
    assert result.complete and order == ALL


def test_a_generation_makes_no_model_call_and_completes_while_the_model_throws(tmp_path) -> None:
    """A1（实时化方案第 4 节"换代里没有模型调用"）：模型客户端一调就抛，这一代照样写成——机会表、日子预期、完成标记都写出，
    而且一次都没调到模型（把换代里任一处模型调用放回去，``attempts`` 就不是 0）。取代原来的"探模型、每晚新增词表失败不停
    这一代"：那两步不在换代里了。"""

    order: list[str] = []
    site, hits = site_with_three_bedtimes(tmp_path)
    scene, provider = nightly(tmp_path, hits, site)
    attempts: list[str] = []

    async def exploding(request):  # type: ignore[no-untyped-def]
        attempts.append("call")
        raise RuntimeError("the model must not be called inside a generation")

    provider.complete_async = exploding  # type: ignore[method-assign]
    publish_tonight(site)
    run = assembled_with(tmp_path, site, scene, models_down=lambda: False, queue_owed=lambda: {"mapping": 1})

    class Foresight:
        def run(self, *, night, **_materials):  # type: ignore[no-untyped-def]
            order.append("机会表")
            return OpportunityReport(night)

    class Days:
        async def run_once(self):  # type: ignore[no-untyped-def]
            order.append("日子")
            return DayReport(datetime.now().astimezone(), Formed())

    run.retry = StubRetry(order=order)  # type: ignore[assignment]
    run.settlement = StubSettlement(order)  # type: ignore[assignment]
    run.foresight = Foresight()  # type: ignore[assignment]
    run.days = Days()  # type: ignore[assignment]
    result = generate(run)
    assert result.complete and run.marks.finished(TODAY) is True and attempts == []
    assert order == ["重试", "结算", "机会表", "日子"]
    assert outcomes(result)[SCENE] == DONE and run.scene.relations.nights("session") == (TODAY,)
    assert result.queue_owed == {"mapping": 1}  # 换代时模型队列各类还欠着几项记进结果


def test_an_unwritten_opportunity_table_or_day_expectation_means_no_generation_today(tmp_path) -> None:
    """E5：机会表、日子预期必须成——没成就不写完成标记（今天没有这一代）。"""

    order: list[str] = []
    run = wired(tmp_path, order)

    class Failing:
        def run(self, *, night, **_materials):  # type: ignore[no-untyped-def]
            return OpportunityReport(night, failed=("session: 机会表: ForesightError",))

    run.foresight = Failing()  # type: ignore[assignment]
    result = generate(run)
    assert result.failed == OPPORTUNITIES and run.marks.finished(TODAY) is None
    assert DAYS not in outcomes(result) and MARK not in outcomes(result)

    order.clear()
    run = wired(tmp_path / "b", order)

    class Stale:
        async def run_once(self):  # type: ignore[no-untyped-def]
            return DayReport(datetime.now().astimezone(), Formed(), "今天这一代的机会表还没有")

    run.days = Stale()  # type: ignore[assignment]
    result = generate(run)
    assert result.failed == DAYS and outcomes(result)[DAYS] == FAILED and run.marks.finished(TODAY) is None


def test_a_scene_step_failing_falls_back_and_the_table_is_still_written(tmp_path) -> None:
    """E3 / E4：语义树那一步里抛了（关系判定）——走退路，机会表照样拿到时间线写出来。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    stage = run.scene.relation_stage

    async def broken(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ModelUnreachable("model: ModelAuthenticationError: denied")

    stage.run = broken  # type: ignore[method-assign]
    result = generate(run)
    assert outcomes(result)[SCENE] == DEGRADED and "b4" in str(result.step(SCENE).reason)  # type: ignore[union-attr]
    assert result.complete and "机会表" in order


def test_completion_marks_answer_finished_in_three_values_and_refuse_a_corrupt_mark(tmp_path) -> None:
    marks = GenerationMarks(tmp_path / "generation", artifact_root=tmp_path)
    day = date(2026, 8, 18)
    assert marks.finished(day) is None  # 一个标记都没有：认不出
    marks.write(GenerationMark(day=day, finished_at=NOW))
    assert marks.finished(day) is True
    assert marks.finished(day - timedelta(days=1)) is None  # 早于第一个标记
    assert marks.finished(day + timedelta(days=1)) is False  # 晚于第一个标记而没有标记
    (tmp_path / "generation" / "reports").mkdir()  # 子目录不是标记
    assert marks.days() == (day,)
    marks.path(day + timedelta(days=2)).write_text('{"schema": "x"}', encoding="utf-8")
    with pytest.raises(GenerationMarkError):  # 问到的那一天的标记坏了：明确报错，不当"做过"（F6）
        marks.finished(day + timedelta(days=2))
    assert marks.prune(day + timedelta(days=400), 366) == 2


# ── 第一轮修复（04-修复清单 F1、F2、F4）──────────────────────────────────


def test_a_closed_today_without_yesterday_does_not_trigger(tmp_path) -> None:
    """F4（codex R3）：定稿日里只有今天、没有昨天时不换代——昨天必须已定稿，并且从最早定稿日连续到昨天。"""

    _site, run = assembled(tmp_path, closed_days=lambda: (TODAY,))
    assert run.ready().day is None
    _site, run = assembled(tmp_path / "b", closed_days=lambda: (TODAY - timedelta(days=2), TODAY))
    assert run.ready().day is None


def test_a_corrupt_mark_is_reported_not_taken_for_done(tmp_path) -> None:
    """F6、F7：完成标记坏了（截断、重复键）——``exists`` / ``finished`` 读出校验、明确报错，不吞成"做过"；换代不把它当已做。"""

    marks = GenerationMarks(tmp_path / "generation", artifact_root=tmp_path)
    day = TODAY
    marks.write(GenerationMark(day=day - timedelta(days=1), finished_at=NOW))
    marks.path(day).write_text('{"schema": "generation_mark_v1"', encoding="utf-8")
    with pytest.raises(GenerationMarkError):
        marks.exists(day)
    with pytest.raises(GenerationMarkError):
        marks.finished(day)
    good = marks.path(day - timedelta(days=1)).read_text(encoding="utf-8")
    marks.path(day).write_text(good.replace("{\n", '{\n  "day": "x",\n', 1), encoding="utf-8")
    with pytest.raises(GenerationMarkError, match="repeats"):
        marks.read(day)
    _site, run = assembled(tmp_path / "run")
    run.marks.path(TODAY).parent.mkdir(parents=True, exist_ok=True)
    run.marks.path(TODAY).write_text("{", encoding="utf-8")
    assert run.ready().day == TODAY  # 坏的标记不当"今天做过"


# ── 第二轮修复（06-第二轮修复清单 G1）────────────────────────────────────


def test_a_backing_off_model_queue_holds_the_relations_to_the_previous_generation(tmp_path) -> None:
    """G1（实时化方案第 2 节"模型调不通 → 关系沿用上一次判定的"，裁定 96）：A1 起换代自己看不到模型通不通——换代那一刻同一
    进程的模型队列正在整体退避，这一代不判关系：不写这一代的关系表（同一代不重写，判出来就一直留着），时间线照算、机会表退到
    上一代关系表，计入 ``relations_previous`` 并报出原因，日子预期照建。队列没在退避（模型只是慢）照判。"""

    def with_table_night(run, order):  # type: ignore[no-untyped-def]
        relations = run.scene.relations

        class Foresight:
            def run(self, *, night, **_materials):  # type: ignore[no-untyped-def]
                order.append("机会表")
                used = relations.latest_night("session", before=night + timedelta(days=1))
                return OpportunityReport(night, lanes={"session": LaneOpportunitySummary(0, 0, 0, 0, used)})

        run.foresight = Foresight()  # type: ignore[assignment]

    for name, down in (("ok", False), ("down", True)):
        order: list[str] = []
        run = wired(tmp_path / name, order, models_down=lambda down=down: down)
        with_table_night(run, order)
        result = generate(run)
        assert result.complete and order[-2:] == ["机会表", "日子"], name
        scene = result.step(SCENE)
        assert scene is not None
        if not down:
            assert run.scene.relations.nights("session") == (TODAY,), name
            assert result.fallbacks[FALLBACK_RELATIONS] == 0 and scene.report["relations_held"] == "", name
            continue
        assert run.scene.relations.nights("session") == (), name  # 这一代不写关系表
        assert result.fallbacks[FALLBACK_RELATIONS] == 1 and result.relation_nights == {"session": None}, name
        assert scene.outcome == DEGRADED and "b4: 关系沿用上一代" in str(scene.reason), name
        assert "模型调不通" in str(scene.report["relations_held"]), name
        assert any("关系沿用上一代" in item for item in result.problems), name
