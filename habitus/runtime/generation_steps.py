"""换代各步的做法（``runtime.generation`` 按先后调它们）：每步拿换代本身（``gen``）取它要的部件，把这一步的读数写进 ``report``；
这一代不做它抛 ``Skip``、没做成但不是异常抛 ``Degrade``，退路计数记进 ``result.fallbacks``。必须成与尽力而为由 ``Generation``
按 ``MUST`` 判，这里不判。"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import TYPE_CHECKING

from habitus.runtime.foresight_nightly import OpportunityReport
from habitus.runtime.generation_lease import LeaseBusyError
from habitus.runtime.generation_marks import GenerationMark
from habitus.runtime.generation_result import (
    FALLBACK_ACTIVE,
    FALLBACK_CALENDAR,
    FALLBACK_COMPACTION,
    FALLBACK_HISTORY,
    FALLBACK_RELATIONS,
    FALLBACK_SCENE,
    FALLBACK_SETTLEMENT,
    FALLBACK_UNSTAMPED,
    LOCK_BUSY,
    MODEL_UNREACHABLE,
    NO_MATERIALS,
    NO_RECORDS,
    QUEUE_BACKING_OFF,
    RETRY_REDUCING,
    RETRY_UNFINISHED,
    UNREACHABLE_ERRORS,
    Degrade,
    GenerationResult,
    NotReady,
    Skip,
    retry_report,
)
from habitus.runtime.generation_summary import (
    day_summary,
    kept_errors,
    opportunity_summary,
    pending_by_branch,
    retry_summary,
    scene_summary,
    settlement_summary,
)
from habitus.runtime.generation_vocabulary import Counted
from habitus.runtime.scene_night import NightReport
from habitus.series import EventSeries, GapKind

if TYPE_CHECKING:
    from habitus.runtime.generation import Generation

# ── 1–2：重读放假数据、重试收尾 ─────────────────────────────────────


async def calendar_step(gen: Generation, report: dict[str, object], result: GenerationResult) -> None:
    """重读放假数据。读不了的年份日历自己跳过；再有别的失败照用上一次读到的日历。"""

    if gen.refresh_calendar is None:
        raise Skip("没接放假数据")
    try:
        await asyncio.to_thread(gen.refresh_calendar)
    except Exception:
        result.fallbacks[FALLBACK_CALENDAR] += 1
        raise
    report["refreshed"] = True


async def retry_step(
    gen: Generation, report: dict[str, object], day: date, deferred: list[str], result: GenerationResult
) -> None:
    """重试收尾（through = 昨天，裁定 67、68）：不试——能重试的由模型队列就近试过了；这里只把"重试到哪一天"推到昨天，那一天
    及以前还在堆里的丢弃、写到一半的撤回、按类计数（``RetryStage.close``，不调模型、不跑归约）。抛了、或就近重试写回的判断
    还在等归约（重试收尾没动）——这一代跳过结算（守住"那天的重试收尾 → 那天的结算"，裁定 68 第 3 条），下一代补结。"""

    if gen.retry is None:
        raise Skip("没接重试")
    through = day - timedelta(days=1)
    report["through"] = through.isoformat()
    try:
        report["pending"] = await asyncio.to_thread(pending_by_branch, gen.retry, through)
        closed = await asyncio.to_thread(gen.retry.close, through=through)
    except BaseException:
        deferred.append(RETRY_UNFINISHED)
        raise
    report["retry"] = retry_summary(closed)
    if not closed.ready:
        # 就近重试写回的判断还在等归约（归约正做到一半、或它的锁被占）：重试收尾没动那几天的项。现在结算，那条判断可能
        # 落进已经结算的日子（裁定 68 第 3 条）——这一代的结算留到下次（阶段 1 已有的退路），其余各步照做、照写完成标记：
        # 归约里要调模型，换代不等它
        deferred.append(RETRY_UNFINISHED)
        raise Degrade(RETRY_REDUCING)


# ── 3：结算 ─────────────────────────────────────────────────────────────


async def settle_step(
    gen: Generation, report: dict[str, object], deferred: list[str], result: GenerationResult
) -> None:
    """结算（排在读序列与机会表之前：结算读的是上一代的表，找原因要在同一代读到它结出的落空）。"""

    settlement = gen.settlement
    if settlement is None:
        raise Skip("没接预测层")
    if deferred:
        result.fallbacks[FALLBACK_SETTLEMENT] += 1
        raise Skip(deferred[0])
    try:
        table = settlement.opportunities.latest(settlement.lane) if settlement.opportunities else None
        # 结算读到的机会表是哪一晚（必须边"结算 → 写新机会表"：它早于这一代要写的那一晚）
        report["opportunity_night"] = None if table is None else table.night.isoformat()
        outcome = await asyncio.to_thread(settlement.run_once)
        still = await asyncio.to_thread(
            lambda: sum(len(settlement.ledger.unsettled_expectations(day)) for day in outcome.pending_days)
        )
    except BaseException:
        result.fallbacks[FALLBACK_SETTLEMENT] += 1
        raise
    report.update(settlement_summary(outcome, due=outcome.settled + still))


# ── 4：读序列 ───────────────────────────────────────────────────────────


async def series_step(
    gen: Generation, report: dict[str, object], day: date, result: GenerationResult, holder: list[EventSeries]
) -> None:
    """读序列，截止到今天（触发已保证连续定稿到昨天）。树上的编号落树之后不改（词表没有迁移），读的时候不用看词表。"""

    if gen.unreduced_gaps is not None:
        result.unreduced_gaps = await asyncio.to_thread(gen.unreduced_gaps, day)
        report["unreduced_gaps"] = result.unreduced_gaps
    # 全量扫行为树是同步 IO：下沉到线程，不卡预测拍与同一个循环上的在线请求（2026-10-18 四方评审）
    series = await asyncio.to_thread(gen.series.read, cutoff=day)
    holder.append(series)
    result.series_cutoff = series.cutoff
    yesterday = day - timedelta(days=1)
    result.unwatched = [
        (gap.started_at, gap.ended_at)
        for gap in series.gaps
        if gap.kind is GapKind.UNOBSERVED and gap.started_at.date() == yesterday
    ]
    report.update(
        {
            "series_cutoff": series.cutoff.isoformat(),
            "records": len(series.records),
            "unwatched_spans": len(result.unwatched),
            "unwatched_minutes": round(sum((end - start).total_seconds() for start, end in result.unwatched) / 60, 1),
        }
    )
    if not series.records:
        # 只有观测空白、一条记录都没有（上游刚接入）：没有可量的东西，这一代照样算做完
        result.nothing = NO_RECORDS
        report["skipped"] = NO_RECORDS


# ── 5：数在用的词 ─────────────────────────────────────────────────────


async def active_step(
    gen: Generation, report: dict[str, object], series: EventSeries, day: date, result: GenerationResult,
    holder: list[Counted],
) -> None:  # fmt: skip
    """数在用的词（词表方案 2026-11-14 第 5、7 节）：按这一代的序列数一遍、在用状态落盘，数出来的集合交给这一代的语义树。
    纯计数、不调模型。没做成走退路：语义树按盘上已有的在用状态（在用只增不减，下一代重数补上）。"""

    try:
        counted = await asyncio.to_thread(gen.vocabulary.count, series, day)
    except BaseException:
        result.fallbacks[FALLBACK_ACTIVE] += 1
        raise
    holder.append(counted)
    result.vocabulary_version = counted.words.version
    result.active_words, result.waiting_words = len(counted.words.words), counted.waiting
    report.update(
        {
            "vocabulary_version": counted.words.version,
            "active_words": len(counted.words.words),
            "added_words": counted.added,
            "waiting_words": counted.waiting,
        }
    )


# ── 6：语义树 ───────────────────────────────────────────────────────────


async def scene_step(
    gen: Generation,
    report: dict[str, object],
    series: EventSeries,
    counted: list[Counted],
    result: GenerationResult,
    holder: list[NightReport],
) -> None:
    """语义树那一步（``SceneNightlyRun.generate``，没有模型调用）尽力而为：同步词表或关系判定没做成的走退路，时间线与材料
    照样交给机会表。换代那一刻模型队列正在整体退避（模型整个调不通），关系判定这一代不做、沿用上一代（实时化方案第 2 节，
    裁定 96）：不拿这一代没长成的词表与先验判出一张新表——同一代不重写，判出来就一直留着。模型只是慢（没在退避）时
    照判，缺的先验权重按 1。"""

    failures: list[tuple[str, BaseException]] = []
    down = gen.models_down is not None and gen.models_down()
    # 这一代在用的词：数出来的那一份；数没做成就按盘上已有的在用状态
    words = counted[0].words if counted else await asyncio.to_thread(gen.scene.syncing.catalog.current)
    if not counted:
        result.vocabulary_version, result.active_words = words.version, len(words.words)
    try:
        # 整步持语义树写锁：模型队列的落盘等这一步做完，这里读到的语义树是一份停住的样子
        async with gen.scene_writes():
            night = await gen.scene.generate(
                series=series,
                words=words,
                now=gen.clock(),
                failures=failures,
                relations_held=f"{MODEL_UNREACHABLE}（{QUEUE_BACKING_OFF}）" if down else None,
            )
            done = await asyncio.to_thread(gen.scene.hits.days_done)
    except LeaseBusyError as exc:
        raise NotReady(LOCK_BUSY) from exc  # 另一个进程的队列正写着、等了一个租期还没放：下一槽再看
    holder.append(night)
    report.update(scene_summary(night))
    held = [] if night.relations_held is None else [f"b4: 关系沿用上一代（{night.relations_held}）"]
    result.stamped_days = len(done)
    stamped = frozenset(done)
    unstamped = sum(1 for record in series.records if record.day not in stamped)
    result.fallbacks[FALLBACK_UNSTAMPED] += unstamped
    report.update({"stamped_days": len(done), "unstamped_records": unstamped})
    if failures:
        result.fallbacks[FALLBACK_SCENE] += len(failures)
        report["failed_steps"] = [step for step, _exc in failures]
    if failures or held:
        raise Degrade(
            "; ".join(
                [
                    f"{step}: {MODEL_UNREACHABLE}"
                    if isinstance(exc, UNREACHABLE_ERRORS)
                    else f"{step}: {type(exc).__name__}"
                    for step, exc in failures
                ]
                + held
            )
        )


# ── 7–8：机会表 → 单元历史 → 压实 ───────────────────────────────────────


async def opportunities_step(
    gen: Generation,
    report: dict[str, object],
    series: EventSeries,
    scene: list[NightReport],
    result: GenerationResult,
) -> None:
    """机会表（必须成）：写不出就是今天没有这一代。单元历史、压实没做成是退路（表照用、压实跳过），只计数。"""

    foresight = gen.foresight
    if foresight is None:
        raise Skip("没接预测层")
    materials = scene[0].materials if scene else None
    if materials is None:
        raise Degrade(NO_MATERIALS)
    opportunities: OpportunityReport = await asyncio.to_thread(
        foresight.run,
        night=series.cutoff,
        concepts=materials.concepts,
        timelines=materials.timelines,
        recurrences=materials.recurrences,
        coverage=materials.coverage,
    )
    report.update(opportunity_summary(opportunities))
    result.relation_nights = {lane: item.relation_night for lane, item in opportunities.lanes.items()}
    result.fallbacks[FALLBACK_RELATIONS] += sum(
        1 for item in opportunities.lanes.values() if item.relation_night != series.cutoff
    )
    result.fallbacks[FALLBACK_HISTORY] += opportunities.history_skipped
    result.fallbacks[FALLBACK_COMPACTION] += opportunities.compaction_failed
    if opportunities.failed:
        raise Degrade(f"机会表出错: {'; '.join(opportunities.failed)}")


# ── 9–10：日子预期、完成标记 ───────────────────────────────────────────


async def days_step(gen: Generation, report: dict[str, object]) -> None:
    """当天的日子预期（必须成）：这一代的表不是今天的（没写成）算没成。"""

    if gen.days is None:
        raise Skip("没接预测层")
    built = await gen.days.run_once()
    report.update(day_summary(built))
    if built.skipped is not None:
        raise Degrade(built.skipped)


async def mark_step(gen: Generation, report: dict[str, object], result: GenerationResult) -> None:
    """写"这一天换代完成"的标记（必须成）：它在，这一天就不再做；结算认"那天有没有这一代"也读它。"""

    problems = [f"错误·看不到：{kept}"] if (kept := kept_errors(retry_report(result))) is not None else []
    problems.extend(f"{item.step}: {item.reason}" for item in result.degraded if item.reason)
    result.problems = problems
    finished = gen.clock()
    mark = GenerationMark(
        day=result.day,
        finished_at=finished,
        series_cutoff=result.series_cutoff,
        vocabulary_version=result.vocabulary_version,
        relation_nights=dict(result.relation_nights),
        stamped_days=result.stamped_days,
        steps={item.step: {"outcome": item.outcome, "reason": item.reason} for item in result.steps},
        fallbacks={kind: count for kind, count in result.fallbacks.items() if count},
        problems=tuple(item[:500] for item in problems),
    )
    await asyncio.to_thread(gen.marks.write, mark)
    result.finished_at = finished
    closed = None if gen.closed_at is None else gen.closed_at(result.day - timedelta(days=1))
    if closed is not None:
        result.lag_seconds = max(0.0, (finished - closed).total_seconds())
    report.update({"finished_at": finished.isoformat(), "problems": len(problems)})
