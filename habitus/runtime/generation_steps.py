"""换代各步的做法（``runtime.generation`` 按先后调它们）：每步拿换代本身（``gen``）取它要的部件，把这一步的读数写进 ``report``；
这一代不做它抛 ``Skip``、没做成但不是异常抛 ``Degrade``，退路计数记进 ``result.fallbacks``。必须成与尽力而为由 ``Generation``
按 ``MUST`` 判，这里不判。"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import TYPE_CHECKING

from habitus.behavior.reduction import BehaviorReductionBusyError
from habitus.runtime.foresight_nightly import OpportunityReport
from habitus.runtime.generation_marks import GenerationMark
from habitus.runtime.generation_result import (
    FALLBACK_CALENDAR,
    FALLBACK_COMPACTION,
    FALLBACK_HISTORY,
    FALLBACK_MODEL,
    FALLBACK_RELATIONS,
    FALLBACK_SCENE,
    FALLBACK_SETTLEMENT,
    FALLBACK_UNMAPPED,
    FALLBACK_VOCABULARY,
    LABELS,
    LOCK_BUSY,
    MIGRATING,
    MODEL_UNREACHABLE,
    NO_MATERIALS,
    NO_RECORDS,
    RETRY,
    RETRY_UNFINISHED,
    UNREACHABLE_ERRORS,
    VOCABULARY,
    VOCABULARY_MOVED,
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
    vocabulary_summary,
)
from habitus.runtime.scene_night import NightReport
from habitus.series import EventSeries, GapKind

if TYPE_CHECKING:
    from habitus.runtime.generation import Generation

# ── 1–3：探模型、重读放假数据、重试收尾 ───────────────────────────────────


async def probe_step(gen: Generation, report: dict[str, object], result: GenerationResult) -> None:
    """先探一次模型（裁定 94 第 4 条）：调不通只记下，后面调模型的那几步照样试、各自报（模型调不通不停一天）。"""

    if gen.probe is None:
        report["probe"] = "not_wired"
        return
    try:
        await gen.probe()
    except UNREACHABLE_ERRORS:
        report["probe"] = "unreachable"
        result.fallbacks[FALLBACK_MODEL] += 1
        raise
    report["probe"] = "ok"


async def migrate_step(gen: Generation, report: dict[str, object]) -> None:
    """续做没做完的词表迁移（排在最前：迁移改的是树上的编号，结算、读序列都不能读到一半新编号一半旧编号；实时化方案第 3 节）。
    持 sweep 锁；锁被另一进程拿着是还没准备好（这一眼到此为止，用户 11-10 失败重试第 1 条）。没有没做完的迁移时是空操作。
    别的失败只报出：迁移还没做完由结算（跳过、计入"结算留到下次"）与读序列（这一代不成）各自挡下。"""

    report["branches"] = len(gen.migration_jobs)
    for job in gen.migration_jobs:
        try:
            await job()
        except BehaviorReductionBusyError as exc:
            raise NotReady(LOCK_BUSY) from exc
    report["migrating"] = gen.vocabulary()[1]


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
    """重试收尾（through = 昨天）：白天放一边的错误统一看一次（裁定 67、68）。锁被占是还没准备好：这一眼到此为止、下一眼从头做
    （用户 11-10 失败重试第 1 条；等锁的状态留在进程里，下一眼接着等）。模型调不通、抛了——这一代跳过结算（守住"那天的重试收尾
    → 那天的结算"，裁定 68 第 3 条），下一代补结。"""

    if gen.retry is None:
        raise Skip("没接重试")
    through = day - timedelta(days=1)
    report["through"] = through.isoformat()
    try:
        report["pending"] = await asyncio.to_thread(pending_by_branch, gen.retry, through)
        retried = await gen.retry.run_once(through=through)
    except BaseException:
        deferred.append(RETRY_UNFINISHED)
        raise
    report["retry"] = retry_summary(retried)
    if retried.unreachable:
        deferred.append(RETRY_UNFINISHED)
        result.unreachable.append(RETRY)
        raise Degrade(f"{MODEL_UNREACHABLE}: the model is unavailable to the retry")
    if not retried.ready:
        raise NotReady(LOCK_BUSY)


# ── 4：结算 ─────────────────────────────────────────────────────────────


async def settle_step(
    gen: Generation, report: dict[str, object], deferred: list[str], result: GenerationResult
) -> None:
    """结算（排在读序列与机会表之前：结算读的是上一代的表，找原因要在同一代读到它结出的落空）。"""

    settlement = gen.settlement
    if settlement is None:
        raise Skip("没接预测层")
    if not deferred and gen.vocabulary()[1]:
        deferred.append(MIGRATING)  # 续做迁移没做完：结算会整拍跳过，这里明确计入、报出（不再静默）
    if deferred:
        result.fallbacks[FALLBACK_SETTLEMENT] += 1
        raise Degrade(deferred[0]) if deferred[0] == MIGRATING else Skip(deferred[0])
    try:
        async with gen.ledger():
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


# ── 5：词表 ─────────────────────────────────────────────────────────────


async def vocabulary_step(gen: Generation, report: dict[str, object], day: date, result: GenerationResult) -> None:
    """各支的每晚新增、定期拆改（调模型的部分尽力；续做没做完的迁移在最前那一步）。锁被占是还没准备好：这一眼到此为止
    （用户 11-10 失败重试第 1 条）；模型调不通、抛了记下、接着下一支；新增、拆改写下的迁移有没有做完由读序列那一步挡下。"""

    branches: list[object] = []
    report["branches"] = branches
    problems: list[str] = []
    for job in gen.vocabulary_jobs:
        try:
            branches.append(vocabulary_summary(await job(day)))
        except BehaviorReductionBusyError as exc:
            raise NotReady(LOCK_BUSY) from exc
        except UNREACHABLE_ERRORS as exc:
            problems.append(f"{MODEL_UNREACHABLE}: {exc}")
            if VOCABULARY not in result.unreachable:
                result.unreachable.append(VOCABULARY)
        except Exception as exc:  # noqa: BLE001 - 一支的词表整理活没做成不挡这一代
            problems.append(f"{LABELS[VOCABULARY]}出错: {type(exc).__name__}")
    if problems:
        result.fallbacks[FALLBACK_VOCABULARY] += len(problems)
        raise Degrade("; ".join(problems))


# ── 6：读序列 ───────────────────────────────────────────────────────────


async def series_step(
    gen: Generation, report: dict[str, object], day: date, result: GenerationResult, holder: list[EventSeries]
) -> None:
    """读序列，截止到今天（触发已保证连续定稿到昨天）。词表迁移做到一半、或读的时候词表变了：这一代不成（改到一半时
    一半新编号一半旧编号），下一眼从头。"""

    if gen.unreduced_gaps is not None:
        result.unreduced_gaps = await asyncio.to_thread(gen.unreduced_gaps, day)
        report["unreduced_gaps"] = result.unreduced_gaps
    before = gen.vocabulary()
    report["vocabulary_version"] = before[0]
    if before[1]:
        raise Degrade(MIGRATING)
    # 全量扫行为树是同步 IO：下沉到线程，不卡预测拍与同一个循环上的在线请求（2026-10-18 四方评审）
    series = await asyncio.to_thread(gen.series.read, cutoff=day)
    if gen.vocabulary() != before:
        raise Degrade(VOCABULARY_MOVED)
    holder.append(series)
    result.series_cutoff, result.vocabulary_version = series.cutoff, before[0]
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


# ── 7：语义树 ───────────────────────────────────────────────────────────


async def scene_step(
    gen: Generation, report: dict[str, object], series: EventSeries, result: GenerationResult, holder: list[NightReport]
) -> None:
    """语义树各步尽力而为（``SceneNightlyRun.run`` 的 ``failures``）：没做成的走退路，时间线与材料照样交给机会表。
    前面哪一步因模型整个调不通没做成（``result.unreachable``），关系判定这一代不做、沿用上一代（实时化方案第 2 节）：
    不拿这一代没长成的词表判出一张新表——同一代不重写，判出来就一直留着。"""

    failures: list[tuple[str, BaseException]] = []
    unreachable = "、".join(LABELS[name] for name in result.unreachable)
    night = await gen.scene.run(
        series=series,
        now=gen.clock(),
        failures=failures,
        relations_held=f"{MODEL_UNREACHABLE}（{unreachable}）" if unreachable else None,
    )
    holder.append(night)
    report.update(scene_summary(night))
    held = [] if night.relations_held is None else [f"b4: 关系沿用上一代（{night.relations_held}）"]
    done = await asyncio.to_thread(gen.scene.hits.days_done)
    result.stamped_days = len(done)
    stamped = frozenset(done)
    unmapped = sum(1 for record in series.records if record.day not in stamped)
    result.fallbacks[FALLBACK_UNMAPPED] += unmapped
    report.update({"stamped_days": len(done), "unmapped_records": unmapped})
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


# ── 8–9：机会表 → 单元历史 → 压实 ───────────────────────────────────────


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
    async with gen.ledger():
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


# ── 10–11：日子预期、完成标记 ───────────────────────────────────────────


async def days_step(gen: Generation, report: dict[str, object]) -> None:
    """当天的日子预期（必须成）：这一代的表不是今天的（没写成）、词表迁移中都算没成。"""

    if gen.days is None:
        raise Skip("没接预测层")
    async with gen.ledger():
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
    async with gen.ledger():
        await asyncio.to_thread(gen.marks.write, mark)
    result.finished_at = finished
    closed = None if gen.closed_at is None else gen.closed_at(result.day - timedelta(days=1))
    if closed is not None:
        result.lag_seconds = max(0.0, (finished - closed).total_seconds())
    report.update({"finished_at": finished.isoformat(), "problems": len(problems)})
