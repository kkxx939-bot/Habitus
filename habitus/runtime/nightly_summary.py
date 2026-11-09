"""夜线各段原有报告的落盘摘要：只有数、日期与固定文案，进 Job 账（``runtime.nightly_jobs``）、每晚报告文件与 span 属性。

各段的报告对象（``RetryReport``、``SettlementReport``、``VocabularyJobReport``、``NightReport``、``OpportunityReport``、
``DayReport``）原样摊成 JSON 能存的字典，不另算什么；语义树与机会表的材料（概念集、时间线）不进摘要。
"""

from __future__ import annotations

import re
from datetime import date

from habitus.behavior.reduction import VocabularyJobReport
from habitus.behavior.retry import RetryReport, RetryStage
from habitus.runtime.foresight_days import DayReport
from habitus.runtime.foresight_nightly import OpportunityReport
from habitus.runtime.foresight_settlement import SettlementReport
from habitus.runtime.scene_night import NightReport

#: 只进本地每晚报告、不进 span 属性（OTLP 与本地 trace）的报告键：信号是给人看的句子，带概念名、关系两端名、迁移原因这类
#: 用户派生文字（codex R3）。
LOCAL_ONLY_KEYS = frozenset({"signals"})
#: 停下原因里"模型调不通"后面跟的是异常文字（可能带供应商返回的内容）：span 上只留这个原因码，全文在 Job 账与报告里。
#: 与 ``runtime.nightly.MODEL_UNREACHABLE`` 同一句（那边导入本模块，这里不反向导入；测试钉住两者一致）。
UNREACHABLE = "模型调不通"
_UNREACHABLE_DETAIL = re.compile(rf"{re.escape(UNREACHABLE)}: [^）]*")


def reason_code(reason: str | None) -> str:
    """span 上的原因：固定文案与异常类型名原样，"模型调不通: <异常文字>"收成"模型调不通"。"""

    return "" if reason is None else _UNREACHABLE_DETAIL.sub(UNREACHABLE, reason)[:256]


#: 信号逐条进报告，但一段最多留这么多条（报告文件有上限）。
_SIGNALS_KEPT = 200


def pending_by_branch(retry: RetryStage, through: date) -> dict[str, int]:
    """前置处理开始前各支重试堆里属于这一晚（``through`` 及以前）的项数。"""

    return {branch.name: len(branch.backlog.snapshot(through=through).ids) for branch in retry.branches}


def retry_summary(report: RetryReport) -> dict[str, object]:
    return {
        "taken": report.taken,
        "recovered": report.recovered,
        "unavailable": report.unavailable,
        "invalid": report.invalid,
        "deterministic": report.deterministic,
        "leftover": report.leftover,
        "refused": report.refused,
        "waiting": list(report.waiting),
        "reduction_unreachable": report.reduction_unreachable,
    }


def settlement_summary(report: SettlementReport, *, due: int) -> dict[str, object]:
    return {
        "due": due,
        "settled": report.settled,
        "outcomes": dict(sorted(report.outcomes.items())),
        "rulings": report.rulings,
        "omissions": report.omissions,
        "feedback": report.feedback,
        "pending_days": [day.isoformat() for day in report.pending_days],
    }


def vocabulary_summary(report: object) -> dict[str, object]:
    """一支的词表整理活；不是 ``VocabularyJobReport`` 的（测试替身）只记它是什么。"""

    if not isinstance(report, VocabularyJobReport):
        return {"report": type(report).__name__}
    return {
        "days": [day.isoformat() for day in report.days],
        "model_calls": report.model_calls,
        "pool": report.pool,
        "created": list(report.created),
        "revision_due": report.revision_due,
        "split": report.split,
        "merged": report.merged,
        "moves": report.moves,
        "signals": list(report.signals[:_SIGNALS_KEPT]),
    }


def scene_summary(report: NightReport) -> dict[str, object]:
    return {
        "night": report.night.isoformat(),
        "concepts": report.concepts,
        "mapped_days": [day.isoformat() for day in report.mapped_days],
        "mapped": report.mapped,
        "resumed": report.resumed,
        "refreshed": report.refreshed,
        "unresolved": report.unresolved,
        "model_failed": report.model_failed,
        "inconsistent": report.inconsistent,
        "backlog": report.backlog,
        "remapped_days": report.remapped_days,
        "relations": {lane: dict(sorted(counts.items())) for lane, counts in sorted(report.relations.items())},
        "transitions": report.transitions,
        "signals": list(report.signals[:_SIGNALS_KEPT]),
    }


def opportunity_summary(report: OpportunityReport) -> dict[str, object]:
    return {
        "night": report.night.isoformat(),
        "lanes": {
            lane: {
                "relations": item.relations,
                "chances": item.chances,
                "hits": item.hits,
                "kinds": item.kinds,
                "relation_night": None if item.relation_night is None else item.relation_night.isoformat(),
                "moments": item.moments,
                "compacted": item.compacted,
            }
            for lane, item in sorted(report.lanes.items())
        },
        "failed": list(report.failed),
        "signals": list(report.signals[:_SIGNALS_KEPT]),
    }


def day_summary(report: DayReport) -> dict[str, object]:
    return {
        "built": report.skipped is None,
        "formed": report.formed.formed,
        "recorded": report.formed.recorded,
        "by_source": dict(sorted(report.formed.by_source.items())),
        "skipped": report.skipped,
    }


def kept_errors(retried: object) -> str | None:
    """前置处理之后仍保持「错误·看不到」、要人看的（裁定 94 第 4 条）：我们自己处理不了的是 bug，内容安全拦下的单独列，格式不合
    夜里再试仍不合的也列出来；都没有是 None。读的是 Job 账里前置处理那一段报告的重试计数。"""

    if not isinstance(retried, dict):
        return None
    counts = {name: retried.get(name, 0) for name in ("deterministic", "refused", "invalid")}
    if not any(isinstance(value, int) and value for value in counts.values()):
        return None
    return f"处理不了 {counts['deterministic']}，内容安全 {counts['refused']}，格式不合 {counts['invalid']}"


__all__ = [
    "LOCAL_ONLY_KEYS",
    "reason_code",
    "day_summary",
    "kept_errors",
    "opportunity_summary",
    "pending_by_branch",
    "retry_summary",
    "scene_summary",
    "settlement_summary",
    "vocabulary_summary",
]
