"""换代各步原有报告的落盘摘要：只有数、日期与固定文案，进完成标记（``runtime.generation_marks``）、每代报告文件与 span 属性。

各段的报告对象（``RetryReport``、``SettlementReport``、``NightReport``、``OpportunityReport``、
``DayReport``）原样摊成 JSON 能存的字典，不另算什么；语义树与机会表的材料（概念集、时间线）不进摘要。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from typing import cast

from habitus.behavior.retry import RetryReport, RetryStage
from habitus.foresight.loss import LossGroup
from habitus.runtime.foresight_days import DayReport
from habitus.runtime.foresight_loss import LedgerLoss
from habitus.runtime.foresight_nightly import OpportunityReport
from habitus.runtime.foresight_settlement import SettlementReport
from habitus.runtime.scene_night import NightReport

#: 只进本地每代报告、不进 span 属性（OTLP 与本地 trace）的报告键：信号是给人看的句子，带概念名、关系两端名、迁移原因这类
#: 用户派生文字（codex R3）；loss 按行为的明细（裁定 114）以行为编号为键，摊开会让每个词成一个属性名。
LOCAL_ONLY_KEYS = frozenset({"signals", "by_kind"})
#: 失败原因里"模型调不通"后面跟的是异常文字（可能带供应商返回的内容）：span 上只留这个原因码，全文在每代报告里。
#: ``runtime.generation`` 记"模型调不通"用的就是这一句。
UNREACHABLE = "模型调不通"
_UNREACHABLE_DETAIL = re.compile(rf"{re.escape(UNREACHABLE)}: [^）]*")


def reason_code(reason: str | None) -> str:
    """span 上的原因：固定文案与异常类型名原样，"模型调不通: <异常文字>"收成"模型调不通"。"""

    return "" if reason is None else _UNREACHABLE_DETAIL.sub(UNREACHABLE, reason)[:256]


#: 信号逐条进报告，但一段最多留这么多条（报告文件有上限）。
_SIGNALS_KEPT = 200


def pending_by_branch(retry: RetryStage, through: date) -> dict[str, int]:
    """重试收尾开始前各支重试堆里属于这一代（``through`` 及以前）的项数：就近没补回、封口时要丢弃的。"""

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
        "loss": None if report.loss is None else loss_summary(report.loss),
    }


def loss_summary(found: LedgerLoss) -> dict[str, object]:
    """loss 那一节（裁定 114）：总的、按来路、说出口与没说出口、按行为（只留本地），每组带刻数与分量；旁边放尺子看不到的——
    漏报、没量到的预期与刻、结果不知道的刻。小数留 6 位。"""

    loss = found.loss
    if loss is None:
        return {"days": found.days, "failed": found.failed}

    def groups(items: Mapping[str, LossGroup]) -> dict[str, object]:
        return {name: _loss_group(item) for name, item in items.items()}

    return {
        "days": found.days,
        "omissions": found.omissions,
        "expectations": loss.expectations,
        "unmeasured_expectations": loss.unmeasured_expectations,
        "moments": loss.moments,
        "unknown_moments": loss.unknown_moments,
        "unmeasured_moments": loss.unmeasured_moments,
        "total": None if loss.total is None else _loss_group(loss.total),
        "by_source": groups(loss.by_source),
        "by_spoken": groups(loss.by_spoken),
        "by_kind": groups(loss.by_kind),
    }


def _loss_group(item: LossGroup) -> dict[str, object]:
    return {
        "moments": item.moments,
        "weight": round(item.weight, 6),
        **{
            name: {"log_loss": round(score.log_loss, 6), "brier": round(score.brier, 6)}
            for name, score in (("unit", item.unit), ("account", item.account), ("raw", item.raw))
        },
    }


def scene_summary(report: NightReport) -> dict[str, object]:
    return {
        "night": report.night.isoformat(),
        "concepts": report.concepts,
        # 封口：盖了章的日子、写了的命中记录、封不了的日子、用时
        "sealed_days": report.sealed_days,
        "seal_rewritten": report.seal_rewritten,
        "unsealed_days": report.unsealed_days,
        "seal_seconds": round(report.seal_seconds, 3),
        "relations": {lane: dict(sorted(counts.items())) for lane, counts in sorted(report.relations.items())},
        "transitions": report.transitions,
        "relations_held": report.relations_held or "",
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
        "degraded": list(report.degraded),
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
    """重试收尾之后仍保持「错误·看不到」、要人看的（裁定 94 第 4 条）：我们自己处理不了的是 bug，内容安全拦下的单独列，格式不合
    再试仍不合的也列出来；都没有是 None。读的是这一代重试收尾那一步报告的重试计数（含模型队列就近试出的）。"""

    if not isinstance(retried, dict):
        return None
    counts = {name: retried.get(name, 0) for name in ("deterministic", "refused", "invalid")}
    if not any(isinstance(value, int) and value for value in counts.values()):
        return None
    return f"处理不了 {counts['deterministic']}，内容安全 {counts['refused']}，格式不合 {counts['invalid']}"


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    """报告摊成 span 属性：嵌套键用点连，列表记条数（短的字符串列表另记一份逗号连起来的），None 不记；``LOCAL_ONLY_KEYS``
    里的键（信号文字）整段跳过。"""

    found: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LOCAL_ONLY_KEYS:
                continue  # 带用户派生文字（概念名、关系两端、迁移原因）：只进本地每代报告，不进 span（codex R3）
            found.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list | tuple):
        found[f"{prefix}.count"] = len(value)
        if value and all(isinstance(item, str) for item in value):
            found[prefix] = ",".join(cast(list[str], list(value)))[:1_000]
        elif value and all(isinstance(item, dict) for item in value):
            for index, item in enumerate(value):
                found.update(flatten(item, f"{prefix}.{index}"))
    elif value is not None:
        found[prefix] = value
    return found


__all__ = [
    "LOCAL_ONLY_KEYS",
    "UNREACHABLE",
    "flatten",
    "reason_code",
    "day_summary",
    "kept_errors",
    "loss_summary",
    "opportunity_summary",
    "pending_by_branch",
    "retry_summary",
    "scene_summary",
    "settlement_summary",
]
