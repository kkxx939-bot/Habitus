"""封口判据：机械推论，不含任何语义判断。

一条判断要等它的 ``evidence_ready_at`` 同时老出两类"还在路上"的输入的窗口才封口落树：

1. **墙钟**：尚未交付的观测。依赖一条上游契约假设：交付以真实送达时刻标注 ``available_at``、
   不回填过去的时刻（观测层只强制 ``available_at >= occurred_at``，"不回填"**没有守卫**——已列入上游
   契约缺口，接入真实观测源时要么在投递口加校验、要么推翻本条判据）。
2. **已交付、尚未融合完成的观测**：机械下界是**全部尚未被任何融合回执覆盖的观测的最早
   ``available_at``**（还没写判断的交付都没有回执，一网打尽）。

两者取最小、再回退一个 lookback，得到封口视界；``evidence_ready_at`` 早于视界的判断机械闭合。
lookback 只准从 ``behavior.fusion.config`` import，组合根喂进来的是同一个数。这个窗口原是逐帧融合"往回看多久的
先前判断"的窗口（出了窗口就不会再被后续判断指向）；两条 lane 都不产出判断间关系之后，它只剩"落树之前等多久"。

会话 lane 的记录一送到就封口（裁定 31）：一轮一条、送到时就是完整的，记录不等 lookback（``arrival_horizon``，只等
还没融合完的观测）。物理 lane 的记录仍等满 lookback。定稿日（``closure``）两条 lane 都仍看带 lookback 的视界：
只改落树的时刻，不改哪天什么时候定稿。例：19:55 送到的一轮 19:55 之后的第一轮 sweep 就落树，不再等到 20:55。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.reduction.record import ReducibleJudgement


def seal_horizon(
    *,
    now: datetime,
    frontier_cutoff: datetime | None,
    lookback_seconds: float,
) -> datetime:
    """返回封口视界；``frontier_cutoff`` 为空表示融合队列已清空。"""

    if isinstance(lookback_seconds, bool) or not isinstance(lookback_seconds, int | float):
        raise BehaviorReductionError("lookback_seconds must be a number")
    if lookback_seconds <= 0:
        raise BehaviorReductionError("lookback_seconds must be positive")
    return arrival_horizon(now=now, frontier_cutoff=frontier_cutoff) - timedelta(seconds=float(lookback_seconds))


def arrival_horizon(*, now: datetime, frontier_cutoff: datetime | None) -> datetime:
    """一送到就封口的那一支（会话 lane）用的记录视界：墙钟与还没融合完的观测取最小，不再回退 lookback。"""

    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise BehaviorReductionError("now must be a timezone-aware datetime")
    horizon = now.astimezone(UTC)
    if frontier_cutoff is not None:
        if not isinstance(frontier_cutoff, datetime) or frontier_cutoff.utcoffset() is None:
            raise BehaviorReductionError("frontier_cutoff must be a timezone-aware datetime")
        horizon = min(horizon, frontier_cutoff.astimezone(UTC))
    return horizon


def sealed_indexes(occurrences: Sequence[ReducibleJudgement], horizon: datetime) -> tuple[int, ...]:
    """已封口的可读判断在 ``occurrences`` 里的下标（升序）。"""

    return tuple(index for index, item in enumerate(occurrences) if item.evidence_ready_at < horizon)


def sealed_gaps(gaps: Sequence[ReducibleJudgement], horizon: datetime) -> tuple[ReducibleJudgement, ...]:
    """已封口的空白段，按 ``(evidence_ready_at, judgement_id)`` 排。"""

    return tuple(
        sorted(
            (item for item in gaps if item.evidence_ready_at < horizon),
            key=lambda item: (item.evidence_ready_at, item.judgement_id),
        )
    )


__all__ = ["arrival_horizon", "seal_horizon", "sealed_gaps", "sealed_indexes"]
