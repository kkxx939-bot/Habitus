"""封口判据：机械推论，不含任何语义判断。

融合的一切跨窗口引用（continues / supersedes / 跨段 concurrent 等）只能指向
``recent_before(cutoff, lookback)`` 选出的上下文，截断用 ``evidence_ready_at``，其中 cutoff =
该段观测的最早 ``available_at``。因此一条判断"再也不可能被引用"当且仅当它的
``evidence_ready_at`` 同时老出了两类未来引用源的窗口：

1. **墙钟**：尚未交付的观测。依赖一条上游契约假设：交付以真实送达时刻标注 ``available_at``、
   不回填过去的时刻（观测层只强制 ``available_at >= occurred_at`` 与不超前于 ``recorded_at``，
   "不回填"**没有守卫**——已列入上游契约缺口，接入真实观测源时要么在投递口加校验、要么推翻
   本条判据）。
2. **已交付、尚未融合完成的观测**：它们将来会切成段，段的 cutoff 是这些观测的历史
   ``available_at``——只看墙钟会把还能被这些段引用的链提前封口。这一类的机械下界是
   **全部尚未被任何融合回执覆盖的观测的最早 ``available_at``**（排队中、已入队未提交、
   甚至还没被入队扫描捡走的，全都没有回执，一网打尽；不能只看最老未提交作业——补发场景下
   段的 cutoff 不随队列序单调）。

两者取最小、再回退一个 lookback，得到封口视界；``evidence_ready_at`` 早于视界的判断机械闭合。
lookback 只准从 ``behavior.fusion.config`` import——融合"还能续"与归约"已封口"必须是同一个数。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from behavior.document.link import BehaviorLinkType
from behavior.reduction.chains import BehaviorChain, ChainAssembly
from behavior.reduction.errors import BehaviorReductionError
from behavior.reduction.record import ReducibleJudgement


def seal_horizon(
    *,
    now: datetime,
    frontier_cutoff: datetime | None,
    lookback_seconds: float,
) -> datetime:
    """返回封口视界；``frontier_cutoff`` 为空表示融合队列已清空。"""

    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise BehaviorReductionError("now must be a timezone-aware datetime")
    if isinstance(lookback_seconds, bool) or not isinstance(lookback_seconds, int | float):
        raise BehaviorReductionError("lookback_seconds must be a number")
    if lookback_seconds <= 0:
        raise BehaviorReductionError("lookback_seconds must be positive")
    horizon = now.astimezone(UTC)
    if frontier_cutoff is not None:
        if not isinstance(frontier_cutoff, datetime) or frontier_cutoff.utcoffset() is None:
            raise BehaviorReductionError("frontier_cutoff must be a timezone-aware datetime")
        horizon = min(horizon, frontier_cutoff.astimezone(UTC))
    return horizon - timedelta(seconds=float(lookback_seconds))


def sealed_chain_indexes(assembly: ChainAssembly, horizon: datetime) -> tuple[int, ...]:
    """已封口且**可发布**的链下标。

    封口本身只看链上最晚判断是否出窗；可发布还要求批内引用关系闭合（见
    ``closed_under_links``），这就是"封口按依赖拓扑排"的机械实现。指向消费账本里已落盘目标
    的边不在这里出现（那些判断已不在待归约集合中）。
    """

    sealed = {
        index
        for index, chain in enumerate(assembly.chains)
        if _sealed(chain, horizon)
    }
    return closed_under_links(assembly, sealed)


def closed_under_links(assembly: ChainAssembly, candidates: set[int]) -> tuple[int, ...]:
    """把候选集剔到批内引用闭合的不动点；runner 缩批后也必须用它重闭合。

    两条闭合规则：
    1. 前向边的目标必须同批（目标未就绪则本链推迟）；
    2. **concurrent_with 对称封口**：并行边可能由早链声明、组装时翻挂到晚链——早链若先封口
       单独发布，它的判断（边的唯一载体）随链消费，下一轮晚链落盘时边已无声蒸发（评审两路
       独立抓到的接缝）。所以边的目标（更早链）也必须等宿主链同批封口，两边一起走。
    """

    kept = set(candidates)
    concurrent_hosts: dict[int, list[int]] = {}
    for index in range(len(assembly.chains)):
        for kind, target_index in assembly.cross_links_of(index):
            if kind == BehaviorLinkType.CONCURRENT_WITH.value:
                concurrent_hosts.setdefault(target_index, []).append(index)
    changed = True
    while changed:
        changed = False
        for index in tuple(kept):
            targets_ready = all(
                target in kept for _kind, target in assembly.cross_links_of(index)
            )
            hosts_ready = all(host in kept for host in concurrent_hosts.get(index, ()))
            if not (targets_ready and hosts_ready):
                kept.discard(index)
                changed = True
    return tuple(sorted(kept))


def sealed_gaps(
    gaps: Sequence[ReducibleJudgement], horizon: datetime
) -> tuple[ReducibleJudgement, ...]:
    """已封口的没读懂段：出窗前它仍可能被 supersedes 认领（后来读懂了），必须等。"""

    return tuple(
        sorted(
            (item for item in gaps if item.evidence_ready_at < horizon),
            key=lambda item: (item.evidence_ready_at, item.judgement_id),
        )
    )


def _sealed(chain: BehaviorChain, horizon: datetime) -> bool:
    return chain.newest_evidence_at < horizon


__all__ = ["closed_under_links", "seal_horizon", "sealed_chain_indexes", "sealed_gaps"]
