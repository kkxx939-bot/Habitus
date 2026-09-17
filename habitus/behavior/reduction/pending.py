"""哪些判断还没归约：判断存储里有、消费账本里没有的那些，按链组好。

归约每轮的第一步就是它；此刻场景补"未封口"那一截的读口也是它（组合根经它把链读成行）。两处共用
这一个口径，"封没封口"就只有一种答法：归约改了判据，读口跟着变，不必再抄一份。

坏记录单条隔离：一条污染不许瘫痪整轮归约（判断存储无删除，整轮失败 = 永久停摆）。隔离的记录不被
消费，每轮都会再次报出——持续可见，等人处置。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.reduction.chains import ChainAssembly, assemble_chains
from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.reduction.record import ReducibleJudgement, parse_judgement_record


@dataclass(frozen=True)
class PendingJudgements:
    """还没归约的判断组成的链，以及这一轮被隔离的坏记录（一条一句）。"""

    assembly: ChainAssembly
    quarantined: tuple[str, ...]


def pending_judgements(
    judgements: BehaviorJudgementStore,
    ledger: BehaviorReductionLedger,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> PendingJudgements:
    """存储里的判断减去账本里已消费的，解析、并链。``checkpoint`` 每 500 条调一次（长循环里续租约）。"""

    consumed = ledger.consumed_judgement_ids()
    records: list[ReducibleJudgement] = []
    quarantined: list[str] = []
    for position, raw in enumerate(judgements.list()):
        if checkpoint is not None and position % 500 == 0:
            checkpoint()
        if raw.get("judgement_id") in consumed:
            continue
        try:
            records.append(parse_judgement_record(raw))
        except BehaviorReductionError as exc:
            quarantined.append(f"judgement {raw.get('judgement_id')} quarantined: {exc}")
    return PendingJudgements(assembly=assemble_chains(tuple(records)), quarantined=tuple(quarantined))


__all__ = ["PendingJudgements", "pending_judgements"]
