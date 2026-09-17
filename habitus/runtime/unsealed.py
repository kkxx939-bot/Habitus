"""此刻场景里"还没封口"的那一截：把判断存储里尚未归约的链读成 ``UnsealedRow``。

行为树上只有归约发布过的 occurrence；最近一段（融合静默期 + 归约回看窗）里的判断还在判断存储
里等封口。此刻场景要把它们补上，否则"到此刻已发生"永远缺最新的几条。

这是一座桥，所以住在组合根：behavior 对 foresight 零知识（不能产 ``UnsealedRow``），foresight 不读
行为侧的任何存储（架构测试钉死）。桥上**一个判断都不发明**，三样都直接调归约自己的东西：

- 哪些判断算"没封口"、几条算一条：``reduction.pending_judgements``——归约每轮第一步用的同一个函数
  （存储里有且账本里没有；supersedes 替换、continues 并链；坏记录隔离）。
- 一条链的起点与最后看到：``BehaviorChain.head.started_at`` 与 ``BehaviorChain.last_observed_at``，
  occurrence 上写的就是这两个。
- 行为名落到 kind：词表 ``token_for``；落不到就带原名进场景、标未归类，不猜。

没读懂的判断（``behavior`` 为 None）是一段空白，按空白行给出。隔离的坏记录归约每轮都会报，这里不报第二遍。
"""

from __future__ import annotations

from datetime import UTC, datetime

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.model import BehaviorKindRegistry
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.reduction.pending import pending_judgements
from habitus.foresight import UnsealedRow


class UnsealedFromJudgements:
    """``UnsealedReader`` 的生产实现：判断存储 + 消费账本 + 词表。"""

    def __init__(
        self, judgements: BehaviorJudgementStore, ledger: BehaviorReductionLedger, kinds: BehaviorKindStore
    ) -> None:
        if not isinstance(judgements, BehaviorJudgementStore):
            raise TypeError("judgements must be a BehaviorJudgementStore")
        if not isinstance(ledger, BehaviorReductionLedger):
            raise TypeError("ledger must be a BehaviorReductionLedger")
        if not isinstance(kinds, BehaviorKindStore):
            raise TypeError("kinds must be a BehaviorKindStore")
        self.judgements = judgements
        self.ledger = ledger
        self.kinds = kinds

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
        """与 ``[since, until]`` 有交集的未封口链与空白，按开始时刻排。"""

        for label, value in (("since", since), ("until", until)):
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f"{label} must be a timezone-aware datetime")
        begin, finish = since.astimezone(UTC), until.astimezone(UTC)
        assembly = pending_judgements(self.judgements, self.ledger).assembly
        registry = self.kinds.read().registry
        rows: list[UnsealedRow] = []
        for chain in assembly.chains:
            name = str(chain.head.behavior)
            rows.append(
                UnsealedRow(
                    name=name,
                    kind_token=_token(registry, name),
                    started_at=chain.head.started_at,
                    last_observed_at=chain.last_observed_at,
                    summary=chain.head.summary,
                )
            )
        rows.extend(
            UnsealedRow(
                name=None,
                kind_token=None,
                started_at=gap.started_at,
                last_observed_at=gap.last_observed_at,
                summary=None,
            )
            for gap in assembly.gaps
        )
        inside = [row for row in rows if _instant(row.last_observed_at) >= begin and _instant(row.started_at) <= finish]
        return tuple(sorted(inside, key=lambda row: (_instant(row.started_at), row.name or "")))


def _token(registry: BehaviorKindRegistry, name: str) -> str | None:
    try:
        return registry.token_for(name)
    except (TypeError, ValueError):
        return None


def _instant(value: datetime) -> datetime:
    return value.astimezone(UTC)


__all__ = ["UnsealedFromJudgements"]
