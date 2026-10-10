"""一个关系"怎么量"的共同口子：短跨度的层、长跨度的状态、带条件的拆分，都走同一套把关（语义树新方案 ``13`` ③：
"调节条件、组合前因走同一套（含前向验证）"）。

每种量法给出：独立单位数与块数、效应（实验组率、对照率）、两侧 p、最小可达 p（剔除用）、区间（第 3 道）、
异质性与去一块（第 2 道）、子集（前向验证只看某天之后、维持检验只看最近几个块）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from habitus.scene.relations import stats
from habitus.scene.relations.spans import Stratum


class Measure(Protocol):
    @property
    def units(self) -> int: ...

    @property
    def blocks(self) -> int: ...

    def effect(self) -> stats.Effect: ...

    def tails(self, *, exact_limit: int) -> tuple[float, float]: ...

    def minimum_p(self) -> float: ...

    def interval(
        self, *, level: float, rounds: int, seed: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]: ...

    def heterogeneity(self) -> float | None: ...

    def leave_one_out(self, *, upward: bool) -> bool: ...

    def subset(self, *, since: date | None = None, last_blocks: int | None = None) -> Measure: ...

    def same_event_share(self) -> float | None: ...


@dataclass(frozen=True)
class ChainMeasure:
    """短跨度：每次（合并后的）A 是一层，层内病例交叉比（``spans``、``stats``）。"""

    strata: tuple[Stratum, ...]

    @property
    def units(self) -> int:
        return len(self.strata)

    @property
    def blocks(self) -> int:
        return len({item.block for item in self.strata})

    def effect(self) -> stats.Effect:
        return stats.effect(self.strata)

    def tails(self, *, exact_limit: int) -> tuple[float, float]:
        if not self.strata:
            return 1.0, 1.0
        return stats.tails(self.strata, exact_limit=exact_limit)

    def minimum_p(self) -> float:
        return stats.minimum_p(self.strata)

    def interval(
        self, *, level: float, rounds: int, seed: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if not self.strata:
            return None, None
        return stats.interval(self.strata, level=level, rounds=rounds, seed=seed)

    def heterogeneity(self) -> float | None:
        return stats.heterogeneity(self.strata)

    def leave_one_out(self, *, upward: bool) -> bool:
        return stats.leave_one_block_out(self.strata) and (stats.effect(self.strata).difference > 0.0) == upward

    def subset(self, *, since: date | None = None, last_blocks: int | None = None) -> ChainMeasure:
        items: Sequence[Stratum] = self.strata
        if since is not None:
            items = [item for item in items if item.chain.day >= since]
        if last_blocks is not None and items:
            newest = max(item.block for item in items)
            items = [item for item in items if item.block > newest - last_blocks]
        return ChainMeasure(tuple(items))

    def same_event_share(self) -> float | None:
        positives = [item for item in self.strata if item.treated]
        return sum(item.same_event for item in positives) / len(positives) if positives else None


__all__ = ["ChainMeasure", "Measure"]
