"""一个关系"怎么量"的共同口子：短跨度的层、长跨度的状态、带条件的拆分，都走同一套把关（语义树新方案 ``13`` ③：
"调节条件、组合前因走同一套（含前向验证）"）。

每种量法给出：独立单位数与块数、效应（实验组率、对照率）、两侧 p、最小可达 p（剔除用）、区间（第 3 道）、
异质性与去一块（第 2 道）、子集（前向验证与维持检验都只看某天之后的新数据）。
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


@dataclass(frozen=True)
class PairedMeasure:
    """同一条链的两道对照各量一遍（``spans.Pairing``：同一天别的事做完的时刻；别的周同周几同钟点、当天没做 A），
    **两道都显著才算**：两侧 p 各取两道里大的，最小可达 p 取大的（任何一道给不出对照，这条关系就样本不够），
    效应与区间取两道里效应弱的那道（是它卡住了结论），去一块两道都要稳，异质性取更不稳的那道。
    单位数与块数是 A 的次数——两道量的是同一批 A。"""

    first: ChainMeasure
    second: ChainMeasure

    @property
    def units(self) -> int:
        return self.first.units

    @property
    def blocks(self) -> int:
        return self.first.blocks

    @property
    def parts(self) -> tuple[ChainMeasure, ChainMeasure]:
        return self.first, self.second

    def weaker(self) -> ChainMeasure:
        """效应（实验组率减对照率）小的那道。"""

        return min(self.parts, key=lambda part: part.effect().difference)

    def effect(self) -> stats.Effect:
        return self.weaker().effect()

    def tails(self, *, exact_limit: int) -> tuple[float, float]:
        first, second = (part.tails(exact_limit=exact_limit) for part in self.parts)
        return max(first[0], second[0]), max(first[1], second[1])

    def minimum_p(self) -> float:
        return max(part.minimum_p() for part in self.parts)

    def interval(
        self, *, level: float, rounds: int, seed: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        return self.weaker().interval(level=level, rounds=rounds, seed=seed)

    def heterogeneity(self) -> float | None:
        found = [value for value in (part.heterogeneity() for part in self.parts) if value is not None]
        return min(found) if found else None

    def leave_one_out(self, *, upward: bool) -> bool:
        return all(part.leave_one_out(upward=upward) for part in self.parts)

    def subset(self, *, since: date | None = None, last_blocks: int | None = None) -> PairedMeasure:
        return PairedMeasure(
            first=self.first.subset(since=since, last_blocks=last_blocks),
            second=self.second.subset(since=since, last_blocks=last_blocks),
        )

    def same_event_share(self) -> float | None:
        return self.first.same_event_share()


__all__ = ["ChainMeasure", "Measure", "PairedMeasure"]
