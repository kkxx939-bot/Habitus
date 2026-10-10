"""长跨度：比"标志之间的状态"（语义树新方案 ``13`` ①）。

不用固定的 30 / 90 / 180 天窗口——那等于按钟表定有效期，和"失效看中间发生了什么"冲突。一个前因给出若干**段状态**（日子的区间），
量 B 在状态里的日发生率对状态外的日发生率；只数人在场的日子（与短跨度同一个理由）。

标志只用机械可判的：

- **新类第一次出现**（``FIRST_SEEN``）：记录开始后至少过了 max(14 天, 这一类的 3 个块长) 之后的第一次才算——实测 16 类里 13 类在头 7 天
  "第一次出现"，那是开始记录，不是第一次发生。状态 = 那天到截止日。
- **一段状态**（``HIGH_STATE`` / ``LOW_STATE``）：对这一类每天的次数做变点检测（二分切段，泊松似然，切一刀要换得
  ``2·ln(天数)`` 以上的偏差下降——BIC；每段至少 1 个块长）。最长的那段是它的常态，比常态多的段是"集中期"、少的是"停了一阵"。
  例："排查问题"第 8 周一周 18 条、前后几周零星几条 → 一段集中排查期。
- **久别重来**（``RETURN``）：隔了超过预测树的复发窗口再出现——与预测树认的"重新开始"同一个判断。状态 = 重来那天到下一次久别。
- **情境概念**（``SITUATION``）：情境成立的日子（那天有记录判过它、而且成立），连着的几天是一段。

**怎么比**（裁定 27 第 4 条，S4）：每段状态和它前后紧挨着的、同样多个人在场的平常日子比（块内对照：慢慢变多变少的
趋势、不同时期的忙闲都在块里扣掉）；几天并成一个单位，单位长由 B 自己的扎堆程度定（``unit_length``）。各段一张四格表，
零假设下状态里的阳性单位数是超几何分布，各段卷起来得 p。p 能小到多少取决于单位数，不取决于天数能挪出几个位置——
原先的循环平移最小只能到 2 /（天数 + 1），54 天时 0.036，再强的关系也过不了线。

"第一次出现某个对象"（某个项目）要等融合产出对象字段再做：概要里同一个项目有好几种叫法，还会改名，现在认会认错。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta

from habitus.scene.relations import stats
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.timeline import LaneTimeline


@dataclass(frozen=True)
class Episode:
    """一段状态：``[start, end)`` 两个日子之间。"""

    start: date
    end: date

    def holds(self, day: date) -> bool:
        return self.start <= day < self.end


def episodes(
    line: LaneTimeline,
    concept: str,
    segment: Segment,
    *,
    block_days: float,
    first_seen: tuple[int, int],
    recurrence_days: float,
) -> tuple[Episode, ...]:
    """这个前因在这一种标志下的全部状态。"""

    days = line.days()
    if not days:
        return ()
    end = max(days) + timedelta(days=1)
    if segment is Segment.SITUATION:
        return _situation_runs(line, concept, days)
    counts = _daily_counts(line, concept, days)
    seen = [day for day in days if counts[day]]
    if not seen:
        return ()
    if segment is Segment.FIRST_SEEN:
        settle = max(first_seen[0], math.ceil(first_seen[1] * block_days))
        first = seen[0]
        return (Episode(first, end),) if (first - days[0]).days >= settle else ()
    if segment is Segment.RETURN:
        starts = [
            later for earlier, later in zip(seen, seen[1:], strict=False) if (later - earlier).days > recurrence_days
        ]
        bounds = [*starts, end]
        found = []
        for start, following in zip(bounds, bounds[1:], strict=False):
            last = max(day for day in seen if start <= day < following)
            found.append(Episode(start, last + timedelta(days=1) if following != end else end))
        return tuple(found)
    return _states(days, counts, segment, minimum=max(1, math.ceil(block_days)))


@dataclass(frozen=True)
class EpisodeMeasure:
    """长跨度的量法：每段状态和它前后紧挨着的平常日子比（块内对照），人在场的日子才算。

    ``unit_days`` 天并成一个单位（单位里 B 来过就算阳性），由 B 自己日序列的扎堆程度定（``unit_length``）：
    相邻单位之间不再明显相关，单位才能当独立样本数。``since`` 之后的状态日才算实验组（前向验证、维持检验的新数据）；
    对照取状态前后的平常日子，不受 ``since`` 限制（新的那段状态往往从旧数据里延续过来）。
    """

    days: tuple[date, ...]
    outcomes: tuple[int | None, ...]
    episodes: tuple[Episode, ...]
    unit_days: int = 1
    since: date | None = None

    @property
    def units(self) -> int:
        return sum(len(table.treated) for table in self._tables())

    @property
    def blocks(self) -> int:
        """单位按 B 的扎堆长度切过，彼此近似独立：一个单位就是一块。"""

        return self.units

    def effect(self) -> stats.Effect:
        tables = self._tables()
        treated = [value for table in tables for value in table.treated]
        controls = [value for table in tables for value in table.controls]
        return stats.Effect(antecedents=len(treated), treated_rate=_mean(treated), control_rate=_mean(controls))

    def tails(self, *, exact_limit: int) -> tuple[float, float]:
        """各段一张四格表：在零假设下（状态内外这几个单位可交换），状态里的阳性数服从超几何分布；各段独立，卷起来。
        ``exact_limit`` 不用：卷积总是精确算（长跨度的单位数最多几百）。"""

        tables = self._tables()
        if not tables:
            return 1.0, 1.0
        distribution = _convolved(tables)
        observed = sum(sum(table.treated) for table in tables)
        upper = sum(mass for count, mass in distribution.items() if count >= observed)
        lower = sum(mass for count, mass in distribution.items() if count <= observed)
        return min(1.0, upper), min(1.0, lower)

    def minimum_p(self) -> float:
        """每段都取到最极端（状态里的单位全是阳性、或全是阴性）的概率，取小的一侧乘 2。"""

        tables = self._tables()
        if not tables:
            return 1.0
        highest = math.prod(table.null()[max(table.null())] for table in tables)
        lowest = math.prod(table.null()[min(table.null())] for table in tables)
        return min(1.0, 2.0 * min(highest, lowest))

    def interval(
        self, *, level: float, rounds: int, seed: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """每段之内，状态单位与对照单位各自有放回重抽，百分位区间。"""

        tables = self._tables()
        if not tables or rounds <= 0:
            return None, None
        generator = random.Random(seed)
        differences: list[float] = []
        relatives: list[float] = []
        for _ in range(rounds):
            treated = [table.treated[generator.randrange(len(table.treated))] for table in tables for _ in table.treated]
            controls = [
                table.controls[generator.randrange(len(table.controls))] for table in tables for _ in table.controls
            ]
            result = stats.Effect(antecedents=len(treated), treated_rate=_mean(treated), control_rate=_mean(controls))
            differences.append(result.difference)
            if result.relative is not None:
                relatives.append(result.relative)
        ordered = sorted(differences)
        tail = (1.0 - level) / 2.0
        spread = (_quantile(ordered, tail), _quantile(ordered, 1.0 - tail))
        if len(relatives) < rounds * 0.9:
            return spread, None
        ordered_relatives = sorted(relatives)
        return spread, (_quantile(ordered_relatives, tail), _quantile(ordered_relatives, 1.0 - tail))

    def heterogeneity(self) -> float | None:
        return None  # 段数很少，看"去掉一段方向变不变"（``leave_one_out``）

    def leave_one_out(self, *, upward: bool) -> bool:
        """去掉任何一段，方向都不变。只有一段状态时，把它切成前后两半，两半各自对着同一批对照都朝这个方向。"""

        tables = self._tables()
        if len(tables) >= 2:
            parts = [[table for table in tables if table is not dropped] for dropped in tables]
        elif tables and len(tables[0].treated) >= 2:
            (only,) = tables
            half = len(only.treated) // 2
            parts = [
                [_Table(only.treated[:half], only.controls)],
                [_Table(only.treated[half:], only.controls)],
            ]
        else:
            return False
        for rest in parts:
            treated = [value for table in rest for value in table.treated]
            controls = [value for table in rest for value in table.controls]
            difference = _mean(treated) - _mean(controls)
            if difference == 0.0 or (difference > 0.0) != upward:
                return False
        return True

    def subset(self, *, since: date | None = None, last_blocks: int | None = None) -> EpisodeMeasure:
        start = self.since
        if since is not None:
            start = since if start is None else max(start, since)
        if last_blocks is not None:
            starts = sorted(unit for table in self._tables(start) for unit in table.starts)
            if starts:
                newest = starts[-last_blocks:][0]
                start = newest if start is None else max(start, newest)
        return replace(self, since=start)

    def same_event_share(self) -> float | None:
        return None

    def _tables(self, since: date | None = None) -> tuple[_Table, ...]:
        """每段状态一张表：状态里（``since`` 之后）的单位、前后各取同样多个人在场的平常日子做对照的单位。"""

        since = self.since if since is None else since
        index = {day: position for position, day in enumerate(self.days)}
        inside = [any(item.holds(day) for item in self.episodes) for day in self.days]
        found: list[_Table] = []
        for item in self.episodes:
            positions = [index[day] for day in self.days if item.holds(day)]
            if not positions:
                continue
            length = len(positions)
            before = [p for p in range(max(0, positions[0] - length), positions[0]) if not inside[p]]
            after = [p for p in range(positions[-1] + 1, min(len(self.days), positions[-1] + 1 + length)) if not inside[p]]
            treated_positions = [p for p in positions if since is None or self.days[p] >= since]
            treated = self._units(treated_positions)
            controls = [*self._units(before), *self._units(after)]
            if treated and controls:
                found.append(
                    _Table(
                        tuple(value for value, _start in treated),
                        tuple(value for value, _start in controls),
                        tuple(start for _value, start in treated),
                    )
                )
        return tuple(found)

    def _units(self, positions: Sequence[int]) -> list[tuple[int, date]]:
        """连续的 ``unit_days`` 天并成一个单位：来过就是 1；都没来、但有判不了的那天，这个单位不算。"""

        found: list[tuple[int, date]] = []
        for offset in range(0, len(positions), self.unit_days):
            chunk = positions[offset : offset + self.unit_days]
            values = [self.outcomes[p] for p in chunk]
            if any(value == 1 for value in values):
                found.append((1, self.days[chunk[0]]))
            elif all(value == 0 for value in values):
                found.append((0, self.days[chunk[0]]))
        return found


@dataclass(frozen=True)
class _Table:
    treated: tuple[int, ...]
    controls: tuple[int, ...]
    starts: tuple[date, ...] = ()

    def null(self) -> dict[int, float]:
        """零假设下状态里阳性单位数的分布（超几何）。"""

        size = len(self.treated) + len(self.controls)
        positive = sum(self.treated) + sum(self.controls)
        drawn = len(self.treated)
        total = math.comb(size, drawn)
        low, high = max(0, drawn - (size - positive)), min(drawn, positive)
        return {
            count: math.comb(positive, count) * math.comb(size - positive, drawn - count) / total
            for count in range(low, high + 1)
        }


def _convolved(tables: Sequence[_Table]) -> dict[int, float]:
    distribution = {0: 1.0}
    for table in tables:
        following: dict[int, float] = {}
        for total, mass in distribution.items():
            for count, probability in table.null().items():
                following[total + count] = following.get(total + count, 0.0) + mass * probability
        distribution = following
    return distribution


def unit_length(days: Sequence[date], outcomes: Sequence[int | None], found: Sequence[Episode]) -> int:
    """几天并成一个单位：去掉状态内外的均值差之后，B 的日序列相隔 k 天的自相关还超过它自己一个标准误，就再加一天。

    扎堆的 B（"这一周特别多"）相邻几天不独立；当成独立日子算，假关系会多报（合成数据：两态扎堆的 B，
    逐日算时名义 1% 实际 7%）。扣掉状态内外的差，是不让真关系本身把单位拉长。"""

    inside = [any(item.holds(day) for item in found) for day in days]
    means = {
        flag: _mean([value for value, held in zip(outcomes, inside, strict=True) if held is flag and value is not None])
        for flag in (True, False)
    }
    residual = [None if value is None else value - means[held] for value, held in zip(outcomes, inside, strict=True)]
    defined = [value for value in residual if value is not None]
    variance = sum(value * value for value in defined) / len(defined) if defined else 0.0
    if variance <= 0.0:
        return 1
    lag = 1
    while lag < len(days) // 4:
        pairs = [
            (first, second)
            for first, second in zip(residual, residual[lag:], strict=False)
            if first is not None and second is not None
        ]
        if not pairs:
            break
        correlation = sum(first * second for first, second in pairs) / (len(pairs) * variance)
        if correlation <= 1.0 / math.sqrt(len(pairs)):
            break
        lag += 1
    return lag


def measure_of(line: LaneTimeline, consequent: str, found: tuple[Episode, ...]) -> EpisodeMeasure:
    days = line.days()
    outcomes = tuple(_daily_outcome(line, consequent, day) for day in days)
    return EpisodeMeasure(days=days, outcomes=outcomes, episodes=found, unit_days=unit_length(days, outcomes, found))


def _daily_outcome(line: LaneTimeline, consequent: str, day: date) -> int | None:
    """那天 B 来没来；没来而那天有 B 判不了的记录，算不知道。"""

    if any(mark.day == day for mark in line.marks.get(consequent, [])):
        return 1
    start = line.day_start(day)
    if any(start <= slot < start + line.slots_per_day for slot in line.unknown.get(consequent, [])):
        return None
    return 0


def _daily_counts(line: LaneTimeline, concept: str, days: Sequence[date]) -> dict[date, int]:
    counts = dict.fromkeys(days, 0)
    for mark in line.marks.get(concept, []):
        if mark.day in counts:
            counts[mark.day] += 1
    return counts


def _situation_runs(line: LaneTimeline, concept: str, days: Sequence[date]) -> tuple[Episode, ...]:
    held = line.situations.get(concept, {})
    found: list[Episode] = []
    start: date | None = None
    last: date | None = None
    for day in days:
        if held.get(day):
            start = day if start is None else start
            last = day
        elif start is not None and last is not None:
            found.append(Episode(start, last + timedelta(days=1)))
            start = last = None
    if start is not None and last is not None:
        found.append(Episode(start, last + timedelta(days=1)))
    return tuple(found)


def _states(days: Sequence[date], counts: dict[date, int], segment: Segment, *, minimum: int) -> tuple[Episode, ...]:
    """变点检测：二分切段（泊松似然、BIC），每段至少 ``minimum`` 天；最长那段是常态，多于 / 少于它的是状态。"""

    values = [counts[day] for day in days]
    pieces = _segments(values, 0, len(values), minimum, penalty=2.0 * math.log(max(len(values), 2)))
    if len(pieces) < 2:
        return ()
    normal = max(pieces, key=lambda piece: (piece[1] - piece[0], -piece[0]))
    normal_rate = _rate(values, *normal)
    wanted = [
        piece
        for piece in pieces
        if (
            _rate(values, *piece) > normal_rate
            if segment is Segment.HIGH_STATE
            else _rate(values, *piece) < normal_rate
        )
    ]
    merged: list[tuple[int, int]] = []
    for low, high in wanted:
        if merged and merged[-1][1] == low:
            merged[-1] = (merged[-1][0], high)
        else:
            merged.append((low, high))
    return tuple(Episode(days[low], days[high - 1] + timedelta(days=1)) for low, high in merged)


def _segments(values: Sequence[int], low: int, high: int, minimum: int, *, penalty: float) -> list[tuple[int, int]]:
    best_gain, best_cut = 0.0, None
    whole = _log_likelihood(values, low, high)
    for cut in range(low + minimum, high - minimum + 1):
        gain = 2.0 * (_log_likelihood(values, low, cut) + _log_likelihood(values, cut, high) - whole)
        if gain > best_gain:
            best_gain, best_cut = gain, cut
    if best_cut is None or best_gain <= penalty:
        return [(low, high)]
    return _segments(values, low, best_cut, minimum, penalty=penalty) + _segments(
        values, best_cut, high, minimum, penalty=penalty
    )


def _log_likelihood(values: Sequence[int], low: int, high: int) -> float:
    """一段日次数按一个泊松率的对数似然（去掉与切法无关的项）。"""

    total = sum(values[low:high])
    length = high - low
    if total == 0 or length == 0:
        return 0.0
    return total * math.log(total / length) - total


def _rate(values: Sequence[int], low: int, high: int) -> float:
    return sum(values[low:high]) / (high - low)


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


__all__ = ["Episode", "EpisodeMeasure", "episodes", "measure_of", "unit_length"]
