"""单元的边（裁定 61 第 1、3、7 条；方案 30 第二、三节）：权重从单元历史的各刻按边数出来，不存参数、不按账折。

- **一刻**（``PastMoment``）：历史上这个行为的一次机会——哪一刻、在场的那一组输入（签名）、下一步来没来、分量（定它来没来的
  那一笔事实的原始强度，与账同一把尺）。每晚按当晚的语义树重标（``foresight.history``）。
- **一条边** = 单元的一个输入（一本账的身份，或 ``↓`` + 压低的关系身份）：它的历史就是签名里含它的那几刻（``edge_moments``）。
- **读数**（``fold_sequence``）：一串 (来没来, 分量) 只读最近 M 项（``Evidence.window``，窗外一项的分量已低于 α），按名次遗忘
  （裁定 57），Beta 后验给上下界；末尾那串连错在之前的读数下几乎不可能（Beta-二项）时，之前的证据打折——"规律断了"
  （裁定 54）。边与单元各层读的是同一个函数，打折算在正在读的那串上，不落盘。
- 调多少不另写公式：权重 w 是这串 fold 出来的均值，每来一项就是 w ← w + η·s·(y − w)，η = s ÷ (有效样本 + s)（方案第三节）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from habitus.foresight.errors import ForesightError
from habitus.foresight.reliability import Reliability, miss_run_probability, reading
from habitus.foundation.lifecycle import decay, rank_by

#: 压低的关系在签名里的前缀（与账身份区分开）。
DOWN = "↓"


@dataclass(frozen=True)
class PastMoment:
    """历史上的一刻：哪一刻（这一段里最早形成的那条预期）、在场的那一组、下一步来没来、分量（原始强度，不打折）。"""

    at: datetime
    signature: frozenset[str]
    came: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.at, datetime) or self.at.utcoffset() is None:
            raise ForesightError("a moment needs a timezone-aware time")
        if not isinstance(self.came, bool):
            raise ForesightError("a moment's came must be a boolean")
        if isinstance(self.weight, bool) or not isinstance(self.weight, int | float) or not 0.0 <= self.weight <= 1.0:
            raise ForesightError("a moment's weight must lie in [0, 1]")

    @property
    def ups(self) -> int:
        return sum(1 for item in self.signature if not item.startswith(DOWN))

    @property
    def downs(self) -> int:
        return sum(1 for item in self.signature if item.startswith(DOWN))


@dataclass(frozen=True)
class Folded:
    """一串刻 fold 出来的读数：读数、规律断了时断点在哪（那串连错第一刻的时刻；没断为 None）、断点之前乘了多少（1 = 没断）、
    读进来了几刻。"""

    reading: Reliability
    break_at: datetime | None = None
    discount: float = 1.0
    count: int = 0


def ordered(moments: Iterable[PastMoment]) -> tuple[PastMoment, ...]:
    """按时刻、再按在场那一组排好（同一时刻的几刻先后与输入顺序无关）。"""

    return tuple(item.item for item in rank_by(moments, key=lambda moment: (moment.at, sorted(moment.signature))))


def edge_moments(moments: Iterable[PastMoment], key: str) -> tuple[PastMoment, ...]:
    """一条边的历史：签名里含它的那几刻（按时刻排）。"""

    return ordered(moment for moment in moments if key in moment.signature)


def fold_sequence(moments: Sequence[PastMoment], *, alpha: float, half_life: float, window: int) -> Folded:
    """``moments`` 已按先后排好。只读最近 ``window`` 刻；每刻的分量 = 它的分量 × 0.5^((最后一刻的名次 − 它的名次) / H)。

    连错按分量数：末尾连着的没来的几刻分量相加（分量 0 的不打断也不加）；来了的一刻就打断。这串连错在之前的读数下
    （Beta-二项，含不确定）的概率 q 小于 α ÷ 之前的有效样本数（每晚都在看，按看的次数校正）就断了：之前的证据乘 q·n / α。
    刚过线是轻罚，再错一次再重一截。"""

    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ForesightError("window must be a positive integer")
    if not 0.0 < alpha < 0.5:
        raise ForesightError("alpha must lie in (0, 0.5)")
    recent = tuple(moments[-window:])
    last = max(0, len(recent) - 1)
    weights = [moment.weight * decay(last - index, half_life=half_life) for index, moment in enumerate(recent)]
    start = len(recent)
    strength = 0.0
    for index in range(len(recent) - 1, -1, -1):
        if recent[index].came:
            break
        if recent[index].weight > 0.0:
            start = index
            strength += recent[index].weight
    before_k = sum(weights[index] for index in range(start) if recent[index].came)
    before_n = sum(weights[:start])
    run_k = sum(weights[index] for index in range(start, len(recent)) if recent[index].came)
    run_n = sum(weights[start:])
    if strength == 0.0 or start == 0:
        return Folded(reading(before_k + run_k, before_n + run_n, alpha=alpha), count=len(recent))
    line = alpha / max(before_n, 1.0)
    chance = miss_run_probability(min(before_k, before_n), before_n, strength)
    if chance >= line:
        return Folded(reading(before_k + run_k, before_n + run_n, alpha=alpha), count=len(recent))
    factor = chance / line
    combined = reading(before_k * factor + run_k, before_n * factor + run_n, alpha=alpha)
    return Folded(combined, break_at=recent[start].at, discount=factor, count=len(recent))


def edge_reading(
    moments: Iterable[PastMoment],
    key: str,
    *,
    alpha: float,
    half_life: float,
    window: int,
    visible_at: datetime | None = None,
) -> Folded:
    """一条边此刻的读数：它的那几刻（``visible_at`` 给了就只到那一刻之前，诊断重建"那一刻的边"用）fold 出来。"""

    mine = edge_moments(moments, key)
    if visible_at is not None:
        mine = tuple(moment for moment in mine if moment.at < visible_at)
    return fold_sequence(mine, alpha=alpha, half_life=half_life, window=window)


__all__ = ["DOWN", "Folded", "PastMoment", "edge_moments", "edge_reading", "fold_sequence", "ordered"]
