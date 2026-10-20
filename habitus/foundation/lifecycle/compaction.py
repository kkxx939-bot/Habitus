"""精确压实：窗外的旧证据压成一对累计值 (k, n)，读数与逐笔全读相同。

每来一次这条轴的机会，全部旧证据整体乘 0.5^(1/H)，再加上这一笔——所以"先把前面压成累计值、再接着
加后面的"和"从头逐笔加"是同一串运算，结果逐位相同。全读与压实两条路径都必须走 ``fold``，不能一条
用名次权重求和、另一条用累计值，那样浮点误差会让贴着线的读数翻档。

强度为 0 的一笔（例如「提醒后发生」在预测轴上）照样是一次机会：旧证据照乘，只是不加分量。
不是机会的（作废、观测空白）由调用方挡在外面，不进 ``fold``。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from habitus.foundation.lifecycle._validation import require_half_life

_SLACK = 1e-9  # k 与 n 的比较给浮点留的余量


@dataclass(frozen=True)
class Tally:
    """一条轴的累计：``k`` 是加权的"好"，``n`` 是加权的全部机会。"""

    k: float = 0.0
    n: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (("k", self.k), ("n", self.n)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{label} must be a finite number")
            if value < 0:
                raise ValueError(f"{label} must be non-negative")
        if self.k > self.n + _SLACK:
            raise ValueError("k must not exceed n")


def step_factor(half_life: float) -> float:
    """来一次机会，旧证据乘多少。"""

    require_half_life(half_life)
    return 0.5 ** (1.0 / half_life)


def fold(tally: Tally, *, came: bool, strength: float, half_life: float) -> Tally:
    """这条轴来了一次机会：旧值整体衰减一步，再加这一笔（``came`` 为真才进 k）。"""

    if isinstance(strength, bool) or not isinstance(strength, int | float) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    factor = step_factor(half_life)
    return Tally(k=tally.k * factor + (strength if came else 0.0), n=tally.n * factor + strength)


def fold_all(events: Iterable[tuple[bool, float]], *, half_life: float, start: Tally | None = None) -> Tally:
    """按先后把一串 (来没来, 强度) 依次 ``fold`` 进 ``start``。"""

    tally = start if start is not None else Tally()
    for came, strength in events:
        tally = fold(tally, came=came, strength=strength, half_life=half_life)
    return tally


def advance(tally: Tally, steps: int, *, half_life: float) -> Tally:
    """之后又来了 ``steps`` 次机会、但不计它们的分量时，这份累计还剩多少（看"过去某一刻的值"往后推）。"""

    factor = _steps_factor(steps, half_life)
    return Tally(k=tally.k * factor, n=tally.n * factor)


def rewind(tally: Tally, steps: int, *, half_life: float) -> Tally:
    """``advance`` 的逆：从此刻的累计反推 ``steps`` 次机会之前它是多少（诊断重建"那一刻的账"用）。"""

    factor = _steps_factor(steps, half_life)
    return Tally(k=tally.k / factor, n=tally.n / factor)


def _steps_factor(steps: int, half_life: float) -> float:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    return step_factor(half_life) ** steps


__all__ = ["Tally", "advance", "fold", "fold_all", "rewind", "step_factor"]
