"""把四层数字与四层背景装配成一个候选的证据。

编排而已：数字在 ``numbers``、背景在 ``context``、类型在 ``model``，本模块只负责"按同一个
槽把它们对齐"，并且保证**钟面的映射只有一种算法**——槽与周几由预测树自己的 ``SlotKey.of``
按树发布的槽宽算出来，不在这里另写一遍取模。

纯函数：不读时钟、不碰存储、不调模型。此刻是谁、树钉哪一代、缓存怎么建，全部由组合根决定。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from habitus.foresight.context import LayerBackground, layer_background
from habitus.foresight.errors import ForesightError
from habitus.foresight.model import LAYER_NAMES, Moment
from habitus.foresight.numbers import CellIndex, provenance
from habitus.prediction.model import SlotKey
from habitus.scene.views import DayIndexCache


@dataclass(frozen=True)
class CandidateEvidence:
    """一个候选在这一刻的证据：四层数字，每层配着**算出那个数的那几天**的历史背景。

    背景与层绑在一起，而不是摊成一个大列表——判断者要能看出"这条历史是给邻域那个 6/20
    作证的"，而不是面对一堆来源不明的往事。
    """

    kind_token: str
    moment: Moment
    layers: tuple[LayerBackground, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind_token, str) or not self.kind_token:
            raise ForesightError("candidate kind_token must be non-empty text")
        if tuple(item.layer.name for item in self.layers) != LAYER_NAMES:
            raise ForesightError("candidate evidence must carry all four layers in order")

    @property
    def ungrouped(self) -> tuple[date, ...]:
        """四层合起来有数、却没有语义背景的日子。"""

        return tuple(sorted({day for item in self.layers for day in item.layer.ungrouped}))


def moment_at(at: datetime, *, slot_minutes: int, day_note: str | None = None) -> Moment:
    """把一个**主体本地时刻**落到钟面上。槽与周几走树的映射，不另写一份。"""

    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ForesightError("moment must be a timezone-aware local datetime")
    key = SlotKey.of(at, slot_minutes=slot_minutes)
    return Moment(at=at, slot=key.slot, weekday=key.weekday, day_note=day_note)


def candidate_evidence(
    cells: CellIndex,
    kind_token: str,
    moment: Moment,
    cache: DayIndexCache,
    *,
    half_width: int,
    window_days: int,
    transition_window_seconds: float,
    max_days_per_layer: int,
) -> CandidateEvidence:
    """一个候选的四层数字 + 四层背景。

    ``cache.covered`` 同时充当"这天归组了没有"的谓词与取背景的入口，所以数字那边算出的
    ``ungrouped`` 与背景那边真取到的日子必然一致——两处用两个判据是这一整套最容易出的错。
    """

    slot = SlotKey(weekday=moment.weekday, slot=moment.slot)
    layers = provenance(cells, kind_token, slot, half_width=half_width, grouped=cache.covered)
    backgrounds = tuple(
        layer_background(
            layer,
            kind_token,
            cache,
            slot_minutes=cells.tree.slot_minutes,
            slot_index=moment.slot,
            half_width=half_width,
            window_days=window_days,
            transition_window_seconds=transition_window_seconds,
            max_days=max_days_per_layer,
        )
        for layer in layers
    )
    return CandidateEvidence(kind_token=kind_token, moment=moment, layers=backgrounds)


def candidates_evidence(
    cells: CellIndex,
    kind_tokens: Sequence[str],
    moment: Moment,
    cache: DayIndexCache,
    *,
    half_width: int,
    window_days: int,
    transition_window_seconds: float,
    max_days_per_layer: int,
) -> tuple[CandidateEvidence, ...]:
    """一次查询里装配多个候选：树、格子索引与日缓存都只建一次。

    顺序按传进来的顺序，不重排也不去重——谁是候选、按什么排，是上游的事。
    """

    return tuple(
        candidate_evidence(
            cells,
            kind_token,
            moment,
            cache,
            half_width=half_width,
            window_days=window_days,
            transition_window_seconds=transition_window_seconds,
            max_days_per_layer=max_days_per_layer,
        )
        for kind_token in kind_tokens
    )


__all__ = ["CandidateEvidence", "candidate_evidence", "candidates_evidence", "moment_at"]
