"""loss 尺子（裁定 114）：单元说"这个行为这一刻有几成会来"，实际来没来，算对数损失与 Brier。纯函数、纯值，只量、不当门。

量哪个数：递送那一拍单元替一条预期算出的两个读数（``ledger.Readings``，随那一拍的递送落盘）——单元读数（开不开口看它）
与这条预期自己那本账的读数；再加上预期上已有的原始命中率（``hits / opportunities``）当基线。三个数在同一批"一刻"上算。

结果按哪份口径：单元历史那一份三值口径（``history.record_of``：来了 / 没来 / 不知道，带分量），单元读数就是按它学的；
"不知道"不计分。

单位是一刻：同一个行为、说中的范围连成一片的几条预期算一刻，一次发生只算给一刻（``history.labelled_moments``，裁定 81）。
一刻里有几条递送带读数时取决定开不开口实际用的那一次：决定时刻最晚的那条（时窗末重算、成员接手都晚于时窗开）；同一拍并列的
依据（不是「单元已选」）优先，再取预期身份最小的。这一刻的来路是那条递送所属预期的来路。

看不到的：一刻里没有一条递送带读数的不计分、单独数（从没在开着的时窗里被看到、后来才并进已决定的单元）；说出口的那几刻
提醒后才做的命中被拿掉、提醒后仍没做的落空留着，所以"说出口"与"没说出口"分开各算一组；漏报（没建预期的发生）不在这里，
由组合根数好放在旁边。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from datetime import datetime

from habitus.foresight.history import Moment, MomentRecord, labelled_moments
from habitus.foresight.ledger.model import Delivery, Expectation
from habitus.foresight.units import UNIT, UNIT_CITE, connected

#: 对数损失里把概率夹在 [下限, 1 − 下限]：单元与账的读数有 Jeffreys 先验，不会正好是 0 或 1；原始命中率（20/20 这种）会。
#: 夹住之后一次说反的损失封顶 −ln(1e-4) ≈ 9.21。固定值，不进配置（裁定 114）。
LOG_LOSS_FLOOR = 1e-4
#: 原始命中率在机会数为 0 时取的值：与读数在没有样本时的均值相同（(0 + ½) / (0 + 1)），这一刻照样进同一批。
NO_HISTORY = 0.5
SPOKEN = "说出口"
UNSPOKEN = "没说出口"


@dataclass(frozen=True)
class Score:
    """一组里一个概率的两个分：带分量的平均对数损失与平均 Brier。"""

    log_loss: float
    brier: float


@dataclass(frozen=True)
class LossGroup:
    """一组"一刻"的分：几刻、分量合计，以及单元读数、账的读数、原始命中率三个概率各自的分。"""

    moments: int
    weight: float
    unit: Score
    account: Score
    raw: Score


@dataclass(frozen=True)
class ScoredMoment:
    """计了分的一刻：哪个行为、来路、说没说出口、来没来、分量，以及当时的三个概率。"""

    kind_token: str
    at: datetime
    expectation_ids: tuple[str, ...]
    source: str
    spoken: bool
    came: bool
    weight: float
    unit: float
    account: float
    raw: float


@dataclass(frozen=True)
class Loss:
    """一段账上的 loss。``expectations`` 是参与的预期条数，``unmeasured_expectations`` 是其中一条带读数的递送都没有的；
    ``moments`` 是结果知道的刻（计分的加没量到的），``unknown_moments`` 是结果不知道、不计分的刻，``unmeasured_moments`` 是
    结果知道但一条读数都没有、不计分的刻。``total`` 没有计分的刻时为 None。"""

    expectations: int
    unmeasured_expectations: int
    moments: int
    unknown_moments: int
    unmeasured_moments: int
    total: LossGroup | None
    by_source: Mapping[str, LossGroup] = field(default_factory=dict)
    by_spoken: Mapping[str, LossGroup] = field(default_factory=dict)
    by_kind: Mapping[str, LossGroup] = field(default_factory=dict)
    scored: tuple[ScoredMoment, ...] = ()


def score(probability: float, came: bool, weight: float) -> tuple[float, float]:
    """一刻的（对数损失 × 分量, Brier × 分量）；对数损失前把概率夹在 [``LOG_LOSS_FLOOR``, 1 − ``LOG_LOSS_FLOOR``]。"""

    clamped = min(max(probability, LOG_LOSS_FLOOR), 1.0 - LOG_LOSS_FLOOR)
    log_loss = -math.log(clamped if came else 1.0 - clamped)
    return weight * log_loss, weight * (probability - (1.0 if came else 0.0)) ** 2


def group(moments: Sequence[ScoredMoment]) -> LossGroup | None:
    """一组刻的分（按给的顺序加，同一份输入结果逐位相同）；空组是 None。"""

    if not moments:
        return None
    weight = sum(item.weight for item in moments)

    def scored(pick: str) -> Score:
        parts = [score(getattr(item, pick), item.came, item.weight) for item in moments]
        return Score(sum(part[0] for part in parts) / weight, sum(part[1] for part in parts) / weight)

    return LossGroup(len(moments), weight, scored("unit"), scored("account"), scored("raw"))


def measure(
    records: Iterable[MomentRecord], expectations: Mapping[str, Expectation], deliveries: Iterable[Delivery]
) -> Loss:
    """``records`` 是结完的预期按单元历史口径记下的结果（``history.record_of``），``expectations`` 按身份给它们的预期，
    ``deliveries`` 是它们（与它们引的依据）的递送。"""

    by_id: dict[str, list[Delivery]] = {}
    for delivery in deliveries:
        by_id.setdefault(delivery.expectation_id, []).append(delivery)
    spoken_ids = {identity for identity, made in by_id.items() if any(item.spoken for item in made)}
    every = tuple(records)
    by_kind: dict[str, list[MomentRecord]] = {}
    for record in every:
        by_kind.setdefault(record.kind_token, []).append(record)
    found: list[ScoredMoment] = []
    groups = known = unmeasured = 0
    for kind, items in sorted(by_kind.items()):
        groups += len(connected(items))
        for moment in labelled_moments(items, present=lambda _group: frozenset()):
            known += 1
            scored = _scored(kind, moment, by_id, expectations, spoken_ids)
            if scored is None:
                unmeasured += 1
            else:
                found.append(scored)
    found.sort(key=lambda item: (item.kind_token, item.at, item.expectation_ids))
    seen = {item.expectation_id for item in every}
    unmeasured_expectations = sum(
        1 for identity in seen if not any(item.loss_readings is not None for item in by_id.get(identity, ()))
    )
    return Loss(
        expectations=len(seen),
        unmeasured_expectations=unmeasured_expectations,
        moments=known,
        unknown_moments=groups - known,
        unmeasured_moments=unmeasured,
        total=group(found),
        by_source=_split(found, lambda item: item.source),
        by_spoken=_split(found, lambda item: SPOKEN if item.spoken else UNSPOKEN),
        by_kind=_split(found, lambda item: item.kind_token),
        scored=tuple(found),
    )


def _scored(
    kind: str,
    moment: Moment,
    by_id: Mapping[str, Sequence[Delivery]],
    expectations: Mapping[str, Expectation],
    spoken_ids: Set[str],
) -> ScoredMoment | None:
    """一刻计分用的那一次读数（没有一条递送带读数是 None），连同来路、说没说出口、原始命中率。"""

    made = [item for identity in moment.expectation_ids for item in by_id.get(identity, ())]
    measured = [(item, item.loss_readings) for item in made if item.loss_readings is not None]
    if not measured:
        return None
    # 决定时刻最晚的那次；同一拍并列时依据优先；再并列时 ``made`` 按预期身份排好，max 取第一个
    chosen, readings = max(measured, key=lambda pair: (pair[0].decided_at, pair[0].reason != UNIT))
    expectation = expectations[chosen.expectation_id]
    raw = expectation.hits / expectation.opportunities if expectation.opportunities > 0 else NO_HISTORY
    # 说出口：这一刻里有一条说了，或「单元已选」引的依据说了（结算同一个认法，``foresight_settlement.Asked``）
    cited = {
        cite.removeprefix(UNIT_CITE)
        for item in made
        if item.reason == UNIT
        for cite in item.evidence
        if cite.startswith(UNIT_CITE)
    }
    return ScoredMoment(
        kind,
        moment.past.at,
        moment.expectation_ids,
        expectation.source,
        bool(spoken_ids & ({*moment.expectation_ids} | cited)),
        moment.past.came,
        moment.past.weight,
        readings.unit.mean,
        readings.account.mean,
        raw,
    )


def _split(moments: Sequence[ScoredMoment], key: Callable[[ScoredMoment], str]) -> dict[str, LossGroup]:
    parts: dict[str, list[ScoredMoment]] = {}
    for item in moments:
        parts.setdefault(key(item), []).append(item)
    return {name: found for name, items in sorted(parts.items()) if (found := group(items)) is not None}


__all__ = [
    "LOG_LOSS_FLOOR",
    "NO_HISTORY",
    "SPOKEN",
    "UNSPOKEN",
    "Loss",
    "LossGroup",
    "Score",
    "ScoredMoment",
    "group",
    "measure",
    "score",
]
