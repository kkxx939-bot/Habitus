"""递法（``19-预测层方案.md`` 第二节；裁定 48–51、61）：每条边（前因 → 事）两档，影子 → 提醒；级从边的读数算出来，不配次数。

| 档 | 做什么 | 怎么上来 | 怎么下去 |
|---|---|---|---|
| 0 影子 | 建预期、记账、结算，不说 | 新边从这里起 | — |
| 1 提醒 | 该来了还没来时主动说 | 边的读数下界 L ≥ θ | 边的读数上界 U < θ |

边的读数从单元历史的各刻按边 fold 出来（``foresight.edges``：只读最近 M 刻、按名次遗忘、"规律断了"打折），后验
``Beta(k + ½, n − k + ½)``，上下界取 α 与 1 − α 分位数。线 ``θ = 1 − r``，r 是可接受的误提醒率：下界过线 = 有 1 − α 的把握
这条边说中的比例不低于 θ。这一版所有边同一条线（``Tolerance``，裁定 58 第 1 条）。

- **滞回**：L ≥ θ 升、U < θ 降，中间维持上一次的级（上一次的级来自账上"级的变化"记录）；免得在线附近来回抖。
- **提醒反馈挂在单元上**（裁定 61）：说了他老不做，降的是整个行为单元的档（``units.unit_level``），不在这里。
- **人的设置按行为**：「以后都提醒我」只跳过攒把握（裁定 60 第 8 条）、「以后别提了」恒为影子（触发也不再为它建预期），
  「照常」撤销。

本模块纯函数、对存储零知识。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from habitus.foresight.edges import PastMoment, edge_reading
from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import LEVELS, AccountLevel, Delivery, Expectation, UnitSetting, account_key
from habitus.foresight.reliability import Reliability


@dataclass(frozen=True)
class Tolerance:
    """这一版所有边同一条线（裁定 58 第 1 条）：θ = 1 − 可接受的误提醒率。原定按这件事的代价比与提醒的效果在范围里波动——
    扣发与急缓都先不做，送达接上之后再用模型答一句"要不要赶紧提醒"分两档。"""

    line: float

    def __post_init__(self) -> None:
        if isinstance(self.line, bool) or not isinstance(self.line, int | float) or not 0.0 < self.line < 1.0:
            raise ForesightError("the line must lie in (0, 1)")

    @property
    def threshold(self) -> float:
        return float(self.line)


@dataclass(frozen=True)
class Evidence:
    """读数的两个值：α（借关系引擎的前向验证显著性）与遗忘的半衰期 ``half_life``——几次机会（``config.foresight.half_life_opportunities``，
    临时值，裁定 57：按这本账自己的机会数，不按天）。"""

    alpha: float
    half_life: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 0.5:
            raise ForesightError("alpha must lie in (0, 0.5)")
        if isinstance(self.half_life, bool) or not isinstance(self.half_life, int | float) or self.half_life <= 0:
            raise ForesightError("half_life must be a positive number of opportunities")

    @property
    def window(self) -> int:
        """窗口 M = ⌈H · log₂(1/α)⌉ 次机会：窗外一笔的分量已低于 α。"""

        return math.ceil(self.half_life * math.log2(1.0 / self.alpha))


@dataclass(frozen=True)
class Standing:
    """一条边此刻的位置：级、线、读数、生效的设置；规律断了时断点在哪（最近那串连错的第一刻）、断点之前的刻乘了多少（1 = 没断）。"""

    level: int
    threshold: float
    prediction: Reliability
    setting: str | None = None
    break_at: datetime | None = None
    discount: float = 1.0


def climb(
    moments: Sequence[PastMoment],
    account: str,
    *,
    now: datetime,
    tolerance: Tolerance,
    evidence: Evidence,
    setting: UnitSetting | None = None,
    previous: AccountLevel | None = None,
) -> Standing:
    """``now`` 那一刻这条边（``account``）在哪一档：从这个行为的单元历史 ``moments`` 里签名含它的那几刻读。
    ``setting`` 是这个行为的设置，``previous`` 是账上最近一条"级的变化"。"""

    folded = edge_reading(
        moments, account, alpha=evidence.alpha, half_life=evidence.half_life, window=evidence.window, visible_at=now
    )
    prediction = folded.reading
    threshold = tolerance.threshold
    chosen = setting.setting if setting is not None and setting.at <= now and setting.setting != "照常" else None
    if chosen == "以后都提醒我":
        level = 1
    elif chosen == "以后别提了" or prediction.upper < threshold:
        level = 0
    elif prediction.lower >= threshold:
        level = 1
    else:
        level = 1 if previous is not None and previous.level == 1 else 0
    return Standing(
        level=level,
        threshold=threshold,
        prediction=prediction,
        setting=chosen,
        break_at=folded.break_at,
        discount=folded.discount,
    )


def decide(
    expectation: Expectation,
    level: int,
    *,
    now: datetime,
    spoken_today: int,
    daily_cap: int,
    text: str | None = None,
    evidence: Sequence[str] = (),
) -> Delivery:
    """对一条到了该说时候的预期的递送决定。``spoken_today`` 是今天已经说过几次（保护闸）。"""

    if level not in LEVELS:
        raise ForesightError(f"unknown delivery level: {level!r}")
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ForesightError("decide needs a timezone-aware now")
    identity = expectation.expectation_id
    cited = tuple(evidence)
    if level == 0:
        return Delivery(identity, now, 0, False, "影子", 0.0, evidence=cited)
    if spoken_today >= daily_cap:
        return Delivery(identity, now, level, False, "名额已满", 0.0, evidence=cited)
    return Delivery(identity, now, level, True, "说了", 1.0, text, evidence=cited)


__all__ = ["Evidence", "Tolerance", "Standing", "account_key", "climb", "decide"]
