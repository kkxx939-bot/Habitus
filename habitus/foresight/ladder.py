"""递法（``19-预测层方案.md`` 第二节；裁定 48–51）：每本账（触发 → 事）两档，影子 → 提醒；级从账上的事实算出来，不配次数。

| 档 | 做什么 | 怎么上来 | 怎么下去 |
|---|---|---|---|
| 0 影子 | 建预期、记账、结算，不说 | 新账从这里起 | — |
| 1 提醒 | 该来了还没来时主动说 | 预测轴下界 L ≥ θ | 预测轴或提醒组上界 U < θ |

三条轴各一本小账（裁定 58 第 5–7 条）：**预测轴**（他自己做了没、离窗多近）管升；**提醒组**（说了之后做了没、有没有明确
拒绝；单位权重）管降——它就是这本账说出口之后的误提醒率；**扣发组**（该说而随机扣下没说之后做了没；单位权重）是提醒组的
对照，``Standing.effect`` = 两者完成率之差，只给人看、不进升降。提醒后才做的在预测轴上是中性（提醒的功劳不算习惯本身）。
每条事实的权重是遗忘乘强度（``reliability.weight``），后验 ``Beta(k + ½, n − k + ½)``，上下界取 α 与 1 − α 分位数。
线 ``θ = 1 − r``，r 是可接受的误提醒率（说了而他其实不会做的比例）：下界过线 = 有 1 − α 的把握这本账说中的比例不低于
θ，也就是误提醒率不高于 r。线每本账一条，按这件事本身的代价比与提醒实际的效果算、在上下限之间波动（``Tolerance``，裁定 58）。

- **滞回**：L ≥ θ 升、U < θ 降，中间维持上一次的级（上一次的级来自账上"级的变化"记录）；免得在线附近来回抖。
- **说错是扣分、可恢复，不清零**（裁定 54）：「这次不用」只是提醒组上一笔坏，账在原地往下移，靠新命中与遗忘回来。
- **规律断了**（裁定 54）：预测轴最近连着几次都是坏的（按分量数：后补只算它那一折），而这在断点之前的读数下（Beta-二项，含不确定）几乎不可能——
  概率 q 小于 α / 断点前的有效样本数（每晚都在看，按看的次数校正）——就把断点之前的事实乘 q·n / α：刚过线是轻罚，
  再错一次再重一截。天天做、命中 88% 的连漏 4 次才算断，每周、命中 90% 的连漏 4 周；本来就弱的要更多次。
- **人的设置**：「以后都提醒我」恒为提醒、「以后别提了」恒为影子（触发也不再为它建预期），「照常」撤销。
- **同一条预期同一条轴后写的覆盖先写的**：过期待决之后的后补、回话改判，不改旧记录，再写一条；「这次不用」例外，
  之后的观测（夜里结出的落空）盖不掉它。

本模块纯函数、对存储零知识。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import (
    COST_RATIO_RANGE,
    LEVELS,
    AccountFact,
    AccountLevel,
    AccountSetting,
    AccountSeverity,
    Delivery,
    Expectation,
    account_key,
)
from habitus.foresight.reliability import Reliability, estimate, miss_run_probability, weight

PREDICTION = "预测"
REMINDED = "提醒组"
HELD_OUT = "扣发组"
DECLINE = "这次不用"
#: 改判一条已有事实的种类：同一时刻落账时排在原判之后（同一晚先结成过期待决、又裁成后补，后补要覆盖它）。
REVISIONS = frozenset({"后补", "已做过", "忘了"})


@dataclass(frozen=True)
class Tolerance:
    """每本账一条线，在产品定的上下限之间波动（裁定 58 第 1 条：不写死、不按容忍度、不按这本账的准确率、不按回话）。

    线 θ = 1 / (1 + c × e)：c 是这件事本身的代价比（漏一次相当于错提醒几次，模型按账判、到期重判、人可改；没判过取
    ``unjudged``），e 是提醒的效果（提醒组完成率 − 扣发组完成率的上界，``Standing`` 算；没有读数时 1）。说中的好处 c × e
    大过说错的代价 1 才开口（Elkan 2001 的代价阈值）；原来两档是 e = 1 时的两个点：c = 2 → 0.33、c = 0.5 → 0.67。
    上下限是产品值（保护闸），从回放定。"""

    floor: float
    ceiling: float
    unjudged: float

    def __post_init__(self) -> None:
        for name in ("floor", "ceiling"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value < 1.0:
                raise ForesightError(f"line {name} must lie in (0, 1)")
        if self.floor >= self.ceiling:
            raise ForesightError("the line floor must lie below its ceiling")
        low, high = COST_RATIO_RANGE
        if isinstance(self.unjudged, bool) or not isinstance(self.unjudged, int | float) or not low <= self.unjudged <= high:
            raise ForesightError(f"the unjudged cost ratio must lie in [{low}, {high}]")

    def threshold(self, cost_ratio: float | None, effect: float = 1.0) -> float:
        if not 0.0 <= effect <= 1.0:
            raise ForesightError("the effect of reminding must lie in [0, 1]")
        gain = (self.unjudged if cost_ratio is None else cost_ratio) * effect
        return min(self.ceiling, max(self.floor, 1.0 / (1.0 + gain)))


@dataclass(frozen=True)
class Evidence:
    """读账的两个借用值：α（关系引擎的前向验证显著性）与遗忘的半衰期（预测树的 ``decay_half_life_days``）。"""

    alpha: float
    half_life_days: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 0.5:
            raise ForesightError("alpha must lie in (0, 0.5)")
        if self.half_life_days <= 0:
            raise ForesightError("half_life_days must be positive")


@dataclass(frozen=True)
class Standing:
    """一本账此刻的位置：级、这本账的线、三条轴的读数、生效的设置；规律断了时断点在哪（最近那串连错的第一次）、断点之前的
    事实乘了多少（1 = 没断）。"""

    level: int
    threshold: float
    prediction: Reliability
    reminded: Reliability
    held_out: Reliability
    setting: str | None = None
    break_at: datetime | None = None
    discount: float = 1.0

    @property
    def effect_bound(self) -> float:
        """线里用的提醒效果：效果的上界（样本少时宽、线只动一点），见 ``effect_bound``。"""

        return effect_bound(self.reminded, self.held_out)

    @property
    def effect(self) -> float | None:
        """提醒的效果：说了之后的完成率减扣下没说之后的（两组都有过才有读数）。"""

        if self.reminded.n <= 0 or self.held_out.n <= 0:
            return None
        return self.reminded.mean - self.held_out.mean


def effect_bound(reminded: Reliability, held_out: Reliability) -> float:
    """提醒效果的上界：提醒组完成率的上界减扣发组完成率的下界，截在 [0, 1]。两组有一组没读数就是 1，线按这件事本身的代价比来；
    样本少时上界宽、线只动一点；数据清楚说明提醒对他没用（说不说他都一样）时上界才掉下来、把线明显抬起来。"""

    if reminded.n <= 0 or held_out.n <= 0:
        return 1.0
    return min(1.0, max(0.0, reminded.upper - held_out.lower))


def current_facts(facts: Iterable[AccountFact]) -> tuple[AccountFact, ...]:
    """同一条预期同一条轴只留最后写的那一条（按时刻；同刻时改判排在原判之后，再按给出的先后）。「这次不用」是人明确表态，
    不被之后的观测盖掉：夜里给同一条预期结出的「落空」时刻更晚，但压回不能因此撤销。"""

    latest: dict[tuple[str, str], AccountFact] = {}
    ordered = sorted(
        enumerate(facts),
        key=lambda pair: (pair[1].kind == DECLINE, pair[1].precedence, pair[1].kind in REVISIONS, pair[0]),
    )
    for item in ordered:
        latest[(item[1].expectation_id, item[1].axis)] = item[1]
    return tuple(sorted(latest.values(), key=lambda fact: fact.at))


def axis_reading(facts: Iterable[AccountFact], axis: str, *, now: datetime, evidence: Evidence) -> Reliability:
    return estimate(
        (
            (weight(item.at, now, strength=item.strength, half_life_days=evidence.half_life_days), item.good)
            for item in facts
            if item.axis == axis and item.at <= now
        ),
        alpha=evidence.alpha,
    )


def climb(
    facts: Iterable[AccountFact],
    *,
    tolerance: Tolerance,
    evidence: Evidence,
    now: datetime,
    severity: AccountSeverity | None = None,
    setting: AccountSetting | None = None,
    previous: AccountLevel | None = None,
) -> Standing:
    """此刻这本账在哪一档。``severity`` 只用来选线（过期的不算）；``previous`` 是账上最近一条"级的变化"。"""

    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ForesightError("climb needs a timezone-aware now")
    live = tuple(item for item in current_facts(facts) if item.at <= now)
    predicted, break_at, discount = _after_break(
        tuple(item for item in live if item.axis == PREDICTION), now=now, evidence=evidence
    )
    prediction = axis_reading(predicted, PREDICTION, now=now, evidence=evidence)
    reminded = axis_reading(live, REMINDED, now=now, evidence=evidence)
    held_out = axis_reading(live, HELD_OUT, now=now, evidence=evidence)
    judged = severity.cost_ratio if severity is not None and severity.current(now) else None
    threshold = tolerance.threshold(judged, effect_bound(reminded, held_out))
    chosen = setting.setting if setting is not None and setting.at <= now and setting.setting != "照常" else None
    if chosen == "以后都提醒我":
        level = 1
    elif chosen == "以后别提了":
        level = 0
    elif prediction.upper < threshold or (reminded.n > 0 and reminded.upper < threshold):
        level = 0
    elif prediction.lower >= threshold:
        level = 1
    else:
        level = 1 if previous is not None and previous.level == 1 else 0
    return Standing(
        level=level,
        threshold=threshold,
        prediction=prediction,
        reminded=reminded,
        held_out=held_out,
        setting=chosen,
        break_at=break_at,
        discount=discount,
    )


def _after_break(
    facts: tuple[AccountFact, ...], *, now: datetime, evidence: Evidence
) -> tuple[tuple[AccountFact, ...], datetime | None, float]:
    """预测轴最近那串连错在断点之前的读数下几乎不可能时，断点之前的事实打折。先后按机会的时刻排（改判不挪位置）。
    连错按分量数：每一笔坏记它的强度（后补只算它打的那一折，没看到的、作废的是 0、不打断）；做了的一笔（强度再低也是做了）
    就打断。"""

    ordered = sorted(facts, key=lambda item: item.at)
    run = 0.0
    start = len(ordered)
    for index in range(len(ordered) - 1, -1, -1):
        item = ordered[index]
        if item.good:
            break
        if item.strength <= 0.0:
            continue
        run += item.strength
        start = index
    if run == 0.0:
        return facts, None, 1.0
    before = ordered[:start]
    reading = axis_reading(before, PREDICTION, now=now, evidence=evidence)
    line = evidence.alpha / max(reading.n, 1.0)
    chance = miss_run_probability(reading.k, reading.n, run)
    if chance >= line:
        return facts, None, 1.0
    factor = chance / line
    discounted = tuple(replace(item, strength=item.strength * factor) for item in before)
    return (*discounted, *ordered[start:]), ordered[start].at, factor


def decide(
    expectation: Expectation,
    level: int,
    *,
    now: datetime,
    spoken_today: int,
    daily_cap: int,
    holdout_rate: float = 0.0,
    draw: float = 1.0,
    text: str | None = None,
    evidence: Sequence[str] = (),
) -> Delivery:
    """对一条到了该说时候的预期的递送决定。``spoken_today`` 是今天已经说过几次（保护闸）；``holdout_rate`` 是这本账的随机
    扣发比例（重的比轻的小，由调用方按重轻取，裁定 58 第 7 条）；``draw`` 是 [0, 1) 抽样。"""

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
    p_speak = 1.0
    if not 0.0 <= holdout_rate < 1.0:
        raise ForesightError("holdout_rate must lie in [0, 1)")
    if holdout_rate > 0.0:
        p_speak = 1.0 - holdout_rate
        if draw < holdout_rate:
            return Delivery(identity, now, level, False, "随机扣发", p_speak, evidence=cited)
    return Delivery(identity, now, level, True, "说了", p_speak, text, evidence=cited)


__all__ = [
    "DECLINE",
    "HELD_OUT",
    "PREDICTION",
    "REMINDED",
    "REVISIONS",
    "Evidence",
    "Tolerance",
    "Standing",
    "account_key",
    "axis_reading",
    "climb",
    "current_facts",
    "effect_bound",
    "decide",
]
