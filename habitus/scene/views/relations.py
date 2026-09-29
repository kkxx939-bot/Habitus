"""relations/：每条假设的当前读数——账本存事实，这里把事实读成强度、类型、稳定性、剂量、归因。

全部读时算、可重算；一个数都不落回账本。

**先过累积度门槛，再谈数**（用户 09-26 裁定：少样本不出数）。门槛双维度：进账的机会至少 ``min_count`` 次、
且锚落在至少 ``min_blocks`` 个不同的块里。块长 = max(1 天, 这本账锚间隔的 p50)（就是前件自己的复发间隔：
喝水那类一天几次的块是 1 天、四天就够，打球那类一周一次的块是 7 天、要四周）。门槛之下是三态之"证据不够"：
不出强度，只报攒了多少、还差多少，并给上一级（父概念 / 少一个元素的子集）的读数当指针——预测层按七i 三退回上一级用。

**假设分两型**（用户 09-27 裁定）：``expected_at`` 有数的是**节律型**，走下面三套；``expected_at is None`` 的是
**无节律型**（约球→打球、挂号→就诊），后件没节律，"比平时多几成"无从谈起——它只读 ``FulfilmentReading``：
兑现率（兑现 /（兑现 + 释放 + 被关掉 + 仍立着且已经立了超过对照间隔的））、到达者的兑现间隔中位、最久立了多少小时，
对照 = 后件本来多久真的发生一次（相邻机会间隔 ÷ 峰概率）。**不读 p1−p0、不读类型**。

三个方面三套算法（用户裁定，不共用公式）：

- **概率**（同一本账同时给出**兑现间隔**）：按"已观测地过了几次机会"走 Kaplan–Meier，后件来了是事件（第 k 步），右删失
  在第 c 步退出。``p1 = 1 − S(expected_at)``；对照 ``p0`` = 各承诺快照里第 expected_at 个机会的 probability 的均值；
  强度 = p1 − p0；区间 = 按块 bootstrap。第二读数：KM 中位步数 + 到达者的中位小时；S 没降到 0.5 就明说"中位没到"。
- **时刻**：``observed_at − 那次机会的 at``（两个绝对时刻的差，不再是环形量），中位数，按块 bootstrap。
- **次数**：``count − 那几次机会的 probability 之和``，均值，按块 bootstrap。

**类型**（只对概率）：区间含零 → 无；负 → 抑制；正 → PN = (p1−p0)/p1、PS = (p1−p0)/(1−p0)（外生 + 单调下的界），
PN 高 PS 低 → 使能，PS 高 → 促进。**稳定性**：按承诺里的情境快照分两层，各过分层门槛且区间不重叠 → 被调节；记下
测过哪些、哪些因太少跳过。**剂量**：前件带档时按档的数值下界排序，≥3 档、单调且与方向同号才算趋势。**归因**
（七i 四）：与本条共用同一次证据的其他假设 C，本账按"那次证据是否也开了 C 的承诺"分两层 = {A,C} vs {A,¬C}，能分开就
说明谁在起作用。**干预分层**：提醒过 / 没提醒各一份。分层各自过门槛、各自种子。

删失与关闭：概率的右删失**进 n**（它有信息）；时刻的删失、次数的删失不进。承诺里出现别的假设指纹、对照跨了两代树 → 标出来。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from functools import partial
from types import MappingProxyType

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import (
    ASPECT_SEPARATOR,
    ELEMENT_SEPARATOR,
    Antecedent,
    Aspect,
    Direction,
    Hypothesis,
)
from habitus.scene.ledger.model import Claim, ClaimRef, Intervention, Outcome, Settlement
from habitus.scene.ledger.opening import LedgerConfig
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.views.stats import (
    Interval,
    SurvivalStep,
    block_bootstrap,
    effective_counts,
    is_monotonic,
    jeffreys_interval,
    kaplan_meier,
    mean,
    median,
    survival_at,
    survival_median,
    widen,
)

HOURS_PER_DAY = 24.0


@dataclass(frozen=True)
class ViewsConfig:
    """全部是待定值，在重放上定；这里只是默认。"""

    #: 累积度门槛（用户 09-27 定成 3 次 / 跨 2 块）：过了就出数，"这个数信不信"归预测层判。
    #: 实测代价（对照 p0=0.50）：n=3 时真有效应抓到 55.5%、真没效应误报 25%——误报率跟着离散格子跳，
    #: 不是单调的（n=3 只有 k=3 或 k=0 才够窄，真没效应时那两种合起来正好 2/8）。定这么低的前提是
    #: 基准写假设时不穷举连线：常识猜对七成时 n=3 出数里假的只占 16%，纯随机配对则是 90%。
    min_count: int = 3
    #: 块那一维留着：挡掉"3 次全挤在同一天"那类（用户：再怎么高频也不能在一天内说明什么）。
    #: 块长 = max(1 天, 前件复发间隔 p50)，所以一天一次或更慢的行为就是"至少跨两天"。
    min_blocks: int = 2
    split_min: int = 3
    split_min_blocks: int = 2
    interval_level: float = 0.90
    bootstrap_samples: int = 1000
    enabling_pn: float = 0.5
    promoting_ps: float = 0.5
    #: 一条概率承诺"命运已定"要等几次机会 —— 组合根要让它等于 ``LedgerConfig.censor_after``。读概率时只收已经定了的那些
    #: （对称截断）：后件来了当晚就 OCCURRED 进账、没来要等这么多次机会才 CENSORED，不截断就是最近一段只进事件、
    #: p1 系统偏高（真 0.333 读 0.375，评审 C-4 实测）。
    settlement_horizon: int = 10

    def __post_init__(self) -> None:
        for label in ("min_count", "min_blocks", "split_min", "split_min_blocks", "bootstrap_samples", "settlement_horizon"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        for label in ("interval_level", "enabling_pn", "promoting_ps"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{label} must be a positive number")
        if not self.interval_level < 1.0:
            raise ValueError("interval_level lies in (0, 1)")

    @property
    def main_gate(self) -> tuple[int, int]:
        return (self.min_count, self.min_blocks)

    @property
    def split_gate(self) -> tuple[int, int]:
        return (self.split_min, self.split_min_blocks)


def require_aligned(views: ViewsConfig, ledger: LedgerConfig) -> None:
    """读侧的对称截断点必须与写侧的删失点是**同一个数**：``settlement_horizon == censor_after``。组合根先调它。

    两者是同一件事的两头：写侧等过 ``censor_after`` 个已观测机会没来就右删失；读侧只收"到那一刻命运已定"的承诺
    （评审 C-4）。不等的后果是系统性偏——读侧比写侧短，最近那一段只进事件不进删失，p1 偏高；读侧比写侧长，
    已经删失了的承诺还被当"命运未定"扣掉，n 偏小。两个数都是待定值（重放上定），但"必须相等"与定成几无关，
    所以这条不是配置而是不变量。
    """

    if views.settlement_horizon != ledger.censor_after:
        raise ValueError(
            f"settlement_horizon ({views.settlement_horizon}) must equal LedgerConfig.censor_after ({ledger.censor_after}): "
            "the read side's truncation point and the write side's censoring point are the same moment"
        )


class TypeReading(str, Enum):
    NONE = "none"
    INHIBITING = "inhibiting"
    ENABLING = "enabling"
    PROMOTING = "promoting"


Pair = tuple[Claim, Settlement]


@dataclass(frozen=True)
class Accumulation:
    """攒到哪了：进账几次、跨了几个块、块多长；门槛是多少。"""

    count: int
    blocks: int
    block_hours: float
    needed_count: int
    needed_blocks: int

    @property
    def sufficient(self) -> bool:
        return self.count >= self.needed_count and self.blocks >= self.needed_blocks

    @property
    def short_by_count(self) -> int:
        return max(0, self.needed_count - self.count)

    @property
    def short_by_blocks(self) -> int:
        return max(0, self.needed_blocks - self.blocks)


@dataclass(frozen=True)
class Account:
    """一条假设的账，承诺、结算、提醒已经 join 好；``block_hours`` 是整本账定下的块长（分层子账沿用）。"""

    hypothesis_identity: str
    claims: tuple[Claim, ...]
    settlements: Mapping[ClaimRef, Settlement]
    interventions: Mapping[ClaimRef, tuple[Intervention, ...]]
    block_hours: float

    @property
    def settled(self) -> tuple[Pair, ...]:
        return tuple((claim, self.settlements[claim.ref]) for claim in self.claims if claim.ref in self.settlements)

    @property
    def open(self) -> int:
        return sum(1 for claim in self.claims if claim.ref not in self.settlements)

    @property
    def antecedent_uris(self) -> frozenset[str]:
        return frozenset(uri for claim in self.claims for uri in claim.antecedent_uris)

    @property
    def fingerprints(self) -> frozenset[str]:
        return frozenset(claim.hypothesis_fingerprint for claim in self.claims)

    @property
    def control_generations(self) -> frozenset[str]:
        """这本账的对照出自哪几代预测树。**跨天多于一代是正常的**——夜批每晚重建树，代名带构建时刻。"""

        return frozenset(claim.control.generation for claim in self.claims if claim.control is not None)

    @property
    def generations_split_within_a_day(self) -> tuple[str, ...]:
        """**同一个触发日**里出现了两代对照：那一晚崩在开承诺中途、重建树之后又重跑了一次（横切第 3 条"开承诺在重建树
        之后"被破坏）。这才是要标出来的；按整本账判"多于一代"在每晚重建的设计下必然为真，会让 ``unrefuted`` 恒假、
        profile 永远空（评审 A-1 实测）。"""

        by_day: dict[object, set[str]] = {}
        for claim in self.claims:
            if claim.control is not None:
                by_day.setdefault(claim.ref.day, set()).add(claim.control.generation)
        return tuple(sorted({item for generations in by_day.values() if len(generations) > 1 for item in generations}))

    def where(self, keep: Callable[[Claim], bool]) -> Account:
        claims = tuple(claim for claim in self.claims if keep(claim))
        return Account(self.hypothesis_identity, claims, self.settlements, self.interventions, self.block_hours)

    def block_of(self, claim: Claim) -> int:
        """这条承诺的锚落在第几个块（从整本账最早的锚起算）。"""

        first = min(item.anchor for item in self.claims).astimezone(UTC)
        elapsed = (claim.anchor.astimezone(UTC) - first).total_seconds() / 3600.0
        return int(elapsed // self.block_hours)


def block_hours_of(claims: Sequence[Claim]) -> float:
    """块长 = max(1 天, 锚间隔的 p50)：一天几次的行为块是 1 天，一周一次的块是 7 天。"""

    anchors = sorted(claim.anchor.astimezone(UTC) for claim in claims)
    gaps = [(later - earlier).total_seconds() / 3600.0 for earlier, later in zip(anchors, anchors[1:], strict=False)]
    if not gaps:
        return HOURS_PER_DAY
    return max(HOURS_PER_DAY, median(gaps))


def load_account(ledger: LedgerStore, hypothesis_identity: str) -> Account:
    claims = ledger.claims_for(hypothesis_identity)
    settlements = {}
    interventions = {}
    for claim in claims:
        settlement = ledger.read_settlement(claim.ref)
        if settlement is not None:
            settlements[claim.ref] = settlement
        reminders = ledger.interventions_for(claim.ref)
        if reminders:
            interventions[claim.ref] = reminders
    return Account(hypothesis_identity, claims, MappingProxyType(settlements), MappingProxyType(interventions), block_hours_of(claims))


# ── 进账的机会 ────────────────────────────────────────────────────────────────


def informative_pairs(account: Account, aspect: Aspect) -> tuple[Pair, ...]:
    """进 n 的机会：概率——后件来了、或右删失且至少已观测地过了一次机会；时刻——观测到 / 缺席；次数——计数。都要带对照快照。"""

    found: list[Pair] = []
    for claim, settlement in account.settled:
        if claim.control is None:
            continue
        if aspect is Aspect.PROBABILITY:
            if settlement.outcome is Outcome.OCCURRED or (settlement.outcome is Outcome.CENSORED and settlement.observed_passes > 0):
                found.append((claim, settlement))
        elif aspect is Aspect.TIMING:
            if settlement.outcome in (Outcome.OBSERVED, Outcome.ABSENT):
                found.append((claim, settlement))
        elif settlement.outcome is Outcome.COUNTED:
            found.append((claim, settlement))
    return tuple(found)


def accumulation_of(account: Account, pairs: Sequence[Pair], gate: tuple[int, int]) -> Accumulation:
    blocks = {account.block_of(claim) for claim, _s in pairs} if pairs else set()
    return Accumulation(count=len(pairs), blocks=len(blocks), block_hours=account.block_hours, needed_count=gate[0], needed_blocks=gate[1])


def settled_pairs(account: Account) -> tuple[Pair, ...]:
    """无节律型进账的机会：兑现（OCCURRED）、释放（RELEASED）、被生命周期关掉（CENSORED）。没有对照也进——
    对照只用来算"后件本来多久一次"，缺了它照样能读兑现率。"""

    return tuple(
        (claim, settlement)
        for claim, settlement in account.settled
        if settlement.outcome in (Outcome.OCCURRED, Outcome.RELEASED, Outcome.CENSORED)
    )


# ── 强度 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Strength:
    aspect: Aspect
    accumulation: Accumulation
    interval: Interval | None
    control: float | None
    unit: str
    #: 概率：p1 = 到第 expected_at 次机会为止已发生的比例（KM）；事件数与右删失数；兑现间隔的第二读数。
    p1: float | None = None
    #: 整条离散生存曲线（用户 09-27 定"逐次印，不是累积"）：每一步"当时还在等几条、这一次来了几条"。
    #: 只印第 expected_at 那一格会把"往后推"读成"不发生"——六次里 4 次吃了早餐、2 次推到第二天，
    #: 单看第 1 格是 33%，逐次看是 2/6 → 2/4 → 0/2，"他不是不吃，是推了一天"才读得出来。
    #: 后面几格不能直接归给这条前件（可能是别的行为在起作用），它是定位工具不是归因（用户补的那一条）。
    curve: tuple[SurvivalStep, ...] = ()
    events: int = 0
    censored: int = 0
    median_steps: int | None = None
    median_hours: float | None = None
    #: 时刻：缺席的次数（那次机会已观测地过了、后件没来）。
    absent: int = 0
    #: 有机会、但那条承诺没有对照快照（新概念、树上还没数）→ 不参与。
    without_control: int = 0
    #: 命运还没定（第 ``settlement_horizon`` 次机会还没过完）→ 不参与，等它定了再算（对称截断）。
    immature: int = 0
    #: 快照铺不到第 ``expected_at`` 次机会 → 不参与。与写侧那条"快照太短就删失"成对，读侧也要能看见条数。
    short_snapshot: int = 0
    agrees_with_prior: bool | None = None

    @property
    def sufficient(self) -> bool:
        return self.accumulation.sufficient


def strength_of(
    account: Account,
    hypothesis: Hypothesis,
    config: ViewsConfig,
    *,
    gate: tuple[int, int] | None = None,
    seed: str | None = None,
    now: datetime | None = None,
) -> Strength:
    """按方面算强度。没过累积度门槛 → 区间 None，其余计数照报（三态之"证据不够"）。

    ``now`` 给了就对概率方面做**对称截断**：只收命运已定的承诺（见 ``ViewsConfig.settlement_horizon``）。不给就不截断
    （旧行为，只在调用方确信账里没有"刚开还没到期"的承诺时用）。
    """

    resolved_gate = gate or config.main_gate
    seed_text = seed or account.hypothesis_identity
    aspect = hypothesis.aspect
    if hypothesis.is_open_ended:
        # 无节律型不走强度这条路（不读 p1−p0、不读类型）：这里只给计数，数由 ``fulfilment_of`` 读。留一法与分账
        # 拿区间是 None 当"读不出"，于是无节律型自然不参与稳定性扫与借力。
        pairs = settled_pairs(account)
        return Strength(
            aspect,
            accumulation_of(account, pairs, resolved_gate),
            None,
            None,
            "fulfilment",
            events=sum(1 for _c, item in pairs if item.outcome is Outcome.OCCURRED),
        )
    pairs = informative_pairs(account, aspect)
    without_control = sum(1 for claim, s in account.settled if claim.control is None and s.outcome is not Outcome.CENSORED)
    accumulation = accumulation_of(account, pairs, resolved_gate)
    if aspect is Aspect.PROBABILITY:
        return _probability_strength(account, hypothesis, pairs, accumulation, config, seed_text, without_control, now)
    if aspect is Aspect.TIMING:
        return _timing_strength(account, hypothesis, pairs, accumulation, config, seed_text, without_control)
    return _count_strength(account, hypothesis, pairs, accumulation, config, seed_text, without_control)


def layer_reading(
    account: Account,
    hypothesis: Hypothesis,
    config: ViewsConfig,
    *,
    gate: tuple[int, int] | None = None,
    seed: str | None = None,
    now: datetime | None = None,
) -> LayerReading:
    """分层读数：节律型给强度（p1−p0），无节律型给兑现率。两者都有 ``.interval`` 与 ``.accumulation``，
    所以“两层比区间”那段判据一个字不用改（P-1：用户 09-27 定“四处”一起按型分派）。

    ``seed`` 只对节律型有意义：它的区间要随机重采样（分层各自种子，第 2 批的 2-16）；
    兑现率用的是 Jeffreys 解析区间，没有随机性，不需要种子——不是漏了。
    """

    if hypothesis.is_open_ended:
        return fulfilment_of(account, hypothesis, config, gate=gate, now=now)
    return strength_of(account, hypothesis, config, gate=gate, seed=seed, now=now)


def settled_by(claim: Claim, config: ViewsConfig, now: datetime) -> bool:
    """这条概率承诺的命运在 ``now`` 之前已经定了吗：第 ``settlement_horizon`` 次机会已经结束（或快照本来就铺到头了）。

    与 ``settle_probability`` 的收口点是同一个：到那一刻它要么已经 OCCURRED，要么被右删失。
    """

    snapshot = claim.control
    if snapshot is None:
        return False
    boundary = snapshot.at(config.settlement_horizon)
    return (boundary.span.end if boundary is not None else snapshot.last_end) <= now


def _probability_strength(
    account: Account,
    hypothesis: Hypothesis,
    pairs: Sequence[Pair],
    accumulation: Accumulation,
    config: ViewsConfig,
    seed: str,
    without_control: int,
    now: datetime | None,
) -> Strength:
    step = hypothesis.expected_at or 1
    ready = [(claim, s) for claim, s in pairs if now is None or settled_by(claim, config, now)]
    immature = len(pairs) - len(ready)
    short_snapshot = sum(1 for claim, _s in ready if claim.control is not None and claim.control.at(step) is None)
    usable = [(claim, s) for claim, s in ready if claim.control is not None and claim.control.at(step) is not None]
    events = sum(1 for _c, s in usable if s.outcome is Outcome.OCCURRED)
    censored = len(usable) - events
    accumulation = accumulation_of(account, usable, (accumulation.needed_count, accumulation.needed_blocks))
    if not usable:
        return Strength(
            hypothesis.aspect, accumulation, None, None, "probability", events=events, censored=censored,
            without_control=without_control, immature=immature, short_snapshot=short_snapshot,
        )
    p0 = mean([claim.control.at(step).probability for claim, _s in usable])  # type: ignore[union-attr]
    p1 = _probability_at(usable, step)
    latencies = [s.latency_hours for _c, s in usable if s.outcome is Outcome.OCCURRED and s.latency_hours is not None]
    curve = kaplan_meier(*_steps(usable))
    median_steps = survival_median(curve)
    median_hours = median(latencies) if latencies else None
    if not accumulation.sufficient or p1 is None:
        return Strength(
            hypothesis.aspect, accumulation, None, p0, "probability", p1=p1, events=events, censored=censored,
            median_steps=median_steps, median_hours=median_hours, without_control=without_control,
            immature=immature, short_snapshot=short_snapshot, curve=curve,
        )
    blocks = [account.block_of(claim) for claim, _s in usable]
    # 区间 = 分块 bootstrap（治天间成簇）∪ Jeffreys（治二值小样本的塌缩）。只用 bootstrap 的话，六条全到 / 全没到时
    # 每次重采样都给同一个数 → 零宽区间 → 必然"显著"（零效应假显著率 p0=0.88 时实测 50.7%，名义 10%）。
    resampled = block_bootstrap(
        usable, blocks, partial(_probability_effect, step=step), seed=seed, samples=config.bootstrap_samples, level=config.interval_level
    )
    counts = effective_counts(curve, step)
    shifted = None
    if counts is not None:
        rate = jeffreys_interval(counts.events, counts.remainder, level=config.interval_level)
        shifted = Interval(p1 - p0, rate.low - p0, rate.high - p0)
    interval = _usable_interval(widen(resampled, shifted) if resampled is not None else shifted)
    return Strength(
        hypothesis.aspect,
        accumulation,
        interval,
        p0,
        "probability",
        p1=p1,
        events=events,
        censored=censored,
        median_steps=median_steps,
        median_hours=median_hours,
        without_control=without_control,
        immature=immature,
        short_snapshot=short_snapshot,
        curve=curve,
        agrees_with_prior=None if interval is None else _agrees(interval, hypothesis.direction),
    )


def _usable_interval(interval: Interval | None) -> Interval | None:
    """零宽区间不是"确定"，是"这个方法在这批数据上算不出宽度"——一律当没有区间（连带 ``unrefuted`` 与 profile 排序）。"""

    if interval is None or interval.width <= 0.0:
        return None
    return interval


def _steps(pairs: Sequence[Pair]) -> tuple[list[int], list[int]]:
    """事件在第几步（已观测过的机会数 + 1）、右删失过了几步。"""

    event_steps = [s.observed_passes + 1 for _c, s in pairs if s.outcome is Outcome.OCCURRED]
    censored_steps = [s.observed_passes for _c, s in pairs if s.outcome is Outcome.CENSORED]
    return event_steps, censored_steps


def _probability_at(pairs: Sequence[Pair], step: int) -> float | None:
    survival = survival_at(kaplan_meier(*_steps(pairs)), step)
    return None if survival is None else 1.0 - survival


def _probability_effect(pairs: Sequence[Pair], *, step: int) -> float | None:
    p1 = _probability_at(pairs, step)
    if p1 is None:
        return None
    p0 = mean([claim.control.at(step).probability for claim, _s in pairs])  # type: ignore[union-attr]
    return p1 - p0


def _timing_strength(account: Account, hypothesis: Hypothesis, pairs: Sequence[Pair], accumulation: Accumulation, config: ViewsConfig, seed: str, without_control: int) -> Strength:
    step = hypothesis.expected_at or 1
    observed: list[Pair] = []
    values: list[float] = []
    for claim, s in pairs:
        target = None if claim.control is None else claim.control.at(step)
        if s.outcome is Outcome.OBSERVED and s.observed_at is not None and target is not None:
            observed.append((claim, s))
            values.append((s.observed_at - target.at).total_seconds() / 60.0)
    absent = sum(1 for _c, s in pairs if s.outcome is Outcome.ABSENT)
    return _bootstrapped(account, hypothesis, observed, values, median, "minutes", accumulation, config, seed, without_control, absent=absent)


def _count_strength(account: Account, hypothesis: Hypothesis, pairs: Sequence[Pair], accumulation: Accumulation, config: ViewsConfig, seed: str, without_control: int) -> Strength:
    first = hypothesis.expected_at or 1
    usable: list[Pair] = []
    values: list[float] = []
    short_snapshot = 0
    for claim, s in pairs:
        expected = None if claim.control is None else claim.control.expected_count(first, hypothesis.horizon)
        if expected is None:
            short_snapshot += 1
        if s.count is None or expected is None:
            continue
        usable.append((claim, s))
        values.append(float(s.count) - expected)
    accumulation = accumulation_of(account, usable, (accumulation.needed_count, accumulation.needed_blocks))
    return _bootstrapped(
        account, hypothesis, usable, values, mean, "times", accumulation, config, seed, without_control, short_snapshot=short_snapshot
    )


def _bootstrapped(
    account: Account,
    hypothesis: Hypothesis,
    pairs: Sequence[Pair],
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    unit: str,
    accumulation: Accumulation,
    config: ViewsConfig,
    seed: str,
    without_control: int,
    *,
    absent: int = 0,
    short_snapshot: int = 0,
) -> Strength:
    """``accumulation`` 按进账的全部机会算（时刻的缺席也是攒到的证据），区间只用有数值的那些；有数值的太少也不出区间。"""

    enough_values = len(values) >= config.split_min and len({block for block in (account.block_of(claim) for claim, _s in pairs)}) >= config.split_min_blocks
    if not values or not accumulation.sufficient or not enough_values:
        return Strength(
            hypothesis.aspect, accumulation, None, 0.0 if values else None, unit, absent=absent,
            without_control=without_control, short_snapshot=short_snapshot,
        )
    blocks = [account.block_of(claim) for claim, _s in pairs]
    interval = _usable_interval(block_bootstrap(values, blocks, statistic, seed=seed, samples=config.bootstrap_samples, level=config.interval_level))
    return Strength(
        hypothesis.aspect,
        accumulation,
        interval,
        0.0,
        unit,
        absent=absent,
        without_control=without_control,
        short_snapshot=short_snapshot,
        agrees_with_prior=None if interval is None else _agrees(interval, hypothesis.direction),
    )


def _agrees(interval: Interval, direction: Direction) -> bool | None:
    if interval.contains_zero:
        return None
    return (interval.point > 0) == (direction is Direction.UP)


# ── 类型 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TypeReadout:
    """从账读出的类型，连同它依据的两个数。PN/PS 是外生 + 单调假设下的界，不是点识别。"""

    reading: TypeReading
    pn: float | None
    ps: float | None


def read_type(strength: Strength, config: ViewsConfig) -> TypeReadout | None:
    """只对概率方面：区间含零 → 无；负 → 抑制；正 → PN = (p1−p0)/p1、PS = (p1−p0)/(1−p0)。"""

    if strength.aspect is not Aspect.PROBABILITY or strength.unit != "probability":
        return None
    if strength.interval is None or strength.control is None or strength.p1 is None:
        return None
    if strength.interval.contains_zero:
        return TypeReadout(TypeReading.NONE, None, None)
    if strength.interval.high < 0:
        return TypeReadout(TypeReading.INHIBITING, None, None)
    p1, p0 = strength.p1, strength.control
    pn = (p1 - p0) / p1 if p1 > 0 else 0.0
    ps = (p1 - p0) / (1.0 - p0) if p0 < 1.0 else 0.0
    reading = TypeReading.ENABLING if pn >= config.enabling_pn and ps < config.promoting_ps else TypeReading.PROMOTING
    return TypeReadout(reading, pn, ps)


# ── 无节律型：兑现率与兑现间隔 ───────────────────────────────────────────────────


@dataclass(frozen=True)
class FulfilmentReading:
    """无节律型（``expected_at is None``）的读数（用户 09-27 裁定）：只有兑现率与兑现间隔，不读 p1−p0、不读类型。

    分母 = 兑现 + 释放 + **仍立着且已经立了超过对照间隔的**（裁定原文的三项）。还没立到对照间隔的不进分母——
    "约了今晚去打球，现在下午三点"既不算兑现也不算没兑现。

    **被生命周期关掉的（``CENSORED``）只计数、不进分母**：``close_claim`` 的关闭理由是自由文本，账本分不出
    "约了三个月没去，不等了"（是一次没兑现，该进分母）与"这条假设整个退役了"（跟他去没去无关，不该进）。
    分不出就一律算会在后者上把兑现率压低，所以照裁定的三项算，这一项等生命周期那份定出"关闭有哪几种"再定。
    今天它恒为 0（``close_claim`` 没有生产调用者）。

    ``median_hours`` 是**到达者**的中位（没兑现的那些不混进来，由 ``rate`` 与 ``longest_standing_hours`` 表达）；
    ``control_hours`` = 后件本来多久真的发生一次 = 相邻机会间隔 ÷ 峰概率（见 ``expected_wait_hours``），没快照就没对照。
    """

    accumulation: Accumulation
    fulfilled: int
    released: int
    #: 被生命周期关掉的条数。**只报数、不进分母**（见上）；今天恒为 0。
    closed: int
    standing: int
    standing_total: int
    rate: float | None
    interval: Interval | None
    median_hours: float | None
    control_hours: float | None
    longest_standing_hours: float | None

    @property
    def sufficient(self) -> bool:
        return self.accumulation.sufficient

    @property
    def unrefuted(self) -> bool:
        """过门槛 ∧ 兑现率算出了区间 ∧ 有对照且中位兑现间隔短于对照间隔（"比它平时自己来得快"）。"""

        return (
            self.sufficient
            and self.interval is not None
            and self.control_hours is not None
            and self.median_hours is not None
            and self.median_hours < self.control_hours
        )


#: 一层的读数：节律型给强度（p1−p0），无节律型给兑现率。两者都有 ``.interval`` 与 ``.accumulation``，
#: 所以"两层比区间"的判据共用一套。
LayerReading = Strength | FulfilmentReading


def expected_wait_hours(claims: Sequence[Claim]) -> float | None:
    """对照：后件**本来多久真的发生一次** = 相邻机会间隔 ÷ 峰概率（小时）。没有一份带两个机会的快照 → None。

    问的不是"它多久有一次机会"，而是"它自己多久来一次"（用户 09-27 裁定）：打球的峰一天一个、每个峰的概率 0.20，
    机会间隔 24 小时，真的去打球的期望等待是 24 / 0.20 = 120 小时。按间隔本身当对照的话，约球之后 52.8 小时去了
    会被判成"不如平时快"——而它平时要等五天。几何等待的直接算法，与节律型的 ``p0`` 用同一份快照数据
    （机会的 ``probability``），不新加字段、不碰预测树。
    """

    waits: list[float] = []
    for claim in claims:
        if claim.control is None or len(claim.control.opportunities) < 2:
            continue
        items = claim.control.opportunities
        gaps = [(later.at - earlier.at).total_seconds() / 3600.0 for earlier, later in zip(items, items[1:], strict=False)]
        waits.append(median(gaps) / median([item.probability for item in items]))
    return median(waits) if waits else None


def fulfilment_of(
    account: Account,
    hypothesis: Hypothesis,
    config: ViewsConfig,
    *,
    gate: tuple[int, int] | None = None,
    now: datetime | None = None,
) -> FulfilmentReading:
    """读无节律型的账。``now`` 不给就不把仍立着的算进分母（没有"立了多久"这个量，宁可不出数也不抬高兑现率）。"""

    if not hypothesis.is_open_ended:
        raise ValueError("fulfilment_of reads open-ended hypotheses (expected_at=None) only")
    resolved_gate = gate or config.main_gate
    pairs = settled_pairs(account)
    fulfilled = [(claim, item) for claim, item in pairs if item.outcome is Outcome.OCCURRED]
    released = sum(1 for _c, item in pairs if item.outcome is Outcome.RELEASED)
    # 被生命周期关掉的只报数，不进分母（见类 docstring）。
    shut = [(claim, item) for claim, item in pairs if item.outcome is Outcome.CENSORED]
    control_hours = expected_wait_hours(account.claims)
    standing_claims = tuple(claim for claim in account.claims if claim.ref not in account.settlements)
    waited = [max(0.0, (now - claim.anchor).total_seconds() / 3600.0) for claim in standing_claims] if now is not None else []
    # 没有对照时把全部仍立着的都算进分母（保守：宁可把兑现率读低，不凭空抬高）。``now`` 没给就一条都不算——
    # "立了多久"这个量不存在时，猜它超没超过对照间隔只会抬高兑现率。
    threshold = control_hours if control_hours is not None else 0.0
    overdue = tuple(claim for claim, hours in zip(standing_claims, waited, strict=False) if hours > threshold)
    shut_claims = {claim.ref for claim, _s in shut}
    counted: list[Claim] = [claim for claim, _s in pairs if claim.ref not in shut_claims]
    counted.extend(overdue)
    blocks = {account.block_of(claim) for claim in counted} if counted else set()
    accumulation = Accumulation(
        count=len(counted), blocks=len(blocks), block_hours=account.block_hours, needed_count=resolved_gate[0], needed_blocks=resolved_gate[1]
    )
    latencies = [item.latency_hours for _c, item in fulfilled if item.latency_hours is not None]
    rate = len(fulfilled) / len(counted) if counted else None
    interval = None
    if counted and accumulation.sufficient:
        interval = _usable_interval(jeffreys_interval(len(fulfilled), len(counted) - len(fulfilled), level=config.interval_level))
    return FulfilmentReading(
        accumulation=accumulation,
        fulfilled=len(fulfilled),
        released=released,
        closed=len(shut),
        standing=len(overdue),
        standing_total=len(standing_claims),
        rate=rate,
        interval=interval,
        median_hours=median(latencies) if latencies else None,
        control_hours=control_hours,
        longest_standing_hours=max(waited) if waited else None,
    )


# ── 上一级的指针 ───────────────────────────────────────────────────────────────


def borrow_candidates(hypothesis: Hypothesis, concepts: ConceptSet) -> tuple[str, ...]:
    """账薄时可以退回去用的上一级假设身份：单前件退到不带档 / 父概念，多前件退到少一个元素的子集。"""

    found: list[str] = []
    if len(hypothesis.antecedents) == 1:
        item = hypothesis.antecedents[0]
        if item.identity in concepts and item.grade is not None:
            found.append(_identity_with(hypothesis, (Antecedent(item.concept),)))
        if item.identity in concepts:
            for parent in concepts.ancestors(item.identity):
                found.append(_identity_with(hypothesis, (Antecedent(concepts[parent].name),)))
    else:
        for index in range(len(hypothesis.antecedents)):
            subset = hypothesis.antecedents[:index] + hypothesis.antecedents[index + 1 :]
            if any(concepts[a.identity].role.is_behavior for a in subset if a.identity in concepts):
                found.append(_identity_with(hypothesis, subset))
    return tuple(dict.fromkeys(found))


def _identity_with(hypothesis: Hypothesis, antecedents: tuple[Antecedent, ...]) -> str:
    """按 ``hypotheses`` 那边的分隔符拼上一级的身份；分隔符一改这里跟着变，不手抄。

    机会段用**本条自己的**（``opportunity_token``）：上一级猜的是别的机会数时就找不到它，于是不给指针——
    那是对的，借来的应该是同一个量（A-7 已定"拿子的口径读父账读出来的不是上一级的读数"）。
    """

    leaf = (
        ELEMENT_SEPARATOR.join(item.leaf_token for item in sorted(antecedents, key=lambda a: a.identity))
        + ASPECT_SEPARATOR
        + hypothesis.aspect.value
        + ASPECT_SEPARATOR
        + hypothesis.opportunity_token
    )
    return f"{hypothesis.consequent_identity}/{leaf}"


@dataclass(frozen=True)
class FallbackReading:
    """自己没攒够时给预测层的指针：上一级哪条、留一法之后它的读数。"""

    identity: str
    strength: Strength


def fallback_reading(
    account: Account,
    hypothesis: Hypothesis,
    ledger: LedgerStore,
    concepts: ConceptSet,
    config: ViewsConfig,
    hypotheses: Mapping[str, Hypothesis],
    now: datetime | None = None,
) -> FallbackReading | None:
    """账最厚、且过了门槛的上一级；留一法 = 去掉与本条共用任何一条出力 occurrence 的承诺（七f-6）。

    **用父假设本体的口径读父账**：父猜的可能是别的机会数（父 expected_at=2、子 1），拿子的口径读父账读出来的不是
    "上一级的读数"（父自己读 −0.13，子拿去读同一本账得 −0.88，评审 A-7）。父假设不在 ``hypotheses`` 里（只有账、
    没有假设）就不给指针——那种账没人能解释它在量什么。
    """

    own = account.antecedent_uris
    best: FallbackReading | None = None
    for identity in borrow_candidates(hypothesis, concepts):
        parent_hypothesis = hypotheses.get(identity)
        if parent_hypothesis is None:
            continue
        parent = load_account(ledger, identity).where(partial(_avoids, uris=own))
        strength = strength_of(parent, parent_hypothesis, config, seed=identity, now=now)
        if not strength.sufficient or strength.interval is None:
            continue
        if best is None or strength.accumulation.count > best.strength.accumulation.count:
            best = FallbackReading(identity, strength)
    return best


# ── 分账 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Moderation:
    situation: str
    with_situation: LayerReading
    without_situation: LayerReading


@dataclass(frozen=True)
class StabilityReport:
    """稳定性扫的结果：被哪些情境调节、测过哪些、哪些因两层里有一层没攒够而跳过。对照没按情境分（刀 5 前的诚实标注）。"""

    moderations: tuple[Moderation, ...]
    tested: tuple[str, ...]
    skipped: tuple[str, ...]
    control_split_by_situation: bool = False
    #: 每个**测过**的情境的两层（不管区间有没有分开）。份额要它：七f-3 定的"调节 {A,B} vs {A,¬B}"就是
    #: 这两层的差（"已经晚睡了，出差还额外压多少"），而 ``moderations`` 只留区间分开的那些，
    #: 所以差算得出但印不出来。两层各自有区间，差真不真由"区间不重叠"判。
    layers: tuple[Moderation, ...] = ()


def stability_scan(account: Account, hypothesis: Hypothesis, concepts: ConceptSet, config: ViewsConfig, now: datetime | None = None) -> StabilityReport:
    """对每个在承诺情境快照里出现过的情境概念分账；两层都过分层门槛且区间不重叠 → 被它调节。"""

    situations = sorted({concepts[name].identity for claim in account.claims for name in claim.situation_snapshot if name in concepts and concepts[name].role.is_situation})
    moderations: list[Moderation] = []
    layers: list[Moderation] = []
    tested: list[str] = []
    skipped: list[str] = []
    for situation in situations:
        label = concepts[situation].name  # 给人读的是概念名，不是规范身份
        with_it = account.where(partial(_has_situation, situation=situation, concepts=concepts))
        without = account.where(partial(_lacks_situation, situation=situation, concepts=concepts))
        a = layer_reading(with_it, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{situation}|with", now=now)
        b = layer_reading(without, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{situation}|without", now=now)
        if a.interval is None or b.interval is None:
            skipped.append(label)
            continue
        tested.append(label)
        layers.append(Moderation(label, a, b))
        if a.interval.disjoint_from(b.interval):
            moderations.append(Moderation(label, a, b))
    return StabilityReport(tuple(moderations), tuple(tested), tuple(skipped), layers=tuple(layers))


@dataclass(frozen=True)
class DoseReading:
    concept: str
    by_grade: tuple[tuple[str, LayerReading], ...]
    monotonic: bool


def dose_readings(account: Account, hypothesis: Hypothesis, concepts: ConceptSet, config: ViewsConfig, now: datetime | None = None) -> tuple[DoseReading, ...]:
    """前件（假设里没写死档的那些）按承诺记下的档分账，看强度是否随档单调。档按**数值下界**排，不按声明顺序。"""

    found: list[DoseReading] = []
    for item in hypothesis.antecedents:
        if item.grade is not None or item.identity not in concepts or not concepts[item.identity].grades:
            continue
        if any(grade.wraps_midnight for grade in concepts[item.identity].grades):
            continue  # 跨午夜的档在数轴上排不出顺序，见 unordered_dose_concepts
        order = [grade.name for grade in sorted(concepts[item.identity].grades, key=lambda grade: grade.lower)]
        by_grade: list[tuple[str, LayerReading]] = []
        for grade in order:
            stratum = account.where(partial(_has_grade, identity=item.identity, grade=grade))
            strength = layer_reading(stratum, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{item.identity}|{grade}", now=now)
            if strength.interval is not None:
                by_grade.append((grade, strength))
        if len(by_grade) >= 2:
            points = [s.interval.point for _g, s in by_grade if s.interval is not None]
            # 单调**且与基准猜的方向同号**才算趋势：一条从 −0.68 走到 −0.08 的曲线在数值上单调，但它说的是
            # "档越重影响越小"，与"熬得越晚越不吃"相反，不该升可信度。
            trend = is_monotonic(points) and _agrees_direction(points, hypothesis.direction)
            found.append(DoseReading(item.concept, tuple(by_grade), trend))
    return tuple(found)


def unordered_dose_concepts(hypothesis: Hypothesis, concepts: ConceptSet) -> tuple[str, ...]:
    """前件里**档跨午夜**的那些概念：它们不出剂量读数，要报出来。

    剂量读的是"档越重、效应越往一边走"，所以档必须能排成一条阶梯——而阶梯是按 ``grade.lower``
    在数轴上排的。一个 22:00–02:00 的档 ``lower`` 是 1320、``upper`` 是 120，与一个 02:00–04:00 的档
    （``lower`` 120）比大小，排出来是"凌晨在深夜之前"，那条"趋势"就是假的。
    所以这一项不出数；要出，得先给绝对时刻的档定一个参照点（相对常态的档不受影响，它们不跨午夜）。
    """

    return tuple(
        item.concept
        for item in hypothesis.antecedents
        if item.grade is None and item.identity in concepts and any(grade.wraps_midnight for grade in concepts[item.identity].grades)
    )


def _agrees_direction(points: Sequence[float], direction: Direction) -> bool:
    """档越重、效应越往基准说的那一边走。"""

    if len(points) < 2:
        return False
    delta = points[-1] - points[0]
    return delta > 0 if direction is Direction.UP else delta < 0


def _has_situation(claim: Claim, situation: str, concepts: ConceptSet) -> bool:
    return any(concepts[name].identity == situation for name in claim.situation_snapshot if name in concepts)


def _lacks_situation(claim: Claim, situation: str, concepts: ConceptSet) -> bool:
    return not _has_situation(claim, situation, concepts)


def _has_grade(claim: Claim, identity: str, grade: str) -> bool:
    return any(hit.identity == identity and hit.grade == grade for hit in claim.antecedent_hits)


# ── 归因：共享证据的调节对照 ───────────────────────────────────────────────────


class EvidenceIndex:
    """出力 occurrence → 哪些假设的承诺用过它。一次扫盘建好、全部假设共用（不然每条都扫一遍整本账）。"""

    def __init__(self, ledger: LedgerStore) -> None:
        self._by_uri: dict[str, set[str]] = {}
        self._uris_by_hypothesis: dict[str, set[str]] = {}
        for identity in ledger.hypotheses():
            uris = {uri for claim in ledger.claims_for(identity) for uri in claim.antecedent_uris}
            self._uris_by_hypothesis[identity] = uris
            for uri in uris:
                self._by_uri.setdefault(uri, set()).add(identity)

    def sharing(self, hypothesis_identity: str, uris: Iterable[str]) -> tuple[str, ...]:
        others = {identity for uri in uris for identity in self._by_uri.get(uri, ()) if identity != hypothesis_identity}
        return tuple(sorted(others))

    def uris_of(self, hypothesis_identity: str) -> frozenset[str]:
        return frozenset(self._uris_by_hypothesis.get(hypothesis_identity, ()))


@dataclass(frozen=True)
class SharedEvidence:
    """与本条共用同一次证据的另一条假设 C，以及 {A,C} vs {A,¬C} 那条调节对照（七i 四）。"""

    other: str
    shared_claims: int
    with_other: LayerReading
    without_other: LayerReading

    @property
    def separable(self) -> bool | None:
        """能不能分出谁在起作用：两层都过门槛才有答案；区间不重叠 → C 在场时确实不一样。"""

        if self.with_other.interval is None or self.without_other.interval is None:
            return None
        return self.with_other.interval.disjoint_from(self.without_other.interval)


def shared_evidence(
    account: Account,
    hypothesis: Hypothesis,
    index: EvidenceIndex,
    config: ViewsConfig,
    hypotheses: Mapping[str, Hypothesis],
    now: datetime | None = None,
) -> tuple[SharedEvidence, ...]:
    """七i 四要的那条对照：**另一个原因概念**（熬夜工作）命中了同一次 occurrence，按"那次证据是否也开了它的承诺"
    分两层 = {A,C} vs {A,¬C}，看得出谁在起作用。

    排除三类不是"另一个原因"的：
    - **同一个前件集合**的兄弟假设（晚睡→起床 / 咖啡 / 就寝）——那是七i 一的多影响，它们的证据 100% 重合，
      "不在场"那层恒空，印出来只有"分不开"三个字的噪音（每条关系多印 3 行，评审 A-6/C-5）；
    - 前件集合是本条**子集或超集**的（{晚睡,出差}→早餐）——那是七i 二的多体账，与 ``stability_scan`` 按「出差中」分账重复；
    - **后件不同**的——"是睡得晚还是工作到晚在起作用"是对同一顿早饭说的。
    """

    mine = frozenset(item.identity for item in hypothesis.antecedents)
    found: list[SharedEvidence] = []
    for other in index.sharing(account.hypothesis_identity, account.antecedent_uris):
        peer = hypotheses.get(other)
        if peer is None or peer.consequent_identity != hypothesis.consequent_identity:
            continue
        theirs_antecedents = frozenset(item.identity for item in peer.antecedents)
        if theirs_antecedents <= mine or mine <= theirs_antecedents:
            continue
        theirs = index.uris_of(other)
        with_other = account.where(partial(_touches, uris=theirs))
        without_other = account.where(partial(_avoids, uris=theirs))
        found.append(
            SharedEvidence(
                other=other,
                shared_claims=len(with_other.claims),
                with_other=layer_reading(with_other, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{other}|with", now=now),
                without_other=layer_reading(without_other, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{other}|without", now=now),
            )
        )
    return tuple(found)


def _touches(claim: Claim, uris: frozenset[str]) -> bool:
    return bool(set(claim.antecedent_uris) & uris)


def _avoids(claim: Claim, uris: frozenset[str]) -> bool:
    return not _touches(claim, uris)


def reminded_split(account: Account, hypothesis: Hypothesis, config: ViewsConfig, now: datetime | None = None) -> tuple[LayerReading, LayerReading]:
    """干预分层：提醒过 / 没提醒各一份（各自过分层门槛，不够就是区间 None）。"""

    reminded = account.where(lambda claim: claim.ref in account.interventions)
    quiet = account.where(lambda claim: claim.ref not in account.interventions)
    return (
        layer_reading(reminded, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|reminded", now=now),
        layer_reading(quiet, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|quiet", now=now),
    )


# ── 读数 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RelationReading:
    hypothesis_identity: str
    aspect: Aspect
    strength: Strength
    type_reading: TypeReadout | None
    stability: StabilityReport
    dose: tuple[DoseReading, ...]
    shared: tuple[SharedEvidence, ...]
    reminded: tuple[LayerReading, LayerReading]
    open_claims: int
    #: 无节律型才有：兑现率与兑现间隔（节律型是 None）。有它的时候 ``strength`` 只带计数，``type_reading`` 一定是 None。
    fulfilment: FulfilmentReading | None = None
    fallback: FallbackReading | None = None
    #: 前件里档跨午夜的概念：剂量阶梯排不出顺序，所以那几个不出剂量读数（见 ``unordered_dose_concepts``）。
    unordered_dose: tuple[str, ...] = ()
    #: 账里有按**别的**指纹开的承诺——同身份的假设改过第几次机会或方向之后，旧口径的观测还留在盘上。
    stale_fingerprints: tuple[str, ...] = ()
    #: 这本账的对照出自哪几代预测树（出处，跨天多代是正常的——树每晚重建）。
    control_generations: tuple[str, ...] = ()
    #: **同一个触发日**里出现了两代对照：那一晚的承诺开在树重建的两边，这一批不可信。
    mixed_control_generations: tuple[str, ...] = ()

    @property
    def mixed_fingerprints(self) -> bool:
        return bool(self.stale_fingerprints)

    @property
    def clean(self) -> bool:
        """这本账的口径没混：没有按旧指纹开的承诺，同一触发日里没有两代对照。"""

        return not self.stale_fingerprints and not self.mixed_control_generations

    @property
    def significant(self) -> bool:
        """**只说"这个数不等于零"**，不说它换了条件还成不成立——那是稳定性，两件事（用户 09-27 要求分开印）。"""

        if self.fulfilment is not None:
            return self.clean and self.fulfilment.interval is not None and self.fulfilment.sufficient
        return self.clean and self.strength.interval is not None and not self.strength.interval.contains_zero

    @property
    def stability_untested(self) -> bool:
        """一个情境都没测过（没有可分层的情境，或每一层都太少）——"没被推翻"里最容易被误读成"已证实"的就是这一段。"""

        return not self.stability.tested

    @property
    def unrefuted(self) -> bool:
        """过了门槛、区间不含零、没被任何测过的情境调节、账没混口径。**不是"已证实"**：没测过的情境不算。

        无节律型另一套判据（没有 p1−p0 可言）：见 ``FulfilmentReading.unrefuted``。"""

        if self.fulfilment is not None:
            return self.fulfilment.unrefuted and not self.stale_fingerprints and not self.mixed_control_generations
        return (
            self.strength.interval is not None
            and not self.strength.interval.contains_zero
            and not self.stability.moderations
            and not self.stale_fingerprints
            and not self.mixed_control_generations
        )


def read_relation(
    hypothesis: Hypothesis,
    *,
    ledger: LedgerStore,
    concepts: ConceptSet,
    config: ViewsConfig | None = None,
    index: EvidenceIndex | None = None,
    hypotheses: Mapping[str, Hypothesis] | None = None,
    now: datetime | None = None,
) -> RelationReading:
    """读一条关系。``hypotheses`` 是这一轮在读的假设集（用来查上一级本体、辨别兄弟假设）；``now`` 给了就做对称截断。"""

    resolved = config or ViewsConfig()
    known = dict(hypotheses or {})
    known.setdefault(hypothesis.identity, hypothesis)
    account = load_account(ledger, hypothesis.identity)
    strength = strength_of(account, hypothesis, resolved, now=now)
    # 无节律型的四处分层也照跑，只是每层读的是兑现率而不是 p1−p0（P-1，用户 09-27 定"四处"）。
    fulfilment = fulfilment_of(account, hypothesis, resolved, now=now) if hypothesis.is_open_ended else None
    fallback = (
        None
        if hypothesis.is_open_ended or strength.sufficient
        else fallback_reading(account, hypothesis, ledger, concepts, resolved, known, now)
    )
    return RelationReading(
        hypothesis_identity=hypothesis.identity,
        aspect=hypothesis.aspect,
        strength=strength,
        type_reading=read_type(strength, resolved),
        stability=stability_scan(account, hypothesis, concepts, resolved, now),
        dose=dose_readings(account, hypothesis, concepts, resolved, now),
        unordered_dose=unordered_dose_concepts(hypothesis, concepts),
        shared=shared_evidence(account, hypothesis, index or EvidenceIndex(ledger), resolved, known, now),
        reminded=reminded_split(account, hypothesis, resolved, now),
        open_claims=account.open,
        fulfilment=fulfilment,
        fallback=fallback,
        stale_fingerprints=tuple(sorted(account.fingerprints - {hypothesis.fingerprint})),
        control_generations=tuple(sorted(account.control_generations)),
        mixed_control_generations=account.generations_split_within_a_day,
    )


def read_relations(
    hypotheses: Iterable[Hypothesis],
    *,
    ledger: LedgerStore,
    concepts: ConceptSet,
    config: ViewsConfig | None = None,
    now: datetime | None = None,
) -> tuple[RelationReading, ...]:
    """一轮读全部关系：证据倒排索引与假设集只建一次，两者都给每条读数用。"""

    items = tuple(hypotheses)
    known = {item.identity: item for item in items}
    index = EvidenceIndex(ledger)
    return tuple(
        read_relation(item, ledger=ledger, concepts=concepts, config=config, index=index, hypotheses=known, now=now) for item in items
    )


__all__ = [
    "Account",
    "Accumulation",
    "DoseReading",
    "EvidenceIndex",
    "FallbackReading",
    "FulfilmentReading",
    "LayerReading",
    "Moderation",
    "RelationReading",
    "SharedEvidence",
    "StabilityReport",
    "Strength",
    "TypeReading",
    "TypeReadout",
    "ViewsConfig",
    "block_hours_of",
    "borrow_candidates",
    "expected_wait_hours",
    "fulfilment_of",
    "layer_reading",
    "settled_by",
    "settled_pairs",
    "dose_readings",
    "fallback_reading",
    "informative_pairs",
    "load_account",
    "read_relation",
    "read_relations",
    "read_type",
    "reminded_split",
    "require_aligned",
    "unordered_dose_concepts",
    "shared_evidence",
    "stability_scan",
    "strength_of",
]
