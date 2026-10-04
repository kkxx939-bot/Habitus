"""一条假设的账在内存里的样子：承诺、结算、提醒 join 好；块（样本的独立单位）；进账的机会。

**块 = 同一个后果窗口**（2026-09-30 二-6）：等同一次就寝的几条承诺是一块——它们不是独立样本。承诺带对照快照时块号就是
第 1 个落点的窗口；没有快照（无节律型、要不到窗口）退回按锚的时间块（max(1 天, 锚间隔 p50)）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from types import MappingProxyType

from habitus.scene.hypotheses.model import Aspect
from habitus.scene.ledger.model import Claim, ClaimRef, Intervention, Outcome, Settlement
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.views.stats import median

HOURS_PER_DAY = 24.0

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
        """这条承诺属于第几个块：**同一个后果窗口 = 同一块**（二-6）；没有快照的按锚落在第几个时间块。

        两种块号不混：窗口块从 0 起编号，时间块接在窗口块之后编号——否则"第 0 个窗口"与"第 0 个时间块"会被当成同一块。
        """

        if claim.control is not None:
            return self._window_blocks[claim.control.opportunities[0].span.start.astimezone(UTC)]
        first = min(item.anchor for item in self.claims).astimezone(UTC)
        elapsed = (claim.anchor.astimezone(UTC) - first).total_seconds() / 3600.0
        return len(self._window_blocks) + int(elapsed // self.block_hours)

    @cached_property
    def _window_blocks(self) -> Mapping[datetime, int]:
        starts = sorted({claim.control.opportunities[0].span.start.astimezone(UTC) for claim in self.claims if claim.control is not None})
        return {start: index for index, start in enumerate(starts)}


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
            # 窗口账（2026-10-01）：来了 / 看清了没来 进 n；没看清的不进分母分子
            if settlement.outcome in (Outcome.OCCURRED, Outcome.ABSENT):
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



__all__ = ["HOURS_PER_DAY", "Account", "Accumulation", "Pair", "accumulation_of", "block_hours_of", "informative_pairs", "load_account", "settled_pairs"]
