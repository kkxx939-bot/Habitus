"""预测错了的诊断（裁定 54、55）：每一次落空先判"和我们的数据有没有关系"，有关的指出是哪一层，无关的老实记"说不清"。纯函数、只读。

诊断从不往树上写、不改账：账上的改判在结算时已经做了（前因不成立 → 作废，窗口里有观测空白 → 中性）。这里只把已有字段装成
一次落空的"事实"（``MissFacts``），按顺序过关：

1. 前因不成立（结算已记作废）；2. 他可能做了、我们没看到（窗里有观测空白，或回了「已做过」）；3. 他开始了但放弃了；
4. 推迟（之后补做了：后补）；5. 规律断了（账在那一刻判出断点、这次在断点之后）；
6. 树上已成立的原因：关系已不在机会表里（失效 / 前向未复现）、这一类在词表里已拆改；
7. 疑点（只报不改）：惯常的前因这次没来、今天的日型和历史那几次都不一样、历史那几次里有宿主助手先通知的；
8. 都不是：说不清，与我们的数据无关。

落空的原因不另分类：预测之后的纠错由单元每晚按当晚的语义树重标历史、再数完成（裁定 55 新版第 2、7 条），这里只回答"数据可不可信、
是哪一层"。「预算内」（这次落空之后账的下界仍在线上：配置允许的那一份误提醒）只是标注，单独计数。
另给一张校准表：按（触发种类 × 实现版本）切片，比"触发算出的命中率"与"账上实际结出的"，算出的落在实际的区间外就标出来
（计算的 bug，报给人改代码，不自动改）。

哪些结过的预期进校准表（``calibration_sample``）、哪些算一次落空（``counts_as_miss``）、一次落空身上的标志怎么从账、树、边的位置
读出来（``miss_facts``）也在这里；组合根（``runtime/foresight_settlement.py``）只取材料——账上现在算数的那笔、窗里的行、依据那几条
在树上的样子、落空那一刻边的位置。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol

from habitus.foresight.clock import is_opportunity
from habitus.foresight.facts import slice_of
from habitus.foresight.ladder import Standing
from habitus.foresight.ledger.model import NATURAL_HITS, VOID, AccountFact, Expectation, relation_backed
from habitus.foresight.ledger.settle import NOT_DONE
from habitus.foresight.reliability import Reliability, estimate

#: 诊断看最近几天结过的预期；重建"落空那一刻的账"时再往前看几天（最晚时限在那一刻之后的预期那时还没结出来）。
#: 两者之和是诊断够得着的最早一天：账本压实只压它之前的（``runtime.foresight_compaction``）。
DIAGNOSIS_DAYS = 30
DEADLINE_LOOKBACK_DAYS = 30
DIAGNOSIS_REACH_DAYS = DIAGNOSIS_DAYS + DEADLINE_LOOKBACK_DAYS

VOIDED = "前因不成立"
POSTPONED = "推迟"
UNSEEN = "他可能做了、我们没看到"
ABANDONED = "开始了但放弃了"
BROKEN = "规律断了"
BUDGET = "预算内"
RELATION_GONE = "关系不成立了"
RECLASSIFIED = "归类变了"
ANTECEDENT_ABSENT = "疑点：惯常的前因没来"
DAY_TYPE = "疑点：日型不同"
NOTICE = "疑点：由外部通知发起"
UNEXPLAINED = "说不清"
CAUSES = (
    VOIDED,
    UNSEEN,
    ABANDONED,
    POSTPONED,
    BROKEN,
    RELATION_GONE,
    RECLASSIFIED,
    ANTECEDENT_ABSENT,
    DAY_TYPE,
    NOTICE,
    UNEXPLAINED,
)


@dataclass(frozen=True)
class MissFacts:
    """一次落空（到最晚时限没发生）身上能读到的事实。``spoken`` = 这一次说出口了（说错的提醒）；影子里的落空一样诊断。"""

    expectation_id: str
    account: str
    kind_token: str
    day: date
    slice: str
    spoken: bool
    voided: bool = False
    unseen: bool = False
    abandoned: bool = False
    broken: bool = False
    within_budget: bool = False
    relation_gone: bool = False
    reclassified: bool = False
    antecedent_absent: bool = False
    day_type_differs: bool = False
    notice_driven: bool = False
    #: 之后补做了（这次落空现在算数的那条是「后补」）。
    postponed: bool = False


def cause_of(miss: MissFacts) -> str:
    """按顺序过关，第一个成立的就是它。"""

    for flag, cause in (
        (miss.voided, VOIDED),
        (miss.unseen, UNSEEN),
        (miss.abandoned, ABANDONED),
        (miss.postponed, POSTPONED),
        (miss.broken, BROKEN),
        (miss.relation_gone, RELATION_GONE),
        (miss.reclassified, RECLASSIFIED),
        (miss.antecedent_absent, ANTECEDENT_ABSENT),
        (miss.day_type_differs, DAY_TYPE),
        (miss.notice_driven, NOTICE),
    ):
        if flag:
            return cause
    return UNEXPLAINED


@dataclass(frozen=True)
class Settled:
    """校准表的一行原料：一条结过的预期——它在哪个切片、触发当时算出的命中率、这一次自然地成没成。"""

    slice: str
    implementation: str
    claimed: float
    hit: bool


@dataclass(frozen=True)
class CalibrationRow:
    slice: str
    implementation: str
    expectations: int
    claimed: float
    realized: Reliability
    off: bool


@dataclass(frozen=True)
class Diagnosis:
    misses: tuple[tuple[MissFacts, str], ...]
    calibration: tuple[CalibrationRow, ...]

    def counts(self, *, spoken_only: bool = False) -> Mapping[str, int]:
        found: dict[str, int] = dict.fromkeys(CAUSES, 0)
        for miss, cause in self.misses:
            if miss.spoken or not spoken_only:
                found[cause] += 1
        return found

    def within_budget(self, *, spoken_only: bool = False) -> int:
        """标了预算内的落空有几次（标注，不是一关）。"""

        return sum(1 for miss, _cause in self.misses if miss.within_budget and (miss.spoken or not spoken_only))


def calibration(rows: Iterable[Settled], *, alpha: float) -> tuple[CalibrationRow, ...]:
    """按（切片, 实现版本）把"算出的"与"实际的"并排；算出的落在实际的 1 − 2α 区间外就标出来（样本少时区间宽，标不出来）。"""

    groups: dict[tuple[str, str], list[Settled]] = {}
    for row in rows:
        groups.setdefault((row.slice, row.implementation), []).append(row)
    found = []
    for (name, implementation), items in sorted(groups.items()):
        claimed = sum(item.claimed for item in items) / len(items)
        realized = estimate(((1.0, item.hit) for item in items), alpha=alpha)
        found.append(
            CalibrationRow(
                slice=name,
                implementation=implementation,
                expectations=len(items),
                claimed=claimed,
                realized=realized,
                off=not realized.lower <= claimed <= realized.upper,
            )
        )
    return tuple(found)


class WindowRow(Protocol):
    """这条预期说中的范围里树上的一行（看他是不是开始了又放弃了）。"""

    @property
    def kind_token(self) -> str: ...

    @property
    def status(self) -> str | None: ...

    @property
    def at(self) -> datetime: ...


@dataclass(frozen=True)
class BasisSeen:
    """依据里的一次历史发生在树上的样子：那一天日历的说法、谁先发起的（行为树 ``prompted_by``）。"""

    day_note: str | None
    prompted_by: str | None


def calibration_sample(item: Expectation, fact: AccountFact) -> Settled | None:
    """这条结过的预期进不进校准表：按账上现在算数的那笔（重结、后补都改的是它，不按当初的结算记录）。不算一次机会的
    （作废、观测空白，``clock.is_opportunity``）、提醒后才做的、触发时没有历史机会可比的都不进。"""

    if not is_opportunity(fact) or fact.kind == "提醒后发生" or item.opportunities <= 0:
        return None
    return Settled(slice_of(item), item.implementation, item.hits / item.opportunities, fact.kind in NATURAL_HITS)


def counts_as_miss(fact: AccountFact, outcome: str) -> bool:
    """这条结过的预期算不算一次落空：现在算数的那笔是过期待决或后补，或者结成过期待决、后来因前因不成立作废了。"""

    return fact.kind in ("过期待决", "后补") or (fact.kind == VOID and outcome == "过期待决")


def miss_facts(
    item: Expectation,
    fact: AccountFact,
    *,
    spoken: bool,
    standing: Standing,
    current: Callable[[str], Sequence[str]],
    window: Iterable[WindowRow],
    today_note: str | None,
    basis: Iterable[BasisSeen],
    listed: set[str] | None,
) -> MissFacts:
    """一次落空身上的事实。``fact`` 是账上现在算数的那笔；``standing`` 是落空那一刻（最晚时限）这条边的位置；``window`` 是起算
    那天到最晚时限那天树上的行；``today_note`` 是预期那一天日历的说法，``basis`` 是依据里树上还找得到的那几次；``listed`` 是
    机会表里现在还有的关系（没有机会表是 None：说不清，不判关系没了）。"""

    voided = fact.kind == VOID
    kinds = {item.kind_token, *current(item.kind_token)}
    seen = tuple(basis)
    notes = {found.day_note for found in seen}
    return MissFacts(
        expectation_id=item.expectation_id,
        account=item.account,
        kind_token=item.kind_token,
        day=item.day,
        slice=slice_of(item),
        spoken=spoken,
        voided=voided,
        # 观测空白结出的强度 0 的过期待决：与「一次机会」同一口径（``clock.is_opportunity``），作废另算
        unseen=not voided and not is_opportunity(fact),
        abandoned=any(
            row.kind_token in kinds and row.status in NOT_DONE and item.covers(row.at.astimezone(UTC)) for row in window
        ),
        broken=standing.break_at is not None and item.formed_at >= standing.break_at,
        within_budget=standing.prediction.lower >= standing.threshold,
        relation_gone=relation_backed(item.source) and listed is not None and item.trigger not in listed,
        reclassified=tuple(current(item.kind_token)) != (item.kind_token,),
        antecedent_absent=any(a.upward and not a.present for a in item.antecedents),
        day_type_differs=today_note is not None and bool(notes) and today_note not in notes,
        notice_driven=any(found.prompted_by == "host_agent" for found in seen),
        postponed=fact.kind == "后补",
    )


def diagnose(misses: Iterable[MissFacts], settled: Iterable[Settled], *, alpha: float) -> Diagnosis:
    return Diagnosis(
        misses=tuple((miss, cause_of(miss)) for miss in misses),
        calibration=calibration(settled, alpha=alpha),
    )


__all__ = [
    "CAUSES",
    "DEADLINE_LOOKBACK_DAYS",
    "DIAGNOSIS_DAYS",
    "DIAGNOSIS_REACH_DAYS",
    "BasisSeen",
    "CalibrationRow",
    "Diagnosis",
    "MissFacts",
    "Settled",
    "WindowRow",
    "calibration",
    "calibration_sample",
    "cause_of",
    "counts_as_miss",
    "diagnose",
    "miss_facts",
]
