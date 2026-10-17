"""预测错了的诊断（裁定 54、55）：每一次落空先判"和我们的数据有没有关系"，有关的指出是哪一层，无关的老实记"说不清"。纯函数、只读。

诊断从不往树上写、不改账：账上的改判在结算时已经做了（前因不成立 → 作废，窗口里有观测空白 → 中性），原因在机会表那一拍已经
对照答案找过、挂在那次落空上（``foresight.causes``）。这里只把已有字段装成一次落空的"事实"（``MissFacts``），按顺序过关：

1. 前因不成立（结算已记作废）；2. 他可能做了、我们没看到（窗里有观测空白，或回了「已做过」）；3. 他开始了但放弃了；
4. 对照答案找到的原因（推迟 = 后补；同一件事、线索没了、别的起因、分支有条件 / 条件不够、还没进关联、没看到下一步）；
5. 规律断了（还没找过原因的：账在那一刻判出断点、这次在断点之后）；
6. 树上已成立的原因：关系已不在机会表里（失效 / 前向未复现）、这一类在词表里已拆改；
7. 疑点（只报不改）：惯常的前因这次没来、今天的日型和历史那几次都不一样、历史那几次里有宿主助手先通知的；
8. 都不是：说不清，与我们的数据无关。

「预算内」（这次落空之后账的下界仍在线上：配置允许的那一份误提醒）只是标注，不截断找原因（裁定 55 第 6 条），单独计数。
另给三张表：替代表——每本账"落空后接着做 X"与"命中后接着做 X"的比例，单侧精确检验（Fisher）p < α 才标出来（"B 没来时他常接着做 X"：
共同原因或替代；它的前因是"B 没来"，不在行为树上，语义树不检验，所以在这里攒，裁定 55 第 3 条 ⑤）；校准表——按（触发种类 × 实现版本）切片，比"触发算出的命中率"与"账上实际结出的"，算出的落在实际的区间外就标出来
（计算的 bug，报给人改代码，不自动改）；调整表——每种原因被用来"今天又在、不说"了几次、其中他果然没做的几次（下一次的答案验这次调整）。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date

from habitus.foresight.ledger.model import CAUSE_KINDS
from habitus.foresight.reliability import Reliability, estimate

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
#: 对照答案找到的原因（与 ``ledger.model.CAUSE_KINDS`` 同序，外加推迟）。
FOUND = (POSTPONED, *CAUSE_KINDS)
CAUSES = (
    VOIDED,
    UNSEEN,
    ABANDONED,
    *FOUND,
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
    #: 对照答案找到的原因种类（``FOUND`` 之一）；还没找过是 None。
    found: str | None = None


def cause_of(miss: MissFacts) -> str:
    """按顺序过关，第一个成立的就是它。"""

    for flag, cause in (
        (miss.voided, VOIDED),
        (miss.unseen, UNSEEN),
        (miss.abandoned, ABANDONED),
    ):
        if flag:
            return cause
    if miss.found is not None:
        return miss.found
    for flag, cause in (
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
class Adjustment:
    """一种原因被用来"上次的原因今天又在、不说"了几次（``held``），其中他果然没做的几次（``right``）。"""

    held: int = 0
    right: int = 0


@dataclass(frozen=True)
class Followed:
    """替代表的一行原料：一条结过的预期（自然命中或看得见的落空），时窗开始之后他接着做的第一件别的事（没有是 None）。"""

    account: str
    next_kind: str | None
    missed: bool


@dataclass(frozen=True)
class SubstitutionRow:
    """一本账 × 接着做的 X：落空里接着做 X 的次数 / 落空数，命中里的同样两个数，两边的读数（给人看）；``p_value`` 是"落空后比命中后
    更常接着做 X"的单侧精确检验（Fisher）；``flagged`` = p < α。不用两边区间不重叠：Jeffreys 区间在一两次时太松，
    甲盘探针里落空 1 次、命中 8 次就被标了（精确检验 p = 1/9）。"""

    account: str
    next_kind: str
    after_misses: int
    misses: int
    after_hits: int
    hits: int
    when_missed: Reliability
    when_hit: Reliability
    p_value: float
    flagged: bool


@dataclass(frozen=True)
class Diagnosis:
    misses: tuple[tuple[MissFacts, str], ...]
    calibration: tuple[CalibrationRow, ...]
    adjustments: Mapping[str, Adjustment] = field(default_factory=dict)
    substitutions: tuple[SubstitutionRow, ...] = ()

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


def substitutions(rows: Iterable[Followed], *, alpha: float) -> tuple[SubstitutionRow, ...]:
    """按账把"落空后接着做 X"与"命中后接着做 X"并排。只列落空里出现过的 X；样本少时区间宽，标不出来。"""

    by_account: dict[str, list[Followed]] = {}
    for row in rows:
        by_account.setdefault(row.account, []).append(row)
    found = []
    for account, items in sorted(by_account.items()):
        missed = [item for item in items if item.missed]
        hit = [item for item in items if not item.missed]
        for kind in sorted({item.next_kind for item in missed if item.next_kind is not None}):
            after_misses = sum(1 for item in missed if item.next_kind == kind)
            after_hits = sum(1 for item in hit if item.next_kind == kind)
            when_missed = estimate(((1.0, item.next_kind == kind) for item in missed), alpha=alpha)
            when_hit = estimate(((1.0, item.next_kind == kind) for item in hit), alpha=alpha)
            chance = fisher_greater(after_misses, len(missed), after_hits, len(hit))
            found.append(
                SubstitutionRow(
                    account,
                    kind,
                    after_misses,
                    len(missed),
                    after_hits,
                    len(hit),
                    when_missed,
                    when_hit,
                    chance,
                    flagged=bool(hit) and chance < alpha,
                )
            )
    return tuple(found)


def fisher_greater(a: int, n: int, c: int, m: int) -> float:
    """单侧 Fisher 精确检验：两组合起来接着做 X 的有 a + c 次，落空那组（n 次）里至少有 a 次的概率（超几何分布）。"""

    total, hits = n + m, a + c
    if n == 0 or hits == 0:
        return 1.0
    denominator = math.comb(total, hits)
    return sum(math.comb(n, k) * math.comb(m, hits - k) for k in range(a, min(n, hits) + 1)) / denominator


def diagnose(
    misses: Iterable[MissFacts],
    settled: Iterable[Settled],
    *,
    alpha: float,
    held: Iterable[tuple[str, bool]] = (),
    followed: Iterable[Followed] = (),
) -> Diagnosis:
    """``held`` 是每一次"上次的原因今天又在、不说"：（那条原因的种类, 他果然没做）；``followed`` 是替代表的原料。"""

    adjustments: dict[str, Adjustment] = {}
    for kind, right in held:
        before = adjustments.get(kind, Adjustment())
        adjustments[kind] = Adjustment(before.held + 1, before.right + int(right))
    return Diagnosis(
        misses=tuple((miss, cause_of(miss)) for miss in misses),
        calibration=calibration(settled, alpha=alpha),
        adjustments=adjustments,
        substitutions=substitutions(followed, alpha=alpha),
    )


__all__ = [
    "CAUSES",
    "FOUND",
    "Adjustment",
    "Followed",
    "SubstitutionRow",
    "CalibrationRow",
    "Diagnosis",
    "MissFacts",
    "Settled",
    "calibration",
    "cause_of",
    "diagnose",
    "fisher_greater",
    "substitutions",
]
