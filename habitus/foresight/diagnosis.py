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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

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


def diagnose(misses: Iterable[MissFacts], settled: Iterable[Settled], *, alpha: float) -> Diagnosis:
    return Diagnosis(
        misses=tuple((miss, cause_of(miss)) for miss in misses),
        calibration=calibration(settled, alpha=alpha),
    )


__all__ = [
    "CAUSES",
    "CalibrationRow",
    "Diagnosis",
    "MissFacts",
    "Settled",
    "calibration",
    "cause_of",
    "diagnose",
]
