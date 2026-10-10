"""关系的状态逐晚累积（语义树新方案 ``13`` ③）：第 N 晚 = 第 N−1 晚 + 当晚检验（+ 预测层反馈，第 9 步接）。

```
样本不够 / 已检 ──当晚过第 1 道──▶ 候选（记发现的那晚 N₀）
候选 ──N₀ 之后新到的独立 A 第一次攒够 5 次：单侧检验──┬─ 过 ─▶ 等第 2、3 道（全部数据）都过 ─▶ 成立
                                                    └─ 不过 ─▶ 前向未复现（记那晚 R）
成立 ──只看最近 8 个块（维持检验）：区间离 0 最远的一端都够不上第 3 道 ──▶ 失效（记那晚 R）
失效 / 前向未复现 ──只用 R 之后的数据又过第 1 道──▶ 候选（数据变了还能回来）
```

- **又回来要靠新数据**：失效与前向未复现之后，第 1 道只看 R 之后的数据。不然旧数据里那段强关系会让它每晚都"又过第 1 道"、
  每次都在前向验证上被否（合成数据实测：埋 25 天后撤掉的关系，失效后两个月里被"重新发现"了 12 次），而每一次都是又给了
  自己一次碰运气的机会。所以"前向未复现"要跨夜记着，不能退回成"已检"就忘了。
- **候选与成立不进当晚的检验族**：它们各走各的（前向验证、维持检验），成立之后不再进全局 FDR。

- **前向验证只判一次**：在新数据第一次攒够 5 次独立的 A、并且跨了不止一个块的那一晚判（方案："攒够没有，次数与块数
  两维都要够"；真实数据上一个高频前因一天就能来 5 次，那不是 5 个独立样本），过就是过、不过就退回。每晚拿越攒越多的新数据反复验，
  等于给自己很多次机会碰出一晚显著——这正是前向验证要防的事（评审模拟：58 晚里"至少一晚过线"的概率是单晚的 5–12 倍）。
- **成立之后不再进全局检验族**：每晚只做维持检验。失效不数晚数：只是不显著（证据弱）不算失效。
- 关系身份 = （lane, 前因, 后果, 跨度段）；条件（调节）由第 6 步加。
- 样本不够与已检两种不跨夜保留任何东西（每晚从数据重算），只有候选、成立、失效带着历史。
- 当晚测不了的关系（概念从概念集里没了）原样带着，不改状态。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum

from habitus.scene.relations import stats
from habitus.scene.relations.engine import (
    LaneTests,
    RelationKey,
    RelationTest,
    Subset,
    Verdict,
    gone,
)


class Status(str, Enum):
    SPARSE = "sparse"
    TESTED = "tested"
    CANDIDATE = "candidate"
    ESTABLISHED = "established"
    EXPIRED = "expired"
    #: 候选在前向验证上没复现。
    REJECTED = "rejected"

    @property
    def carried(self) -> bool:
        """跨夜带着历史的四种。"""

        return self in (Status.CANDIDATE, Status.ESTABLISHED, Status.EXPIRED, Status.REJECTED)

    @property
    def dormant(self) -> bool:
        """失效与前向未复现：要在那之后的新数据上重新显著才回得来。"""

        return self in (Status.EXPIRED, Status.REJECTED)


@dataclass(frozen=True)
class Relation:
    """一条关系在第 ``night`` 晚的样子。``tonight`` 是当晚的检验读数（候选的第 2、3 道按它的方向判过）。"""

    key: RelationKey
    status: Status
    upward: bool
    tonight: RelationTest | None = None
    discovered_on: date | None = None
    forward_passed_on: date | None = None
    established_on: date | None = None
    expired_on: date | None = None
    rejected_on: date | None = None
    #: 只靠先验才过的第 1 道（不加权就过不了）：打标签给人看，回测时单独比（先验有没有用要单变量验）。
    prior_only: bool = False
    #: 前向验证（只在候选上）：发现之后新数据上的读数。
    forward: Subset | None = None
    #: 维持检验（只在成立 / 刚失效的上）：上一次判过之后攒的这一段新数据的读数。
    maintenance: Subset | None = None
    #: 前向验证要攒几次独立的前因（发现那晚按功效算定，之后不变）。
    forward_needed: int | None = None
    #: 维持检验判到哪一晚（下一段只看这之后的新数据）；连续几段说没了。
    maintained_through: date | None = None
    misses: int = 0

    @property
    def reset_on(self) -> date | None:
        """失效 / 前向未复现那晚：重新发现只看这之后的数据。"""

        return self.expired_on if self.status is Status.EXPIRED else self.rejected_on


@dataclass(frozen=True)
class Transition:
    """状态迁移日志的一条：哪条关系哪晚从什么到什么、为什么。"""

    night: date
    key: RelationKey
    before: Status | None
    after: Status
    reason: str


@dataclass(frozen=True)
class LaneState:
    """一条 lane 一晚的关系表。样本不够的只记各段的条数：它们不跨夜带任何东西、每晚从数据重算，逐条存会让一晚的文件
    以兆计（真实会话 lane 带长跨度与条件约 7,800 个检验，九成以上样本不够，一晚 2.8 MB）。"""

    lane: str
    night: date
    family: int
    relations: tuple[Relation, ...]
    #: 样本不够的条数，按跨度段。
    sparse: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(relation.status is Status.SPARSE for relation in self.relations):
            raise StateError("sparse relations are counted, not stored")
        object.__setattr__(self, "sparse", dict(sorted(self.sparse.items())))

    def carried(self) -> dict[RelationKey, Relation]:
        return {relation.key: relation for relation in self.relations if relation.status.carried}

    def of(self, status: Status) -> tuple[Relation, ...]:
        return tuple(relation for relation in self.relations if relation.status is status)

    def count(self, status: Status) -> int:
        return sum(self.sparse.values()) if status is Status.SPARSE else len(self.of(status))


class StateError(ValueError):
    """折叠的顺序不对：派生树按固定先后处理已封口的历史，不往回改。"""


def fold(
    previous: LaneState | None, tests: LaneTests, weights: Mapping[RelationKey, float] | None = None
) -> tuple[LaneState, tuple[Transition, ...]]:
    """第 ``tests.cutoff`` 晚的状态与这一晚的迁移。``previous`` 是这条 lane 之前最近的一晚（没有就是第一晚）。"""

    if previous is not None:
        if previous.lane != tests.line.lane:
            raise StateError("a lane folds onto its own history only")
        if previous.night >= tests.cutoff:
            raise StateError(f"night {tests.cutoff} does not come after the stored night {previous.night}")
    carried: Mapping[RelationKey, Relation] = previous.carried() if previous is not None else {}
    since = {key: relation.reset_on for key, relation in carried.items() if relation.reset_on is not None}
    outside = frozenset(
        key for key, relation in carried.items() if relation.status in (Status.CANDIDATE, Status.ESTABLISHED)
    )
    night = tests.night(weights, since=since, outside=outside)
    # 每晚同时算不加权的：只靠先验才过第 1 道的，发现时打"靠先验"
    plain = night if not weights else tests.night(None, since=since, outside=outside)
    unweighted = frozenset(test.key for test in plain.tests if test.verdict is Verdict.SIGNIFICANT)
    tonight = {test.key: test for test in night.tests}
    relations: list[Relation] = []
    sparse: dict[str, int] = {}
    transitions: list[Transition] = []
    for key in sorted(set(tonight) | set(carried)):
        before = carried.get(key)
        after, reason = _step(before, tonight.get(key), tests, night.cutoff)
        if after.status is Status.CANDIDATE and after.discovered_on == night.cutoff and key not in unweighted:
            after = replace(after, prior_only=True)
            reason = f"{reason}（靠先验：不加权过不了）" if reason else reason
        if after.status is Status.SPARSE:
            sparse[key.segment.value] = sparse.get(key.segment.value, 0) + 1
        else:
            relations.append(after)
        if reason is not None:
            transitions.append(
                Transition(
                    night=night.cutoff,
                    key=key,
                    before=before.status if before is not None else _plain(tonight.get(key)),
                    after=after.status,
                    reason=reason,
                )
            )
    state = LaneState(
        lane=night.lane, night=night.cutoff, family=night.family, relations=tuple(relations), sparse=sparse
    )
    return state, tuple(transitions)


def _step(
    before: Relation | None, test: RelationTest | None, tests: LaneTests, night: date
) -> tuple[Relation, str | None]:
    """一条关系这一晚怎么走；返回（新样子, 迁移原因——不迁移是 None）。"""

    if _retired(test.key if test is not None else before.key if before is not None else None, tests):
        # 词表拆改把前因或后果的类停用了：这条关系随之停用（裁定 27 第 9 条）——不是"效应没了"，新类上要从头发现、前向验证
        if before is not None and before.status in (Status.CANDIDATE, Status.ESTABLISHED):
            return replace(before, status=Status.EXPIRED, expired_on=night, maintenance=None), "概念停用（词表拆改），关系随之停用"
        if before is not None:
            return before, None
    if test is None:
        if before is None:
            raise StateError("a relation with no history must have a test tonight")
        return before, None  # 概念不在了，测不了：原样带着
    if before is None or before.status.dormant:
        if test.verdict is Verdict.SIGNIFICANT:
            reason = "过了第 1 道" if before is None else f"{before.reset_on} 之后的新数据上又过了第 1 道"
            needed = _forward_needed(test, tests)
            return Relation(
                key=test.key,
                status=Status.CANDIDATE,
                upward=test.upward,
                tonight=test,
                discovered_on=night,
                forward_needed=needed,
            ), f"{reason}（前向验证要攒 {needed} 次）"
        if before is not None:
            return replace(before, tonight=test, maintenance=None, forward=None), None
        return Relation(key=test.key, status=_plain(test), upward=test.upward, tonight=test), None
    if before.status is Status.CANDIDATE:
        return _candidate(before, test, tests, night)
    return _established(before, test, tests, night)


def _forward_needed(test: RelationTest, tests: LaneTests) -> int:
    """前向验证要攒几次：按发现那晚效应区间里离 0 近的那一端（没有区间就用点估计）算功效（裁定 27 第 3 条）。
    用离 0 近的一端，是不让发现那晚的运气把要攒的次数算少。"""

    thresholds = tests.config.thresholds
    point = test.effect.treated_rate - test.effect.control_rate
    if test.interval is None:
        near = point if test.upward else -point
    else:
        near = test.interval[0] if test.upward else -test.interval[1]
    return stats.required_antecedents(
        test.effect.control_rate,
        near if near > 0.0 else abs(point),
        upward=test.upward,
        alpha=thresholds.forward_alpha,
        power=thresholds.forward_power,
        minimum=thresholds.forward_min_antecedents,
        maximum=thresholds.forward_max_antecedents,
    )


def _candidate(before: Relation, test: RelationTest, tests: LaneTests, night: date) -> tuple[Relation, str | None]:
    """前向验证、第 2 道、第 3 道**同一晚只判一次**（裁定 27 第 3 条）：新数据攒够那一晚判，过了成立，任何一道没过就退回。
    每晚重判等于给自己很多次碰运气的机会（第四轮评审算法 4：一条关系成立那晚异质性恰好过线，全量数据上 p = 0.004）。"""

    thresholds = tests.config.thresholds
    judged = tests.assess(replace(test, upward=before.upward))
    current = replace(before, tonight=judged)
    assert before.discovered_on is not None
    forward = tests.evaluate(before.key, upward=before.upward, since=before.discovered_on, purpose="forward")
    current = replace(current, forward=forward)
    needed = before.forward_needed or thresholds.forward_min_antecedents
    if forward.antecedents < needed or forward.blocks < 2:
        return current, None  # 新数据还不够（次数与块数两维都要够：一天里的几次不是几个独立样本），接着等
    if forward.p_value >= thresholds.forward_alpha:
        rejected = replace(current, status=Status.REJECTED, rejected_on=night)
        return rejected, f"前向验证没复现（新数据 {forward.antecedents} 次，p = {forward.p_value:.3g}）"
    if not (judged.stable and judged.large_enough):
        missing = "时间上不稳" if not judged.stable else "影响不够大"
        rejected = replace(current, status=Status.REJECTED, rejected_on=night)
        return rejected, f"前向验证复现（新数据 {forward.antecedents} 次），但{missing}"
    established = replace(
        current, status=Status.ESTABLISHED, forward_passed_on=night, established_on=night, maintained_through=night
    )
    return established, f"前向验证复现（新数据 {forward.antecedents} 次），时间上稳、影响够大"


def _established(before: Relation, test: RelationTest, tests: LaneTests, night: date) -> tuple[Relation, str | None]:
    """维持检验：上一次判过之后攒的新数据满一段（``maintenance_blocks`` 个块）才判一次；连续两段都说没了才失效（裁定 27 第 3 条）。
    每晚看最近几块等于每晚给它一次被运气判掉的机会（第四轮评审算法 5：效应恒定的真关系 120 晚里最多 58% 被判失效）。"""

    config = tests.config
    thresholds = config.thresholds
    window = tests.evaluate(before.key, upward=before.upward, since=before.maintained_through, purpose="maintain")
    current = replace(before, tonight=test, maintenance=window)
    if window.blocks < thresholds.maintenance_blocks or window.antecedents < thresholds.forward_min_antecedents:
        return current, None  # 这一段还没攒满
    missed = gone(window.interval, window.relative_interval, upward=before.upward, config=config)
    # 说没了就记一段；这一段单独看仍显著（明确还在）才清零；说不清的（区间宽）不动——不然一段样本少的窗口就把"连续"打断了
    if missed:
        misses = before.misses + 1
    elif window.p_value < thresholds.forward_alpha:
        misses = 0
    else:
        misses = before.misses
    current = replace(current, maintained_through=night, misses=misses)
    if misses >= 2:
        return replace(current, status=Status.EXPIRED, expired_on=night), "连续两段新数据都说它没了"
    return current, ("这一段新数据说它没了，再看下一段" if missed else None)


def _retired(key: RelationKey | None, tests: LaneTests) -> bool:
    if key is None:
        return False
    concepts = tests.concepts
    return any(name in concepts and concepts[name].retired for name in (key.antecedent, key.consequent))


def _plain(test: RelationTest | None) -> Status:
    if test is None or test.verdict is Verdict.SPARSE:
        return Status.SPARSE
    return Status.TESTED


__all__ = ["LaneState", "Relation", "StateError", "Status", "Transition", "fold"]
