"""关系的状态逐晚累积（语义树新方案 ``13`` ③）：第 N 晚 = 第 N−1 晚 + 当晚检验（+ 预测层反馈，第 9 步接）。

```
样本不够 / 已检 ──当晚过第 1 道──▶ 候选（记发现的那晚 N₀）
候选 ──N₀ 之后新到的独立 A 攒够按功效定的次数（至少 2 块）：单侧检验──┬─ 过 ─▶ 等第 2、3 道（全部数据）都过 ─▶ 成立
                                                                    └─ 不过 ─▶ 前向未复现（记那晚 R）
成立 ──维持检验：从上一次判过起不重叠地攒满一段（成立那晚按功效算好的次数、至少 2 块），连续两段区间都够不上第 3 道
      ──▶ 失效（记那晚 R）
失效 / 前向未复现 ──只用 R 之后的数据又过第 1 道──▶ 候选（数据变了还能回来）
```

- **又回来要靠新数据**：失效与前向未复现之后，第 1 道只看 R 之后的数据。不然旧数据里那段强关系会让它每晚都"又过第 1 道"、
  每次都在前向验证上被否（合成数据实测：埋 25 天后撤掉的关系，失效后两个月里被"重新发现"了 12 次），而每一次都是又给了
  自己一次碰运气的机会。所以"前向未复现"要跨夜记着，不能退回成"已检"就忘了。
- **候选与成立不进当晚的检验族**：它们各走各的（前向验证、维持检验），成立之后不再进全局 FDR。

- **前向验证只判一次**：在新数据第一次攒够按功效算的次数、跨了不止一个块、并且这几层全中能到 α 的那一晚判（方案："攒够没有，次数与块数
  两维都要够"；真实数据上一个高频前因一天就能来 5 次，那不是 5 个独立样本），过就是过、不过就退回。每晚拿越攒越多的新数据反复验，
  等于给自己很多次机会碰出一晚显著——这正是前向验证要防的事（评审模拟：58 晚里"至少一晚过线"的概率是单晚的 5–12 倍）。
- **成立之后不再进全局检验族**：每晚只做维持检验。失效不数晚数：只是不显著（证据弱）不算失效。
- 关系身份 = （lane, 前因, 后果, 跨度段）；条件（调节）由第 6 步加。几个条件合起来的只长在已成立的带条件关系上
  （裁定 61 第 6 条、62 第 1 条）：这一晚之后少任意一个条件的父关系没有一条还成立的，组合整行清掉（迁移写原因）。
- 成立的关系每晚也判第 2、3 道（``assess``）：``large_enough`` 是预测层分"原因"与"巧合"的依据（裁定 62 第 3 条），
  不能成立之后就一直是 None。
- 样本不够与已检两种不跨夜保留任何东西（每晚从数据重算），只有候选、成立、失效带着历史。
- 当晚测不了的关系（概念从概念集里没了）原样带着，不改状态。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum

from habitus.scene.concepts.model import members_retired
from habitus.scene.relations import stats
from habitus.scene.relations.engine import (
    LaneTests,
    RelationKey,
    RelationTest,
    Subset,
    Verdict,
    gone,
)
from habitus.scene.relations.longspan import MEASURE_VERSIONS

#: 维持检验一段至少几块：一天里的几次不是几个独立样本（与前向验证的 ``blocks ≥ 2`` 同一条理由）。
_MIN_MAINTENANCE_BLOCKS = 2


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
    #: 维持检验每段要攒几次独立的前因（成立那晚按功效算定，之后不变；None = 还没成立）。
    maintenance_needed: int | None = None

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
    #: 这份状态按哪一版词表算的。下一晚据它算"这之间哪些类的成员变了"——不读概念集的同步进度：同一晚重跑时同步进度早已推进，
    #: 读它会让第二趟看不到变化、接回没退回的旧状态，把退回悄悄撤销（2026-10-18 四方评审）。
    vocabulary: int = 0
    #: 各段量法的版本（键是段名）。判据一改只清那一段的行：关系键里不含段的边界，不清的话旧行会换到新判据下的段上
    #: 接着做维持检验（26 号清单 3.5）。没写的段按版本 1。
    measure_versions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(relation.status is Status.SPARSE for relation in self.relations):
            raise StateError("sparse relations are counted, not stored")
        if isinstance(self.vocabulary, bool) or not isinstance(self.vocabulary, int) or self.vocabulary < 0:
            raise StateError("a lane state's vocabulary version is a non-negative integer")
        object.__setattr__(self, "sparse", dict(sorted(self.sparse.items())))
        if any(
            not isinstance(name, str) or isinstance(version, bool) or not isinstance(version, int) or version < 1
            for name, version in self.measure_versions.items()
        ):
            raise StateError("measure versions map segment names to positive integers")
        object.__setattr__(self, "measure_versions", dict(sorted(self.measure_versions.items())))

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
        previous = _drop_stale_measures(previous)
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
        if before is not None and key not in tonight and _expired_row(before, tests):
            # 再也用不上的行清掉（26 号清单 5.6、5.7）；清掉还在用的行时留一条迁移，说清楚为什么
            if before.status in (Status.CANDIDATE, Status.ESTABLISHED):
                transitions.append(
                    Transition(
                        night=night.cutoff,
                        key=key,
                        before=before.status,
                        after=Status.EXPIRED,
                        reason="它依赖的段或条件已经不在了（变点重算后整段没了、条件被拒或消失），这一行清掉",
                    )
                )
            continue
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
    relations, dropped = _drop_orphaned_combinations(relations, night.cutoff)
    state = LaneState(
        lane=night.lane,
        night=night.cutoff,
        family=night.family,
        relations=tuple(relations),
        sparse=sparse,
        measure_versions=MEASURE_VERSIONS,
    )
    return state, (*transitions, *dropped)


def _drop_orphaned_combinations(relations: list[Relation], night: date) -> tuple[list[Relation], list[Transition]]:
    """几个条件合起来的关系，这一晚之后少任意一个条件的父关系没有一条还成立的，整行清掉（组合只长在成立的关系上，
    父退回它跟着退回）。按条件个数从少到多判，三层的组合随两层的一起清。"""

    kept = list(relations)
    dropped: list[Transition] = []
    for size in sorted({len(item.key.conditions) for item in kept if len(item.key.conditions) >= 2}):
        established = {item.key for item in kept if item.status is Status.ESTABLISHED}
        remaining: list[Relation] = []
        for item in kept:
            key = item.key
            if len(key.conditions) == size and not any(parent in established for parent in key.parents()):
                dropped.append(
                    Transition(
                        night=night,
                        key=key,
                        before=item.status,
                        after=Status.EXPIRED,
                        reason="它长在上面的那条关系不再成立，这个组合一起退回、整行清掉",
                    )
                )
                continue
            remaining.append(item)
        kept = remaining
    return kept, dropped


def _drop_stale_measures(previous: LaneState) -> LaneState:
    """某一段的量法改了版本：那一段带着的行清掉，在新判据下从数据重新发现。关系键里不含段的边界，不清的话旧行会
    换到新判据下的段上接着做维持检验（26 号清单 3.5）。其余段原样带着。"""

    stale = {
        segment for segment, version in MEASURE_VERSIONS.items() if previous.measure_versions.get(segment, 1) < version
    }
    if not stale:
        return previous
    kept = tuple(relation for relation in previous.relations if relation.key.segment.value not in stale)
    return replace(previous, relations=kept, measure_versions={**previous.measure_versions, **MEASURE_VERSIONS})


def revert_touched(
    previous: LaneState, touched: frozenset[str], night: date
) -> tuple[LaneState, tuple[Transition, ...]]:
    """词表拆分 / 合并改了某些类的成员（裁定 59）：以这些概念为前因或后果的候选与成立退回候选，发现日记成 ``night``，
    在新成员上重新做前向验证——这期间不开口，影子照记。否则"查看信息"拆出"看日程"之后，"起床 → 查看信息"还靠搬走的那批
    记录成立着、照样开口，要等维持检验连判两段才失效。只改运行中的状态，不迁移任何旧数据。"""

    reverted: list[Relation] = []
    transitions: list[Transition] = []
    for relation in previous.relations:
        key = relation.key
        if relation.status in (Status.CANDIDATE, Status.ESTABLISHED) and ({key.antecedent, key.consequent} & touched):
            reverted.append(
                replace(
                    relation,
                    status=Status.CANDIDATE,
                    discovered_on=night,
                    forward=None,
                    forward_passed_on=None,
                    established_on=None,
                    maintenance=None,
                    maintained_through=None,
                    misses=0,
                )
            )
            transitions.append(
                Transition(
                    night=night,
                    key=key,
                    before=relation.status,
                    after=Status.CANDIDATE,
                    reason="词表拆改改了类的成员，退回候选、在新成员上重新做前向验证",
                )
            )
        else:
            reverted.append(relation)
    return replace(previous, relations=tuple(reverted)), tuple(transitions)


def _step(
    before: Relation | None, test: RelationTest | None, tests: LaneTests, night: date
) -> tuple[Relation, str | None]:
    """一条关系这一晚怎么走；返回（新样子, 迁移原因——不迁移是 None）。"""

    if _retired(test.key if test is not None else before.key if before is not None else None, tests):
        # 词表拆改把前因或后果的类停用了：这条关系随之停用（裁定 27 第 9 条）——不是"效应没了"，新类上要从头发现、前向验证
        if before is not None and before.status in (Status.CANDIDATE, Status.ESTABLISHED):
            return replace(
                before, status=Status.EXPIRED, expired_on=night, maintenance=None
            ), "概念停用（词表拆改），关系随之停用"
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
        minimum=1,
        maximum=thresholds.forward_max_antecedents,
    )


def _maintenance_needed(test: RelationTest, tests: LaneTests) -> int:
    """维持检验每段要攒几次（26 号清单 5.3）：零假设是"效应还像成立那晚那么大"，备择是"回落到对照率"，方向朝回落那一侧；
    功效与 α 与前向验证同一对。成立那晚算好存进 ``Relation.maintenance_needed``，之后每段按它判——不再借固定的块数与前向的次数。
    功效按二项检验算、实际判"没了"用 ``gone()`` 的区间，两者不是同一条规则：模拟测试兜底（``test_relation_power_thresholds``）。"""

    thresholds = tests.config.thresholds
    established = test.effect.treated_rate
    control = test.effect.control_rate
    return stats.required_antecedents(
        established,
        abs(established - control),
        upward=not test.upward,
        alpha=thresholds.forward_alpha,
        power=thresholds.forward_power,
        minimum=1,
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
    needed = before.forward_needed if before.forward_needed is not None else _forward_needed(judged, tests)
    if forward.antecedents < needed or forward.blocks < 2:
        return current, None  # 新数据还不够（次数与块数两维都要够：一天里的几次不是几个独立样本），接着等
    if forward.minimum_p > thresholds.forward_alpha:
        # 次数够了，但每层对照太少，这几层就算全中也到不了 α（例：每层 2 个对照、攒 2 次，全中也只有 (1/3)² ≈ 0.11）：
        # 判了必输，再攒，不判。要攒的次数按"对照很多"的二项算，不知道以后每层有几个对照；匹配设计的样本量本该带每层对照数
        # （L10.15，用户 10-19 定；文献见 27 号调研）。
        return current, None
    if forward.p_value >= thresholds.forward_alpha:
        rejected = replace(current, status=Status.REJECTED, rejected_on=night)
        return rejected, f"前向验证没复现（新数据 {forward.antecedents} 次，p = {forward.p_value:.3g}）"
    if not (judged.stable and judged.large_enough):
        missing = "时间上不稳" if not judged.stable else "影响不够大"
        rejected = replace(current, status=Status.REJECTED, rejected_on=night)
        return rejected, f"前向验证复现（新数据 {forward.antecedents} 次），但{missing}"
    maintenance = _maintenance_needed(judged, tests)
    established = replace(
        current,
        status=Status.ESTABLISHED,
        forward_passed_on=night,
        established_on=night,
        maintained_through=night,
        maintenance_needed=maintenance,
    )
    return (
        established,
        f"前向验证复现（新数据 {forward.antecedents} 次），时间上稳、影响够大（维持检验每段攒 {maintenance} 次）",
    )


def _established(before: Relation, test: RelationTest, tests: LaneTests, night: date) -> tuple[Relation, str | None]:
    """维持检验：上一次判过之后攒的新数据满一段（成立那晚按功效算好的次数、至少 2 块）才判一次；连续两段都说没了才失效
    （裁定 27 第 3 条）。每晚看最近几块等于每晚给它一次被运气判掉的机会（第四轮评审算法 5：效应恒定的真关系 120 晚里最多
    58% 被判失效）。成立在 v4 之前、没存次数的，按这一晚的读数补算一次存上。
    当晚的读数带第 2、3 道（``assess``）：``large_enough`` 要每晚都有（裁定 62 第 3 条），带滞回——昨晚够大的今晚要远端也掉线才
    翻成不够（裁定 63 第 3 条），免得线附近的关系一晚是原因、一晚是巧合，机会表与单元跟着进出。"""

    config = tests.config
    thresholds = config.thresholds
    previously_large = before.tonight.large_enough if before.tonight is not None else None
    judged = tests.assess(replace(test, upward=before.upward), previously_large=previously_large)
    needed = before.maintenance_needed
    if needed is None:
        needed = _maintenance_needed(judged, tests)
        before = replace(before, maintenance_needed=needed)
    window = tests.evaluate(before.key, upward=before.upward, since=before.maintained_through, purpose="maintain")
    current = replace(before, tonight=judged, maintenance=window)
    if window.blocks < _MIN_MAINTENANCE_BLOCKS or window.antecedents < needed:
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


def _expired_row(before: Relation, tests: LaneTests) -> bool:
    """冻住的行（当晚没有读数）还留不留：停用概念的休眠行永远不会再用；依赖的段或条件已经不在的行也不会。只是前因这阵子
    没有新链的照旧冻着；休眠行的 ``reset_on`` 留着，重新显著时从它之后的新数据算。"""

    if before.status.dormant and _retired(before.key, tests):
        return True
    return not tests.still_defined(before.key)


def _retired(key: RelationKey | None, tests: LaneTests) -> bool:
    if key is None:
        return False
    concepts = tests.concepts
    return any(name in concepts and members_retired(concepts, name) for name in (key.antecedent, key.consequent))


def _plain(test: RelationTest | None) -> Status:
    if test is None or test.verdict is Verdict.SPARSE:
        return Status.SPARSE
    return Status.TESTED


__all__ = ["LaneState", "Relation", "StateError", "Status", "Transition", "fold", "revert_touched"]
