"""定期拆改（设计 四、裁定 17）：模型只提差分 → 证据规则 → 重判只在封闭集合里 → 锚点自检 → 冷却期与改动上限。

放行全用用户自己的数据：
- 拆分：切出的两边都跨天反复出现、提醒句不同（``evidence.split_shortfall``）；
- 合并：同一对在不同的拆改周期里都被提出（``JobState.merge_supported``）；
- 每一处改动都过锚点自检（``anchors.AnchorGate``）。

一次拆改的全部改动合成"下一版"交出去（一份事实）；改树、写日志由归约侧执行。没有被接受的改动时不出版本，
但仍更新定时状态（上次拆改时刻、本周期提过的合并）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller, KindModelOutputError
from habitus.behavior.kinds.changes import (
    AddClass,
    ChangeReason,
    Move,
    Operation,
    RetireClass,
    ReviseClass,
    VersionRecord,
    apply,
    merge_operations,
    split_operations,
)
from habitus.behavior.kinds.classify.classifier import DaytimeClassifier
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, ClassOrigin, Vocabulary
from habitus.behavior.kinds.revision.anchors import AnchorGate
from habitus.behavior.kinds.revision.evidence import split_shortfall
from habitus.behavior.kinds.revision.facts import Member, members_of
from habitus.behavior.kinds.revision.prompt import NEW, REVISION_SYSTEM_PROMPT, render_user_message, revision_schema
from habitus.behavior.kinds.revision.proposals import (
    MergeProposal,
    Proposal,
    ReviseProposal,
    SplitProposal,
    parse_proposals,
)
from habitus.behavior.kinds.revision.rejudge import rejudge
from habitus.behavior.kinds.schedule import JobState, MergePair


@dataclass(frozen=True)
class RevisionResult:
    record: VersionRecord | None
    state: JobState
    model_calls: int
    signals: tuple[str, ...]


@dataclass(frozen=True)
class _Step:
    operations: tuple[Operation, ...]
    moves: tuple[Move, ...]
    touched: frozenset[ClassId]


@dataclass
class _Run:
    """一次拆改的上下文：原词表、成员、定时状态、本周期、冷却中的类、续约钩子，以及一路累积的结果。"""

    before: Vocabulary
    members: Sequence[Member]
    state: JobState
    period: str
    frozen: frozenset[ClassId]
    now: datetime
    checkpoint: Callable[[], object]
    working: Vocabulary
    steps: list[_Step] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    calls: int = 0

    @property
    def taken(self) -> frozenset[ClassId]:
        return frozenset().union(*(step.touched for step in self.steps)) if self.steps else frozenset()


class Reviser:
    def __init__(self, caller: KindModelCaller, anchors: AnchorGate) -> None:
        if not isinstance(caller, KindModelCaller):
            raise TypeError("caller must be KindModelCaller")
        if not isinstance(anchors, AnchorGate):
            raise TypeError("anchors must be AnchorGate")
        self.caller = caller
        self.anchors = anchors
        self.classifier: DaytimeClassifier = anchors.classifier
        self.config: BehaviorKindConfig = caller.config

    async def revise(
        self,
        vocabulary: Vocabulary,
        members: Sequence[Member],
        records: Sequence[VersionRecord],
        state: JobState,
        *,
        now: datetime,
        period: str,
        checkpoint: Callable[[], object],
    ) -> RevisionResult:
        frozen = cooling_classes(records, now=now, days=self.config.cooldown_days)
        run = _Run(vocabulary, members, state, period, frozen, now, checkpoint, vocabulary)
        proposed_merges: set[MergePair] = set()
        cap = self.config.max_changes_per_revision
        for lane in Lane:
            if not vocabulary.active(lane):
                continue
            if len(run.steps) >= cap:
                run.signals.append(f"kind_revision_cap_reached {cap}")
                break
            proposals = await self._propose(lane, run)
            proposed_merges |= {(str(p.source), str(p.target)) for p in proposals if isinstance(p, MergeProposal)}
            for proposal in proposals:
                if len(run.steps) >= cap:
                    run.signals.append(f"kind_revision_cap_reached {cap}")
                    break
                accepted = await self._consider(proposal, lane, run)
                if accepted is not None:
                    run.steps.append(accepted)
                    run.working = _applied(run.working, accepted.operations, now)
        next_state = state.after_revision(now, period, frozenset(proposed_merges), keep=self.config.merge_evidence_runs)
        if not run.steps:
            return RevisionResult(None, next_state, run.calls, tuple(run.signals))
        record = VersionRecord(
            vocabulary.version + 1,
            now,
            ChangeReason.REVISION,
            tuple(operation for step in run.steps for operation in step.operations),
            tuple(move for step in run.steps for move in step.moves),
            note=f"定期拆改 {len(run.steps)} 处",
        )
        return RevisionResult(record, next_state, run.calls, tuple(run.signals))

    async def _consider(self, proposal: Proposal, lane: Lane, run: _Run) -> _Step | None:
        label = f"{type(proposal).__name__} {proposal.source}"
        if proposal.touched & (run.frozen | run.taken):
            run.signals.append(f"kind_revision_skipped {label}: class is cooling down or already changed this run")
            return None
        if isinstance(proposal, MergeProposal):
            pair = (str(proposal.source), str(proposal.target))
            if not run.state.merge_supported(pair, runs=self.config.merge_evidence_runs, period=run.period):
                run.signals.append(
                    f"kind_revision_merge_waiting {pair[0]}->{pair[1]}: needs proposals in earlier periods"
                )
                return None
        try:
            step = await self._build(proposal, run)
            candidate = _applied(run.working, step.operations, run.now)  # 撞名、指向不存在的类等在这里暴露：整条丢掉
        except BehaviorKindError as exc:
            run.signals.append(f"kind_revision_invalid {label}: {exc}")
            return None
        if step.touched & (run.frozen | run.taken):
            run.signals.append(f"kind_revision_skipped {label}: moves entries into a cooling or already changed class")
            return None
        check = await self.anchors.check(
            lane, run.before, candidate, step.touched | run.taken, run.members, checkpoint=run.checkpoint
        )
        run.calls += check.model_calls
        if not check.passed:
            run.signals.append(f"kind_revision_anchor_rejected {label}: {check.describe()}")
            return None
        run.signals.append(f"kind_revision_accepted {label}: {proposal.reason} ({check.describe()})")
        return step

    async def _build(self, proposal: Proposal, run: _Run) -> _Step:
        own = members_of(run.members, str(proposal.source))
        if isinstance(proposal, MergeProposal):
            moves = tuple(Move(m.occurrence, m.token, str(proposal.target)) for m in own)
            return _Step(merge_operations(proposal.source, proposal.target), moves, proposal.touched)
        if isinstance(proposal, SplitProposal):
            return await self._build_split(proposal, own, run)
        assert isinstance(proposal, ReviseProposal)
        revised = proposal.revised.to_class(proposal.source, origin=run.working.get(proposal.source).origin)
        neighbours = [run.working.get(e.goes_to) for e in revised.excludes if e.goes_to is not None]
        closed: tuple[BehaviorClass, ...] = (revised, *(item for item in neighbours if item.active))
        moves, result = await rejudge(
            own, closed, keep=proposal.source, classifier=self.classifier, checkpoint=run.checkpoint
        )
        run.calls += result.model_calls
        touched = proposal.touched | {ClassId.parse(m.target) for m in moves}
        return _Step((ReviseClass(revised),), moves, frozenset(touched))

    async def _build_split(self, proposal: SplitProposal, own: Sequence[Member], run: _Run) -> _Step:
        new_id = run.working.next_id(proposal.source.lane)
        if any(target == NEW for _, target in proposal.new.excludes):
            raise BehaviorKindError("a split-out class cannot exclude itself")
        new = proposal.new.to_class(new_id, origin=ClassOrigin.SPLIT)
        origin = run.working.get(proposal.source).origin
        narrowed = proposal.narrowed.to_class(proposal.source, origin=origin, new_id=new_id)
        moves, result = await rejudge(
            own, (narrowed, new), keep=proposal.source, classifier=self.classifier, checkpoint=run.checkpoint
        )
        run.calls += result.model_calls
        moved = {move.occurrence for move in moves if move.target == str(new_id)}
        shortfall = split_shortfall(
            new,
            narrowed,
            [m for m in own if m.occurrence in moved],
            [m for m in own if m.occurrence not in moved],
            min_count=self.config.recurrence_min_count,
            min_days=self.config.recurrence_min_days,
        )
        if shortfall is not None:
            raise BehaviorKindError(f"split lacks evidence: {shortfall}")
        return _Step(split_operations(narrowed, new), moves, proposal.touched | {new_id})

    async def _propose(self, lane: Lane, run: _Run) -> tuple[Proposal, ...]:
        classes = {f"C{i}": item for i, item in enumerate(run.before.active(lane), start=1)}
        counts = {key: _count(members_of(run.members, str(item.id))) for key, item in classes.items()}
        samples = {
            key: _spread(members_of(run.members, str(item.id)), self.config.revision_samples_per_class)
            for key, item in classes.items()
        }
        run.calls += 1
        try:
            raw: list[dict[str, Any]] = await self.caller.call(
                system=REVISION_SYSTEM_PROMPT,
                user=render_user_message(classes, counts, samples, max_steps=self.config.sample_max_steps),
                schema=revision_schema(tuple(classes), max_examples=self.config.max_examples),
                name="behavior_kind_revision",
                validator=_validated,
            )
        except KindModelOutputError as exc:
            run.signals.append(f"kind_revision_rejected {lane.value}: {exc}")
            return ()
        proposals, dropped = parse_proposals(raw, classes, max_examples=self.config.max_examples)
        run.signals.extend(dropped)
        return proposals


def cooling_classes(records: Sequence[VersionRecord], *, now: datetime, days: int) -> frozenset[ClassId]:
    """最近 ``days`` 天里被每晚新增或定期拆改碰过的类。"""

    since = now - timedelta(days=days)
    touched: set[ClassId] = set()
    for record in records:
        if record.at < since:
            continue
        for operation in record.operations:
            if isinstance(operation, AddClass | ReviseClass):
                touched.add(operation.item.id)
            elif isinstance(operation, RetireClass):
                touched.add(operation.class_id)
            else:
                touched.update((operation.source, operation.target))
    return frozenset(touched)


def _applied(vocabulary: Vocabulary, operations: Sequence[Operation], at: datetime) -> Vocabulary:
    return apply(vocabulary, VersionRecord(vocabulary.version + 1, at, ChangeReason.REVISION, tuple(operations)))


def _count(members: Sequence[Member]) -> tuple[int, int]:
    return len(members), len({member.day for member in members})


def _spread(members: Sequence[Member], limit: int) -> tuple[Member, ...]:
    """按日子与地址排好，均匀取 ``limit`` 条（确定性，覆盖整段时间而不是只取开头）。"""

    ordered = sorted(members, key=lambda member: (member.day, member.occurrence))
    if len(ordered) <= limit:
        return tuple(ordered)
    step = len(ordered) / limit
    return tuple(ordered[int(index * step)] for index in range(limit))


def _validated(parsed: object) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict) or set(parsed) != {"proposals"} or not isinstance(parsed["proposals"], list):
        raise BehaviorKindError("revision output must be an object with a `proposals` list")
    if not all(isinstance(item, dict) for item in parsed["proposals"]):
        raise BehaviorKindError("revision proposals must be objects")
    return list(parsed["proposals"])


__all__ = ["Reviser", "RevisionResult", "cooling_classes"]
