"""定期拆改（设计 四、裁定 17；拆分一条 2026-10-12 按 `18` 第四节改）：

拆分：**模型分组、算法定拆**——每个在用类取近期窗口内的成员交给模型按"同一件事"分组（``grouping``），
每组数条数、跨天数；够复现门槛、又不是整个类的组各成一个新类，剩下的留在收窄的原类。模型从不被问"该不该拆"
（v2 由模型按语言规则提拆分，35 晚 4 次提案 0 次拆分）。
合并 / 改判据：仍由模型看各类摘要与抽样提差分（``prompt``）。
之后对每一处改动一样：重判只在封闭集合里 → 证据规则复核 → 锚点自检 → 冷却期与改动上限。

放行全用用户自己的数据：
- 拆分：每个切出的组和留下的那边都跨天反复出现、提醒句不同（``evidence.split_shortfall``）；
- 合并：同一对在不同的拆改周期里都被提出（``JobState.merge_supported``）；
- 每一处改动都过锚点自检（``anchors.AnchorGate``）。

一次拆改的全部改动合成"下一版"交出去（一份事实）；改树、写日志由归约侧执行。没有被接受的改动时不出版本，
但仍更新定时状态（上次拆改时刻、本周期提过的合并）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller, KindModelOutputError
from habitus.behavior.kinds.changes import (
    AddClass,
    Branch,
    ChangeReason,
    Move,
    Operation,
    RetireClass,
    ReviseClass,
    VersionRecord,
    apply,
    merge_operations,
)
from habitus.behavior.kinds.classify.classifier import DaytimeClassifier
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, ClassOrigin, Exclusion, Vocabulary
from habitus.behavior.kinds.revision.anchors import AnchorGate
from habitus.behavior.kinds.revision.evidence import split_shortfall
from habitus.behavior.kinds.revision.facts import Member, members_of
from habitus.behavior.kinds.revision.grouping import Group, group_members, sentence_key
from habitus.behavior.kinds.revision.prompt import NEW, REVISION_SYSTEM_PROMPT, render_user_message, revision_schema
from habitus.behavior.kinds.revision.proposals import (
    Draft,
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
            await self._split_by_grouping(lane, run)
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
        # 基线是本轮已采纳改动之后的词表（``run.working``），不是开跑时的：锚点自检量的是"这一处改动"拉走了谁
        check = await self.anchors.check(
            lane, run.working, candidate, step.touched | run.taken, run.members, checkpoint=run.checkpoint
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

    async def _split_by_grouping(self, lane: Lane, run: _Run) -> None:
        """拆分：对每个在用类的近期成员分组，按复现定哪些组切出去（`18` 4.1 第 1–3 步），再走 ``_consider`` 的后半段。"""

        since = (run.now - timedelta(days=self.config.revision_window_days)).date()
        active = run.before.active(lane)
        for item in active:
            if len(run.steps) >= self.config.max_changes_per_revision:
                return
            if item.id in run.frozen or item.id in run.taken:
                run.signals.append(
                    f"kind_revision_split_skipped {item.id}: class is cooling down or already changed this run"
                )
                continue
            recent = [member for member in members_of(run.members, str(item.id)) if member.day >= since]
            if len(recent) < self.config.grouping_floor:
                continue
            groups, calls, signals = await group_members(
                self.caller,
                item,
                active,
                recent,
                batch_size=self.config.grouping_batch_size,
                max_steps=self.config.sample_max_steps,
                max_examples=self.config.max_examples,
            )
            run.calls += calls
            run.signals.extend(signals)
            proposal = self._split_from_groups(item, groups, run)
            if proposal is None:
                continue
            accepted = await self._consider(proposal, lane, run)
            if accepted is not None:
                run.steps.append(accepted)
                run.working = _applied(run.working, accepted.operations, run.now)

    def _split_from_groups(self, item: BehaviorClass, groups: Sequence[Group], run: _Run) -> SplitProposal | None:
        """够复现门槛的组各成一个新类；分组说"就是已有的某一类"的组搬过去（目标冷却中或本轮已改过就先留下）。原类留给谁：

        - 有一组的提醒句就是原类的提醒句 → 它留在原类（原类条目换成它刷新过的）；
        - 没有，而且不够格的零散记录很少（少于一个类的复现数）→ 最大的够格组留在原类；
        - 零散记录多 → 原类条目不动、当零散记录的桶，够格的组全部切出去——不把几十条一次性的事改名成某一组的名字。
        要切出去、搬出去的组一个都没有就不拆。
        """

        if not groups:
            return None
        movable: list[Group] = []
        for group in groups:
            if group.into is None:
                continue
            if group.into in run.frozen or group.into in run.taken:
                run.signals.append(
                    f"kind_revision_split_blocked {item.id}: {group.count} entries belong to {group.into}, which is cooling down or already changed this run"
                )
            else:
                movable.append(group)
        fresh = [group for group in groups if group.into is None]
        if len(groups) == 1 and not movable:
            run.signals.append(f"kind_revision_split_none {item.id}: one group")
            return None
        recurring = sorted(
            (
                group
                for group in fresh
                if group.recurs(min_count=self.config.recurrence_min_count, min_days=self.config.recurrence_min_days)
            ),
            key=lambda group: (-group.count, -group.days, group.draft.name),
        )
        shape = "、".join(
            f"{group.draft.name} {group.count} 条跨 {group.days} 天" + (f"（→ {group.into}）" if group.into else "")
            for group in groups
        )
        if not recurring and not movable:
            run.signals.append(f"kind_revision_split_waiting {item.id}: {len(groups)} groups, 0 recur ({shape})")
            return None
        parked = sum(
            group.count for group in groups if group.into is not None and group not in movable
        )  # 目标冷却中，先留下、不算零散
        leaving = sum(group.count for group in recurring) + sum(group.count for group in movable)
        residual = sum(group.count for group in groups) - leaving - parked
        own = [
            group
            for group in recurring
            if sentence_key(group.draft.reminder) == sentence_key(item.reminder) or group.draft.name == item.name
        ]
        if own:
            keep, narrowed = own[0], own[0].draft
        elif recurring and residual < self.config.recurrence_min_count:
            keep, narrowed = recurring[0], recurring[0].draft
        else:
            keep, narrowed = None, _draft_of(item)
        news = [group for group in recurring if group is not keep]
        if not news and not movable:
            run.signals.append(
                f"kind_revision_split_waiting {item.id}: {len(groups)} groups, 1 recur, {residual} scattered ({shape})"
            )
            return None
        kept = keep.draft.name if keep is not None else f"原条目（{residual} 条零散记录）"
        parts = [f"{group.draft.name} {group.count} 条跨 {group.days} 天" for group in news]
        parts += [f"{group.count} 条搬去已有的 {group.into}" for group in movable]
        reason = "按分组与复现：" + "；".join(parts) + f"；留在原类的是 {kept}"
        return SplitProposal(
            item.id,
            narrowed,
            tuple(group.draft for group in news),
            reason,
            into=tuple(group.into for group in movable if group.into),
        )

    async def _build_split(self, proposal: SplitProposal, own: Sequence[Member], run: _Run) -> _Step:
        lane = proposal.source.lane
        first = run.working.next_id(lane).number
        new_ids = tuple(ClassId(lane, first + offset) for offset in range(len(proposal.new)))
        if any(target == NEW for draft in proposal.new for _, target in draft.excludes):
            raise BehaviorKindError("a split-out class cannot exclude itself")
        news = tuple(
            draft.to_class(new_id, origin=ClassOrigin.SPLIT)
            for draft, new_id in zip(proposal.new, new_ids, strict=True)
        )
        original = run.working.get(proposal.source)
        targets = tuple(run.working.get(class_id) for class_id in proposal.into)  # 搬去的已有类：进封闭集合，不另建
        narrowed = proposal.narrowed.to_class(proposal.source, origin=original.origin)
        pointers = tuple(Exclusion(item.name, item.id) for item in (*news, *targets))
        narrowed = replace(narrowed, excludes=_merged_excludes(pointers, narrowed.excludes, original.excludes))
        moves, result = await rejudge(
            own,
            (narrowed, *news, *targets),
            keep=proposal.source,
            classifier=self.classifier,
            checkpoint=run.checkpoint,
        )
        run.calls += result.model_calls
        moved_to = {str(new.id): {move.occurrence for move in moves if move.target == str(new.id)} for new in news}
        moved_anywhere = set().union(*moved_to.values()) if moved_to else set()
        kept = [member for member in own if member.occurrence not in moved_anywhere]
        for new in news:
            shortfall = split_shortfall(
                new,
                narrowed,
                [member for member in own if member.occurrence in moved_to[str(new.id)]],
                kept,
                min_count=self.config.recurrence_min_count,
                min_days=self.config.recurrence_min_days,
            )
            if shortfall is not None:
                raise BehaviorKindError(f"split lacks evidence: {shortfall}")
        operations: list[Operation] = []
        for new in news:
            operations += (AddClass(new), Branch(proposal.source, new.id))
        operations += (Branch(proposal.source, target.id) for target in targets)
        operations.append(ReviseClass(narrowed))
        return _Step(tuple(operations), moves, proposal.touched | set(new_ids))

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


def _draft_of(item: BehaviorClass) -> Draft:
    """原类条目原样写成草稿（原类当零散记录的桶时用）。"""

    return Draft(
        name=item.name,
        criterion=item.criterion,
        reminder=item.reminder,
        excludes=tuple((exclusion.text, exclusion.goes_to) for exclusion in item.excludes),
        examples=item.examples,
    )


def _merged_excludes(*lists: Sequence[Exclusion]) -> tuple[Exclusion, ...]:
    """按「不含」的文字去重，先来的保留去向：指向本次新类的 → 分组给的 → 原类自己的。"""

    found: dict[str, Exclusion] = {}
    for items in lists:
        for item in items:
            found.setdefault(item.text, item)
    return tuple(found.values())


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
