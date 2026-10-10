"""每晚只新增（设计 三）：待定池整体交模型聚组 → 复现够了（次数 + 天数）建新类 → 产出"下一版"。

已有的类每晚不动。模型说某一组其实就是已有的某一类时，这组成员交白天归类重判一次（``recheck``）：
白天归类判进某类的迁进那一类，仍不归已有类的放行、照常按复现门槛长新类；每条只重判一次。

产出是一份事实：要追加的那一版（新增的类 + 待定条目改到新编号的迁移）与这一晚重判过的条目。改树、写日志、清池、
给池里条目打"已重判"标记由归约侧执行。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller, KindModelOutputError
from habitus.behavior.kinds.changes import AddClass, ChangeReason, Move, VersionRecord
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import Lane, pending_token
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, ClassOrigin, Exclusion, Vocabulary
from habitus.behavior.kinds.nightly.prompt import NIGHTLY_SYSTEM_PROMPT, nightly_schema, render_user_message
from habitus.behavior.kinds.nightly.recheck import ContentOf, PendingRecheck
from habitus.behavior.kinds.pending import PendingEntry, PendingPool


@dataclass(frozen=True)
class NightlyResult:
    """``record`` 为空表示今晚不新增也不迁移；``rechecked`` 是这一晚重判过、仍留在池里的条目（归约侧给它们打标记）；
    ``model_calls`` 与 ``signals`` 给监控。"""

    record: VersionRecord | None
    model_calls: int
    signals: tuple[str, ...]
    rechecked: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Group:
    members: tuple[PendingEntry, ...]
    draft: dict[str, Any]


class NightlyGrower:
    def __init__(self, caller: KindModelCaller, recheck: PendingRecheck) -> None:
        if not isinstance(caller, KindModelCaller):
            raise TypeError("caller must be KindModelCaller")
        if not isinstance(recheck, PendingRecheck):
            raise TypeError("recheck must be PendingRecheck")
        self.caller = caller
        self.recheck = recheck
        self.config: BehaviorKindConfig = caller.config

    async def grow(
        self,
        pool: PendingPool,
        vocabulary: Vocabulary,
        *,
        now: datetime,
        content_of: ContentOf,
        checkpoint: Callable[[], object],
    ) -> NightlyResult:
        signals: list[str] = []
        calls = 0
        added: list[BehaviorClass] = []
        moves: list[Move] = []
        rechecked: set[str] = set()
        for lane in Lane:
            entries = pool.lane(lane)
            if not entries:
                continue
            calls += 1
            groups, flagged = await self._groups(lane, entries, vocabulary, signals)
            if flagged:
                released, assigned, extra = await self._resolve(flagged, vocabulary, signals, content_of, checkpoint)
                calls += extra
                groups += released
                moves += assigned
                rechecked |= {
                    entry.occurrence for group in released for entry in group.members if not entry.rechecked
                }
            draft_vocabulary = vocabulary
            for group in groups:
                created = self._promote(lane, group, draft_vocabulary, signals)
                if created is None:
                    continue
                added.append(created)
                draft_vocabulary = Vocabulary(
                    draft_vocabulary.version,
                    {**draft_vocabulary.classes, created.id: created},
                    draft_vocabulary.branches,
                )
                moves += [Move(entry.occurrence, pending_token(lane), str(created.id)) for entry in group.members]
        if not moves:
            return NightlyResult(None, calls, tuple(signals), frozenset(rechecked))
        record = VersionRecord(
            vocabulary.version + 1,
            now,
            ChangeReason.NIGHTLY,
            tuple(AddClass(item) for item in added),
            tuple(moves),
            note=f"每晚新增 {len(added)} 类，迁出待定 {len(moves)} 条",
        )
        return NightlyResult(record, calls, tuple(signals), frozenset(rechecked))

    async def _resolve(
        self,
        flagged: Sequence[_Group],
        vocabulary: Vocabulary,
        signals: list[str],
        content_of: ContentOf,
        checkpoint: Callable[[], object],
    ) -> tuple[list[_Group], list[Move], int]:
        """认作已有类的组：没重判过的成员交白天归类重判；归进类的迁走，其余（含以前重判过的）放行、照常长新类。"""

        fresh = [entry for group in flagged for entry in group.members if not entry.rechecked]
        result = await self.recheck.recheck(fresh, vocabulary, content_of=content_of, checkpoint=checkpoint)
        signals.extend(result.signals)
        released: list[_Group] = []
        assigned: list[Move] = []
        for group in flagged:
            for entry in group.members:
                if entry.occurrence in result.assigned:
                    target = result.assigned[entry.occurrence]
                    assigned.append(Move(entry.occurrence, pending_token(entry.lane), target))
            free = tuple(
                entry for entry in group.members if entry.rechecked or entry.occurrence in result.released
            )
            if free:
                released.append(_Group(free, {**group.draft, "existing": None}))
            signals.append(
                f"kind_pending_rechecked {group.draft['name']!r}: "
                f"{sum(1 for e in group.members if e.occurrence in result.assigned)} assigned, {len(free)} released"
            )
        return released, assigned, result.model_calls

    async def _groups(
        self, lane: Lane, entries: Sequence[PendingEntry], vocabulary: Vocabulary, signals: list[str]
    ) -> tuple[list[_Group], list[_Group]]:
        """模型聚组；返回（新事的组，认作已有类的组）。"""

        classes = {f"C{i}": item for i, item in enumerate(vocabulary.active(lane), start=1)}
        keyed = {f"P{i}": entry for i, entry in enumerate(entries, start=1)}
        try:
            drafts = await self.caller.call(
                system=NIGHTLY_SYSTEM_PROMPT,
                user=render_user_message(classes, keyed),
                schema=nightly_schema(tuple(keyed), tuple(classes), max_examples=self.config.max_examples),
                name="behavior_kind_nightly",
                validator=lambda parsed: _validated(parsed, keyed, classes),
            )
        except KindModelOutputError as exc:
            signals.append(f"kind_nightly_rejected {lane.value}: {exc}")
            return [], []
        groups: list[_Group] = []
        flagged: list[_Group] = []
        for draft in drafts:
            members = tuple(keyed[key] for key in draft["members"])
            existing = draft["existing"]
            if existing is not None:
                signals.append(
                    f"kind_pending_matches_existing {lane.value}: {len(members)} entries look like "
                    f"{classes[existing].id} {classes[existing].name!r}; rechecking"
                )
            draft = {
                **draft,
                "excludes": [
                    (item["text"], classes[item["goes_to"]].id if item["goes_to"] else None)
                    for item in draft["excludes"]
                ],
            }
            (groups if existing is None else flagged).append(_Group(members, draft))
        return groups, flagged

    def _promote(self, lane: Lane, group: _Group, vocabulary: Vocabulary, signals: list[str]) -> BehaviorClass | None:
        count = len(group.members)
        days = len({entry.day for entry in group.members})
        name = group.draft["name"].strip()
        if count < self.config.recurrence_min_count or days < self.config.recurrence_min_days:
            signals.append(f"kind_pending_group_waiting {lane.value} {name!r}: {count} entries over {days} days")
            return None
        try:
            created = BehaviorClass(
                id=vocabulary.next_id(lane),
                name=name,
                criterion=group.draft["criterion"].strip(),
                reminder=group.draft["reminder"].strip(),
                excludes=tuple(Exclusion(text.strip(), target) for text, target in group.draft["excludes"]),
                examples=tuple(dict.fromkeys(text.strip() for text in group.draft["examples"]))[
                    : self.config.max_examples
                ],
                origin=ClassOrigin.PROMOTED,
            )
            Vocabulary(vocabulary.version, {**vocabulary.classes, created.id: created}, vocabulary.branches)
        except BehaviorKindError as exc:
            signals.append(f"kind_pending_group_invalid {lane.value} {name!r}: {exc}")
            return None
        signals.append(f"kind_promoted {created.id} {name!r}: {count} entries over {days} days")
        return created


def _validated(
    parsed: object, keyed: Mapping[str, PendingEntry], classes: Mapping[str, BehaviorClass]
) -> list[dict[str, Any]]:
    """每条待定恰好在一组；编号都认识。"""

    if not isinstance(parsed, Mapping) or set(parsed) != {"groups"} or not isinstance(parsed["groups"], list):
        raise BehaviorKindError("nightly output must be an object with a `groups` list")
    seen: list[str] = []
    for group in parsed["groups"]:
        if not isinstance(group, Mapping):
            raise BehaviorKindError("nightly group must be an object")
        members = group.get("members")
        if not isinstance(members, list) or not members or any(member not in keyed for member in members):
            raise BehaviorKindError("nightly group members must be listed pending entries")
        seen += members
        for name in ("name", "criterion", "reminder"):
            if not isinstance(group.get(name), str) or not group[name].strip():
                raise BehaviorKindError(f"nightly group {name} must be non-empty text")
        existing = group.get("existing")
        if existing is not None and existing not in classes:
            raise BehaviorKindError("nightly group existing must be a listed class or null")
        for exclusion in group.get("excludes") or []:
            if not isinstance(exclusion, Mapping) or (
                exclusion.get("goes_to") is not None and exclusion["goes_to"] not in classes
            ):
                raise BehaviorKindError("nightly exclusion target must be a listed class or null")
    if sorted(seen) != sorted(keyed):
        raise BehaviorKindError("every pending entry must be in exactly one group")
    return [dict(group) for group in parsed["groups"]]


__all__ = ["NightlyGrower", "NightlyResult"]
