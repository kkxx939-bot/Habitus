"""定期拆改（裁定 17）：合并按周期攒证据、拆分要两边都跨天反复出现且提醒句不同、锚点自检、封闭集合重判、
冷却期（包括重判移入的目标类）、改动上限、定时。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller
from habitus.behavior.kinds.changes import AddClass, ChangeReason, VersionRecord, apply
from habitus.behavior.kinds.classify import DaytimeClassifier, OccurrenceContent
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import BehaviorClass, ClassOrigin, Vocabulary
from habitus.behavior.kinds.revision import AnchorGate, Member, Reviser, RevisionResult, cooling_classes
from habitus.behavior.kinds.schedule import JobState, nightly_due, revision_due, revision_period
from tests.unit.behavior.kinds_fixtures import (
    EDIT,
    NOW,
    RESEARCH,
    RoutingProvider,
    client_for,
    edit_class,
    research_class,
)

COUNT = ClassId(Lane.SESSION, 3)
CLEANUP = ClassId(Lane.SESSION, 4)
PERIOD = "2026-09-30"
EARLIER = "2026-09-23"
PAIR = ("s-k0003", "s-k0002")


def count_class() -> BehaviorClass:
    return BehaviorClass(COUNT, "统计代码规模", "数代码行数", "该统计代码规模了", origin=ClassOrigin.PROMOTED)


def vocabulary() -> Vocabulary:
    return Vocabulary(version=2, classes={EDIT: edit_class(), RESEARCH: research_class(), COUNT: count_class()})


def member(key: str, token: str, name: str, day: int) -> Member:
    return Member(f"behavior://{key}", token, Lane.SESSION, date(2026, 9, day), OccurrenceContent(name=name))


EDITS = [member(f"e{d}", "s-k0001", name, d) for d, name in enumerate(("改记忆召回", "改调度器", "改配置"), start=1)]
CLEANS = [
    member(f"x{d}", "s-k0001", name, d) for d, name in enumerate(("清理构建产物", "清理缓存", "清理日志"), start=4)
]
OTHERS = [
    member("r1", "s-k0002", "读源码", 1),
    member("r2", "s-k0002", "读论文", 2),
    member("c1", "s-k0003", "数行数", 3),
    member("c2", "s-k0003", "数文件", 4),
]
MEMBERS = [*EDITS, *CLEANS, *OTHERS]
ROUTES: dict[str, tuple[str, ...]] = {
    **{m.content.name: ("修改代码",) for m in EDITS},
    **{m.content.name: ("清理", "修改代码") for m in CLEANS},
    "读源码": ("调研",),
    "读论文": ("调研",),
    "数行数": ("统计代码规模",),
    "数文件": ("统计代码规模",),
}
EVIDENCE = JobState(revision_at=NOW - timedelta(days=7), merge_history=((EARLIER, frozenset({PAIR})),))


def proposal(kind: str, cls: str, *, target: str | None = None, draft: Any = None, new: Any = None) -> dict[str, Any]:
    return {"reason": "抽样里的原话", "kind": kind, "class": cls, "target": target, "draft": draft, "new_class": new}


def entry(name: str, *, goes_to: str | None = None, reminder: str | None = None) -> dict[str, Any]:
    excludes = [{"text": "清理", "goes_to": goes_to}] if goes_to else []
    return {
        "name": name,
        "criterion": f"{name}的判据",
        "reminder": reminder or f"该{name}了",
        "excludes": excludes,
        "examples": [],
    }


SPLIT = proposal("split", "C1", draft=entry("修改代码", goes_to="NEW"), new=entry("清理"))


def revise(
    proposals: list[dict[str, Any]],
    *,
    routes: Mapping[str, tuple[str, ...]] = ROUTES,
    members: Sequence[Member] = MEMBERS,
    state: JobState | None = None,
    records: tuple[VersionRecord, ...] = (),
    config: BehaviorKindConfig | None = None,
) -> tuple[RevisionResult, RoutingProvider, list[int]]:
    provider = RoutingProvider([{"proposals": proposals}], routes)
    caller = KindModelCaller(client_for(provider), config=config or BehaviorKindConfig(merge_evidence_runs=2))
    reviser = Reviser(caller, AnchorGate(DaytimeClassifier(caller)))
    beats: list[int] = []
    result = asyncio.run(
        reviser.revise(
            vocabulary(),
            members,
            records,
            state or JobState(),
            now=NOW,
            period=PERIOD,
            checkpoint=lambda: beats.append(1),
        )
    )
    return result, provider, beats


def test_merge_waits_for_an_earlier_period_and_remembers_this_one() -> None:
    result, provider, _ = revise([proposal("merge", "C3", target="C2")])
    assert result.record is None and provider.calls == 1
    assert any("kind_revision_merge_waiting s-k0003->s-k0002" in s for s in result.signals)
    assert result.state.merge_history == ((PERIOD, frozenset({PAIR})),) and result.state.revision_at == NOW


def test_proposals_within_the_same_period_count_once() -> None:
    same = JobState(merge_history=((PERIOD, frozenset({PAIR})),))
    result, _, _ = revise([proposal("merge", "C3", target="C2")], state=same)
    assert result.record is None and any("merge_waiting" in s for s in result.signals)
    assert result.state.merge_history == ((PERIOD, frozenset({PAIR})),)


def test_supported_merge_passes_the_anchor_check_and_moves_members() -> None:
    result, provider, beats = revise([proposal("merge", "C3", target="C2")], state=EVIDENCE)
    record = result.record
    assert record is not None and record.reason is ChangeReason.REVISION
    assert provider.classify_calls == 3 and result.model_calls == 4 and beats  # 锚点：旧清单两遍 + 新清单一遍
    assert [(m.occurrence, m.target) for m in record.moves] == [
        ("behavior://c1", "s-k0002"),
        ("behavior://c2", "s-k0002"),
    ]
    after = apply(vocabulary(), record)
    assert not after.get(COUNT).active and after.descendants(COUNT) == (RESEARCH,)
    assert any("kind_revision_accepted" in s and "anchors 6, pulled 0" in s for s in result.signals)


def test_split_with_evidence_rejudges_only_inside_the_closed_set() -> None:
    result, provider, _ = revise([SPLIT])
    record = result.record
    assert record is not None
    rejudge_prompt = provider.prompts[1].split("## 待归类的记录")[0]
    assert "修改代码" in rejudge_prompt and "清理" in rejudge_prompt and "调研" not in rejudge_prompt
    assert {(m.occurrence, m.target) for m in record.moves} == {(m.occurrence, str(CLEANUP)) for m in CLEANS}
    after = apply(vocabulary(), record)
    assert after.get(EDIT).excludes[0].goes_to == CLEANUP and after.descendants(EDIT) == (EDIT, CLEANUP)


def test_split_needs_both_sides_to_recur_across_days() -> None:
    result, _, _ = revise([SPLIT], members=[*EDITS, *CLEANS[:2], *OTHERS])
    assert result.record is None
    assert any("split lacks evidence: split-out side has 2 entries over 2 days" in s for s in result.signals)


def test_split_needs_a_different_reminder() -> None:
    same = proposal("split", "C1", draft=entry("修改代码", goes_to="NEW"), new=entry("清理", reminder="该修改代码了。"))
    result, _, _ = revise([same])
    assert result.record is None and any("reminders are the same" in s for s in result.signals)


def test_anchor_check_rejects_a_change_that_pulls_other_classes() -> None:
    routes = {**ROUTES, "数行数": ("清理", "统计代码规模"), "数文件": ("清理", "统计代码规模")}
    result, _, _ = revise([SPLIT], routes=routes)
    assert result.record is None
    assert any(s.startswith("kind_revision_anchor_rejected") and "anchors 4, pulled 2" in s for s in result.signals)


def test_anchor_tolerance_comes_from_config() -> None:
    routes = {**ROUTES, "数行数": ("清理", "统计代码规模")}
    result, _, _ = revise([SPLIT], routes=routes, config=BehaviorKindConfig(max_anchor_pull=0.25))
    assert result.record is not None


def test_unstable_anchors_are_not_counted() -> None:
    routes = {**ROUTES, "读源码": ("读源码",)}  # 旧清单下就归不回原类：不当锚点
    result, _, _ = revise([SPLIT], routes=routes)
    assert result.record is not None and any("anchors 3, pulled 0" in s for s in result.signals)


def test_without_anchors_nothing_is_changed() -> None:
    result, _, _ = revise([SPLIT], members=[*EDITS, *CLEANS])
    assert result.record is None and any("anchors 0" in s for s in result.signals)


def test_classes_changed_recently_are_cooling_down() -> None:
    nightly = VersionRecord(3, NOW - timedelta(days=1), ChangeReason.NIGHTLY, (AddClass(count_class()),))
    result, _, _ = revise([proposal("merge", "C3", target="C2")], state=EVIDENCE, records=(nightly,))
    assert result.record is None and any("cooling down" in s for s in result.signals)


def test_rejudge_may_not_move_entries_into_a_cooling_class() -> None:
    nightly = VersionRecord(3, NOW - timedelta(days=1), ChangeReason.NIGHTLY, (AddClass(research_class()),))
    routes = {**ROUTES, "改配置": ("调研", "修改代码")}
    result, _, _ = revise(
        [proposal("revise", "C1", draft=entry("修改代码", goes_to="C2"))], routes=routes, records=(nightly,)
    )
    assert result.record is None
    assert any("moves entries into a cooling or already changed class" in s for s in result.signals)


def test_rejudge_may_not_move_entries_into_a_class_changed_earlier_in_the_run() -> None:
    routes = {**ROUTES, "改配置": ("调研", "修改代码")}
    proposals = [proposal("merge", "C3", target="C2"), proposal("revise", "C1", draft=entry("修改代码", goes_to="C2"))]
    result, provider, _ = revise(proposals, routes=routes, state=EVIDENCE)
    record = result.record
    assert record is not None and len(record.operations) == 2  # 只有合并：退役 + 分支
    assert any("moves entries into a cooling or already changed class" in s for s in result.signals)
    assert provider.classify_calls == 4  # 锚点在一次拆改里只挑一次；被跳过的提议不再过锚点


def test_change_cap_stops_further_proposals() -> None:
    config = BehaviorKindConfig(merge_evidence_runs=1, max_changes_per_revision=1)
    proposals = [proposal("merge", "C3", target="C2"), proposal("revise", "C1", draft=entry("修改代码"))]
    result, _, _ = revise(proposals, config=config)
    assert result.record is not None and len(result.record.operations) == 2
    assert any(s.startswith("kind_revision_cap_reached") for s in result.signals)


def test_malformed_proposals_are_dropped() -> None:
    result, _, _ = revise([proposal("merge", "C3"), proposal("split", "C1", draft=entry("修改代码"))])
    assert result.record is None
    assert sum(s.startswith("kind_revision_proposal_dropped") for s in result.signals) == 2


def test_a_proposal_that_clashes_with_an_active_name_is_dropped_not_raised() -> None:
    result, _, _ = revise(
        [proposal("revise", "C1", draft=entry("调研"))], config=BehaviorKindConfig(merge_evidence_runs=1)
    )
    assert result.record is None and any(s.startswith("kind_revision_invalid") for s in result.signals)
    assert result.state.revision_at == NOW


def test_merge_evidence_needs_consecutive_earlier_periods() -> None:
    other = frozenset({("s-k0001", "s-k0002")})
    state = JobState(merge_history=(("2026-09-16", frozenset({PAIR})), ("2026-09-23", other)))
    assert not state.merge_supported(PAIR, runs=2, period=PERIOD)
    assert not state.merge_supported(PAIR, runs=3, period=PERIOD)
    later = state.after_revision(NOW, PERIOD, frozenset({PAIR}), keep=2)
    assert [period for period, _ in later.merge_history] == ["2026-09-23", PERIOD]
    assert later.merge_supported(PAIR, runs=2, period="2026-10-07")


def test_revision_period_is_the_date_of_the_latest_scheduled_moment() -> None:
    tz = timezone(timedelta(hours=8))
    assert revision_period(datetime(2026, 10, 7, 4, 0, tzinfo=tz), weekday=2, hour=3) == "2026-10-07"
    assert revision_period(datetime(2026, 10, 7, 2, 0, tzinfo=tz), weekday=2, hour=3) == "2026-09-30"
    assert revision_period(datetime(2026, 10, 12, 9, 0, tzinfo=tz), weekday=2, hour=3) == "2026-10-07"


def test_revision_is_due_once_after_each_scheduled_moment() -> None:
    tz = timezone(timedelta(hours=8))
    wednesday_4am = datetime(2026, 10, 7, 4, 0, tzinfo=tz)
    assert revision_due(wednesday_4am, None, weekday=2, hour=3)
    assert not revision_due(wednesday_4am, wednesday_4am - timedelta(minutes=30), weekday=2, hour=3)
    assert revision_due(wednesday_4am + timedelta(days=2), wednesday_4am - timedelta(days=6), weekday=2, hour=3)
    assert not revision_due(
        datetime(2026, 10, 7, 2, 0, tzinfo=tz), datetime(2026, 10, 1, 3, 30, tzinfo=tz), weekday=2, hour=3
    )


def test_nightly_is_due_once_a_day_after_its_hour() -> None:
    tz = timezone(timedelta(hours=8))
    three_am = datetime(2026, 10, 7, 3, 0, tzinfo=tz)
    assert nightly_due(three_am, None, hour=2)
    assert not nightly_due(three_am, three_am - timedelta(minutes=30), hour=2)
    assert nightly_due(three_am, three_am - timedelta(hours=2), hour=2)
    assert not nightly_due(datetime(2026, 10, 7, 1, 0, tzinfo=tz), datetime(2026, 10, 6, 2, 30, tzinfo=tz), hour=2)


def test_the_anchor_cache_follows_the_recent_members_not_just_the_vocabulary_version() -> None:
    """同一版词表：同一批最近成员复用上次挑出的锚点（不再问模型）；下一周成员换了就重挑（裁定 17②；第二轮评审 P2-5）。"""

    provider = RoutingProvider([], ROUTES)
    caller = KindModelCaller(client_for(provider), config=BehaviorKindConfig())
    gate = AnchorGate(DaytimeClassifier(caller))

    def check(members: Sequence[Member]) -> int:
        before = provider.classify_calls
        asyncio.run(gate.check(Lane.SESSION, vocabulary(), vocabulary(), frozenset(), members, checkpoint=lambda: None))
        return provider.classify_calls - before

    first = check(MEMBERS)
    repeated = check(MEMBERS)
    next_week = check([*MEMBERS, member("e9", "s-k0001", "改调度器", 9)])
    assert repeated < first  # 同一批成员：挑锚点那两遍不再问
    assert next_week > repeated  # 成员换了：重新挑一遍锚点（多了一条成员，分批可能更多）
