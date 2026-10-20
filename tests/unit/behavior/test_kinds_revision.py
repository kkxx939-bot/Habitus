"""定期拆改（裁定 17；拆分 2026-10-12 按 `18` 改为"模型分组、算法定拆"）：合并按周期攒证据；拆分由每个类近期成员的分组
与复现定（每边都要跨天反复出现、提醒句不同，模型不提拆）、锚点自检、封闭集合重判、冷却期（包括重判移入的目标类）、改动上限、定时。

脚本回放的顺序：每个成员够 ``grouping_min_members`` 的类先各吃一份分组答复（只有「修改代码」有 6 条），然后才是提案答复。"""

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
from habitus.behavior.kinds.revision import AnchorGate, Member, Reviser, RevisionResult
from habitus.behavior.kinds.revision.proposals import parse_proposals
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


def proposal(kind: str, cls: str, *, target: str | None = None, draft: Any = None) -> dict[str, Any]:
    return {"reason": "抽样里的原话", "kind": kind, "class": cls, "target": target, "draft": draft}


def entry(name: str, *, goes_to: str | None = None, reminder: str | None = None) -> dict[str, Any]:
    excludes = [{"text": "清理", "goes_to": goes_to}] if goes_to else []
    return {
        "name": name,
        "criterion": f"{name}的判据",
        "reminder": reminder or f"该{name}了",
        "excludes": excludes,
        "examples": [],
    }


def group(
    keys: Sequence[str], name: str, *, reminder: str | None = None, existing: str | None = None
) -> dict[str, Any]:
    """分组答复里的一组：记录编号按（日子，地址）排——「修改代码」的 6 条是 M1–M3 改动、M4–M6 清理。
    ``existing``：分批时归进前几批的某一组（G…）。"""

    return {"members": list(keys), **entry(name, reminder=reminder), "existing": existing}


EDIT_KEYS = ("M1", "M2", "M3")
CLEAN_KEYS = ("M4", "M5", "M6")
#: 两组各 3 条跨 3 天：按复现够拆；留在原类的是名字排前面的「修改代码」，「清理」成新类 s-k0004
SPLIT_GROUPS = {"groups": [group(EDIT_KEYS, "修改代码"), group(CLEAN_KEYS, "清理")]}


def one_group(members: Sequence[Member], config: BehaviorKindConfig) -> list[dict[str, Any]]:
    """默认的分组答复：「修改代码」窗口内的成员全在一组（不拆）；成员不够分组门槛就没有这份答复。"""

    since = (NOW - timedelta(days=config.revision_window_days)).date()
    recent = [m for m in members if m.token == str(EDIT) and m.day >= since]
    if len(recent) < config.grouping_floor:
        return []
    return [{"groups": [group([f"M{i}" for i in range(1, len(recent) + 1)], "修改代码")]}]


def revise(
    proposals: list[dict[str, Any]],
    *,
    groups: Sequence[dict[str, Any]] | None = None,
    routes: Mapping[str, tuple[str, ...]] = ROUTES,
    members: Sequence[Member] = MEMBERS,
    state: JobState | None = None,
    records: tuple[VersionRecord, ...] = (),
    config: BehaviorKindConfig | None = None,
) -> tuple[RevisionResult, RoutingProvider, list[int]]:
    config = config or BehaviorKindConfig(merge_evidence_runs=2)
    bodies = [*(one_group(members, config) if groups is None else groups), {"proposals": proposals}]
    provider = RoutingProvider(bodies, routes)
    caller = KindModelCaller(client_for(provider), config=config)
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
    assert result.record is None and provider.calls == 2  # 「修改代码」分组一次 + 提案一次
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
    assert (
        provider.classify_calls == 3 and result.model_calls == 5 and beats
    )  # 分组 + 提案 + 锚点：旧清单两遍、新清单一遍
    assert [(m.occurrence, m.target) for m in record.moves] == [
        ("behavior://c1", "s-k0002"),
        ("behavior://c2", "s-k0002"),
    ]
    after = apply(vocabulary(), record)
    assert not after.get(COUNT).active and after.descendants(COUNT) == (RESEARCH,)
    assert any("kind_revision_accepted" in s and "anchors 6, pulled 0" in s for s in result.signals)


def test_split_comes_from_grouping_and_rejudges_only_inside_the_closed_set() -> None:
    """模型分组、算法定拆：两组各 3 条跨 3 天 → 「清理」切出去成 s-k0004，原类条目换成留下那组的、带「不含 → 新类」。"""

    result, provider, _ = revise([], groups=[SPLIT_GROUPS])
    record = result.record
    assert record is not None
    grouping_prompt = provider.prompts[0]
    assert "## 记录" in grouping_prompt and "改记忆召回" in grouping_prompt and "清理日志" in grouping_prompt
    rejudge_prompt = next(p for p in provider.prompts if "## 待归类的记录" in p).split("## 待归类的记录")[0]
    assert "修改代码" in rejudge_prompt and "清理" in rejudge_prompt and "调研" not in rejudge_prompt
    assert {(m.occurrence, m.target) for m in record.moves} == {(m.occurrence, str(CLEANUP)) for m in CLEANS}
    after = apply(vocabulary(), record)
    narrowed = after.get(EDIT)
    assert any(e.text == "清理" and e.goes_to == CLEANUP for e in narrowed.excludes)
    assert (
        narrowed.name == "修改代码" and narrowed.examples == ()
    )  # 原类条目换成留下那组的（例子由分组给，原例子可能已属新类）
    assert after.get(CLEANUP).origin is ClassOrigin.SPLIT and after.descendants(EDIT) == (EDIT, CLEANUP)
    assert any("kind_revision_accepted SplitProposal s-k0001: 按分组与复现" in s for s in result.signals)


def test_split_waits_until_at_least_two_groups_recur() -> None:
    """分组分出了两组，但一组只有 1 条：按复现不够拆（门槛 2 条跨 2 天，裁定 59），这周等着（信号里写明每组的条数与天数）。"""

    groups = {"groups": [group(("M1", "M2", "M3", "M4", "M5"), "修改代码"), group(("M6",), "清理")]}
    result, provider, _ = revise([], groups=[groups])
    assert result.record is None and provider.classify_calls == 0  # 没走到重判与锚点
    assert any(
        "kind_revision_split_waiting s-k0001: 2 groups, 1 recur" in s and "清理 1 条跨 1 天" in s
        for s in result.signals
    )


def test_split_is_dropped_when_rejudge_leaves_a_side_too_thin() -> None:
    """分组说两边各 3 条，封闭集合重判只把 1 条清理归过去：证据规则复核不过（门槛 2 条跨 2 天），整条拆分丢掉。"""

    routes = {**ROUTES, "清理日志": ("修改代码",), "清理缓存": ("修改代码",)}
    result, _, _ = revise([], groups=[SPLIT_GROUPS], routes=routes)
    assert result.record is None
    assert any("split lacks evidence: split-out side has 1 entries over 1 days" in s for s in result.signals)


def test_groups_with_the_same_reminder_are_one_group_and_do_not_split() -> None:
    same = {"groups": [group(EDIT_KEYS, "修改代码"), group(CLEAN_KEYS, "清理", reminder="该修改代码了。")]}
    result, _, _ = revise([], groups=[same])
    assert result.record is None and any("kind_revision_split_none s-k0001: one group" in s for s in result.signals)


def test_a_class_splits_several_ways_in_one_revision() -> None:
    """三组各 3 条跨 3 天：名字排前面的「修改代码」留在原类，「写文档」「清理」各成一类（s-k0004、s-k0005），一次拆改做完。"""

    docs = [
        member(f"d{d}", "s-k0001", name, d)
        for d, name in enumerate(("写接口文档", "写部署文档", "写变更说明"), start=7)
    ]
    routes = {**ROUTES, **{m.content.name: ("写文档", "修改代码") for m in docs}}
    groups = {"groups": [group(EDIT_KEYS, "修改代码"), group(CLEAN_KEYS, "清理"), group(("M7", "M8", "M9"), "写文档")]}
    result, _, _ = revise([], groups=[groups], routes=routes, members=[*MEMBERS, *docs])
    record = result.record
    assert record is not None
    kinds = [type(op).__name__ for op in record.operations]
    assert kinds == ["AddClass", "Branch", "AddClass", "Branch", "ReviseClass"]
    after = apply(vocabulary(), record)
    docs_id, clean_id = ClassId(Lane.SESSION, 4), ClassId(Lane.SESSION, 5)
    assert after.get(docs_id).name == "写文档" and after.get(clean_id).name == "清理"
    assert {(m.occurrence, m.target) for m in record.moves} == {
        *((m.occurrence, str(clean_id)) for m in CLEANS),
        *((m.occurrence, str(docs_id)) for m in docs),
    }
    assert {e.goes_to for e in after.get(EDIT).excludes} >= {docs_id, clean_id}
    assert after.descendants(EDIT) == (EDIT, docs_id, clean_id)


def test_grouping_only_looks_at_the_recent_window_but_rejudges_everything() -> None:
    """窗口外的旧成员不进分组（按月的事要攒够几次，窗口默认 120 天），但拆分时它仍在原类的封闭集合里重判。"""

    old = member("o1", "s-k0001", "改旧年的配置", 1)
    old = Member(old.occurrence, old.token, old.lane, date(2026, 1, 1), old.content)
    routes = {**ROUTES, "改旧年的配置": ("修改代码",)}
    result, provider, _ = revise([], groups=[SPLIT_GROUPS], routes=routes, members=[old, *MEMBERS])
    assert result.record is not None
    assert "改旧年的配置" not in provider.prompts[0]  # 分组提示词
    rejudge_prompt = next(p for p in provider.prompts if "## 待归类的记录" in p)
    assert "改旧年的配置" in rejudge_prompt


def test_grouping_is_batched_and_later_batches_see_earlier_groups() -> None:
    """成员多于一批时分批问；后一批看得见前几批的组（G…），答 existing 就归进去，提醒句逐字相同的也合并。"""

    config = BehaviorKindConfig(merge_evidence_runs=2, grouping_batch_size=3)
    halves = [{"groups": [group(("M1", "M2", "M3"), "修改代码")]}, {"groups": [group(("M1", "M2", "M3"), "清理")]}]
    result, provider, _ = revise([], groups=halves, config=config)
    assert result.record is not None and sum("## 记录" in p for p in provider.prompts) == 2
    second = next(p for p in provider.prompts if "## 前几批已分出的组" in p)
    assert "G1  修改代码：该修改代码了" in second and "## 前几批" not in provider.prompts[0]
    # 第二批换了措辞但答 existing=G1：归进第一组，整类一组、不拆
    reworded = [
        {"groups": [group(("M1", "M2", "M3"), "修改代码")]},
        {"groups": [group(("M1", "M2", "M3"), "改一改代码", reminder="该改一改代码了", existing="G1")]},
    ]
    result, _, _ = revise([], groups=reworded, config=config)
    assert result.record is None and any("split_none" in s for s in result.signals)
    # 没答 existing、但提醒句逐字相同：也合并
    same_words = [{"groups": [group(("M1", "M2", "M3"), "修改代码")]}] * 2
    result, _, _ = revise([], groups=same_words, config=config)
    assert result.record is None and any("split_none" in s for s in result.signals)


def test_the_group_with_the_class_own_reminder_stays_even_when_scattered_records_are_many() -> None:
    """一组的提醒句就是原类的提醒句 → 它留在原类；零散记录再多也不把原类改名。"""

    odd = [
        member(f"z{d}", "s-k0001", name, d) for d, name in enumerate(("帮我算个税", "查下汇率", "翻译一段话"), start=7)
    ]
    routes = {**ROUTES, **{m.content.name: ("修改代码",) for m in odd}}
    groups = {
        "groups": [
            group(EDIT_KEYS, "修改代码", reminder="该改代码了"),
            group(CLEAN_KEYS, "清理"),
            *(group((f"M{i}",), name) for i, name in zip((7, 8, 9), ("算个税", "查汇率", "翻译"), strict=True)),
        ]
    }
    result, _, _ = revise([], groups=[groups], routes=routes, members=[*MEMBERS, *odd])
    record = result.record
    assert record is not None
    after = apply(vocabulary(), record)
    assert after.get(EDIT).reminder == "该改代码了" and after.get(CLEANUP).name == "清理"
    assert any("留在原类的是 修改代码" in s for s in result.signals)


def test_many_scattered_records_keep_the_original_entry_and_all_recurring_groups_split_out() -> None:
    """没有一组是原类自己的提醒句、零散记录又多：原类条目不动当桶，够格的组全部切出去。"""

    docs = [
        member(f"d{d}", "s-k0001", name, d)
        for d, name in enumerate(("写接口文档", "写部署文档", "写变更说明"), start=7)
    ]
    routes = {**ROUTES, **{m.content.name: ("写文档", "修改代码") for m in docs}}
    groups = {
        "groups": [
            *(group((key,), name) for key, name in zip(EDIT_KEYS, ("改召回", "改调度", "改配置项"), strict=True)),
            group(CLEAN_KEYS, "清理"),
            group(("M7", "M8", "M9"), "写文档"),
        ]
    }
    result, _, _ = revise([], groups=[groups], routes=routes, members=[*MEMBERS, *docs])
    record = result.record
    assert record is not None
    after = apply(vocabulary(), record)
    source = after.get(EDIT)
    assert source.name == "修改代码" and source.reminder == "该改代码了" and source.examples == edit_class().examples
    assert {after.get(ClassId(Lane.SESSION, n)).name for n in (4, 5)} == {"清理", "写文档"}
    assert any("留在原类的是 原条目（3 条零散记录）" in s for s in result.signals)


def test_a_group_that_is_an_existing_class_moves_there_instead_of_making_a_new_one() -> None:
    """分组答 existing=C…（参考清单里的类）：那组记录搬去已有的类，不另起同名新类（第 35 晚重放：查看信息里 14 条其实是「询问知识性问题」，
    另起新类撞名、整处拆分作废）。参考清单对「修改代码」来说是 C1 调研、C2 统计代码规模。"""

    routes = {**ROUTES, **{m.content.name: ("调研",) for m in CLEANS}}  # 封闭集合里有调研时，清理那三条归调研
    groups = {"groups": [group(EDIT_KEYS, "修改代码"), group(CLEAN_KEYS, "调研", existing="C1")]}
    result, _, _ = revise([], groups=[groups], routes=routes)
    record = result.record
    assert record is not None and [type(op).__name__ for op in record.operations] == ["Branch", "ReviseClass"]
    assert {(m.occurrence, m.target) for m in record.moves} == {(m.occurrence, "s-k0002") for m in CLEANS}
    assert any("3 条搬去已有的 s-k0002" in s for s in result.signals)


def test_moving_into_a_cooling_class_is_held_back() -> None:
    nightly = VersionRecord(3, NOW - timedelta(days=1), ChangeReason.NIGHTLY, (AddClass(research_class()),))
    groups = {"groups": [group(EDIT_KEYS, "修改代码"), group(CLEAN_KEYS, "调研", existing="C1")]}
    result, _, _ = revise([], groups=[groups], records=(nightly,))
    assert result.record is None
    assert any(s.startswith("kind_revision_split_blocked s-k0001: 3 entries belong to s-k0002") for s in result.signals)
    assert any(s.startswith("kind_revision_split_waiting s-k0001") for s in result.signals)


def test_a_rejected_grouping_answer_skips_the_class() -> None:
    bad = {"groups": [group(EDIT_KEYS, "修改代码")]}  # 少了三条记录：每条必须恰好在一组
    result, _, _ = revise([], groups=[bad, bad, bad])
    assert result.record is None
    assert any(s.startswith("kind_revision_grouping_rejected s-k0001") for s in result.signals)


def test_split_proposals_from_the_model_are_rejected_by_the_schema() -> None:
    """提案的 schema 只认 merge / revise：模型答 split 整份答复不合 schema、留信号；解析器里还留着一道同样的闸。"""

    result, _, _ = revise([proposal("split", "C1", draft=entry("修改代码"))])
    assert result.record is None
    assert any(s.startswith("kind_revision_rejected session") for s in result.signals)


def test_the_parser_drops_a_split_kind_with_a_reason() -> None:
    proposals, signals = parse_proposals(
        [proposal("split", "C1", draft=entry("修改代码"))], {"C1": edit_class()}, max_examples=3
    )
    assert proposals == () and len(signals) == 1 and "decided by grouping" in signals[0]


def test_anchor_check_rejects_a_change_that_pulls_other_classes() -> None:
    routes = {**ROUTES, "数行数": ("清理", "统计代码规模"), "数文件": ("清理", "统计代码规模")}
    result, _, _ = revise([], groups=[SPLIT_GROUPS], routes=routes)
    assert result.record is None
    assert any(s.startswith("kind_revision_anchor_rejected") and "anchors 4, pulled 2" in s for s in result.signals)


def test_anchor_tolerance_comes_from_config() -> None:
    routes = {**ROUTES, "数行数": ("清理", "统计代码规模")}
    result, _, _ = revise([], groups=[SPLIT_GROUPS], routes=routes, config=BehaviorKindConfig(max_anchor_pull=0.25))
    assert result.record is not None


def test_unstable_anchors_are_not_counted() -> None:
    routes = {**ROUTES, "读源码": ("读源码",)}  # 旧清单下就归不回原类：不当锚点
    result, _, _ = revise([], groups=[SPLIT_GROUPS], routes=routes)
    assert result.record is not None and any("anchors 3, pulled 0" in s for s in result.signals)


def test_an_anchor_taken_by_an_earlier_accepted_change_is_not_charged_to_later_proposals() -> None:
    """锚点自检的基线是本轮已采纳改动之后的词表：第一处拆分切出「清理」后，「调研」里本来就更像清理的那条被新类接走是合理的，
    不该算成第二处提议（改「统计代码规模」的判据）拉走了锚点（第 35 晚重放：这一条让后面三处拆分全被挡下）。"""

    members = [*MEMBERS, member("r3", "s-k0002", "清理资料", 3)]
    routes = {**ROUTES, "清理资料": ("清理", "调研")}  # 旧清单里没有「清理」→ 归调研、站得稳；「清理」一出现就该归过去
    revise_count = proposal("revise", "C3", draft=entry("统计代码规模"))
    tolerant = BehaviorKindConfig(
        max_anchor_pull=0.25, merge_evidence_runs=2
    )  # 让拆分本身过闸（它合理地拉走 r3 这 1 个锚点）
    result, _, _ = revise([revise_count], groups=[SPLIT_GROUPS], routes=routes, members=members, config=tolerant)
    assert result.record is not None and len(result.record.operations) == 4  # 拆分 3 个操作 + 改判据 1 个
    # 第二处的锚点只剩 r1、r2：r3 在新基线下已归「清理」、站不稳，不再当锚点；老做法会算成 anchors 3, pulled 1 而拒掉
    assert any(
        s.startswith("kind_revision_accepted ReviseProposal s-k0003") and "anchors 2, pulled 0" in s
        for s in result.signals
    )


def test_without_anchors_nothing_is_changed() -> None:
    result, _, _ = revise([], groups=[SPLIT_GROUPS], members=[*EDITS, *CLEANS])
    assert result.record is None and any("anchors 0" in s for s in result.signals)


def test_classes_changed_recently_are_cooling_down() -> None:
    nightly = VersionRecord(3, NOW - timedelta(days=1), ChangeReason.NIGHTLY, (AddClass(count_class()),))
    result, _, _ = revise([proposal("merge", "C3", target="C2")], state=EVIDENCE, records=(nightly,))
    assert result.record is None and any("cooling down" in s for s in result.signals)
    assert any(s.startswith("kind_revision_split_skipped s-k0003") for s in result.signals)  # 分组那条路也说清是被冻住


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
    result, _, _ = revise([proposal("merge", "C3"), proposal("revise", "C1")])  # 合并没目标；改判据没条目
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


def test_anchor_cache_drops_keys_of_older_vocabulary_versions() -> None:
    """26 号清单 6.9：词表版本前进后，旧版本的键再也用不上，清掉；同一版的不同批成员照旧各留各的。"""

    from dataclasses import replace as _replace

    provider = RoutingProvider([], ROUTES)
    caller = KindModelCaller(client_for(provider), config=BehaviorKindConfig())
    gate = AnchorGate(DaytimeClassifier(caller))

    def check(base: Vocabulary, members: Sequence[Member]) -> None:
        asyncio.run(gate.check(Lane.SESSION, base, base, frozenset(), members, checkpoint=lambda: None))

    check(vocabulary(), MEMBERS)
    check(vocabulary(), [*MEMBERS, member("e9", "s-k0001", "改调度器", 9)])
    assert {key[0] for key in gate._stable} == {vocabulary().version} and len(gate._stable) == 2
    newer = _replace(vocabulary(), version=vocabulary().version + 1)
    check(newer, MEMBERS)
    assert {key[0] for key in gate._stable} == {newer.version} and len(gate._stable) == 1
