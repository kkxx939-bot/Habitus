"""上下文视图：历史投影的九个槽位、钟面邻域筛选、此刻视图、逐槽三值表。全部机械、零 LLM。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.scene.grouping import DraftRelation, GroupingAssembly, SceneDraft
from habitus.scene.model import SceneLinkType, SceneRole
from habitus.scene.views import (
    COMPARED_SLOTS,
    ActionRef,
    ContextView,
    DayIndexCache,
    SceneRef,
    Verdict,
    compare,
    context_view,
    history_contexts,
    now_context,
    render_table,
)
from tests.unit.behavior.tree_payloads import occurrence_payload
from tests.unit.scene.fixtures import DAY1, DAY2, DAY3, SUBJECT, Site, at, publish, scripted_by_day

DAY4 = DAY3 + timedelta(days=1)


def grouped_site(tmp_path) -> tuple[Site, dict[str, str]]:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    uris = site.seed()
    assert site.refresh(DAY1, DAY2).published == (DAY1, DAY2)
    return site, uris


def cache_for(site: Site) -> DayIndexCache:
    return DayIndexCache(site.behavior_tree, site.scene_tree, subject=SUBJECT)


def test_context_view_projects_every_slot_from_the_scene_and_the_behaviour_tree(tmp_path) -> None:
    site, uris = grouped_site(tmp_path)
    cache = cache_for(site)

    view = context_view(uris["wash"], cache, window_days=7)

    assert view.kind_token == "洗菜" and view.name == "洗菜" and view.day == DAY2 and view.covered
    assert view.scene == SceneRef(uri=site.scene_tree.read_day(DAY2)[0].uri, label="准备晚饭")
    assert view.role == "essential"
    assert [step.kind_token for step in view.prior_steps] == ["交谈"]
    # 前提：事的 needs 指向买菜这条行为，落到 kind 买菜
    assert [(item.source, item.text, item.target_kinds) for item in view.preconditions] == [("needs", "去超市买菜", ("买菜",))]
    # 起因：事的 results_from 指向采购这件事，落到它留下的待用前提的产生方 kind
    assert [(cause.label, cause.kinds) for cause in view.causes if isinstance(cause, SceneRef)] == [("去超市采购", ("买菜",))]
    assert view.last_time is None  # 回看窗内没有上一次洗菜
    assert view.concurrent == ()
    assert view.subjects == ()  # 只有主体自己：和谁没有可对的值
    assert view.summary
    shopping = context_view(uris["buy"], cache, window_days=7)
    assert shopping.scene is not None and shopping.scene.label == "去超市采购"
    assert shopping.preconditions == () and shopping.prior_steps == ()
    phone = context_view(uris["phone2"], cache, window_days=7)
    assert phone.last_time is not None
    assert phone.last_time.uri == uris["phone1"] and phone.last_time.days_ago == 1 and phone.last_time.scene is None
    with pytest.raises(KeyError):
        context_view("behavior://occurrences/2026/08/16/没有--20260816T000000000000%2B0800.md", cache, window_days=7)


def test_irrelevant_members_keep_the_scene_but_not_its_preconditions_or_causes(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    uris = site.seed()
    original = site.grouper.group

    async def group(payload):
        assembly = await original(payload)
        if payload.day != DAY2:
            return assembly
        dinner = assembly.scenes[0]
        by_name = {row.name: row.no for row in payload.occurrences}
        members = tuple((no, SceneRole.IRRELEVANT if no == by_name["看手机"] else role) for no, role in dinner.members)
        return GroupingAssembly((SceneDraft(dinner.label, members, relations=dinner.relations),), (), ())

    site.grouper.group = group  # type: ignore[method-assign]
    site.refresh(DAY1, DAY2)
    view = context_view(uris["phone2"], cache_for(site), window_days=7)
    assert view.scene is not None and view.scene.label == "准备晚饭" and view.role == "irrelevant"
    assert view.prior_steps == () and view.preconditions == () and view.causes == ()


def test_pending_preconditions_and_short_range_links_are_projected_across_midnight(tmp_path) -> None:
    """前提第二路（同一件事里更早成员留下的待用前提）、行为树自带的 results_from / concurrent_with（跨午夜）。"""

    site = Site(tmp_path, now=at(DAY4, 12, 0))
    writer = BehaviorDocumentWriter(site.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(DAY4, 12, 0))

    def occurrence(name, day, hour, minute, *, links=(), **overrides):
        started = at(day, hour, minute)
        document = writer.publish(
            BehaviorKind.OCCURRENCE,
            occurrence_payload(
                occurred_on=day,
                name=name,
                kind_token=name,
                started_at=started,
                last_observed_at=started + timedelta(minutes=10),
                onset_available_at=started + timedelta(seconds=2),
                basis=(),
                goal=None,
                **overrides,
            ),
            links=links,
        )
        return str(BehaviorURI.from_address(document.address))

    brush = occurrence("刷牙", DAY2, 23, 55, subjects=(SUBJECT, "家庭成员B"))
    wash_face = occurrence("洗脸", DAY3, 0, 3, links=(("concurrent_with", brush),))
    bed = occurrence("上床", DAY3, 0, 10, links=(("results_from", brush),))
    site.grouper.group = scripted_by_day(  # type: ignore[method-assign]
        {
            DAY2: lambda by: GroupingAssembly(
                (SceneDraft("睡前", ((by["刷牙"], SceneRole.ESSENTIAL),), pending_effects=(("牙刷好了", by["刷牙"]),)),), (), ()
            ),
            DAY3: lambda by: GroupingAssembly(
                (
                    SceneDraft(
                        "睡前",
                        ((by["洗脸"], SceneRole.ESSENTIAL), (by["上床"], SceneRole.ESSENTIAL)),
                        relations=(DraftRelation(SceneLinkType.NEEDS, reference_no=1),),
                    ),
                ),
                (),
                (),
            ),
        }
    )
    assert site.refresh(DAY2, DAY3).published == (DAY2, DAY3)
    cache = cache_for(site)

    brush_view = context_view(brush, cache, window_days=7)
    assert [item.uri for item in brush_view.concurrent] == [wash_face]  # 次日指回来的 concurrent_with
    assert brush_view.subjects == ("家庭成员B",)
    bed_view = context_view(bed, cache, window_days=7)
    assert [cause.name for cause in bed_view.causes if isinstance(cause, ActionRef)] == ["刷牙"]  # 跨日 results_from
    assert [item.kind_token for item in bed_view.prior_steps] == ["洗脸"]
    # needs 指向前一天的"睡前"这件事：落到它留下的待用前提的产生方 kind（刷牙）
    assert [(item.source, item.text, item.target_kinds) for item in bed_view.preconditions] == [("needs", "睡前", ("刷牙",))]
    wash_view = context_view(wash_face, cache, window_days=7)
    assert [item.uri for item in wash_view.concurrent] == [brush]


def test_history_contexts_filter_by_kind_coverage_and_slot_neighbourhood(tmp_path) -> None:
    site, uris = grouped_site(tmp_path)
    publish(site.behavior_tree, DAY3, "看手机", 20, 30)  # DAY3 未归组：不是历史视图
    cache = cache_for(site)
    days = (DAY1, DAY2, DAY3)

    all_phone = history_contexts("看手机", cache, days=days, window_days=7)
    assert [view.occurrence_uri for view in all_phone] == [uris["phone1"], uris["phone2"]]
    # 15 分钟槽、半宽 2 槽：20:15 在槽 81，与 20:00 的槽 80 相距 1；21:00 的槽 84 相距 4
    near = history_contexts("看手机", cache, days=days, window_days=7, slot_minutes=15, slot_index=80, slot_half_width=2)
    assert [view.occurrence_uri for view in near] == [uris["phone2"]]
    # 环形：23:55 与 00:00 的槽相邻
    publish(site.behavior_tree, DAY1, "看手机", 23, 55)
    site.refresh(DAY1, force=True)
    cache = cache_for(site)
    wrap = history_contexts("看手机", cache, days=days, window_days=7, slot_minutes=15, slot_index=0, slot_half_width=1)
    assert [view.at.strftime("%H:%M") for view in wrap] == ["23:55"]
    assert history_contexts("不存在", cache, days=days, window_days=7) == ()
    with pytest.raises(ValueError):
        history_contexts("看手机", cache, days=days, window_days=7, slot_minutes=15)
    with pytest.raises(ValueError):
        history_contexts("看手机", cache, days=days, window_days=7, slot_minutes=15, slot_index=0, slot_half_width=-1)


def test_now_context_is_observed_facts_only_and_compare_yields_a_three_valued_table(tmp_path) -> None:
    site, uris = grouped_site(tmp_path)
    # DAY4 晚上：先商量晚餐（交谈，已结束）、正在看手机（ongoing）；候选是"洗菜"
    publish(site.behavior_tree, DAY4, "商量晚餐", 19, 5, kind="交谈", subjects=(SUBJECT, "家庭成员B"))
    phone_now = publish(site.behavior_tree, DAY4, "看手机", 19, 20, status="ongoing")
    cache = cache_for(site)
    history = history_contexts("洗菜", cache, days=(DAY1, DAY2), window_days=7)
    assert len(history) == 1

    now = now_context("洗菜", now=at(DAY4, 19, 26), cache=cache, window_days=7, pending_expiry_days=30)

    assert now.occurrence_uri is None and now.day == DAY4 and not now.covered
    assert now.scene is None  # 本层不推断此刻所在的事
    assert [step.kind_token for step in now.prior_steps] == ["交谈", "看手机"]
    assert now.preconditions == ()  # 待用清单为空（食材已被 DAY2 兑现）
    assert now.today_kinds == frozenset({"交谈", "看手机"})
    assert now.causes == (ActionRef(phone_now, "看手机", "看手机"),)
    assert now.last_time is not None and now.last_time.uri == uris["wash"] and now.last_time.days_ago == 2
    assert now.last_time.scene is not None and now.last_time.scene.label == "准备晚饭"
    assert [item.kind_token for item in now.concurrent] == ["看手机"]  # 只按 status：ongoing 的才算
    assert now.subjects == ()  # 最近一条只有主体自己

    table = compare(now, history)

    verdicts = {row.slot: row.verdict for row in table.rows}
    assert tuple(verdicts) == COMPARED_SLOTS and "time" not in verdicts and table.history_count == 1
    assert table.now is now
    scene_row = table.row("scene")
    assert scene_row.verdict == Verdict.UNKNOWN and scene_row.matched == ("准备晚饭",) and scene_row.evidence == (uris["wash"],)
    assert verdicts["prior_steps"] == Verdict.MATCHED and table.row("prior_steps").matched == ("交谈",)
    assert verdicts["preconditions"] == Verdict.UNMATCHED  # 历史前提是买菜，今天没买、清单也没有
    assert verdicts["causes"] == Verdict.UNMATCHED  # 历史起因是采购（买菜），此刻起因是看手机
    assert verdicts["last_time"] == Verdict.UNKNOWN  # 历史那次没有上一次
    assert verdicts["concurrent"] == Verdict.UNKNOWN
    assert verdicts["subjects"] == Verdict.UNKNOWN
    assert verdicts["fact"] == Verdict.UNKNOWN and table.row("fact").evidence == (uris["wash"],)
    rendered = render_table(table)
    assert "| 所在的事 | 缺信息 | — | 准备晚饭 |" in rendered and "| 前提 | 没对上 |" in rendered


def test_cross_day_pending_preconditions_match_through_the_producer_kind(tmp_path) -> None:
    """待用清单项带产生方的 kind：DAY3 买了菜（未兑现），DAY4 候选洗菜的历史前提"needs 买菜"对上。"""

    site, uris = grouped_site(tmp_path)
    publish(site.behavior_tree, DAY3, "去超市买菜", 15, 0, kind="买菜")
    site.grouper.group = scripted_by_day(  # type: ignore[method-assign]
        {
            DAY3: lambda by: GroupingAssembly(
                (SceneDraft("采购", ((by["去超市买菜"], SceneRole.ESSENTIAL),), pending_effects=(("家里有明天要用的食材", by["去超市买菜"]),)),),
                (),
                (),
            )
        }
    )
    assert site.refresh(DAY3).published == (DAY3,)
    publish(site.behavior_tree, DAY4, "商量晚餐", 19, 5, kind="交谈")
    cache = cache_for(site)
    history = history_contexts("洗菜", cache, days=(DAY1, DAY2), window_days=7)

    now = now_context("洗菜", now=at(DAY4, 19, 26), cache=cache, window_days=7, pending_expiry_days=30)

    assert [(item.text, item.target_kinds) for item in now.preconditions] == [("家里有明天要用的食材", ("买菜",))]
    row = compare(now, history).row("preconditions")
    assert row.verdict == Verdict.MATCHED and row.matched == ("买菜",) and row.evidence == (uris["wash"],)
    assert cache.pending(DAY4, 30) is cache.pending(DAY4, 30)  # 一次查询只算一次


def test_now_context_spans_midnight_and_rejects_history_from_the_future(tmp_path) -> None:
    site, uris = grouped_site(tmp_path)
    publish(site.behavior_tree, DAY3, "商量晚餐", 23, 50, kind="交谈")
    cache = cache_for(site)
    history = history_contexts("洗菜", cache, days=(DAY1, DAY2), window_days=7)

    now = now_context("洗菜", now=at(DAY4, 0, 5), cache=cache, window_days=7, pending_expiry_days=30)

    assert [step.kind_token for step in now.prior_steps] == ["交谈"]  # 昨晚 23:50 在一小时窗内
    assert now.today_kinds == frozenset()
    assert compare(now, history).row("prior_steps").verdict == Verdict.MATCHED
    with pytest.raises(ValueError, match="precede"):
        compare(now_context("洗菜", now=at(DAY2, 19, 0), cache=cache, window_days=7, pending_expiry_days=30), history)
    with pytest.raises(ValueError, match="kind_token"):
        compare(now, (history_contexts("看手机", cache, days=(DAY1,), window_days=7)[0],))
    empty = compare(now_context("不存在", now=at(DAY4, 0, 5), cache=cache, window_days=7, pending_expiry_days=30), ())
    assert {row.verdict for row in empty.rows} == {Verdict.UNKNOWN}


def test_cache_keeps_only_covered_days_and_reads_today_fresh(tmp_path) -> None:
    site, _ = grouped_site(tmp_path)
    cache = cache_for(site)
    assert cache.day(DAY1) is cache.day(DAY1)  # 已归组：缓存
    first = cache.day(DAY4)
    publish(site.behavior_tree, DAY4, "看手机", 9, 0)
    assert cache.day(DAY4) is not first and len(cache.day(DAY4).occurrences) == 1  # 未归组：每次重读
    with pytest.raises(ValueError):
        DayIndexCache(site.behavior_tree, site.scene_tree, subject=" ")


def test_view_model_guards(tmp_path) -> None:
    with pytest.raises(ValueError):
        ContextView(kind_token="", at=at(date(2026, 8, 15), 1, 0))
    view = ContextView(kind_token="x", at=at(date(2026, 8, 15), 13, 7))
    assert (view.weekday, view.minute_of_day) == (5, 13 * 60 + 7)
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    with pytest.raises(ValueError):
        now_context("x", now=at(date(2026, 8, 15), 1, 0).replace(tzinfo=None), cache=cache_for(site), window_days=7, pending_expiry_days=30)

