"""上下文视图：历史投影的槽位、钟面邻域筛选、此刻视图、逐槽三值表，以及按预测树维度对齐的字段、聚合画像与相似情景反查。全部机械、零 LLM。"""

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
    NO_NEIGHBOUR,
    ActionRef,
    ContextView,
    DayIndexCache,
    Neighbour,
    ObservationGap,
    SceneRef,
    Verdict,
    compare,
    context_view,
    history_contexts,
    history_profile,
    now_context,
    render_table,
    similar_scenes,
)
from habitus.scene.views.now import gaps_until
from tests.unit.behavior.tree_payloads import gap_payload, occurrence_payload
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


def test_cache_reads_each_day_once_per_query_and_checks_coverage_by_pointer(tmp_path) -> None:
    site, _ = grouped_site(tmp_path)
    cache = cache_for(site)
    assert cache.day(DAY1) is cache.day(DAY1)  # 已归组：缓存
    first = cache.day(DAY4)
    publish(site.behavior_tree, DAY4, "看手机", 9, 0)
    assert cache.day(DAY4) is first and len(cache.day(DAY4).occurrences) == 0  # 一次查询里今天也只读一次
    assert len(cache_for(site).day(DAY4).occurrences) == 1  # 新查询新缓存才看到新发布
    assert cache.covered(DAY1) and not cache.covered(DAY4) and not cache_for(site).covered(DAY4)  # 只看指针，不解码
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



# ── 按预测树维度对齐的字段与投影（2026-09-12）──────────────────────────────────────────────


def test_projection_exposes_first_of_day_next_steps_and_transition_neighbours(tmp_path) -> None:
    """``first_of_day`` 与树的 first_days 同口径；``next_steps`` 与 ``prior_steps`` 同口径（同一件事里的成员）；
    ``preceding`` / ``following`` 是转移窗口内时间上紧邻的那条，只在给了窗口时才算，三态。"""

    site, uris = grouped_site(tmp_path)
    cache = cache_for(site)

    plain = context_view(uris["wash"], cache, window_days=7)
    assert plain.preceding is None and plain.following is None  # 没给窗口：不算，不是"没有"
    assert plain.first_of_day is True and [step.kind_token for step in plain.next_steps] == ["看手机"]

    wash = context_view(uris["wash"], cache, window_days=7, transition_window_seconds=3600)
    assert wash.preceding == Neighbour(ActionRef(uris["discuss"], "商量晚餐", "交谈"))  # 19:12 → 19:43，31 分钟
    assert wash.following == Neighbour(ActionRef(uris["phone2"], "看手机", "看手机"))  # 19:43 → 20:15，32 分钟
    tight = context_view(uris["wash"], cache, window_days=7, transition_window_seconds=1800)
    assert tight.preceding == Neighbour(None) and tight.preceding.absent and tight.preceding.value == NO_NEIGHBOUR  # 半小时内确认没有
    assert tight.following == Neighbour(None)
    discuss = context_view(uris["discuss"], cache, window_days=7, transition_window_seconds=3600)
    assert discuss.preceding == Neighbour(None) and discuss.following is not None and discuss.following.value == "洗菜"
    assert discuss.prior_steps == () and [step.kind_token for step in discuss.next_steps] == ["洗菜", "看手机"]
    # 20:15 看手机之后 20:30–20:50 有观测空白：一小时窗内没找到下一条，且窗口没看全 → 删失
    phone = context_view(uris["phone2"], cache, window_days=7, transition_window_seconds=3600)
    assert phone.following == Neighbour(None, censored=True) and phone.following.value is None and not phone.following.absent
    with pytest.raises(ValueError):
        context_view(uris["wash"], cache, window_days=7, transition_window_seconds=0)
    with pytest.raises(ValueError, match="one day"):
        history_contexts("洗菜", cache, days=(DAY2,), window_days=7, transition_window_seconds=86_401)
    with pytest.raises(ValueError):
        Neighbour(ActionRef(uris["wash"], "洗菜", "洗菜"), censored=True)


def test_second_occurrence_of_the_day_is_not_first_and_a_hole_between_neighbours_censors_them(tmp_path) -> None:
    site, uris = grouped_site(tmp_path)
    second = publish(site.behavior_tree, DAY2, "洗菜", 20, 20)  # 20:15 看手机之后 5 分钟，同 kind 第二次
    site.refresh(DAY2, force=True)
    cache = cache_for(site)

    views = history_contexts("洗菜", cache, days=(DAY2,), window_days=7, transition_window_seconds=3600)
    assert [(view.occurrence_uri, view.first_of_day) for view in views] == [(uris["wash"], True), (second, False)]
    later = views[1]
    assert later.preceding == Neighbour(ActionRef(uris["phone2"], "看手机", "看手机"))
    # 20:20 之后一小时内没有下一条，而 20:30–20:50 是空白：找不到 + 窗口有洞 → 删失
    assert later.following == Neighbour(None, censored=True)
    # 中间有洞时找到了也删：把空白挪到 20:16–20:19，看手机 → 第二次洗菜 之间断档
    holed = Site(tmp_path / "holed", now=at(DAY3, 12, 0))
    a = publish(holed.behavior_tree, DAY2, "看手机", 20, 15)
    b = publish(holed.behavior_tree, DAY2, "洗菜", 20, 20)
    writer = BehaviorDocumentWriter(holed.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(DAY3, 12, 0))
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY2, started_at=at(DAY2, 20, 16), ended_at=at(DAY2, 20, 19), gap_kind="未观测"))
    holed_cache = cache_for(holed)
    assert context_view(a, holed_cache, window_days=7, transition_window_seconds=3600).following == Neighbour(None, censored=True)
    assert context_view(b, holed_cache, window_days=7, transition_window_seconds=3600).preceding == Neighbour(None, censored=True)


def test_neighbours_follow_the_total_order_skip_partners_and_cross_midnight(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY4, 12, 0))
    brush = publish(site.behavior_tree, DAY2, "刷牙", 23, 55)
    wash_face = publish(site.behavior_tree, DAY3, "洗脸", 0, 3, links=(("concurrent_with", brush),))
    bed = publish(site.behavior_tree, DAY3, "上床", 0, 10)
    twin_a = publish(site.behavior_tree, DAY3, "喝水", 0, 30)
    twin_b = publish(site.behavior_tree, DAY3, "关灯", 0, 30)  # 同一时刻的两条
    cache = cache_for(site)
    window = dict(window_days=7, transition_window_seconds=3600)

    brush_view = context_view(brush, cache, **window)
    assert brush_view.following == Neighbour(ActionRef(bed, "上床", "上床"))  # 跳过并行伙伴，跨午夜
    face_view = context_view(wash_face, cache, **window)
    assert face_view.preceding == Neighbour(None) and face_view.following == Neighbour(ActionRef(bed, "上床", "上床"))
    assert [item.uri for item in face_view.concurrent] == [brush]  # 前一天被指向的并行伙伴也进对称闭包
    bed_view = context_view(bed, cache, **window)
    assert bed_view.preceding == Neighbour(ActionRef(wash_face, "洗脸", "洗脸"))
    # 同刻两条按 (瞬时, URI) 的全序只有一个方向：靠前的那条把靠后的当下一条，反之不然
    first, second = sorted((twin_a, twin_b))
    first_view, second_view = context_view(first, cache, **window), context_view(second, cache, **window)
    assert first_view.following is not None and first_view.following.action is not None and first_view.following.action.uri == second
    assert second_view.preceding is not None and second_view.preceding.action is not None and second_view.preceding.action.uri == first
    assert first_view.preceding == Neighbour(ActionRef(bed, "上床", "上床"))
    assert second_view.following == Neighbour(None)


def test_now_context_exposes_preceding_and_todays_gaps_and_compare_has_a_preceding_row(tmp_path) -> None:
    site, uris = grouped_site(tmp_path)
    publish(site.behavior_tree, DAY4, "商量晚餐", 19, 5, kind="交谈")
    phone_now = publish(site.behavior_tree, DAY4, "看手机", 19, 20, status="ongoing")
    writer = BehaviorDocumentWriter(site.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(DAY4, 20, 0))
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY3, started_at=at(DAY3, 23, 30), ended_at=at(DAY4, 0, 45)))
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY4, started_at=at(DAY4, 19, 0), ended_at=at(DAY4, 19, 4)))
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY4, started_at=at(DAY4, 19, 24), ended_at=at(DAY4, 19, 40)))
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY4, started_at=at(DAY4, 19, 50), ended_at=at(DAY4, 19, 55)))
    cache = cache_for(site)
    history = history_contexts("洗菜", cache, days=(DAY1, DAY2), window_days=7, transition_window_seconds=3600)

    now = now_context("洗菜", now=at(DAY4, 19, 26), cache=cache, window_days=7, pending_expiry_days=30, transition_window_seconds=1800)

    # 正在进行的看手机已判入"同时在做"，不能再当紧邻上一条；上一条是 19:05 的交谈，中间 19:24 起有空白 → 删失
    assert [item.uri for item in now.concurrent] == [phone_now]
    assert now.preceding == Neighbour(None, censored=True)
    assert [(gap.started_at.strftime("%d %H:%M"), gap.ended_at.strftime("%H:%M"), gap.kind) for gap in now.gaps] == [
        ("18 00:00", "00:45", "没读懂"),  # 昨晚开始、跨午夜的空白裁到今天 00:00–00:45
        ("18 19:00", "19:04", "没读懂"),
        ("18 19:24", "19:26", "没读懂"),  # 跨过此刻：截到此刻
    ]  # 19:50 的空白还没到，不算；昨天那一半只在删失检查里用，不在今天的视图上
    assert now.gaps[0].started_at == at(DAY4, 0, 0)
    assert "preceding" in COMPARED_SLOTS
    assert compare(now, history).row("preceding").verdict == Verdict.UNKNOWN  # 此刻删失：缺信息

    clear = now_context("洗菜", now=at(DAY4, 19, 23), cache=cache, window_days=7, pending_expiry_days=30, transition_window_seconds=1800)
    assert clear.preceding is not None and clear.preceding.value == "交谈"
    row = compare(clear, history).row("preceding")
    assert row.verdict == Verdict.MATCHED and row.now == ("交谈",) and row.matched == ("交谈",) and row.evidence == (uris["wash"],)
    absent = now_context("洗菜", now=at(DAY4, 19, 23), cache=cache, window_days=7, pending_expiry_days=30, transition_window_seconds=60)
    assert absent.preceding == Neighbour(None)  # 一分钟内没有、也没有洞：确认没有
    assert compare(absent, history).row("preceding").verdict == Verdict.UNMATCHED
    plain = now_context("洗菜", now=at(DAY4, 19, 23), cache=cache, window_days=7, pending_expiry_days=30)
    assert plain.preceding is None
    assert "| 紧邻上一条 | 对上 | 交谈 | 交谈 |" in render_table(compare(clear, history))
    with pytest.raises(ValueError):
        now_context("洗菜", now=at(DAY4, 19, 26), cache=cache, window_days=7, pending_expiry_days=30, transition_window_seconds=0)


def test_absent_neighbours_compare_as_a_value(tmp_path) -> None:
    """历史上做这件事之前确认什么都没做（∅），今天也什么都没做：对上——∅ 是树上的一条边，不是缺信息。"""

    site, uris = grouped_site(tmp_path)
    cache = cache_for(site)
    history = history_contexts("交谈", cache, days=(DAY2,), window_days=7, transition_window_seconds=1800)
    assert history[0].preceding == Neighbour(None)  # DAY2 19:12 商量晚餐之前半小时没有别的
    publish(site.behavior_tree, DAY4, "看手机", 18, 0)
    now = now_context("交谈", now=at(DAY4, 19, 0), cache=cache_for(site), window_days=7, pending_expiry_days=30, transition_window_seconds=1800)
    row = compare(now, history).row("preceding")
    assert now.preceding == Neighbour(None) and row.verdict == Verdict.MATCHED and row.matched == (NO_NEIGHBOUR,)


def test_gaps_until_clips_to_now_and_drops_future_gaps() -> None:
    gaps = (
        ObservationGap(at(DAY4, 9, 0), at(DAY4, 9, 30), "没读懂"),
        ObservationGap(at(DAY4, 10, 0), at(DAY4, 11, 0), "未观测"),
        ObservationGap(at(DAY4, 12, 0), at(DAY4, 12, 30), "没读懂"),
    )
    clipped = gaps_until(gaps, at(DAY4, 10, 20))
    assert [(gap.ended_at.strftime("%H:%M"), gap.kind) for gap in clipped] == [("09:30", "没读懂"), ("10:20", "未观测")]
    with pytest.raises(ValueError):
        ObservationGap(at(DAY4, 9, 0), at(DAY4, 8, 0), "没读懂")
    with pytest.raises(ValueError):
        ObservationGap(at(DAY4, 9, 0), at(DAY4, 9, 5), "")


def test_day_index_gaps_follow_the_tree_denominator_rules(tmp_path) -> None:
    """零宽丢弃；"没读懂"段里读出了行为的开始就整段作废（我们在看，只是没读懂）。"""

    site, uris = grouped_site(tmp_path)  # DAY2 有 20:30–20:50 的"没读懂"空白，里面没有行为
    writer = BehaviorDocumentWriter(site.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(DAY3, 12, 0))
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY2, started_at=at(DAY2, 19, 40), ended_at=at(DAY2, 19, 50)))  # 洗菜 19:43 在里面 → 作废
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY2, started_at=at(DAY2, 21, 0), ended_at=at(DAY2, 21, 0)))  # 零宽
    writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY2, started_at=at(DAY2, 21, 10), ended_at=at(DAY2, 21, 20), gap_kind="未观测"))
    gaps = cache_for(site).day(DAY2).gaps
    assert [(gap.started_at.strftime("%H:%M"), gap.kind) for gap in gaps] == [("20:30", "没读懂"), ("21:10", "未观测")]
    # 未知词表在写入时就被 schema 挡住（gap_kind 枚举），读侧的硬失败是第二道闸，写不进去所以这里不构造


def test_history_profile_aggregates_by_slot_with_day_capped_time_counts(tmp_path) -> None:
    site, uris = grouped_site(tmp_path)
    cache = cache_for(site)
    views = history_contexts("看手机", cache, days=(DAY1, DAY2), window_days=7, transition_window_seconds=3600)

    profile = history_profile("看手机", views, covered_days=2)

    assert profile.occurrences == 2 and profile.days == 2 and profile.covered_days == 2 and profile.span == (DAY1, DAY2)
    assert profile.first_of_day == 2 and not profile.empty
    assert profile.by_weekday == ((5, 1), (6, 1)) and profile.by_hour == ((20, 1), (21, 1))
    assert profile.scenes == (("准备晚饭", 1),) and profile.roles == (("essential", 1),)  # DAY1 那次未归组
    assert profile.prior_steps == (("交谈", 1), ("洗菜", 1)) and profile.next_steps == ()
    assert profile.preceding == ((NO_NEIGHBOUR, 1), ("洗菜", 1))  # DAY1 21:00 那次一小时内没有上一条也没有洞 → ∅ 也是一个值
    assert profile.following == ((NO_NEIGHBOUR, 1),)  # DAY1 21:00 之后确认没有；DAY2 20:15 之后有空白 → 删失不计
    assert profile.preconditions == (("买菜", 1),) and profile.causes == (("买菜", 1),)  # 准备晚饭 needs / results_from 采购
    assert profile.concurrent == () and profile.subjects == ()
    # 周几 / 小时按不同日子封顶：同一天再看三次手机，周日仍只算 1
    for minute in (0, 20, 40):
        publish(site.behavior_tree, DAY2, "看手机", 21, minute)
    site.refresh(DAY2, force=True)
    more = history_profile("看手机", history_contexts("看手机", cache_for(site), days=(DAY1, DAY2), window_days=7))
    assert more.occurrences == 5 and more.days == 2 and more.by_weekday == ((5, 1), (6, 1)) and more.by_hour == ((20, 1), (21, 2))
    assert more.first_of_day == 2 and more.covered_days is None
    empty = history_profile("看手机", ())
    assert empty.empty and empty.span is None and empty.days == 0 and empty.by_weekday == ()
    with pytest.raises(ValueError):
        history_profile("洗菜", views)
    with pytest.raises(ValueError):
        history_profile("看手机", views, covered_days=-1)
    with pytest.raises(TypeError):
        history_profile("看手机", (object(),))  # type: ignore[arg-type]


def test_similar_scenes_reverse_lookup_orders_by_overlap_then_recency_and_only_counts_steps(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    # DAY2 的看手机归成 irrelevant：它不参与重叠，也不出现在"接下来"
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
    assert site.refresh(DAY1, DAY2).published == (DAY1, DAY2)
    cache = cache_for(site)
    days = (DAY1, DAY2, DAY3)

    dinner_only = similar_scenes(cache, days=days, kinds=frozenset({"交谈"}), limit=5)
    assert [item.label for item in dinner_only] == ["准备晚饭"]
    scene = dinner_only[0]
    assert scene.day == DAY2 and scene.overlap == ("交谈",)
    assert [(member.action.kind_token, member.role) for member in scene.members] == [("交谈", "essential"), ("洗菜", "essential"), ("看手机", "irrelevant")]
    assert [member.action.kind_token for member in scene.following] == ["洗菜"]  # 最后一个重叠步骤之后的步骤，irrelevant 不算
    assert similar_scenes(cache, days=days, kinds=frozenset({"看手机"}), limit=5) == ()  # irrelevant 成员不触发重叠

    both = similar_scenes(cache, days=days, kinds=frozenset({"买菜", "交谈"}), limit=5)
    assert [(item.label, item.overlap) for item in both] == [("准备晚饭", ("交谈",)), ("去超市采购", ("买菜",))]  # 重叠数相同：新近优先
    assert both[1].following == ()  # 采购只有一步，重叠成员之后没有别的
    assert [item.label for item in similar_scenes(cache, days=days, kinds=frozenset({"买菜", "交谈"}), limit=1)] == ["准备晚饭"]
    assert similar_scenes(cache, days=days, kinds=frozenset(), limit=5) == ()
    assert similar_scenes(cache, days=days, kinds=frozenset({"不存在"}), limit=5) == ()
    with pytest.raises(ValueError):
        similar_scenes(cache, days=days, kinds=frozenset({"交谈"}), limit=0)
    with pytest.raises(TypeError):
        similar_scenes(cache, days=days, kinds={"交谈"}, limit=5)  # type: ignore[arg-type]


def test_slot_filter_validates_the_clock_face(tmp_path) -> None:
    site, _ = grouped_site(tmp_path)
    cache = cache_for(site)
    with pytest.raises(ValueError, match="divisor"):
        history_contexts("看手机", cache, days=(DAY1,), window_days=7, slot_minutes=7, slot_index=0)
    with pytest.raises(ValueError, match="clock face"):
        history_contexts("看手机", cache, days=(DAY1,), window_days=7, slot_minutes=15, slot_index=96)


# --- 当地日历：这一天在当地是什么日子 ---------------------------------------------------


class ScriptedCalendar:
    """只对指定的几天说话，并记下自己被问了几次。"""

    def __init__(self, notes: dict[date, str | None]) -> None:
        self.notes = notes
        self.asked: list[date] = []

    def describe(self, day: date) -> str | None:
        self.asked.append(day)
        return self.notes.get(day)


def test_the_day_type_reaches_every_view_the_prediction_tree_has_a_dimension_for(tmp_path) -> None:
    """日型要跟着三个历史维度一起给出来，否则"上次补班日这个时段他做了什么"问不出来。

    它是摆在判断者面前的一个事实，**不参与筛选与排序**——今天像周几由判断者自己判，系统不替
    他把补班的周六改按周一去查树。
    """

    site, uris = grouped_site(tmp_path)
    calendar = ScriptedCalendar({DAY2: "补班日，按周一上班"})
    cache = DayIndexCache(site.behavior_tree, site.scene_tree, subject=SUBJECT, calendar=calendar)

    view = context_view(uris["wash"], cache=cache, window_days=30)
    assert view.day_note == "补班日，按周一上班"
    assert context_view(uris["buy"], cache=cache, window_days=30).day_note is None  # DAY1 是普通日子
    scenes = similar_scenes(cache, days=(DAY1, DAY2), kinds=frozenset({"交谈"}), limit=5)
    assert [(item.day, item.day_note) for item in scenes] == [(DAY2, "补班日，按周一上班")]
    now = now_context("买菜", now=at(DAY2, 21, 0), cache=cache, window_days=30, pending_expiry_days=30)
    assert now.day_note == "补班日，按周一上班"


def test_without_calendar_data_the_day_type_is_explicitly_empty(tmp_path) -> None:
    """没有日历数据时视图上恒为空——那正是我们此刻能说的全部，不是漏了一个字段。"""

    site, uris = grouped_site(tmp_path)
    assert context_view(uris["wash"], cache=cache_for(site), window_days=30).day_note is None


def test_a_blank_note_is_no_note_and_a_non_text_note_is_refused(tmp_path) -> None:
    """空白的注记当作没话说；返回非文本是插进来的日历自己坏了，当场报出来而不是往视图里塞。"""

    site, uris = grouped_site(tmp_path)
    blank = DayIndexCache(site.behavior_tree, site.scene_tree, subject=SUBJECT, calendar=ScriptedCalendar({DAY2: "   "}))
    assert context_view(uris["wash"], cache=blank, window_days=30).day_note is None
    broken = DayIndexCache(site.behavior_tree, site.scene_tree, subject=SUBJECT, calendar=ScriptedCalendar({DAY2: 7}))  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        context_view(uris["wash"], cache=broken, window_days=30)


def test_the_calendar_is_asked_once_per_day_per_query(tmp_path) -> None:
    """日历跟着索引走同一套缓存语义：一次查询里一天只问一次，换查询才重新问。

    读时算是为了永远跟上更新后的调休表；但"一次查询内部"仍要是同一个答案，不然同一次判断里
    两个维度可能拿到不同的日型。
    """

    site, _ = grouped_site(tmp_path)
    calendar = ScriptedCalendar({DAY2: "补班日"})
    cache = DayIndexCache(site.behavior_tree, site.scene_tree, subject=SUBJECT, calendar=calendar)
    cache.day(DAY2)
    cache.day(DAY2)
    cache.day(DAY1)
    assert calendar.asked == [DAY2, DAY1]
