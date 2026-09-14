"""读时投影：行为侧的上下文，按预测树算候选的维度对齐。全部机械、零 LLM。

这一组守的是四块非平凡逻辑——它们都是与预测树"同口径"的对齐点，漂了不会有任何地方报错：
钟面邻域过滤、并行关系的对称闭包、主体剔除、观测空白的反证规则。
"""

from __future__ import annotations

import pytest

from habitus.scene.views import ContextView, DayIndexCache, Neighbour, context_view, history_contexts
from tests.unit.behavior.tree_payloads import gap_payload
from tests.unit.scene.fixtures import DAY1, DAY2, SUBJECT, Site, at, publish


def site(tmp_path) -> Site:
    return Site(tmp_path, now=at(DAY2, 23, 0))


def cache_for(ground: Site) -> DayIndexCache:
    return DayIndexCache(ground.behavior_tree, subject=SUBJECT)


def test_only_the_clock_face_neighbourhood_is_taken(tmp_path) -> None:
    """槽的口径与预测树同一公式：``minute_of_day // slot_minutes``，环形距离 ≤ 半宽。

    同一天同一 kind 两条落在不同槽——不按槽过滤的话，远处那条也会混进这一格的背景。
    """

    ground = site(tmp_path)
    near = publish(ground.behavior_tree, DAY1, "洗菜", 20, 0)
    publish(ground.behavior_tree, DAY1, "洗菜", 8, 0)  # 槽 32，离槽 80 有 48 格

    views = history_contexts(
        "洗菜", cache_for(ground), days=(DAY1,), window_days=7, slot_minutes=15, slot_index=80, slot_half_width=2
    )

    assert [view.occurrence_uri for view in views] == [near]


def test_the_neighbourhood_wraps_around_midnight(tmp_path) -> None:
    """环形：23:55 与 00:00 的槽相邻。"""

    ground = site(tmp_path)
    late = publish(ground.behavior_tree, DAY1, "洗菜", 23, 55)

    views = history_contexts(
        "洗菜", cache_for(ground), days=(DAY1,), window_days=7, slot_minutes=15, slot_index=0, slot_half_width=1
    )

    assert [view.occurrence_uri for view in views] == [late]


def test_a_one_way_concurrent_link_is_seen_from_both_ends(tmp_path) -> None:
    """行为树只存前向边。并行是对称的事实，投影时要取对称闭包——只认单向的话，被指的那条
    看不见指它的那条，而"同时在做什么"对两边都成立。"""

    ground = site(tmp_path)
    earlier = publish(ground.behavior_tree, DAY1, "听播客", 19, 40)
    later = publish(ground.behavior_tree, DAY1, "洗菜", 19, 43, links=(("concurrent_with", earlier),))
    cache = cache_for(ground)

    assert [item.uri for item in context_view(later, cache, window_days=7).concurrent] == [earlier]
    assert [item.uri for item in context_view(earlier, cache, window_days=7).concurrent] == [later]


def test_the_subject_is_dropped_from_who_else_was_there(tmp_path) -> None:
    """"和谁"是主体之外的人。主体总在 subjects 里，不剔掉的话每一条都会说"和自己"。"""

    ground = site(tmp_path)
    uri = publish(ground.behavior_tree, DAY1, "商量晚餐", 19, 12, subjects=(SUBJECT, "家庭成员B"))

    assert context_view(uri, cache_for(ground), window_days=7).subjects == ("家庭成员B",)


def test_an_unreadable_gap_is_voided_by_a_behaviour_that_started_inside_it(tmp_path) -> None:
    """"没读懂"的那段里若读出了一条行为的开始，整段作废——我们在看，只是没读懂，读出来了就
    证伪了。这条直接决定预测树的曝光分母。"""

    from habitus.behavior import BehaviorDocumentWriter
    from habitus.behavior.model import BehaviorKind
    from habitus.infrastructure.store.locks import ProcessLocalLockStore

    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "洗菜", 20, 10)
    writer = BehaviorDocumentWriter(ground.behavior_tree, ProcessLocalLockStore(), clock=lambda: ground.now)
    writer.publish(
        BehaviorKind.GAP,
        gap_payload(occurred_on=DAY1, started_at=at(DAY1, 20, 0), ended_at=at(DAY1, 20, 30), gap_kind="没读懂"),
    )
    writer.publish(
        BehaviorKind.GAP,
        gap_payload(occurred_on=DAY1, started_at=at(DAY1, 22, 0), ended_at=at(DAY1, 22, 30), gap_kind="没读懂"),
    )

    gaps = cache_for(ground).day(DAY1).gaps

    assert [(gap.started_at.hour, gap.ended_at.hour) for gap in gaps] == [(22, 22)]


def test_causes_come_from_the_behaviour_trees_own_results_from_edge(tmp_path) -> None:
    """起因是语义边，与"此前"不是一回事：一个说因果，一个只说先后。"""

    ground = site(tmp_path)
    talk = publish(ground.behavior_tree, DAY1, "商量晚餐", 19, 12, kind="交谈")
    cook = publish(ground.behavior_tree, DAY1, "查配方", 19, 20, links=(("results_from", talk),))

    view = context_view(cook, cache_for(ground), window_days=7)

    assert [item.uri for item in view.causes] == [talk]


def test_the_neighbours_are_the_ones_immediately_before_and_after(tmp_path) -> None:
    """转移边看的是时间上**紧邻**的上一条与下一条；方向不能弄反。"""

    ground = site(tmp_path)
    before = publish(ground.behavior_tree, DAY1, "商量晚餐", 19, 12, kind="交谈")
    middle = publish(ground.behavior_tree, DAY1, "洗菜", 19, 43)
    after = publish(ground.behavior_tree, DAY1, "煮汤", 19, 55)

    view = context_view(middle, cache_for(ground), window_days=7, transition_window_seconds=3600)

    assert view.preceding == Neighbour(cache_for(ground).day(DAY1).ref(before))
    assert view.following == Neighbour(cache_for(ground).day(DAY1).ref(after))


def test_a_day_with_nothing_of_that_kind_yields_no_views(tmp_path) -> None:
    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "洗菜", 19, 43)

    assert history_contexts("从没做过", cache_for(ground), days=(DAY1,), window_days=7) == ()


def test_a_context_view_is_refused_for_an_unknown_occurrence(tmp_path) -> None:
    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "洗菜", 19, 43)
    with pytest.raises(KeyError):
        context_view("behavior://occurrences/2026/08/15/没有--20260815T000000000000%2B0800.md", cache_for(ground), window_days=7)


def test_the_day_is_read_once_per_query(tmp_path) -> None:
    """一次装配会为每个候选的每一层问同一天很多次；不记住的话就是上万次解码。"""

    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "洗菜", 19, 43)
    cache = cache_for(ground)

    assert cache.day(DAY1) is cache.day(DAY1)


def test_a_view_carries_the_day_note_even_when_there_is_no_calendar(tmp_path) -> None:
    ground = site(tmp_path)
    uri = publish(ground.behavior_tree, DAY1, "洗菜", 19, 43)

    view: ContextView = context_view(uri, cache_for(ground), window_days=7)

    assert view.day_note is None
