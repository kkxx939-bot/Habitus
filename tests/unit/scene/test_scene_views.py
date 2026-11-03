"""读侧的按日索引：一天的原子行为与观测空白。全部机械、零 LLM。

这一组守的是与 temporal / 事件序列"同口径"的对齐点，漂了不会有任何地方报错：
每行带自己那天的次数、观测空白的反证规则、跨多天的空白从中间那天也看得见。
"""

from __future__ import annotations

from habitus.behavior.tree.writer import BehaviorDocumentWriter
from habitus.scene.views import DayIndexCache
from habitus.series.reader import admitted
from tests.unit.behavior.tree_payloads import gap_payload
from tests.unit.kind_ids import kind_id
from tests.unit.scene.fixtures import DAY1, DAY2, SUBJECT, Site, at, publish


def site(tmp_path) -> Site:
    return Site(tmp_path, now=at(DAY2, 23, 0))


def cache_for(ground: Site) -> DayIndexCache:
    return DayIndexCache(ground.behavior_tree, subject=SUBJECT, admits=admitted)


def test_rows_carry_their_own_days_count_of_their_kind(tmp_path) -> None:
    """跨日的行各带各天的次数：周日 23:50 的刷牙数的是周日全天，不借周一的。"""

    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "刷牙", 8, 0)
    late = publish(ground.behavior_tree, DAY1, "刷牙", 23, 50)
    own = publish(ground.behavior_tree, DAY2, "刷牙", 0, 5)
    cache = cache_for(ground)

    assert [(row.at.hour, row.day_count) for row in cache.day(DAY1).rows] == [(8, 2), (23, 2)]
    assert cache.day(DAY1).by_uri[late].day_count == 2
    assert cache.day(DAY2).by_uri[own].day_count == 1 and cache.day(DAY2).by_uri[own].kind_token == kind_id("刷牙")
    assert cache.day(DAY2).by_uri[own].last_observed_at == at(DAY2, 0, 15)


def test_an_unreadable_gap_is_voided_by_a_behaviour_that_started_inside_it(tmp_path) -> None:
    """ "没读懂"的那段里若读出了一条行为的开始，整段作废——我们在看，只是没读懂，读出来了就
    证伪了。这条直接决定观测覆盖（temporal.coverage）的分母。"""

    from habitus.behavior.model import BehaviorKind
    from tests.lock_store import ProcessLocalLockStore

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


def test_the_day_is_read_once_per_query(tmp_path) -> None:
    """一次装配会为每个候选的每一层问同一天很多次；不记住的话就是上万次解码。"""

    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "洗菜", 19, 43)
    cache = cache_for(ground)

    assert cache.day(DAY1) is cache.day(DAY1)


def test_a_day_has_no_note_when_there_is_no_calendar(tmp_path) -> None:
    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "洗菜", 19, 43)

    assert cache_for(ground).day(DAY1).day_note is None


def test_a_gap_that_spans_several_days_is_seen_from_its_middle_days(tmp_path) -> None:
    """空白文档只落在起始日目录；出差三天的"未观测"从第二天读起必须还在，否则第二天会被读成
    "整天都在看、什么都没发生"。"""

    from datetime import timedelta

    from habitus.behavior.model import BehaviorKind
    from tests.lock_store import ProcessLocalLockStore

    ground = site(tmp_path)
    writer = BehaviorDocumentWriter(ground.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(DAY2, 23, 0))
    writer.publish(
        BehaviorKind.GAP,
        gap_payload(
            occurred_on=DAY1 - timedelta(days=2),
            started_at=at(DAY1 - timedelta(days=2), 8, 0),
            ended_at=at(DAY2, 20, 0),
            gap_kind="未观测",
        ),
    )
    middle = cache_for(ground).day(DAY1)
    assert [(gap.started_at.hour, gap.ended_at.hour, gap.kind) for gap in middle.gaps] == [(0, 0, "未观测")]
    assert middle.gaps[0].ended_at == at(DAY1 + timedelta(days=1), 0, 0)


def test_every_day_sees_the_same_gaps_as_the_series(tmp_path) -> None:
    """按天读树与事件序列对空白只有一个读法：跨午夜的「没读懂」两边各被一条行为证伪一半、跨三天的「未观测」，
    每一天 ``DayIndex.gaps`` 都等于序列里落在那一天的段（结算、递送读前者，关系、temporal、机会表读后者）。"""

    from datetime import timedelta

    from habitus.behavior.model import BehaviorKind
    from habitus.series.reader import read_series
    from tests.lock_store import ProcessLocalLockStore

    ground = Site(tmp_path, now=at(DAY2 + timedelta(days=4), 3, 0))
    writer = BehaviorDocumentWriter(ground.behavior_tree, ProcessLocalLockStore(), clock=lambda: ground.now)
    days = [DAY1 + timedelta(days=offset) for offset in range(5)]
    for start, end, kind in (
        ((days[0], 23, 0), (days[1], 2, 0), "没读懂"),
        ((days[1], 23, 0), (days[2], 1, 0), "没读懂"),
        ((days[2], 20, 0), (days[4], 8, 0), "未观测"),
    ):
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(occurred_on=start[0], started_at=at(*start), ended_at=at(*end), gap_kind=kind),
        )
    publish(ground.behavior_tree, days[0], "看书", 21, 0)
    publish(ground.behavior_tree, days[1], "看手机", 0, 30)  # 证伪第一段的后一半
    publish(ground.behavior_tree, days[1], "刷牙", 23, 30)  # 证伪第二段的前一半
    publish(ground.behavior_tree, days[4], "起床", 9, 0)
    series = read_series(ground.behavior_tree, cutoff=days[4] + timedelta(days=1))
    cache = cache_for(ground)
    for day in days:
        assert [(gap.started_at, gap.ended_at, gap.kind) for gap in cache.day(day).gaps] == [
            (gap.started_at, gap.ended_at, gap.kind.value) for gap in series.gaps_on(day)
        ]
    assert [(gap.started_at, gap.ended_at) for gap in cache.day(days[1]).gaps] == []
    assert [(gap.started_at, gap.ended_at) for gap in cache.day(days[2]).gaps] == [
        (at(days[2], 0, 0), at(days[2], 1, 0)),
        (at(days[2], 20, 0), at(days[3], 0, 0)),
    ]
