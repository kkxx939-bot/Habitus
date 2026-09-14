"""待关联清单：树上的出处日减去已关联完成的日期，按候选配额取最早的那几件。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.prediction.model import SlotKey
from habitus.scene import AssociationTask, backlog
from habitus.scene.backlog import BlockedDays, slot_order
from tests.unit.foresight.fixtures import MONDAY, Ground, at, slot_of


class Associated:
    """脚本化的"已关联完成的日期"；生产实现是 ``RegularityTree.days_for``。"""

    def __init__(self, **days: tuple[date, ...]) -> None:
        self.days = {kind: frozenset(value) for kind, value in days.items()}
        self.asked: list[str] = []

    def days_for(self, kind_token: str) -> frozenset[date]:
        self.asked.append(kind_token)
        return self.days.get(kind_token, frozenset())


def weekly_ground(tmp_path, *, weeks: int = 4) -> Ground:
    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=7 * weeks), 12, 0))
    for week in range(weeks):
        ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球")
    return ground


def days(*offsets: int) -> list[date]:
    return [MONDAY + timedelta(days=offset) for offset in offsets]


def test_a_candidate_nobody_has_associated_yet_owes_every_day(tmp_path) -> None:
    """新候选一天都没关联过，它的全部出处日都在待办里——范围是树给的，不是任何日历窗。"""

    tree = weekly_ground(tmp_path).tree()
    tasks = backlog(tree, Associated(), per_candidate=50, limit=50)
    assert [task.day for task in tasks] == days(0, 7, 14, 21)
    assert all(task.slots == (SlotKey(weekday=0, slot=slot_of(19, 0)),) for task in tasks)


def test_only_the_difference_is_owed(tmp_path) -> None:
    """已关联完成的日子不再进清单；重跑一次不会重复关联（差集天然幂等）。"""

    tree = weekly_ground(tmp_path).tree()
    tasks = backlog(tree, Associated(打球=tuple(days(0, 7))), per_candidate=50, limit=50)
    assert [task.day for task in tasks] == days(14, 21)
    assert backlog(tree, Associated(打球=tuple(days(0, 7, 14, 21))), per_candidate=50, limit=50) == ()


def test_each_candidate_gets_its_earliest_ones_first(tmp_path) -> None:
    """候选**内部**必须升序：规律级增量叠加，"朋友教 → 约朋友"的演化顺序靠它。

    这条曾经是假绿——用例只有一个候选，追加顺序（按候选名）与日期顺序恰好同构，所以把截断挪到
    排序之前也照样全绿。现在用两个候选，且**候选名序与日期序相反**。
    """

    ground = weekly_ground(tmp_path)  # 打球：MONDAY 起四个周一
    ground.record(MONDAY, "阿", 8, 0, kind="阿")  # 名字最小，日子最早
    ground.record(MONDAY + timedelta(days=28), "阿", 8, 0, kind="阿")
    tree = ground.tree()
    each = backlog(tree, Associated(), per_candidate=1, limit=50)
    assert [(task.kind_token, task.day) for task in each] == [("阿", MONDAY), ("打球", MONDAY)]
    # 全局上限仍然按时间升序截断，留最早的
    assert [task.day for task in backlog(tree, Associated(), per_candidate=50, limit=2)] == days(0, 0)


def test_an_old_candidate_does_not_starve_a_new_one(tmp_path) -> None:
    """全局按日期截断会让最老的那个候选把预算整段吃光，而"低频但重要"最需要上下文。"""

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=70), 12, 0))
    for offset in range(60):
        ground.record(MONDAY + timedelta(days=offset), "刷牙", 7, 0, kind="刷牙")
    ground.record(MONDAY + timedelta(days=59), "体检", 9, 0, kind="体检")
    tasks = backlog(ground.tree(), Associated(), per_candidate=5, limit=50)
    assert "体检" in {task.kind_token for task in tasks}
    assert sum(1 for task in tasks if task.kind_token == "刷牙") == 5


def test_one_day_two_slots_is_one_task_carrying_both_cells(tmp_path) -> None:
    """同一天做了两次、落在不同槽，仍然只是**一件**待关联的事，但两个格子都要带上。"""

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=2), 12, 0))
    ground.record(MONDAY, "喝水", 9, 0, kind="喝水")
    ground.record(MONDAY, "喝水", 15, 0, kind="喝水")
    tasks = backlog(ground.tree(), Associated(), per_candidate=5, limit=50)
    assert len(tasks) == 1
    assert tasks[0].slots == (SlotKey(weekday=0, slot=slot_of(9, 0)), SlotKey(weekday=0, slot=slot_of(15, 0)))


def test_tasks_are_ordered_by_time_then_slot_then_candidate(tmp_path) -> None:
    """同一天同一槽的两个候选也要有确定的先后，否则每轮顺序随字典序漂。"""

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=3), 12, 0))
    ground.record(MONDAY + timedelta(days=1), "晚饭", 18, 0, kind="晚饭")
    ground.record(MONDAY, "beta", 8, 0, kind="beta")
    ground.record(MONDAY, "alpha", 8, 0, kind="alpha")  # 同日同槽，按 token 定序
    ground.record(MONDAY, "午饭", 12, 0, kind="午饭")
    tasks = backlog(ground.tree(), Associated(), per_candidate=5, limit=50)
    assert [(task.day, task.kind_token) for task in tasks] == [
        (MONDAY, "alpha"),
        (MONDAY, "beta"),
        (MONDAY, "午饭"),
        (MONDAY + timedelta(days=1), "晚饭"),
    ]


def test_earlier_days_cells_never_enter_the_backlog(tmp_path) -> None:
    """ "当天更早已经做过"记到的格子不是一次发生，不该欠它的关联。

    这条曾经是假绿：现场只有一天一次发生，那些格子 ``occurred_days == 0``、根本不会被发布，
    断言恒真。现在用**两天同周几、第二天做得更晚**——第一天在 09:00 那一格上只有 earlier_days，
    把它记进出处的话，08-03 那件任务会挂上一个当天根本没发生的格子。
    """

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=9), 12, 0))
    ground.record(MONDAY, "吃药", 7, 0, kind="吃药")
    ground.record(MONDAY + timedelta(days=7), "吃药", 9, 0, kind="吃药")
    tasks = {task.day: task.slots for task in backlog(ground.tree(), Associated(), per_candidate=5, limit=50)}
    assert tasks[MONDAY] == (SlotKey(weekday=0, slot=slot_of(7, 0)),)
    assert tasks[MONDAY + timedelta(days=7)] == (SlotKey(weekday=0, slot=slot_of(9, 0)),)


def test_each_candidate_is_asked_exactly_once(tmp_path) -> None:
    """每个候选只问一次已关联日期；生产实现每次要走三层目录遍历，N+1 会很疼。"""

    ground = weekly_ground(tmp_path)
    ground.record(MONDAY, "喝水", 9, 0, kind="喝水")
    done = Associated()
    backlog(ground.tree(), done, per_candidate=5, limit=50)
    assert sorted(done.asked) == ["喝水", "打球"] or sorted(done.asked) == ["打球", "喝水"]
    assert len(done.asked) == len(set(done.asked)) == 2


def test_days_for_must_return_plain_dates(tmp_path) -> None:
    """返回 ``datetime`` 会让 ``day in done`` 永远为假——同一批日子每轮都被重新关联。"""

    class Wrong:
        def __init__(self, value: object) -> None:
            self.value = value

        def days_for(self, kind_token: str) -> object:
            return self.value

    tree = weekly_ground(tmp_path).tree()
    moment = datetime(2026, 8, 3, 19, 0, tzinfo=timezone(timedelta(hours=8)))
    for bad in (frozenset({moment}), {moment}, [MONDAY], None):
        with pytest.raises(TypeError):
            backlog(tree, Wrong(bad), per_candidate=5, limit=50)


def test_backlog_guards(tmp_path) -> None:
    tree = weekly_ground(tmp_path).tree()
    for field, bad in (
        ("limit", 0),
        ("limit", -1),
        ("limit", True),
        ("limit", 1.5),
        ("per_candidate", 0),
        ("per_candidate", True),
    ):
        kwargs = {"per_candidate": 5, "limit": 5, field: bad}
        with pytest.raises((ValueError, TypeError)):
            backlog(tree, Associated(), **kwargs)
    with pytest.raises(TypeError):
        backlog(tree, object(), per_candidate=5, limit=5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        backlog(object(), Associated(), per_candidate=5, limit=5)  # type: ignore[arg-type]


def test_association_task_guards() -> None:
    slot = SlotKey(weekday=0, slot=1)
    other = SlotKey(weekday=0, slot=2)
    with pytest.raises(ValueError):
        AssociationTask(kind_token="", day=MONDAY, slots=(slot,))
    with pytest.raises(TypeError):
        AssociationTask(kind_token="打球", day=datetime(2026, 8, 3), slots=(slot,))  # datetime 是 date 子类
    with pytest.raises(TypeError):
        AssociationTask(kind_token="打球", day=MONDAY, slots=[slot])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AssociationTask(kind_token="打球", day=MONDAY, slots=())
    with pytest.raises(ValueError):
        AssociationTask(kind_token="打球", day=MONDAY, slots=(slot, slot))  # 重复
    with pytest.raises(ValueError):
        AssociationTask(kind_token="打球", day=MONDAY, slots=(other, slot))  # 降序
    assert AssociationTask(kind_token="打球", day=MONDAY, slots=(slot, other)).slots == (slot, other)
    assert slot_order(other) > slot_order(slot)


# ── 被挡住的日期 ─────────────────────────────────────────────────────────────


def test_a_blocked_day_does_not_eat_the_candidate_quota(tmp_path) -> None:
    """一件永久失败的任务会永远排在最前面。不给它让路，那个候选从此不再有新的关联。"""

    ground = Ground(tmp_path / "blocked", now=at(MONDAY + timedelta(days=21), 23, 0))
    for week in range(3):
        ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球")
    tree = ground.tree()
    first = backlog(tree, Associated(), per_candidate=1, limit=10)[0]

    blocked: BlockedDays = Associated(打球=(first.day,))
    nexts = backlog(tree, Associated(), per_candidate=1, limit=10, blocked=blocked)

    assert [task.day for task in nexts] == [first.day + timedelta(days=7)]


def test_the_blocked_source_is_asked_once_per_candidate(tmp_path) -> None:
    ground = Ground(tmp_path / "asked", now=at(MONDAY + timedelta(days=7), 23, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")
    ground.record(MONDAY, "洗澡", 20, 0, kind="洗澡")
    blocked = Associated()

    backlog(ground.tree(), Associated(), per_candidate=5, limit=10, blocked=blocked)

    assert sorted(blocked.asked) == ["打球", "洗澡"] and len(blocked.asked) == 2


def test_a_blocked_source_returning_datetimes_is_refused(tmp_path) -> None:
    ground = Ground(tmp_path / "wrong", now=at(MONDAY + timedelta(days=7), 23, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")

    class _Datetimes:
        def days_for(self, kind_token: str) -> frozenset[date]:
            return frozenset({at(MONDAY, 19, 0)})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="blocked days_for"):
        backlog(ground.tree(), Associated(), per_candidate=5, limit=10, blocked=_Datetimes())


def test_a_blocked_source_without_the_method_is_refused(tmp_path) -> None:
    ground = Ground(tmp_path / "nomethod", now=at(MONDAY + timedelta(days=7), 23, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")
    with pytest.raises(TypeError, match="blocked must provide"):
        backlog(ground.tree(), Associated(), per_candidate=5, limit=10, blocked=object())  # type: ignore[arg-type]
