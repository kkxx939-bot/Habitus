"""预测层测试的现场：一棵真实的行为树同时喂出**预测树**与**语义层**，两边是同一批 occurrence。

四层出处的全部意义就是"数字与背景来自同一批日子"，所以夹具不能一边造假树一边造假情景——
两棵树必须从同一棵行为树派生，与夜批的真实顺序一致（行为树封口 → 语义层关联 → 预测树重建）。
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.foresight import AssociatedDays
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.prediction import builder, source
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.model import PredictionTree
from habitus.scene.views import DayIndexCache
from tests.unit.behavior.tree_payloads import gap_payload
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

SLOT_MINUTES = 15


def config(**overrides) -> PredictionTreeConfig:
    """几乎关掉收缩的一组参数：四层的裸比值要能直接手算。"""

    values = dict(
        slot_minutes=SLOT_MINUTES,
        decay_half_life_days=3_650.0,
        recent_half_life_days=14.0,
        recurrence_half_life_days=3_650.0,
        pool_half_width=2,
        shrink_slot_to_pool=0.001,
        shrink_pool_to_weekday=0.001,
        shrink_weekday_to_all_day=0.001,
        laplace_epsilon=0.001,
        transition_window_seconds=7_200.0,
        shrink_edge=0.001,
        recurrence_window_days=90.0,
        rebuild_interval_seconds=86_400.0,
        published_generations=3,
    )
    values.update(overrides)
    return PredictionTreeConfig(**values)


def slot_of(hour: int, minute: int = 0) -> int:
    return (hour * 60 + minute) // SLOT_MINUTES


class Ground:
    """一棵行为树 + 由它派生的语义层与预测树。"""

    def __init__(self, tmp_path: Path, *, now: datetime) -> None:
        self.site = Site(tmp_path, now=now)

    def record(self, day: date, name: str, hour: int, minute: int = 0, *, kind: str | None = None) -> str:
        return publish(self.site.behavior_tree, day, name, hour, minute, kind=kind)

    def gap(
        self, day: date, start_hour: int, start_minute: int, end_hour: int, end_minute: int, *, kind: str = "没读懂"
    ) -> None:
        """一段观测空白：删失与曝光都靠它，没有它测不出"那段没看清"。"""

        writer = BehaviorDocumentWriter(
            self.site.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(day, 23, 59)
        )
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(
                occurred_on=day,
                started_at=at(day, start_hour, start_minute),
                ended_at=at(day, end_hour, end_minute),
                gap_kind=kind,
            ),
        )

    def associated(self, *days: date) -> AssociatedDays:
        """"这个候选哪几天关联完成了"这个事实源。按天关联已删，语义层现在按候选累积，所以它
        接的是候选而不是日子——同一天可能这个候选做完了、那个还没做。"""

        done = frozenset(days)
        return lambda _kind: done

    def tree(self, **overrides) -> PredictionTree:
        snapshot = source.read(self.site.behavior_tree)
        latest = snapshot.latest_day
        assert latest is not None
        return builder.build(
            snapshot,
            config=config(**overrides),
            reference=latest,
            built_at=datetime(2026, 12, 31, tzinfo=UTC),
        )

    def cache(self) -> DayIndexCache:
        return DayIndexCache(self.site.behavior_tree, subject=SUBJECT)


#: 各现场共用的锚点周一。现场常量属于夹具模块，不能挂在某个测试文件上让别人去 import。
MONDAY = date(2026, 8, 3)

__all__ = ["MONDAY", "SLOT_MINUTES", "Ground", "at", "config", "slot_of"]
