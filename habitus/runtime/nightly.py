"""第 N 晚在组合根的串法：行为树只读一次，预测树与语义树用同一份序列（语义树新方案 ``13`` 第四节）。

```
今晚 N = min(主体所在地的"今天", 最后一个已封口日子的次日)   ← 封口由归约说了算（与结算同一个口径）
序列 = series.reader.read_series(行为树, 截止日 N)        ← 行为树每晚只读这一次
预测树 = 用这份序列建、发布（基准日 N 的前一天）
语义树 = SceneNightlyRun.run(序列, 这一代树)               ← B1 同步词表 → B2 写概念 → B3 映射 → B4 关系
```

两棵树对同一条记录的读法一致，由序列保证（CI 钉住）；预测树一个模块都不碰行为树。行为树上还没有一条可用记录时
预测树不发布，语义树这一晚也不跑（没有树就没有节律，也没有可量的东西）。

词表迁移做到一半时不跑（第四轮评审 E3）：迁移先改树上的行、最后才写新版本，做到一半读出来的序列一半新编号一半旧编号；
读序列前后各看一次词表状态（版本、有没有未完成的迁移），不一致就这一晚不发布、不跑语义树，下一次再来。

截止日不能只取"今天"（第四轮评审 E1）：零点刚过时昨天那条链可能还在融合静默期、没归约，夜批把昨天当成完整的一天映射、
盖完成章，之后补发进来的记录就永远不映射了；预测树也会用上没封口的一天。所以只用到最后一个已封口的日子为止。
接进常驻 worker 是第 11 步：这里只把一晚的跑法做成一个可注入的对象。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from habitus.behavior.tree import BehaviorTree
from habitus.prediction.store import PublishedGeneration
from habitus.runtime.prediction import PredictionRebuilder
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.series import EventSeries
from habitus.series.reader import read_series


@dataclass(frozen=True)
class NightOutcome:
    night: date
    series: EventSeries
    published: PublishedGeneration | None
    scene: NightReport | None
    #: 这一晚为什么没跑（没跑是 None）。
    skipped: str | None = None


class Nightly:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        prediction: PredictionRebuilder,
        scene: SceneNightlyRun,
        closed_days: Callable[[], Iterable[date]],
        vocabulary: Callable[[], tuple[int, bool]],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if prediction.behavior_tree is not behavior_tree or scene.behavior_tree is not behavior_tree:
            raise ValueError("the prediction and scene runs must read the same behaviour tree")
        self.behavior_tree = behavior_tree
        self.prediction = prediction
        self.scene = scene
        if not callable(closed_days):
            raise TypeError("closed_days must be callable")
        self.closed_days = closed_days
        if not callable(vocabulary):
            raise TypeError("vocabulary must be callable (version, migrating)")
        #: 词表此刻的（版本, 有没有未完成的迁移）
        self.vocabulary = vocabulary
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def run_once(self) -> NightOutcome:
        closed = sorted(self.closed_days())
        night = self.prediction.tonight()
        if not closed:
            # 一天都还没封口（刚装上）：没有完整的一天可算，这一晚不跑
            return NightOutcome(night, EventSeries(cutoff=night), None, None, "一天都还没封口")
        night = min(night, closed[-1] + timedelta(days=1))
        before = self.vocabulary()
        if before[1]:
            return NightOutcome(night, EventSeries(cutoff=night), None, None, "词表迁移还没做完")
        series = read_series(self.behavior_tree, cutoff=night)
        if self.vocabulary() != before:
            return NightOutcome(night, EventSeries(cutoff=night), None, None, "读序列时词表变了")
        published = self.prediction.build_from(series)
        if published is None:
            return NightOutcome(night=night, series=series, published=None, scene=None)
        tree = self.prediction.store.load_generation(published.generation, expected_digest=published.digest)
        report = await self.scene.run(series=series, tree=tree, now=self._clock())
        return NightOutcome(night=night, series=series, published=published, scene=report)


__all__ = ["NightOutcome", "Nightly"]
