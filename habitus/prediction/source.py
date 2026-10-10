"""把第 N 晚的事件序列转成一次重建的输入快照。

行为树怎么读（重复、「待定」「非事件」、空白、并行链接）只写在 ``series.reader`` 一处，预测树与语义树共用
（语义树新方案 ``13`` 第四节）；本层不再碰行为树，只做预测树自己的那一步翻译：

- 类编号的记录成为 ``ObservedAction``；「待定」成为 ``ObservedUnnamed``（不是候选，只在先后顺序里占位）；
- 并行对从 uri 翻成**排序之后**的下标（配对层的前置条件）；
- ``reminded`` 为真的记录**硬拒**：被提醒之后的发生不属于自然率，而提醒的随机化记录还没建（语义树新方案第五节：
  只算没被提醒的那一栏）。宁可硬拒也不能静默把它数进去——这条护栏必须在提醒功能上线**之前**就在这里挡着；
- 截止日原样带进快照：跨过它的转移窗口没看全，配对层记删失。
"""

from __future__ import annotations

from habitus.prediction.errors import PredictionTreeError
from habitus.prediction.model import BehaviorSnapshot, ObservedAction, ObservedGap, ObservedUnnamed
from habitus.series import EventSeries, SkipReason


def snapshot_of(series: EventSeries) -> BehaviorSnapshot:
    if not isinstance(series, EventSeries):
        raise PredictionTreeError("series must be an EventSeries")
    actions: list[ObservedAction] = []
    unnamed: list[ObservedUnnamed] = []
    rank: dict[str, int] = {}
    for record in series.records:
        if record.reminded:
            raise PredictionTreeError(
                "occurrence is marked as reminded but reminder randomisation is not recorded yet; "
                "counting it would silently pollute the natural rate (see TODO(PRED-TREE-001))"
            )
        if not record.classified:
            unnamed.append(ObservedUnnamed(started_at=record.started_at, day=record.day, lane=record.lane))
            continue
        rank[record.uri] = len(actions)
        actions.append(
            ObservedAction(action=record.kind_token, started_at=record.started_at, day=record.day, lane=record.lane)
        )
    concurrent = {
        (min(rank[left], rank[right]), max(rank[left], rank[right]))
        for left, right in series.concurrent
        if left in rank and right in rank
    }
    return BehaviorSnapshot(
        actions=tuple(actions),
        unnamed=tuple(unnamed),
        gaps=tuple(ObservedGap(started_at=gap.started_at, ended_at=gap.ended_at, watched=gap.watched) for gap in series.gaps),
        concurrent=tuple(sorted(concurrent)),
        skipped_duplicates=series.skipped_count(SkipReason.DUPLICATE),
        cutoff=series.cutoff,
    )


__all__ = ["snapshot_of"]
