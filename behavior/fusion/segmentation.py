"""把观测交付流确定性地切成有界的融合窗口。

## 切段不判断行为是否结束

"这段观测里的行为结束了没有"是语义判断，属于融合层；切段只做**机械分组**——按容量和跨度。
观测空白**原样保留在片段序列里**（片段渲染带相对秒偏移），由能处理不可知的那一层裁量。

这条分工顺带免掉了一个拍不出来的参数：切段不需要一个"正确"的空白阈值，因为空白信息不丢。
密度断裂本身也不能当作行为结束的证据——人走出画面与人离开了家，在数据上完全一样。

## 但切点会尽量落在观测空白处

必须切的时候（达到容量或跨度上限），在尾部一个有界窗口内选择**最大的观测空白**下刀。这仍然
是确定性的、不含语义判断的，却能大幅减少把一次行为拦腰切断的概率。

## 跨交付去重在这里完成

上游滑窗推送会让相邻交付重叠（每 30 秒推送最近 60 秒即有一半重复）。同一条观测在多个交付里
出现时只保留一次，但**全部来源交付都记进 ``source_refs``** 以便审计——去重不是丢弃来源。
"""

from __future__ import annotations

from collections.abc import Container, Sequence
from dataclasses import dataclass

from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.errors import BehaviorFusionError
from behavior.observation import BehaviorObservation, BehaviorObservationEnvelope


@dataclass(frozen=True)
class BehaviorFusionSegment:
    """一次融合的输入：有界的片段序列，以及它们来自哪些交付。"""

    fragments: tuple[BehaviorObservation, ...]
    source_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fragments:
            raise BehaviorFusionError("a fusion segment must not be empty")
        if not self.source_refs:
            raise BehaviorFusionError("a fusion segment must record its source deliveries")

    @property
    def span_seconds(self) -> float:
        return (self.fragments[-1].occurred_at - self.fragments[0].occurred_at).total_seconds()


def segment_observations(
    envelopes: Sequence[BehaviorObservationEnvelope],
    *,
    config: BehaviorFusionConfig | None = None,
    exclude: Container[str] = frozenset(),
) -> tuple[BehaviorFusionSegment, ...]:
    """把若干观测交付合并、去重、定序后切成融合窗口。

    定序键是 ``(occurred_at, observation_id)``：``occurred_at`` 允许并列（同一时刻的视觉与
    听觉观测），并列时按内容身份定序以保证同一批输入总得到同一个切法。

    ``exclude`` 按 ``observation_id`` 剔除已经被处理过的观测。它必须在**观测**粒度上生效而不是
    交付粒度：滑窗推送让每个交付都与前一个重叠，若因为"这个交付里有一条已处理"就跳过整个交付，
    那么从第二个交付开始的每一条新观测都会被静默丢掉——观测本来就不完整，缺失只能记录不能制造。
    """

    resolved = config or BehaviorFusionConfig()
    if not isinstance(resolved, BehaviorFusionConfig):
        raise TypeError("config must be BehaviorFusionConfig")
    if isinstance(envelopes, (str, bytes)) or not isinstance(envelopes, Sequence):
        raise BehaviorFusionError("envelopes must be a sequence")
    if any(not isinstance(item, BehaviorObservationEnvelope) for item in envelopes):
        raise TypeError("envelopes must contain BehaviorObservationEnvelope values")
    if isinstance(exclude, (str, bytes)) or not isinstance(exclude, Container):
        raise TypeError("exclude must be a container of observation identities")

    sources: dict[str, set[str]] = {}
    unique: dict[str, BehaviorObservation] = {}
    for envelope in envelopes:
        for observation in envelope.batch.observations:
            if observation.observation_id in exclude:
                continue
            unique.setdefault(observation.observation_id, observation)
            sources.setdefault(observation.observation_id, set()).add(envelope.source_id)
    if not unique:
        return ()
    ordered = sorted(unique.values(), key=lambda item: (item.occurred_at, item.observation_id))

    segments: list[BehaviorFusionSegment] = []
    remaining = ordered
    while remaining:
        cut = _cut_point(remaining, resolved)
        chunk = remaining[:cut]
        refs = sorted({ref for item in chunk for ref in sources[item.observation_id]})
        segments.append(BehaviorFusionSegment(tuple(chunk), tuple(refs)))
        remaining = remaining[cut:]
    return tuple(segments)


def _cut_point(fragments: Sequence[BehaviorObservation], config: BehaviorFusionConfig) -> int:
    """返回本段应当包含的片段数量。

    先按容量与跨度求出硬上限，再在尾部的有界搜索窗口里挑最大空白处下刀；没有可选空白时就在
    硬上限处切。整个过程只用时间与计数，不含任何语义判断。
    """

    limit = min(len(fragments), config.max_fragments_per_segment)
    origin = fragments[0].occurred_at
    for index in range(1, limit):
        span = (fragments[index].occurred_at - origin).total_seconds()
        if span > config.max_segment_span_seconds:
            limit = index
            break
    if limit >= len(fragments):
        return len(fragments)
    search_start = max(1, limit - config.boundary_search_fragments)
    best_index = limit
    best_gap = -1.0
    for index in range(search_start, limit + 1):
        gap = (fragments[index].occurred_at - fragments[index - 1].occurred_at).total_seconds()
        # 平局取**最靠后**的那个切点。上游定频抽帧，间隔天生全等，于是"挑最大空白"退化成一次
        # 全平局；用严格 ``>`` 会让切点固定落在搜索窗最前端，而不是回退到硬上限：均匀采样下
        # 每段恒定少切一个搜索窗的量，`limit` 小于窗口宽度时更会塌成"每段一条观测"——那正是
        # ``enqueue`` 文档里被当成灾难的那个形态。没有可选空白时就该切在硬上限。
        if gap >= best_gap:
            best_gap = gap
            best_index = index
    return best_index


__all__ = ["BehaviorFusionSegment", "segment_observations"]
