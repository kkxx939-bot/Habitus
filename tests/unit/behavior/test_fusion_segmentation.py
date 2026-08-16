"""切段的确定性行为。

这个模块此前**一条测试都没有**，而它决定了融合层每一次模型调用看到多少观测。零覆盖直接
养出过一个真实缺陷：``_cut_point`` 在"窗口内所有间隔相等"时取窗口最前端而不是硬上限，于是
上游定频抽帧（间隔天生全等）时每段恒定少切一个搜索窗的量，采样一旦稀疏到 ``limit`` 不超过
窗口宽度，更会塌成"每段一条观测"——一次连续行为被逐帧切开、逐帧调模型。

所以这里的测试重点不是"边界值对不对"，而是**上游最常见的那种输入**（等间隔）会得到什么。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.errors import BehaviorFusionError
from behavior.fusion.segmentation import BehaviorFusionSegment, segment_observations
from behavior.observation import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationConfig,
    BehaviorObservationEnvelope,
)
from foundation.integrity import canonical_digest

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone(timedelta(hours=8)))
OBSERVER = "home-a/hall"
OBSERVATION_CONFIG = BehaviorObservationConfig()


def observation(offset: float, *, semantics: str | None = None) -> BehaviorObservation:
    at = NOW + timedelta(seconds=offset)
    return BehaviorObservation.create(
        observer_id=OBSERVER,
        occurred_at=at,
        available_at=at + timedelta(milliseconds=500),
        modality="vision",
        semantics=semantics or f"片段{offset}",
        participants=["家庭成员A"],
        knowledge_state="observed",
        confidence=0.9,
        evidence_refs=[f"cam:{offset}"],
        config=OBSERVATION_CONFIG,
    )


def envelope(
    observations: tuple[BehaviorObservation, ...], seed: str
) -> BehaviorObservationEnvelope:
    return BehaviorObservationEnvelope.create(
        observer_id=OBSERVER,
        protocol="m2bos_behavior_observation_v1",
        batch=BehaviorObservationBatch(observer_id=OBSERVER, observations=observations),
        delivery_id=canonical_digest(seed),
        # 交付时刻必须不早于批内每条语义的可用时刻，否则观测层直接拒收。
        recorded_at=max(item.available_at for item in observations) + timedelta(seconds=1),
        config=OBSERVATION_CONFIG,
    )


def uniform(count: int, step: float) -> tuple[BehaviorObservation, ...]:
    """上游定频抽帧的形态：间隔完全相等。"""

    return tuple(observation(index * step) for index in range(count))


def sizes(segments: tuple[BehaviorFusionSegment, ...]) -> list[int]:
    return [len(item.fragments) for item in segments]


# --- 等间隔输入：这是上游的常态，也是缺陷曾经藏身的地方 -------------------------------------


@pytest.mark.parametrize("step", [2.0, 4.0, 5.0, 20.0, 30.0, 60.0, 300.0])
def test_uniform_sampling_fills_a_segment_to_its_hard_limit(step: float) -> None:
    """间隔全等时没有"最大空白"可挑，就该切在容量/跨度上限，而不是搜索窗最前端。"""

    config = BehaviorFusionConfig()
    allowed = min(
        config.max_fragments_per_segment,
        int(config.max_segment_span_seconds // step) + 1,
    )
    segments = segment_observations([envelope(uniform(allowed * 3, step), "s1")], config=config)
    assert sizes(segments)[0] == allowed


@pytest.mark.parametrize("step", [30.0, 60.0, 300.0])
def test_sparse_uniform_sampling_does_not_collapse_into_one_observation_per_segment(
    step: float,
) -> None:
    """采样稀疏到 ``limit`` 不超过搜索窗宽度时，曾经每段只剩一条观测。

    那等于把折叠这件事整个取消掉：一次连续行为被逐帧切开，逐帧调模型，产出一组互不相认的
    判断——``enqueue`` 的文档正把这个形态列为要避免的灾难。
    """

    config = BehaviorFusionConfig()
    segments = segment_observations([envelope(uniform(120, step), "s1")], config=config)
    assert max(sizes(segments)) > 1
    assert sizes(segments).count(1) <= 1  # 只允许末尾余数是单条


def test_a_real_blank_beats_the_hard_limit() -> None:
    """尾部窗口里存在真空白时仍然优先在空白处下刀——这是"尽量不拦腰切断"的本意。"""

    config = BehaviorFusionConfig(max_fragments_per_segment=20, boundary_search_fragments=8)
    fragments = list(uniform(60, 4.0))
    blank_at = 15
    fragments = fragments[:blank_at] + [
        observation((index * 4.0) + 300.0) for index in range(blank_at, 60)
    ]
    segments = segment_observations([envelope(tuple(fragments), "s1")], config=config)
    assert sizes(segments)[0] == blank_at


def test_the_span_limit_cuts_before_the_capacity_limit() -> None:
    config = BehaviorFusionConfig(
        max_fragments_per_segment=500, max_segment_span_seconds=100, boundary_search_fragments=4
    )
    segments = segment_observations([envelope(uniform(200, 10.0), "s1")], config=config)
    assert sizes(segments)[0] == 11  # 0..100 秒共 11 条，第 12 条越界


def test_the_capacity_limit_cuts_before_the_span_limit() -> None:
    config = BehaviorFusionConfig(
        max_fragments_per_segment=12, max_segment_span_seconds=100_000,
        boundary_search_fragments=4,
    )
    segments = segment_observations([envelope(uniform(50, 1.0), "s1")], config=config)
    assert sizes(segments)[0] == 12


def test_a_short_stream_is_one_segment_and_is_never_cut() -> None:
    segments = segment_observations([envelope(uniform(5, 4.0), "s1")], config=BehaviorFusionConfig())
    assert sizes(segments) == [5]


# --- 去重、来源与顺序 -----------------------------------------------------------------------


def test_overlapping_deliveries_keep_every_observation_exactly_once() -> None:
    """滑窗推送让相邻交付重叠；去重必须在观测粒度上做，不能整个交付跳过。"""

    stream = uniform(6, 5.0)
    segments = segment_observations(
        [envelope(stream[:4], "s1"), envelope(stream[2:], "s2")], config=BehaviorFusionConfig()
    )
    ids = [item.observation_id for item in segments[0].fragments]
    assert len(ids) == len(set(ids)) == 6


def test_deduplication_records_every_source_delivery() -> None:
    """去重不是丢弃来源：重复出现的观测，两个交付都要记进 ``source_refs`` 以便审计。"""

    stream = uniform(4, 5.0)
    first, second = envelope(stream[:3], "s1"), envelope(stream[1:], "s2")
    segments = segment_observations([first, second], config=BehaviorFusionConfig())
    assert segments[0].source_refs == tuple(sorted({first.source_id, second.source_id}))


def test_exclude_drops_processed_observations_without_dropping_their_neighbours() -> None:
    """``exclude`` 按观测生效。曾经按交付生效，导致重叠交付里的新观测被静默丢掉。"""

    stream = uniform(6, 5.0)
    processed = {stream[0].observation_id, stream[1].observation_id}
    segments = segment_observations(
        [envelope(stream, "s1")], config=BehaviorFusionConfig(), exclude=processed
    )
    remaining = [item.observation_id for item in segments[0].fragments]
    assert remaining == [item.observation_id for item in stream[2:]]


def test_everything_excluded_yields_no_segment_rather_than_an_empty_one() -> None:
    stream = uniform(3, 5.0)
    excluded = {item.observation_id for item in stream}
    assert segment_observations([envelope(stream, "s1")], exclude=excluded) == ()


def test_fragments_are_ordered_by_time_then_identity() -> None:
    """同一时刻可以有多条观测（视觉与听觉）；并列时按内容身份定序，同一批输入才总得到同一个切法。"""

    first = observation(0.0, semantics="人走进厨房")
    same_moment = observation(0.0, semantics="听到脚步声")
    later = observation(5.0)
    forward = segment_observations([envelope((first, same_moment, later), "s1")])
    backward = segment_observations([envelope((same_moment, first, later), "s1")])
    assert [item.observation_id for item in forward[0].fragments] == [
        item.observation_id for item in backward[0].fragments
    ]
    assert forward[0].fragments[-1].observation_id == later.observation_id


def test_segments_partition_the_stream_in_order() -> None:
    """切段是机械分组：不重不漏，且顺序与时间一致。"""

    config = BehaviorFusionConfig(max_fragments_per_segment=7, boundary_search_fragments=3)
    stream = uniform(30, 4.0)
    segments = segment_observations([envelope(stream, "s1")], config=config)
    flat = [item.observation_id for segment in segments for item in segment.fragments]
    assert flat == [item.observation_id for item in stream]


# --- 参数校验 -------------------------------------------------------------------------------


def test_no_envelopes_yields_no_segments() -> None:
    assert segment_observations([]) == ()


def test_envelopes_must_be_a_sequence_of_envelopes() -> None:
    with pytest.raises(BehaviorFusionError):
        segment_observations("not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        segment_observations([object()])  # type: ignore[list-item]


def test_a_segment_must_not_be_empty_or_sourceless() -> None:
    with pytest.raises(BehaviorFusionError):
        BehaviorFusionSegment((), ("s1",))
    with pytest.raises(BehaviorFusionError):
        BehaviorFusionSegment((observation(0.0),), ())
