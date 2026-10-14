"""物理 lane 的输入形状：上游的一份交付——行为段若干、覆盖时段若干。

这是对上游的要求（方案第四节、裁定 40），不合要求的整份拒收，不猜、不修：

- 行为段是事件级的完整一段：已经结束、互不重叠（首尾相接在同一瞬间可以）、按时间排好，落在"在看"的覆盖时段里；
- 覆盖时段首尾相接、不留洞：这段时间里每一刻要么在看，要么说明为什么没看；
- 时间一律带偏移；只带结论，不带原始读数（多出来的字段一律拒收）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.lanes.physical.config import PhysicalLaneConfig

SOURCE_PREFIX = "physical/"
_DELIVERY_FIELDS = frozenset({"source", "device", "delivery_id", "sent_at", "segments", "coverage"})
_SEGMENT_FIELDS = frozenset({"id", "behavior", "started_at", "ended_at", "facts"})
_SPAN_FIELDS = frozenset({"started_at", "ended_at", "state"})


class PhysicalDeliveryError(BehaviorFusionError):
    """上游的交付不合要求。"""


class CoverageState(str, Enum):
    """一段时间里上游在不在看；不在看的要说原因。"""

    OBSERVING = "observing"  # 在佩戴且在采集
    NOT_WORN = "not_worn"
    CHARGING = "charging"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class PhysicalSegment:
    """上游认定的一段行为：是什么、从几点到几点。``segment_id`` 是上游给的固定编号。"""

    segment_id: str
    behavior: str
    started_at: datetime
    ended_at: datetime
    facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageSpan:
    """一段覆盖时段。"""

    started_at: datetime
    ended_at: datetime
    state: CoverageState

    @property
    def observing(self) -> bool:
        return self.state is CoverageState.OBSERVING


@dataclass(frozen=True)
class PhysicalDelivery:
    """上游的一份交付。用 ``parse_delivery`` 造，造出来的就是合要求的。"""

    source: str
    device: str
    delivery_id: str
    sent_at: datetime
    segments: tuple[PhysicalSegment, ...]
    coverage: tuple[CoverageSpan, ...]

    @property
    def gaps(self) -> tuple[CoverageSpan, ...]:
        """没在看的那些时段。"""

        return tuple(item for item in self.coverage if not item.observing)


def parse_delivery(value: object, *, config: PhysicalLaneConfig | None = None) -> PhysicalDelivery:
    """校验并读入一份交付；不合要求就抛 ``PhysicalDeliveryError``，整份不收。"""

    limits = config or PhysicalLaneConfig()
    body = _mapping(value, "delivery", _DELIVERY_FIELDS, required=_DELIVERY_FIELDS)
    source = _text(body["source"], "source")
    if not source.startswith(SOURCE_PREFIX) or source == SOURCE_PREFIX:
        raise PhysicalDeliveryError(f"source must start with {SOURCE_PREFIX!r}: this is not a physical-lane delivery")
    sent_at = _moment(body["sent_at"], "sent_at")
    device = _identifier(body["device"], "device", limits.max_device_chars)
    segments = tuple(_segment(item, index, limits) for index, item in enumerate(_items(body["segments"], "segments")))
    coverage = tuple(_span(item, index) for index, item in enumerate(_items(body["coverage"], "coverage")))
    if len(segments) > limits.max_segments:
        raise PhysicalDeliveryError("delivery carries more segments than the configured limit")
    if len(coverage) > limits.max_coverage_spans:
        raise PhysicalDeliveryError("delivery carries more coverage spans than the configured limit")
    _require_tiled(coverage)
    _require_ordered(segments)
    _require_observed(segments, coverage)
    _require_finished(segments, sent_at + timedelta(seconds=limits.max_future_seconds))
    return PhysicalDelivery(
        source=source,
        device=device,
        delivery_id=_text(body["delivery_id"], "delivery_id"),
        sent_at=sent_at,
        segments=segments,
        coverage=coverage,
    )


def _segment(value: object, index: int, limits: PhysicalLaneConfig) -> PhysicalSegment:
    label = f"segments[{index}]"
    body = _mapping(value, label, _SEGMENT_FIELDS, required=_SEGMENT_FIELDS - {"facts"})
    started_at = _moment(body["started_at"], f"{label}.started_at")
    ended_at = _moment(body["ended_at"], f"{label}.ended_at")
    if ended_at < started_at:
        raise PhysicalDeliveryError(f"{label} ends before it starts")
    facts = tuple(_text(item, f"{label}.facts") for item in _items(body.get("facts", ()), f"{label}.facts"))
    if len(facts) > limits.max_facts or any(len(item) > limits.max_fact_chars for item in facts):
        raise PhysicalDeliveryError(f"{label}.facts exceeds the configured limit")
    return PhysicalSegment(
        segment_id=_identifier(body["id"], f"{label}.id", limits.max_segment_id_chars),
        behavior=_text(body["behavior"], f"{label}.behavior"),
        started_at=started_at,
        ended_at=ended_at,
        facts=facts,
    )


def _span(value: object, index: int) -> CoverageSpan:
    label = f"coverage[{index}]"
    body = _mapping(value, label, _SPAN_FIELDS, required=_SPAN_FIELDS)
    started_at = _moment(body["started_at"], f"{label}.started_at")
    ended_at = _moment(body["ended_at"], f"{label}.ended_at")
    if ended_at <= started_at:
        raise PhysicalDeliveryError(f"{label} must end after it starts")
    try:
        state = CoverageState(body["state"])
    except ValueError as exc:
        raise PhysicalDeliveryError(f"{label}.state is not a known coverage state") from exc
    return CoverageSpan(started_at, ended_at, state)


def _require_tiled(coverage: Sequence[CoverageSpan]) -> None:
    """覆盖时段首尾相接、不留洞、不重叠。"""

    for before, after in zip(coverage, coverage[1:], strict=False):
        if after.started_at != before.ended_at:
            raise PhysicalDeliveryError(
                "coverage spans must be contiguous: every moment is either observed or explained"
            )


def _require_ordered(segments: Sequence[PhysicalSegment]) -> None:
    """行为段按时间排好、互不重叠：同一时刻只有一个行为。编号不重复。"""

    for before, after in zip(segments, segments[1:], strict=False):
        if after.started_at < before.ended_at:
            raise PhysicalDeliveryError("segments must be ordered and must not overlap")
    if len({item.segment_id for item in segments}) != len(segments):
        raise PhysicalDeliveryError("segment ids must be unique within a delivery")


def _require_observed(segments: Sequence[PhysicalSegment], coverage: Sequence[CoverageSpan]) -> None:
    """每一段都落在"在看"的时段里（相邻的"在看"时段连起来算一段）。

    段的开始时刻必须真的在"在看"里面（不能正好踩在"在看"结束、"没在看"开始的那一瞬）：下游按开始时刻认这条行为
    在哪个时段，踩在"没在看"的起点上就成了"没在看的时候看见了它"。
    """

    watched: list[tuple[datetime, datetime]] = []
    for span in coverage:
        if not span.observing:
            continue
        if watched and watched[-1][1] == span.started_at:
            watched[-1] = (watched[-1][0], span.ended_at)
        else:
            watched.append((span.started_at, span.ended_at))
    for segment in segments:
        if not any(begin <= segment.started_at < end and segment.ended_at <= end for begin, end in watched):
            raise PhysicalDeliveryError(f"segment {segment.segment_id} lies outside the observed coverage")


def _require_finished(segments: Sequence[PhysicalSegment], latest: datetime) -> None:
    """只送已经结束的段：结束时刻不能晚于这份交付送出的时刻（进行中的不进来）。"""

    for segment in segments:
        if segment.ended_at > latest:
            raise PhysicalDeliveryError(f"segment {segment.segment_id} had not ended when the delivery was sent")


def _mapping(value: object, label: str, allowed: frozenset[str], *, required: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PhysicalDeliveryError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        # 只带结论，不带原始读数：多出来的字段不是"先收着"，是拒收。
        raise PhysicalDeliveryError(f"{label} carries fields outside the contract: {unknown}")
    missing = sorted(required - set(value))
    if missing:
        raise PhysicalDeliveryError(f"{label} is missing {missing}")
    return value


def _items(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PhysicalDeliveryError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalDeliveryError(f"{label} must be non-empty text")
    return " ".join(value.split())


def _identifier(value: object, label: str, limit: int) -> str:
    """设备名、段编号：要拿去当凭据的来源身份与引用，不许带控制字符，长度有上限。"""

    text = _text(value, label)
    if len(text) > limit or any(not character.isprintable() for character in text):
        raise PhysicalDeliveryError(f"{label} must be printable text of at most {limit} characters")
    return text


def _moment(value: object, label: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise PhysicalDeliveryError(f"{label} is not an ISO-8601 time") from exc
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise PhysicalDeliveryError(f"{label} must carry its UTC offset")
    return value


__all__ = [
    "SOURCE_PREFIX",
    "CoverageSpan",
    "CoverageState",
    "PhysicalDelivery",
    "PhysicalDeliveryError",
    "PhysicalSegment",
    "parse_delivery",
]
