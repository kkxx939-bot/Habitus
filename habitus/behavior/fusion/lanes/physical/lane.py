"""物理 lane：上游的一份交付 → 每段行为一条判断、每段没在看的时间一条空白。

规则（方案第四节）：不调模型、不接续、不合并；同一段再送一遍不重复写；上游改判不收。

一份交付除了自己要合约定（``model.parse_delivery``），还不许和这块设备之前收下的东西矛盾——行为树只增不改，
一条行为落在已经记成"没在看"的时段里，下游就读不了这棵树了：

- 新送来的行为，开始时刻落在已经记成没在看的时段里：整份拒收；
- 新送来的行为（新的编号），和已经收下的另一段行为在时间上重叠：整份拒收（同一时刻只有一个行为；首尾相接可以）；
- 新送来的没在看的时段，盖住了已经收下的某段行为的开始时刻：整份拒收；
- 新送来的没在看的时段，和已经记下的没在看的时段有重叠：只落还没记过的那一截（两边都说没在看，取并集，不重复落）。

全是同步的文件读写，调用方放到线程里做。同一个进程里一次只处理一份交付：两份同时进来，各读各的"还没收过"，
同一段会写出两条。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime

from habitus.behavior.fusion.lanes.physical.ledger import AcceptedSpan, SpanKind
from habitus.behavior.fusion.lanes.physical.model import (
    CoverageSpan,
    PhysicalDelivery,
    PhysicalDeliveryError,
    PhysicalSegment,
)
from habitus.behavior.fusion.lanes.physical.recorder import PhysicalSegmentRecorder, UnitDisposition, gap_unit


@dataclass(frozen=True)
class UnitOutcome:
    """一段行为或一段空白的处理结果，供回执与可观测用。"""

    unit: PhysicalSegment | CoverageSpan
    disposition: UnitDisposition


class PhysicalLane:
    """把一份交付里的段依次处理掉。中途出错就整体抛出，调用方之后把同一份交付再送一遍；
    已经写过的段重来时会被认出来、不重复写。"""

    def __init__(self, recorder: PhysicalSegmentRecorder) -> None:
        if not isinstance(recorder, PhysicalSegmentRecorder):
            raise TypeError("recorder must be PhysicalSegmentRecorder")
        self.recorder = recorder
        self._one_at_a_time = threading.Lock()

    def consume(self, delivery: PhysicalDelivery) -> tuple[UnitOutcome, ...]:
        if not isinstance(delivery, PhysicalDelivery):
            raise TypeError("delivery must be PhysicalDelivery")
        with self._one_at_a_time:
            pieces = self._reconciled(delivery)  # 先对一遍，矛盾的整份拒收，什么都不写
            covered = self.recorder.covered()
            outcomes = [
                UnitOutcome(segment, self.recorder.record_segment(delivery.device, segment, covered))
                for segment in delivery.segments
            ]
            for span, remaining in pieces:
                dispositions = [
                    self.recorder.record_gap(delivery.device, started_at, ended_at, span.state, covered)
                    for started_at, ended_at in remaining
                ]
                recorded = UnitDisposition.RECORDED in dispositions
                outcomes.append(
                    UnitOutcome(span, UnitDisposition.RECORDED if recorded else UnitDisposition.ALREADY_RECORDED)
                )
            return tuple(outcomes)

    def _reconciled(self, delivery: PhysicalDelivery) -> list[tuple[CoverageSpan, list[tuple[datetime, datetime]]]]:
        """拿这份交付对这块设备之前收下的时段；返回每段没在看的时间里还没记过的那几截。"""

        ledger, device = self.recorder.ledger, delivery.device
        moments = [edge for segment in delivery.segments for edge in (segment.started_at, segment.ended_at)]
        moments += [edge for span in delivery.coverage for edge in (span.started_at, span.ended_at)]
        if not moments:
            return []
        known = ledger.spans(device, min(moments), max(moments))
        unobserved = [item for item in known if item.kind is SpanKind.UNOBSERVED]
        segments = [item for item in known if item.kind is SpanKind.SEGMENT]
        for segment in delivery.segments:
            if ledger.accepted(device, f"segment#{segment.segment_id}") is not None:
                continue  # 收过的段：是重发还是改判由写的那一步分
            if _starts_inside(segment.started_at, unobserved):
                raise PhysicalDeliveryError(
                    f"segment {segment.segment_id} starts inside a span already recorded as unobserved"
                )
            if any(_overlap(segment.started_at, segment.ended_at, item) for item in segments):
                raise PhysicalDeliveryError(
                    f"segment {segment.segment_id} overlaps a segment already recorded under another id"
                )
        pieces: list[tuple[CoverageSpan, list[tuple[datetime, datetime]]]] = []
        for span in delivery.gaps:
            if ledger.accepted(device, gap_unit(span.started_at, span.ended_at)) is not None:
                pieces.append((span, []))
                continue
            for item in segments:
                if span.started_at <= item.started_at < span.ended_at:
                    raise PhysicalDeliveryError(
                        "an unobserved span covers the start of a segment that was already recorded"
                    )
            pieces.append((span, _subtract(span.started_at, span.ended_at, unobserved)))
        return pieces


def _starts_inside(moment: datetime, spans: list[AcceptedSpan]) -> bool:
    return any(item.started_at <= moment < item.ended_at for item in spans)


def _overlap(started_at: datetime, ended_at: datetime, span: AcceptedSpan) -> bool:
    """两段行为在时间上有没有重叠；首尾相接不算，起止完全相同算。"""

    if (started_at, ended_at) == (span.started_at, span.ended_at):
        return True
    return started_at < span.ended_at and span.started_at < ended_at


def _subtract(started_at: datetime, ended_at: datetime, spans: list[AcceptedSpan]) -> list[tuple[datetime, datetime]]:
    """``[started_at, ended_at)`` 里还没被 ``spans``（按开始时刻排好）盖住的那几截。"""

    remaining: list[tuple[datetime, datetime]] = []
    cursor = started_at
    for item in spans:
        if item.ended_at <= cursor or item.started_at >= ended_at:
            continue
        if item.started_at > cursor:
            remaining.append((cursor, item.started_at))
        cursor = max(cursor, item.ended_at)
        if cursor >= ended_at:
            break
    if cursor < ended_at:
        remaining.append((cursor, ended_at))
    # 小账上的时段是 UTC；切出来的每一截写回上游这段自己的偏移（行为树上的时间都带发生地的本地偏移）。
    zone = started_at.tzinfo
    return [(begin.astimezone(zone), end.astimezone(zone)) for begin, end in remaining]


__all__ = ["PhysicalLane", "UnitOutcome"]
