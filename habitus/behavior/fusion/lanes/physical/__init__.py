"""物理 lane 的融合：上游送来一段行为，写一条判断。

要点（裁定 40）：上游送来的是最终结果（什么行为、几点到几点）
和覆盖时段（哪段时间在看）；融合不调模型，一段转成一条记录，按上游的编号去重，没在看的时间记成空白。

模块分工：``model`` 一份交付的形状与对上游约定的校验；``recorder`` 写判断；``ledger`` 记收过哪些段；``lane`` 把一份交付处理掉。
进程内的正门是 ``Runtime.deliver_physical``。
"""

# TODO(BHV-PHYSICAL-LANE-001): 输入形状未对过上游真实样例；没有 HTTP 端点，"没在看"的原因落树后看不到。

from habitus.behavior.fusion.lanes.physical.config import PhysicalLaneConfig
from habitus.behavior.fusion.lanes.physical.lane import PhysicalLane, UnitOutcome
from habitus.behavior.fusion.lanes.physical.ledger import AcceptedSpan, PhysicalSegmentLedger, SpanKind
from habitus.behavior.fusion.lanes.physical.model import (
    CoverageSpan,
    CoverageState,
    PhysicalDelivery,
    PhysicalDeliveryError,
    PhysicalSegment,
    parse_delivery,
)
from habitus.behavior.fusion.lanes.physical.protocol import PHYSICAL_PROTOCOL
from habitus.behavior.fusion.lanes.physical.recorder import (
    PHYSICAL_FUSION_VERSION,
    PHYSICAL_RULE_VERSION,
    PhysicalSegmentRecorder,
    UnitDisposition,
    physical_observer,
)

__all__ = [
    "PHYSICAL_FUSION_VERSION",
    "PHYSICAL_PROTOCOL",
    "PHYSICAL_RULE_VERSION",
    "AcceptedSpan",
    "CoverageSpan",
    "CoverageState",
    "PhysicalDelivery",
    "PhysicalDeliveryError",
    "PhysicalLane",
    "PhysicalLaneConfig",
    "PhysicalSegment",
    "PhysicalSegmentLedger",
    "PhysicalSegmentRecorder",
    "SpanKind",
    "UnitDisposition",
    "UnitOutcome",
    "parse_delivery",
    "physical_observer",
]
