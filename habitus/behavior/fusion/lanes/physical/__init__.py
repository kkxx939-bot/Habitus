"""物理 lane 的融合：上游送来一段行为，写一条判断。

方案见桌面 ``14-事件融合方案.md`` 第四节与裁定 40。要点：上游送来的是最终结果（什么行为、几点到几点）
和覆盖时段（哪段时间在看）；融合不调模型，一段转成一条记录，按上游的编号去重，没在看的时间记成空白。

模块分工：``model`` 一份交付的形状与对上游约定的校验；``recorder`` 写判断；``ledger`` 记收过哪些段；``lane`` 把一份交付处理掉。

这里的形状是从我们这边要用什么倒推出来的，还没有对过上游（S12）的真实样例；样例来了要逐条对。

TODO(BHV-PHYSICAL-LANE-001)：方案里已定、这里没做的（都不在融合这一层，等融合改完再谈）：
- 入口：进程内的正门是 ``Runtime.deliver_physical``；HTTP 端点还没有（随上游的认证与协议一起定）。
- 空白的原因：没在看的时段落成「未观测」的空白，但"没戴 / 充电 / 无数据"这个原因只留在凭据上，落树后看不到。
- 预测层只接了会话那一支（按不按 lane 分没定），物理这一支的记录不进预测层的证据、也不结算。
- 小账（``ledger``）只增不减，一段一个小文件；什么时候清归统一的生命周期设计。
- 没被任何一份交付覆盖到的时间（数据还没到）不出任何记录；下游怎么把它读成"没看见"是三棵树的事。
"""

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
