"""时间预测树：钟面主干上的计数树。

本包**只算概率、零语义、零 LLM**——它把行为树里的历史行为叠到 (周几 × 时刻) 的钟面上数数，
配上可信度伴随值，每夜整棵重建、原子发布。完整规格见 ``prediction/model.py`` 的
``TODO(PRED-TREE-001)``。

产物是四类可查的数据：这个时刻会开始做什么（节点）、做完这个接着做什么（边）、这件事隔多久
做一次（复发间隔）、到这个点为止今天做了没有（累积率）。树**不判断**重要性、不解释"为什么
规律"、不持有当日状态——那些都属于上层。

# TODO(PRED-DOWNSTREAM-001): 预测算法与执行层（本次实现范围之外，随语义关联层一并设计）。
# 下列内容抢救自随旧实现一并退役的 TODO 与模块 docstring，是**用户裁定过、未被推翻**的设计，
# 重建时按此照做，不要重新发明。
#
# ── 预测算法（原 TODO(PRED-ALGO-003)）─────────────────────────────────────────────────
# - **时间维已不需要从头设计**：本层的危险率 h(t) 就是离散时间危险率，时刻分布由
#   `P(第 t 槽才发生) = h(t)·Π_{s<t}(1−h(s))` 逐槽累乘免费得到，中位数即预计时刻。剩下的
#   只是接线：把该分布接进候选的 expected_delay 与时窗输出。注意"时长"（做多久）与"时刻"
#   （何时开始）之分——公开结果显示前者误差极大（约 35 分钟量级）、后者可预测性高得多，
#   输出的是后者。
# - **查询层**：给定此刻构造各条件值、按该行为的分布查树、按**对数线性**组合得到候选与概率。
#   更接近接口而非新算法。禁止朴素相乘（见 PRED-TREE-001 组合契约；实测朴素相加会灾难性
#   过度自信，log-loss 达基线 13 倍）。
# - **判决层**：门槛不得再拍——概率、边际、支持度三道门用覆盖率-精确率曲线反推定档。原设计里
#   读 Outcome ``valence`` 的"负向结果门"**建议直接删除**：上游观测契约只接受有主体的行为观测、
#   不送环境反馈，该字段没有生产者，那道门从不触发。
# - **反馈闭环**：预测记录目前只记"预测了什么"，没有承载"后来实际发生了什么、用户接受还是
#   拒绝"。而"默认主动提醒 + 用户确认"这一产品形态本身就在持续产生高质量标签，现在全部丢弃。
#   等渗校准若改为按真实接受率拟合，校准的才是执行门真正需要的那个概率。
# - **干预混淆必须在提醒功能上线之前就位**：结算要带"此前有没有提醒"标记，被提醒后的发生
#   不得进入自然率的估计——事后无法把两种数据分开。行为树的 ``reminded`` 字段已就位但恒为
#   False（干预账本未建）。
#
# ── 三档执行（原 Runtime/prediction_execution.py）──────────────────────────────────────
# - 预测之后的行动决策分三档：**沉默 / 主动建议 / 直接执行**。确认与风险逻辑在预测算法**后面**，
#   预测器只产出候选与门槛评估，执行层把结果映射为产品动作。
# - **默认档是主动建议**——向用户提议并等确认，用户本人就是建议的裁决者。只有经长期观察被验证
#   稳定的规律才有资格进入直接执行档，且执行前必须通过 LLM 约束检查（**fail-closed**：查不了
#   就降级为建议，而不是硬执行）。
# - **合规必须被建立而非未被否定**：从未被送审的候选不允许直接执行。
# - **没有任何人工维护的行为风险表**——直接执行资格靠数据挣来。
# - 晋升判据依赖建议确认史（用户对建议的确认/拒绝记录），该存储与回流链路待设计；落地前
#   资格判定缺省为 None，即一切执行意图都降级为建议档。
#
# ── 组合根的两条边界（原 Runtime/prediction_bridge.py 与 prediction_llm.py）────────────
# - **prediction 不得 import memory，memory 也不应知道 prediction**——两者的桥接住在组合根。
#   桥接只做确定性工作：构造行为条件化的检索查询、按相关性过滤命中、把记忆条目整理成**原文**
#   上下文、产出可审计的来源绑定。**刻意不做语义判断**：条目的立场（趋向/回避）与此刻的相关性
#   都依赖情境，属于 LLM 在调用现场的裁量；领域代码不用关键词、正则或预计算映射表重新解释
#   自然语言。
# - **prediction 不得 import ModelClient，全部 LLM 编排住在组合根**。离线批处理把语义理解蒸馏成
#   版本化的确定性表；在线只有两个受控位置：不确定时的顾问（只对既有候选重排）与执行前的约束
#   检查（fail-closed）。
#
# ── 发布协议（原 TODO(PRED-STORE-001) 的教训）────────────────────────────────────────
# - 旧实现按状态逐个成代、独立激活，跨状态没有原子性，重学习发布期间预测会读到**混代**数据并
#   静默失真。本层因此规定：**两阶段发布**（先物化校验，再统一翻转指针，任一失败整批不激活、
#   旧代继续服务），且**一次查询必须钉住一代**。
"""

from prediction.builder import build, config_digest
from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError, PredictionTreeStoreError
from prediction.model import (
    EdgeStatistics,
    IntervalQuantiles,
    NodeCounts,
    NodeStatistics,
    ObservedAction,
    ObservedGap,
    PredictionTree,
    RecurrenceStatistics,
    SlotExposure,
    SlotKey,
)
from prediction.store import PredictionTreeStore, PublishedGeneration

# 门面**刻意不导出** ``prediction.source``：它是本层唯一 import behavior 的模块，
# 挂上门面等于让"只想查一棵已发布的树"的消费者也拖进整张行为树的导入图。
# 需要读行为树的调用方显式 ``from prediction import source``。

__all__ = [
    "EdgeStatistics",
    "IntervalQuantiles",
    "NodeCounts",
    "NodeStatistics",
    "ObservedAction",
    "ObservedGap",
    "PredictionTree",
    "PredictionTreeConfig",
    "PredictionTreeError",
    "PredictionTreeStore",
    "PredictionTreeStoreError",
    "PublishedGeneration",
    "RecurrenceStatistics",
    "SlotExposure",
    "SlotKey",
    "build",
    "config_digest",
]
