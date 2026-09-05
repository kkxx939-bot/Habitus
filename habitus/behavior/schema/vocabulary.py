"""行为语义树受控词表；枚举取值属于不可配置的代码不变量。

旧的任务态词表（completed/failed/cancelled、outcome_type/valence、action 子字段等）随
``TODO(BHV-TREE-REBUILD-001)`` 整体退役——那是 coding agent 的领域模型，人的行为不 ``failed``
也不产生 ``human_response``。新词表逐字沿用融合层的判断词表：写入层零发明，状态从链尾判断
原样搬运。
"""

from __future__ import annotations

import re

SHA256 = re.compile(r"^[0-9a-f]{64}$")

# 这次行为怎么结束的；与 behavior.fusion.judgement.JudgementStatus 逐字一致。
OCCURRENCE_STATUSES = frozenset({"ongoing", "completed", "interrupted", "abandoned"})

# 上面那个结论怎么知道的；``interrupted``（确实没做完）与 ``observation_lost``（不知道）
# 之分是删失纪律的根，混了曝光统计就错。
STATUS_BASES = frozenset({"observed", "inferred", "observation_lost"})

# 观测空白的两种类型：没读懂（融合给出 behavior 为空的判断）、未观测（上游覆盖信号，
# 契约尚未接入，见 TODO(BHV-TREE-REBUILD-001) 上游缺口一节）。
GAP_KINDS = frozenset({"没读懂", "未观测"})

__all__ = [
    "GAP_KINDS",
    "OCCURRENCE_STATUSES",
    "SHA256",
    "STATUS_BASES",
]
