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

# 观测空白的三种类型：没读懂（模型读了、给不出结论）、错误（系统出了问题：调不通、抛异常、文件坏，裁定 94 第 3 条）、
# 未观测（上游覆盖信号，契约尚未接入，见 TODO(BHV-TREE-REBUILD-001) 上游缺口一节）。
GAP_KINDS = frozenset({"没读懂", "错误", "未观测"})
#: 在看、但没看明白的那一种：它可以被那段里读出的行为作废（``behavior.schema.gaps``）。
UNREADABLE_GAP = "没读懂"
#: 在看、但系统出了问题没读成：结算与「没读懂」同为"看不到"，同样可被那段里补回来的行为作废；夜里前置处理统一看（裁定 94 第 3、4 条）。
ERROR_GAP = "错误"
#: 在看、却没读出结论的两种（「没读懂」「错误」）：时窗里有就不开口，被那段里读出的行为作废。
UNSEEN_GAPS = frozenset({UNREADABLE_GAP, ERROR_GAP})

# 这件事是谁发起的（裁定 54）：系统提醒了他 / 宿主助手先通知了他（包裹到了、账单到期）/ 他自己。不知道就不写这个字段。
PROMPTERS = frozenset({"system", "host_agent", "self"})

__all__ = [
    "GAP_KINDS",
    "ERROR_GAP",
    "UNREADABLE_GAP",
    "UNSEEN_GAPS",
    "PROMPTERS",
    "OCCURRENCE_STATUSES",
    "SHA256",
    "STATUS_BASES",
]
