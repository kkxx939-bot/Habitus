"""会话 lane 的融合：读一轮对话，写一条"这一轮在做什么"的判断。

要点（裁定 31–37）：一轮一条、只描述不合并、带语义的判断只在词表；
每轮描述之后在初始词表里选一个词（裁定 101），选词结果随判断一起存。

模块分工：``model`` 输入形状；``material`` 拆轮与取材料；``redaction`` 抹密钥；``prompt`` 提示词与输出校验；
``service`` 调模型描述这一轮；``picking`` 选词口（实现由组合根注入）；``recorder`` 写判断；``lane`` 把它们串起来。
本包不认识会话源：把会话源信封翻成 ``SessionTurn``、接成会话源的消费者，是组合根的事。
"""

# TODO(BHV-SESSION-LANE-001): 挡不住脚本调起的会话；人一开口时服务不知道这一轮开始了。

from habitus.behavior.fusion.lanes.session.config import SessionLaneConfig
from habitus.behavior.fusion.lanes.session.lane import SessionLane, TurnDisposition, TurnHandler, TurnOutcome
from habitus.behavior.fusion.lanes.session.material import render_turn, split_turns
from habitus.behavior.fusion.lanes.session.model import SessionMessage, SessionMessageRole, SessionTurn
from habitus.behavior.fusion.lanes.session.picking import NOTICE_DESCRIPTION, TurnPicker
from habitus.behavior.fusion.lanes.session.prompt import (
    SESSION_PROMPT_VERSION,
    SESSION_SYSTEM_PROMPT,
    TurnDescription,
    assemble_description,
    session_json_schema,
)
from habitus.behavior.fusion.lanes.session.protocol import SESSION_PROTOCOL
from habitus.behavior.fusion.lanes.session.recorder import SESSION_FUSION_VERSION, SessionTurnRecorder, session_observer
from habitus.behavior.fusion.lanes.session.redaction import redact
from habitus.behavior.fusion.lanes.session.reminders import (
    NoReminders,
    PendingReminder,
    ReminderFeedback,
    ReminderResponse,
)
from habitus.behavior.fusion.lanes.session.service import DescribedTurn, SessionTurnDescriber
from habitus.behavior.fusion.lanes.session.unwatched import SessionUnwatchedRecorder

__all__ = [
    "NOTICE_DESCRIPTION",
    "SESSION_FUSION_VERSION",
    "SESSION_PROMPT_VERSION",
    "SESSION_PROTOCOL",
    "SESSION_SYSTEM_PROMPT",
    "DescribedTurn",
    "NoReminders",
    "PendingReminder",
    "ReminderFeedback",
    "ReminderResponse",
    "SessionLane",
    "SessionLaneConfig",
    "SessionMessage",
    "SessionMessageRole",
    "SessionTurn",
    "SessionTurnDescriber",
    "SessionTurnRecorder",
    "SessionUnwatchedRecorder",
    "TurnDescription",
    "TurnDisposition",
    "TurnHandler",
    "TurnOutcome",
    "TurnPicker",
    "assemble_description",
    "redact",
    "render_turn",
    "session_json_schema",
    "session_observer",
    "split_turns",
]
