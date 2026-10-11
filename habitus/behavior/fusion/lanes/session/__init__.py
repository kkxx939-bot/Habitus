"""会话 lane 的融合：读一轮对话，写一条"这一轮在做什么"的判断。

方案见桌面 ``14-事件融合方案.md`` 第五、六节与裁定 31–37。要点：一轮一条、只描述不合并、带语义的判断只在词表。

模块分工：``model`` 输入形状；``material`` 拆轮与取材料；``redaction`` 抹密钥；``prompt`` 提示词与输出校验；
``service`` 调模型（本条 lane 唯一的模型触点）；``recorder`` 写判断；``lane`` 把它们串起来。
本包不认识会话源：把会话源信封翻成 ``SessionTurn``、接成会话源的消费者，是组合根的事。

TODO(BHV-SESSION-LANE-001)：方案里已定、这一轮没做的（裁定 35、31，用户 2026-10-10：「这7个可以，对应地方记todo 就行」）：
- 来源挡不住脚本调起的会话：会话源信封上没有"是不是本人在交互界面说的"这个标识，要插件与服务入口先带上来；
  现在只挡得住子代理（见组合根的桥）。影响：脚本批量调起的会话会被记成这个人的行为，树只增不改，记错了删不掉。
- 人一开口就通知服务"这一轮开始了"：插件已有 ``UserPromptSubmit`` 钩子，但服务没有接口收、归约也不会因此拦住封口。
  影响：助手干活期间系统不知道有事在发生。例：17:58 开口、19:55 做完的一轮，18:30 时系统看到的最近一条仍是 17:57 之前的。
"""

from habitus.behavior.fusion.lanes.session.config import SessionLaneConfig
from habitus.behavior.fusion.lanes.session.lane import SessionLane, TurnDisposition, TurnOutcome
from habitus.behavior.fusion.lanes.session.material import render_turn, split_turns
from habitus.behavior.fusion.lanes.session.model import SessionMessage, SessionMessageRole, SessionTurn
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
from habitus.behavior.fusion.lanes.session.service import DescribedTurn, SessionTurnDescriber

__all__ = [
    "SESSION_FUSION_VERSION",
    "SESSION_PROMPT_VERSION",
    "SESSION_PROTOCOL",
    "SESSION_SYSTEM_PROMPT",
    "DescribedTurn",
    "SessionLane",
    "SessionLaneConfig",
    "SessionMessage",
    "SessionMessageRole",
    "SessionTurn",
    "SessionTurnDescriber",
    "SessionTurnRecorder",
    "TurnDescription",
    "TurnDisposition",
    "TurnOutcome",
    "assemble_description",
    "redact",
    "render_turn",
    "session_json_schema",
    "session_observer",
    "split_turns",
]
