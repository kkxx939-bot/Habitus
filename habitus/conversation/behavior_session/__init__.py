"""行为侧会话 lane 在会话源这边留下的回执（形状与存储）。

本包只有会话源自己的类型，不认识行为侧；把会话源信封交给会话 lane、再把结果写成这份回执的那座桥在组合根
（``runtime/session_lane.py``）。
"""

from habitus.conversation.behavior_session.model import (
    BEHAVIOR_SESSION_OUTPUT_KIND,
    BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
    BehaviorSessionOutput,
    BehaviorSessionTurnRecord,
)
from habitus.conversation.behavior_session.store import BehaviorSessionOutputStore

__all__ = [
    "BEHAVIOR_SESSION_OUTPUT_KIND",
    "BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION",
    "BehaviorSessionOutput",
    "BehaviorSessionOutputStore",
    "BehaviorSessionTurnRecord",
]
