"""会话 lane 的凭据在观测存储里用的协议名。

单独放一个不 import 任何东西的模块：逐帧融合的入队扫描要认它（见到就跳过），不该为此把会话 lane 的其余部分拉进来。
"""

SESSION_PROTOCOL = "habitus_session_turn_v1"

__all__ = ["SESSION_PROTOCOL"]
