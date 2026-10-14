"""会话 lane 的凭据在观测存储里用的协议名。

凭据进了观测存储之后，靠这个名字认出它是哪条 lane 写的。
"""

SESSION_PROTOCOL = "habitus_session_turn_v1"

__all__ = ["SESSION_PROTOCOL"]
