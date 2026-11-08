"""Conversation 生命周期维护各协作者共用的错误类型。"""

from __future__ import annotations


class ConversationLifecycleError(RuntimeError):
    """Conversation 生命周期状态不完整，不能继续压缩或清理。"""


__all__ = ["ConversationLifecycleError"]
