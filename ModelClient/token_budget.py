"""供应商无关的保守 Token 预算估算。"""

from __future__ import annotations

from ModelClient.contracts import ChatRequest


def estimate_text_tokens(value: str) -> int:
    """按 UTF-8 字节给出保守近似；真实 Provider 以后可以在适配层收紧。"""

    if not isinstance(value, str):
        raise TypeError("token estimation input must be text")
    return estimate_utf8_bytes_tokens(len(value.encode("utf-8")))


def estimate_utf8_bytes_tokens(byte_count: int) -> int:
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("UTF-8 byte count must be a non-negative integer")
    return max(1, (byte_count + 2) // 3)


def estimate_chat_request_tokens(request: ChatRequest) -> int:
    """估算完整消息、工具和响应 Schema 的输入 Token。"""

    if not isinstance(request, ChatRequest):
        raise TypeError("request must be a ChatRequest")
    total = 2
    for message in request.messages:
        total += 4 + estimate_text_tokens(message.role) + estimate_text_tokens(message.content or "")
    for tool in request.tools:
        total += estimate_text_tokens(str(tool))
    if request.response_format is not None:
        total += estimate_text_tokens(str(request.response_format.schema))
    return total


__all__ = ["estimate_chat_request_tokens", "estimate_text_tokens", "estimate_utf8_bytes_tokens"]
