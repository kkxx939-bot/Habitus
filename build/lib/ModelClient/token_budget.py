"""供应商无关的保守 Token 预算估算。"""

from __future__ import annotations


def estimate_text_tokens(value: str) -> int:
    """按 UTF-8 字节给出保守近似；真实 Provider 以后可以在适配层收紧。"""

    if not isinstance(value, str):
        raise TypeError("token estimation input must be text")
    return estimate_utf8_bytes_tokens(len(value.encode("utf-8")))


def estimate_utf8_bytes_tokens(byte_count: int) -> int:
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("UTF-8 byte count must be a non-negative integer")
    return max(1, (byte_count + 2) // 3)


def estimate_model_visible_bytes(value: bytes) -> int:
    """估算已冻结、将送入模型上下文的规范 JSON 投影。"""

    if not isinstance(value, bytes) or not value:
        raise ValueError("model-visible token input must be non-empty bytes")
    return 2 + estimate_utf8_bytes_tokens(len(value))


__all__ = [
    "estimate_model_visible_bytes",
    "estimate_text_tokens",
    "estimate_utf8_bytes_tokens",
]
