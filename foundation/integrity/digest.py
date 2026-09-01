"""基于规范 JSON 字节生成稳定摘要。"""

from __future__ import annotations

import hashlib
from typing import Any

from foundation.integrity.canonical_json import canonical_json


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    """返回精确 UTF-8 文本字节的 SHA-256 摘要。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bytes_digest(value: bytes) -> str:
    """对**已经是 UTF-8 字节**的内容直接取摘要，结果与 ``text_digest`` 逐位相同。

    存在的理由是省掉一次无谓的往返：手上已有字节时走 ``text_digest`` 要先 ``decode`` 出一份
    str、函数里再 ``encode`` 回一份 bytes——同一份内容在那一刻躺着三份，纯属白做。预测树的
    发布回读校验实测因此从 4.96× payload 降到 1.00×（``prediction/store.py``）。

    **它不是通用的省内存手段**：一旦这份字节还要被解析成对象，峰值就由对象图决定，摘要走
    哪条路都不影响（见 ``TODO(PRED-STORE-002)`` 的实测）。
    """

    return hashlib.sha256(value).hexdigest()


__all__ = ["bytes_digest", "canonical_digest", "text_digest"]
