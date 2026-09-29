"""向量旁册共用的数值基元：单位化、余弦、float16 + base64 的落盘编码。

两本旁册（词表的 ``kinds.vectors.json``、概念集的 ``concepts/.vectors.json``）都是派生物，存的都是
"名字 → 单位化向量"，落盘精度一律 float16——写入时就按落盘精度截断，同一进程内与重启读回的候选
排序才一致。这里只放与任何一棵树都无关的四个函数。
"""

from __future__ import annotations

import base64
import math
import struct
from collections.abc import Sequence
from operator import mul


class VectorCodecError(ValueError):
    """落盘的向量文本解不回指定维度。"""


def normalized(values: Sequence[float]) -> tuple[float, ...]:
    """单位化；零向量原样返回（余弦为 0）。"""

    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return tuple(float(v) for v in values)
    return tuple(float(v) / norm for v in values)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """两个**已单位化**向量的余弦（点积）。"""

    return sum(map(mul, a, b))


def pack_float16(vector: Sequence[float]) -> str:
    if any(not math.isfinite(value) for value in vector):
        raise VectorCodecError("a stored vector must hold finite values")
    try:
        return base64.b64encode(struct.pack(f"<{len(vector)}e", *vector)).decode("ascii")
    except (OverflowError, struct.error) as exc:
        raise VectorCodecError("a stored vector does not fit float16") from exc


def unpack_float16(text: object, dimension: int) -> tuple[float, ...]:
    if not isinstance(text, str):
        raise VectorCodecError("a stored vector must be base64 text")
    try:
        raw = base64.b64decode(text, validate=True)
        values = struct.unpack(f"<{dimension}e", raw)
    except (ValueError, struct.error) as exc:
        raise VectorCodecError("a stored vector is not decodable") from exc
    return tuple(float(v) for v in values)


__all__ = ["VectorCodecError", "cosine", "normalized", "pack_float16", "unpack_float16"]
