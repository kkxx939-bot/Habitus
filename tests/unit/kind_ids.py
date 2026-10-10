"""测试数据用的类编号：行为树上的 ``kind_token`` 只能是基础词表的编号或占位标记（裁定 18 不留旧口径）。

测试里习惯用行为名（「吃药」）写数据；这里给每个名字配一个**稳定**的编号，同一个名字在任何测试、任何顺序下
都得到同一个编号，断言里用同一个函数换算即可。已经是编号或占位标记的原样返回。
"""

from __future__ import annotations

from hashlib import sha256

from habitus.behavior.kinds.ids import ClassId, KindIdError, Lane, lane_of_token

_SPACE = 900_000
_SEEN: dict[str, str] = {}


def kind_id(name: str, lane: Lane = Lane.SESSION) -> str:
    try:
        lane_of_token(name)
    except KindIdError:
        pass
    else:
        return name
    number = 1_000 + int(sha256(f"{lane.value}:{name}".encode()).hexdigest(), 16) % _SPACE
    token = str(ClassId(lane, number))
    owner = _SEEN.setdefault(token, name)
    if owner != name:
        raise AssertionError(f"test kind ids collide: {owner!r} and {name!r} both map to {token}")
    return token


def kind_names() -> dict[str, str]:
    """到目前为止配过的编号 → 名字：测试里充当词表给出的"编号 → 类名"。"""

    return dict(_SEEN)


__all__ = ["kind_id", "kind_names"]
