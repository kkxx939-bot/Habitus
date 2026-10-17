"""预测层实现的指纹（裁定 54）：预测层这份代码改了，指纹就变——每条预期记下形成它的那份代码，修了 bug 之后按它把预期分成前后
两批对比同一切片的读数（"调对了没有"）。和提示词同一个做法：指纹随正文走，不靠人记得升版本号。

算的是 ``habitus/foresight/`` 整包与组合根里 ``runtime/foresight*.py`` 的源码（按相对路径排序后连同路径一起哈希）。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent.parent


def implementation_fingerprint(root: Path = _PACKAGE) -> str:
    digest = sha256()
    files = sorted([*(root / "foresight").rglob("*.py"), *(root / "runtime").glob("foresight*.py")])
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


IMPLEMENTATION = implementation_fingerprint()

__all__ = ["IMPLEMENTATION", "implementation_fingerprint"]
