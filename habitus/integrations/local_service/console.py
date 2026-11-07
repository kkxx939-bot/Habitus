"""``habitus-server`` 命令行的进程入口：先把标准输出与标准错误定成 UTF-8，再交给 ``cli.main``。

Windows 上 Python 写被管道或重定向的输出时按本地代码页编码（中文系统是 GBK），遇到代码页外的字符直接抛
``UnicodeEncodeError``，交给 Agent 读的输出也会乱码。这里统一改成 UTF-8；POSIX 上默认已是 UTF-8，重设一次无副作用。
"""

from __future__ import annotations

import io
import sys
from collections.abc import Sequence


def use_utf8_streams() -> None:
    """把 stdout / stderr 改成 UTF-8；被替换成不支持 ``reconfigure`` 的对象（测试捕获等）时不动它。"""

    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> None:
    use_utf8_streams()
    from habitus.integrations.local_service.cli import main as cli_main

    cli_main(argv)


__all__ = ["main", "use_utf8_streams"]
