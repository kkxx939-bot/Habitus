"""核心依赖必须写在 ``pyproject.toml`` 的 ``dependencies`` 里，而不是碰巧被别的包带进来。

CI 先装锁文件、再 ``-e . --no-deps``，只删 pyproject 里的一行不会让任何功能测试变红；用户 ``pip install`` 时却会缺包。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# 名字 → 为什么必须是直接依赖
REQUIRED = {
    "tzdata": "Windows 没有系统 IANA 时区库，habitus/config/locale.py 构造时就调 ZoneInfo（W2）",
    "truststore": "network.trust_store: system 读操作系统证书库（W18）",
    "certifi": "network.trust_store: bundled 直接用它（habitus/foundation/network.py，W18）",
    "tzlocal": "habitus-server init 检测系统时区，Windows 上把注册表时区名换成 IANA 名（W3）",
}


def test_core_runtime_packages_are_direct_dependencies() -> None:
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["dependencies"]
    names = {re.split(r"[\s<>=!~;\[]", item, maxsplit=1)[0].casefold() for item in declared}
    missing = {name: reason for name, reason in REQUIRED.items() if name not in names}
    assert not missing, f"pyproject.toml dependencies 缺少：{missing}"
