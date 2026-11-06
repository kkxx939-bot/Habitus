"""没有系统 IANA 时区库时（Windows 的常态），时区配置只靠 ``tzdata`` 包也必须能加载（W2）。"""

import subprocess
import sys

# 在子进程里清空 zoneinfo 的系统搜索路径，模拟 Windows：此时 ZoneInfo 只能从 tzdata 包里找时区。
# 用子进程是为了不污染本进程里 zoneinfo 的路径与缓存。
_SCRIPT = """
import zoneinfo
zoneinfo.reset_tzpath(to=[])
from habitus.config.locale import LocaleConfig
for name in ("Asia/Shanghai", "Asia/Singapore", "America/Los_Angeles"):
    LocaleConfig(timezone=name)
LocaleConfig()
print("ok")
"""


def test_locale_loads_from_tzdata_without_a_system_time_zone_database() -> None:
    result = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=60, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
