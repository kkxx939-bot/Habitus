"""主体所在地：时区与当地日历。

**为什么单独成一组而不是塞进某一层**：语义树与预测层要用它取当地日历（``conditions.calendar``），
预测层要用它造"此刻"的时钟——两层用的必须是同一个地方，分别配两份迟早会对不上。配置单一出处
这条纪律在这里的落法就是：一组 ``locale``，组合根按字段分发给谁需要谁。

本模块不 import ``zoneinfo`` 以外的任何东西，也不读日历文件：时区名只在这里做"IANA 里有没有
这个名字"的校验，日历文件的格式与解析属于 ``conditions.calendar_data``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

from habitus.config.loader import construct_config

# 地区码只做形状校验（ISO 3166-1 两位国家码，或带 3166-2 的下一级：US-CA），不内置国家表：
# 当地日历数据自己知道认哪些地区，配置层替它立一张表只会过期。日历跟着这个人在哪，不跟着服务部署在哪（裁定 59）。
_REGION = re.compile(r"[A-Z]{2}(-[A-Z0-9]{1,3})?")


@cache
def _known_timezones() -> frozenset[str]:
    return frozenset(available_timezones())


@dataclass(frozen=True)
class LocaleConfig:
    """主体的时区、地区码与当地日历数据目录。

    放假安排是外接组件（裁定 59）：``calendar_path`` 指向一个目录，里面每年一份 ``<年>.json``（照当地官方当年公布的安排录入）。
    不给就读随包附带的这个地区的那一份（目前有 CN 2025、2026，照国务院办公厅通知，裁定 80 第 2 条；SG 2025、2026，照新加坡
    人力部公布的公共假期，W3）；哪年没有数据，那一年的日型答不出（未知，不当普通工作日）；周几、几号、月末照常答。

    不写这一组时默认新加坡（W3，用户 2026-11-04 定）；``habitus-server init`` 新建配置时检测系统时区、请用户确认后写进来。
    """

    timezone: str = "Asia/Singapore"
    region: str = "SG"
    calendar_path: str | None = None
    # 没有放假数据时、建预期要一个名义日型的兜底：哪几天算周末（0=周一 … 6=周日）。统计里不用它（未知就是未知）。
    nominal_weekend: tuple[int, ...] = (5, 6)

    def __post_init__(self) -> None:
        if not self.timezone.strip():
            raise ValueError("locale.timezone must be a non-empty IANA time zone name")
        # 按 tzdata 的名字表逐字比对（区分大小写），不用 ZoneInfo 能不能打开来判：macOS 默认文件系统不分大小写，
        # "asia/singapore" 在那里打得开，配置搬到 Linux 就读不出来
        if self.timezone not in _known_timezones():
            raise ValueError(
                f"locale.timezone is not a known IANA time zone (names are case-sensitive): {self.timezone!r}"
            )
        if _REGION.fullmatch(self.region) is None:
            raise ValueError("locale.region must be an ISO 3166 code such as CN, SG or US-CA")
        weekend = self.nominal_weekend
        if not weekend or any(not 0 <= day <= 6 for day in weekend) or len(set(weekend)) != len(weekend):
            raise ValueError("locale.nominal_weekend must list distinct weekdays 0–6")
        object.__setattr__(self, "nominal_weekend", tuple(sorted(weekend)))
        if self.calendar_path is not None and not self.calendar_path.strip():
            raise ValueError("locale.calendar_path must be a non-empty directory path or null")

    def zone(self) -> ZoneInfo:
        """组合根用它造时钟与本地日界；校验已经在构造时做过。"""

        return ZoneInfo(self.timezone)

    @classmethod
    def from_mapping(cls, value: Any) -> LocaleConfig:
        return construct_config(cls, value, "config.locale")


__all__ = ["LocaleConfig"]
