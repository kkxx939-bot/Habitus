"""主体所在地：时区与当地日历。

**为什么单独成一组而不是塞进某一层**：语义树与预测层要用它取当地日历（``conditions.calendar``），
预测层要用它造"此刻"的时钟——两层用的必须是同一个地方，分别配两份迟早会对不上。配置单一出处
这条纪律在这里的落法就是：一组 ``locale``，组合根按字段分发给谁需要谁。

本模块不 import ``zoneinfo`` 以外的任何东西，也不读日历文件：时区名只在这里做"IANA 里有没有
这个名字"的校验，日历文件的格式与解析属于 ``conditions.calendar_data``。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from habitus.config.loader import construct_config

# 地区码只做形状校验（ISO 3166-1 两位国家码，或带 3166-2 的下一级：US-CA），不内置国家表：
# 当地日历数据自己知道认哪些地区，配置层替它立一张表只会过期。日历跟着这个人在哪，不跟着服务部署在哪（裁定 59）。
_REGION = re.compile(r"[A-Z]{2}(-[A-Z0-9]{1,3})?")


@dataclass(frozen=True)
class LocaleConfig:
    """主体的时区、地区码与当地日历数据目录。

    放假安排是外接组件（裁定 59）：``calendar_path`` 指向一个目录，里面每年一份 ``<年>.json``（照国务院当年的通知录入）。
    不给就没有放假数据，日型答不出（未知，不当普通工作日）；周几、几号、月末照常答。
    """

    timezone: str = "Asia/Shanghai"
    region: str = "CN"
    calendar_path: str | None = None
    # 没有放假数据时、建预期要一个名义日型的兜底：哪几天算周末（0=周一 … 6=周日）。统计里不用它（未知就是未知）。
    nominal_weekend: tuple[int, ...] = (5, 6)

    def __post_init__(self) -> None:
        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError("locale.timezone must be a non-empty IANA time zone name")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"locale.timezone is not a known IANA time zone: {self.timezone!r}") from exc
        if not isinstance(self.region, str) or _REGION.fullmatch(self.region) is None:
            raise ValueError("locale.region must be an ISO 3166 code such as CN, SG or US-CA")
        weekend = self.nominal_weekend
        if (
            isinstance(weekend, str | bytes)
            or not isinstance(weekend, Sequence)
            or not weekend
            or any(isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6 for day in weekend)
            or len(set(weekend)) != len(weekend)
        ):
            raise ValueError("locale.nominal_weekend must list distinct weekdays 0–6")
        object.__setattr__(self, "nominal_weekend", tuple(sorted(weekend)))
        if self.calendar_path is not None and (
            not isinstance(self.calendar_path, str) or not self.calendar_path.strip()
        ):
            raise ValueError("locale.calendar_path must be a non-empty directory path or null")

    def zone(self) -> ZoneInfo:
        """组合根用它造时钟与本地日界；校验已经在构造时做过。"""

        return ZoneInfo(self.timezone)

    @classmethod
    def from_mapping(cls, value: Any) -> LocaleConfig:
        return construct_config(cls, value, "config.locale")


__all__ = ["LocaleConfig"]
