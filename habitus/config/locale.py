"""主体所在地：时区与当地日历。

**为什么单独成一组而不是塞进某一层**：情景树读侧要用它标注每一天的日型（``scene.calendar``），
预测层要用它造"此刻"的时钟——两层用的必须是同一个地方，分别配两份迟早会对不上。配置单一出处
这条纪律在这里的落法就是：一组 ``locale``，组合根按字段分发给谁需要谁。

本模块不 import ``zoneinfo`` 以外的任何东西，也不读日历文件：时区名只在这里做"IANA 里有没有
这个名字"的校验，日历文件的格式与解析属于 ``scene.calendar`` 的实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from habitus.config.loader import construct_config

# 地区码只做形状校验（两个大写字母，ISO 3166-1 alpha-2 的形状），不内置国家表：
# 当地日历数据自己知道认哪些地区，配置层替它立一张表只会过期。
_REGION_LENGTH = 2


@dataclass(frozen=True)
class LocaleConfig:
    """主体的时区、地区码与当地日历数据文件。

    ``calendar_path`` 为空即**没有当地日历数据**——那时日型恒为空（``scene.NominalCalendar``），
    这是显式的零修正，不是降级。数据文件的格式尚未定，所以现在给了路径直接拒，而不是静默
    忽略：静默忽略会让人以为调休已经生效。
    """

    timezone: str = "Asia/Shanghai"
    region: str = "CN"
    calendar_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError("locale.timezone must be a non-empty IANA time zone name")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"locale.timezone is not a known IANA time zone: {self.timezone!r}") from exc
        if not isinstance(self.region, str) or len(self.region) != _REGION_LENGTH or not self.region.isalpha() or not self.region.isupper():
            raise ValueError("locale.region must be a two-letter uppercase region code")
        if self.calendar_path is not None:
            if not isinstance(self.calendar_path, str) or not self.calendar_path.strip():
                raise ValueError("locale.calendar_path must be a non-empty path or null")
            raise ValueError(
                "locale.calendar_path is not supported yet; leave it null until the calendar data "
                "format is settled, otherwise the day type would silently stay empty"
            )

    def zone(self) -> ZoneInfo:
        """组合根用它造时钟与本地日界；校验已经在构造时做过。"""

        return ZoneInfo(self.timezone)

    @classmethod
    def from_mapping(cls, value: Any) -> LocaleConfig:
        return construct_config(cls, value, "config.locale")


__all__ = ["LocaleConfig"]
