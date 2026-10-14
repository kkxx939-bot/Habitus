"""物理 lane 的保护上限（都是保护闸，不是语义参数）。"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PhysicalLaneConfig:
    """一份交付与每一段写出来的量的上限。

    - ``max_segments`` / ``max_coverage_spans``：一份交付里最多多少个行为段、多少个覆盖时段。
    - ``max_facts`` / ``max_fact_chars``：一段最多带几条补充事实、每条多少字。
    - ``max_summary_chars``：拼出来的概要的字数上限。
    - ``max_device_chars`` / ``max_segment_id_chars``：设备名、段编号的字数上限（它们要进凭据的来源身份与引用，
      那边有自己的上限；在入口先挡，免得一份交付写到一半才失败）。
    - ``max_future_seconds``：段的结束时刻最多比交付的送出时刻晚多少（上游只送已经结束的；这点余量留给时钟不齐）。
    """

    max_segments: int = 2_000
    max_coverage_spans: int = 2_000
    max_facts: int = 5
    max_fact_chars: int = 80
    max_summary_chars: int = 200
    max_device_chars: int = 120
    max_segment_id_chars: int = 200
    max_future_seconds: int = 300

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"physical lane {item.name} must be a positive integer")


__all__ = ["PhysicalLaneConfig"]
