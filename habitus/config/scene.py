"""语义关联层的运行配置。

本模块只有纯标量、不 import ``scene``（与 ``config/behavior.py`` 同一纪律）。

旧语义树（按候选累积的关联）那一组参数——提示词字符上限、重试、每轮调用数、每批目标数、前因/前提
行数预算——随它整块删掉（2026-09-26）。新语义树按功能分支重建，每支的旋钮（映射召回数、删失覆盖
阈值、残差升级判据……）在那一支落地时再进配置，且数值一律先在重放上定，这里不预留空转的字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config


@dataclass(frozen=True)
class SceneConfig:
    """语义关联层是否启用。"""

    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("scene.enabled must be a boolean")

    @classmethod
    def from_mapping(cls, value: Any) -> SceneConfig:
        return construct_config(cls, value, "config.scene")


__all__ = ["SceneConfig"]
