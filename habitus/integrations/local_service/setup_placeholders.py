"""找出模型路由里还留着向导占位值的配置项（W7；W10a 起 Chat 同样适用）。

非交互 ``init`` 写出的 Chat 与 Embedding 段是不预置厂商的模板：地址与模型名带 ``example.``。占位片段由 Profile 字段的
``placeholder_values`` 声明，这里只按注册表判，不认任何厂商或模型名。doctor、服务启动前检查与 ``init`` 的终端提示
都用它，好在发请求之前就说清"还是占位值、要填哪几项"，而不是等到报 DNS 或连接错误。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from habitus.config import HabitusConfig
from habitus.integrations.local_service.setup_registry import (
    SetupCapability,
    SetupRegistry,
    is_setup_placeholder,
    model_document,
)

_MODEL_CAPABILITIES: tuple[SetupCapability, ...] = ("chat", "embedding", "rerank")


@dataclass(frozen=True)
class UnfilledSetup:
    """一个模型能力还留着占位值：``placeholders`` 是占位项，``to_fill`` 是同一 Profile 里要一起填的必填项。"""

    capability: SetupCapability
    placeholders: tuple[str, ...]
    to_fill: tuple[str, ...]

    def describe(self, config_path: Path | str) -> str:
        prefix = f"models.{self.capability}"
        placeholders = ", ".join(f"{prefix}.{path}" for path in self.placeholders)
        to_fill = ", ".join(f"{prefix}.{path}" for path in self.to_fill)
        return (
            f"{prefix} still holds setup placeholder values ({placeholders}); "
            f"fill in {to_fill} with the real service values in {config_path}"
        )


def unfilled_settings(registry: SetupRegistry, config: HabitusConfig) -> tuple[UnfilledSetup, ...]:
    """按已注册 Profile 的占位片段检查 chat、embedding、rerank 路由；没有对应 Profile 的第三方适配器不查。"""

    result: list[UnfilledSetup] = []
    for capability in _MODEL_CAPABILITIES:
        model = getattr(config.models, capability)
        if model is None:
            continue
        document = model_document(model)
        if not any(profile.matches(document) for profile in registry.profiles(capability, include_unselectable=True)):
            continue
        profile = registry.identify(capability, document)
        placeholders = tuple(
            item.paths[0]
            for item in profile.fields
            if is_setup_placeholder(_text_at(document, item.paths[0]), item.placeholder_values)
        )
        if placeholders:
            to_fill = tuple(item.paths[0] for item in profile.fields if item.kind in {"required_text", "positive_int"})
            result.append(UnfilledSetup(capability, placeholders, to_fill))
    return tuple(result)


def _text_at(document: Mapping[str, object], path: str) -> str:
    current: object = document
    for part in path.split("."):
        current = current.get(part, "") if isinstance(current, Mapping) else ""
    return current if isinstance(current, str) else ""


__all__ = ["UnfilledSetup", "unfilled_settings"]
