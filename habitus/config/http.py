"""Habitus 单用户本地 HTTP 服务配置。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.config.loader import construct_config
from habitus.foundation.network import is_loopback_host


@dataclass(frozen=True)
class HTTPAPIConfig:
    """只描述 loopback HTTP Adapter，不承载记忆或租户领域规则。"""

    host: str = "127.0.0.1"
    port: int = 8787
    max_request_bytes: int = 16 * 1024 * 1024
    # 单独一轮超过 max_request_bytes 时插件分块上传（remember_parts_v1）：暂存在内存里的总字节上限与一次上传的过期时间
    remember_parts_max_bytes: int = 64 * 1024 * 1024
    remember_parts_ttl_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not self.host or self.host != self.host.strip() or any(character.isspace() for character in self.host):
            raise ValueError("host must be normalized non-empty text")
        if not is_loopback_host(self.host):
            raise ValueError("unauthenticated local service must bind to a loopback address")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be an integer between 1 and 65535")
        if not 1_024 <= self.max_request_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_request_bytes must be between 1024 and 67108864")
        if not self.max_request_bytes <= self.remember_parts_max_bytes <= 1024 * 1024 * 1024:
            raise ValueError("remember_parts_max_bytes must be between max_request_bytes and 1073741824")
        if not 1 <= self.remember_parts_ttl_seconds <= 86_400:
            raise ValueError("remember_parts_ttl_seconds must be between 1 and 86400")

    @classmethod
    def from_mapping(cls, value: object) -> HTTPAPIConfig:
        return construct_config(cls, value, "config.http")


__all__ = ["HTTPAPIConfig"]
