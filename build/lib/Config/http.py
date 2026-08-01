"""m2bOS 单用户本地 HTTP 服务配置。"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from Config.loader import construct_config


@dataclass(frozen=True)
class HTTPAPIConfig:
    """只描述 loopback HTTP Adapter，不承载记忆或租户领域规则。"""

    host: str = "127.0.0.1"
    port: int = 8787
    max_request_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.host, str)
            or not self.host
            or self.host != self.host.strip()
            or any(character.isspace() for character in self.host)
        ):
            raise ValueError("host must be normalized non-empty text")
        if not self._is_loopback(self.host):
            raise ValueError("unauthenticated local service must bind to a loopback address")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65_535:
            raise ValueError("port must be an integer between 1 and 65535")
        if (
            isinstance(self.max_request_bytes, bool)
            or not isinstance(self.max_request_bytes, int)
            or not 1_024 <= self.max_request_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("max_request_bytes must be between 1024 and 67108864")

    @classmethod
    def from_mapping(cls, value: object) -> HTTPAPIConfig:
        return construct_config(cls, value, "config.http")

    @staticmethod
    def _is_loopback(host: str) -> bool:
        if host.casefold() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False


__all__ = ["HTTPAPIConfig"]
