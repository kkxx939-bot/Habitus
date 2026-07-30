"""m2bOS HTTP API 的监听与传输安全配置。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from Config.loader import construct_config

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class HTTPAPIConfig:
    """只描述 HTTP Adapter，不承载记忆或租户领域规则。"""

    host: str = "127.0.0.1"
    port: int = 8787
    api_key_env: str = "M2BOS_HTTP_API_KEY"
    operations_api_key_env: str = "M2BOS_HTTP_OPERATIONS_API_KEY"
    max_request_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.host, str)
            or not self.host
            or self.host != self.host.strip()
            or any(character.isspace() for character in self.host)
            or "/" in self.host
        ):
            raise ValueError("host must be a non-empty host name or IP literal")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65_535:
            raise ValueError("port must be an integer between 1 and 65535")
        for name in ("api_key_env", "operations_api_key_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or _ENV_NAME.fullmatch(value) is None:
                raise ValueError(f"{name} must be a normalized environment variable name")
        if self.operations_api_key_env == self.api_key_env:
            raise ValueError("operations_api_key_env must differ from api_key_env")
        if (
            isinstance(self.max_request_bytes, bool)
            or not isinstance(self.max_request_bytes, int)
            or not 1_024 <= self.max_request_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("max_request_bytes must be between 1024 and 67108864")

    @classmethod
    def from_mapping(cls, value: object) -> HTTPAPIConfig:
        return construct_config(cls, value, "config.http")


__all__ = ["HTTPAPIConfig"]
