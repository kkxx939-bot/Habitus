"""无认证 loopback 服务的 Host 与 Origin 请求边界。"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def local_request_violation(*, host_header: str | None, origin_header: str | None) -> str | None:
    """返回本地请求的拒绝原因；无 Origin 的本机客户端请求合法。"""

    if not _loopback_authority(host_header):
        return "Host must identify a loopback address"
    if origin_header is None:
        return None
    if origin_header.strip().casefold() == "null":
        return "Origin must identify a loopback HTTP origin"
    try:
        origin = urlsplit(origin_header)
    except ValueError:
        return "Origin must be a valid loopback HTTP origin"
    if origin.scheme not in {"http", "https"} or not _is_loopback(origin.hostname):
        return "Origin must identify a loopback HTTP origin"
    if origin.username is not None or origin.password is not None:
        return "Origin must not contain user information"
    return None


def _loopback_authority(value: str | None) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        parsed = urlsplit(f"//{value}")
        _ = parsed.port
    except ValueError:
        return False
    return _is_loopback(parsed.hostname)


def _is_loopback(host: str | None) -> bool:
    if not isinstance(host, str) or not host:
        return False
    normalized = host.strip("[]").casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


__all__ = ["local_request_violation"]
