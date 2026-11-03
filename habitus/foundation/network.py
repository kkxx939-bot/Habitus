"""远端地址的明文判定：只有回环地址可以走明文 HTTP。"""

from __future__ import annotations

import ipaddress


def is_loopback_host(hostname: str) -> bool:
    """``localhost`` 或回环 IP 字面量（不做 DNS 解析）：IPv4 的 127.0.0.0/8 与 IPv6 的 ``::1``。

    IPv4 映射的 IPv6 地址（``::ffff:127.0.0.1``）与带作用域的 ``::1%lo0`` 不算：插件那一侧（Node 的 URL 解析）不认它们，
    本地服务若绑在这种地址上，插件读到的连接地址会被拒收。两侧按同一张判例表判
    （``plugins/memory-plugin-shared/test/fixtures/service-contract.json`` 的 ``loopback_hosts``）。
    """

    if hostname.casefold() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and (address.ipv4_mapped is not None or address.scope_id is not None):
        return False
    return address.is_loopback


__all__ = ["is_loopback_host"]
