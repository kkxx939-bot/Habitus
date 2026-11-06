"""模型适配器共用的 httpx 客户端构造：连接池、超时与出站设置（证书库、额外 CA、代理）一处给齐（W18）。

四处构造（chat 的同步与异步、embedding、rerank）以前各写一份、都没传 ``verify`` 与代理，所以只认 certifi、
不读系统证书库。现在全部从这里拿：``verify`` 是按出站设置缓存的 SSL context，配了代理时传 ``proxy``，
没配时不传、httpx 照旧读 ``HTTPS_PROXY`` 等环境变量；目标是回环地址时再用 ``mounts`` 挂成直连，两种代理都绕开。
"""

from __future__ import annotations

import httpx

from habitus.foundation.network import OutboundSettings
from habitus.model_client.config import ProviderConfig


def sync_http_client(route: ProviderConfig, outbound: OutboundSettings) -> httpx.Client:
    """给 ``route`` 这条路由建同步客户端；不跟随重定向。"""

    options = outbound.httpx_options(route.base_url)
    return httpx.Client(
        follow_redirects=False,
        limits=_limits(route),
        timeout=httpx.Timeout(float(route.timeout_seconds)),
        verify=options.verify,
        proxy=options.proxy,
        mounts=options.mounts,
    )


def async_http_client(route: ProviderConfig, outbound: OutboundSettings) -> httpx.AsyncClient:
    """给 ``route`` 这条路由建异步客户端（每个事件循环一个，由适配器缓存）；不跟随重定向。"""

    options = outbound.httpx_options(route.base_url)
    return httpx.AsyncClient(
        follow_redirects=False,
        limits=_limits(route),
        timeout=httpx.Timeout(float(route.timeout_seconds)),
        verify=options.verify,
        proxy=options.proxy,
        mounts=options.mounts,
    )


def _limits(route: ProviderConfig) -> httpx.Limits:
    return httpx.Limits(max_connections=route.max_concurrent, max_keepalive_connections=route.max_concurrent)


__all__ = ["async_http_client", "sync_http_client"]
