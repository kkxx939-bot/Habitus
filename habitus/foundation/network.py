"""出站网络的共用底：回环地址判定，以及调模型服务时的证书库、额外 CA 与代理设置（W18）。"""

from __future__ import annotations

import functools
import ipaddress
import os
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

#: 证书库来源：``system`` 读操作系统证书库（``truststore``：Windows 证书存储、macOS 钥匙串、Linux ``/etc/ssl``）；
#: ``bundled`` 只信 ``certifi`` 自带的那一份，留给没有系统证书库的极简容器。
TRUST_STORES = ("system", "bundled")
#: 代理需要认证时的去处：秘密值只能在 ``credentials`` 段，配置里的 ``network.proxy`` 不收账号密码。
_PROXY_CREDENTIALS_HINT = (
    "must not contain a user name or password; for a proxy that needs Basic authentication put the URL with "
    "credentials in the HTTPS_PROXY environment variable instead"
)


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


class OutboundSettingsError(ValueError):
    """出站设置本身不合法，或它指向的证书文件、证书库此刻用不了。报错写出字段与原因。"""


@dataclass(frozen=True)
class HttpxClientOptions:
    """构造 httpx 客户端要的三项：``verify``（SSL context）、``proxy``、``mounts``（回环直连）。"""

    verify: ssl.SSLContext
    proxy: str | None = None
    mounts: Mapping[str, None] | None = None


@dataclass(frozen=True)
class OutboundSettings:
    """调模型服务（chat、embedding、rerank）时的出站设置；不可变、可哈希，SSL context 按设置缓存。

    - ``trust_store``：见 ``TRUST_STORES``。
    - ``ca_bundle``：在证书库之上**额外**信任的 PEM 文件（公司代理做 TLS 解密时的根证书）。
    - ``proxy``：配了就用它、覆盖 ``HTTPS_PROXY`` 等环境变量；没配就照 httpx 默认读环境变量。
      回环地址（``is_loopback_host``）不论代理来自配置还是环境变量都直连：公司代理到不了用户电脑上的
      回环地址，自部署在本机的模型服务不能被送进去（用户 2026-11-06 定）。
    """

    trust_store: str = "system"
    ca_bundle: Path | None = None
    proxy: str | None = None

    def __post_init__(self) -> None:
        if self.trust_store not in TRUST_STORES:
            raise OutboundSettingsError(
                f"trust_store must be one of {', '.join(TRUST_STORES)}; got {self.trust_store!r}"
            )
        if self.ca_bundle is not None:
            object.__setattr__(self, "ca_bundle", Path(self.ca_bundle).expanduser().absolute())
        if self.proxy is not None:
            _validate_proxy_url(self.proxy)

    def ssl_context(self) -> ssl.SSLContext:
        """按 ``(trust_store, ca_bundle)`` 缓存的客户端 SSL context；同一设置在进程里只建一次。"""

        return _cached_ssl_context(
            self.trust_store, self.ca_bundle, os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_CERT_DIR")
        )

    def httpx_options(self, target_url: str) -> HttpxClientOptions:
        """连 ``target_url`` 这个模型服务的 httpx 客户端参数。

        配了代理就传 ``proxy``；没配时不传，httpx 照旧读环境变量。目标是回环地址时再用 ``mounts`` 把这个主机
        挂成直连（值 ``None`` = 不经代理的默认传输）：httpx 按主机匹配的挂载优先于 ``all://``、``https://``
        这类代理挂载，所以配置的代理和环境变量代理都绕得开。
        """

        context = self.ssl_context()
        hostname = (urlsplit(target_url).hostname or "").casefold()
        if not is_loopback_host(hostname):
            return HttpxClientOptions(verify=context, proxy=self.proxy)
        pattern = f"all://[{hostname}]" if ":" in hostname else f"all://{hostname}"
        return HttpxClientOptions(verify=context, proxy=self.proxy, mounts={pattern: None})


def check_ca_bundle(path: Path) -> None:
    """``ca_bundle`` 现在读得出来吗：存在、是普通文件、里面有 PEM 证书。报错写出路径。"""

    if not path.exists():
        raise OutboundSettingsError(f"ca_bundle file does not exist: {path}")
    if not path.is_file():
        raise OutboundSettingsError(f"ca_bundle must be a regular file: {path}")
    probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    try:
        probe.load_verify_locations(cafile=path)
    except ssl.SSLError as exc:
        raise OutboundSettingsError(f"ca_bundle contains no readable PEM certificate: {path}") from exc
    except OSError as exc:
        raise OutboundSettingsError(f"ca_bundle cannot be read: {path} ({exc.strerror or type(exc).__name__})") from exc


def build_ssl_context(
    trust_store: str, ca_bundle: Path | None, environ: Mapping[str, str] | None = None
) -> ssl.SSLContext:
    """不经缓存现建一个客户端 SSL context（doctor 用它查"此刻建得出来吗"）。

    W18 之前 httpx 用默认 ``verify=True``，会读 ``SSL_CERT_FILE``（没有时 ``SSL_CERT_DIR``）；有人靠它让公司根证书生效。
    这里照同样的先后读这两个环境变量，叠加在证书库之上，旧做法升级后仍然有效。
    """

    if ca_bundle is not None:
        check_ca_bundle(ca_bundle)
    environ = os.environ if environ is None else environ
    if trust_store == "system":
        try:
            import truststore
        except ImportError as exc:
            raise OutboundSettingsError(
                "trust_store=system needs the truststore package, which cannot be imported; reinstall Habitus "
                "or set network.trust_store: bundled"
            ) from exc
        context: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    else:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    if ca_bundle is not None:
        context.load_verify_locations(cafile=ca_bundle)
    _load_environment_certificates(context, environ)
    return context


def _load_environment_certificates(context: ssl.SSLContext, environ: Mapping[str, str]) -> None:
    for variable, keyword in (("SSL_CERT_FILE", "cafile"), ("SSL_CERT_DIR", "capath")):
        value = environ.get(variable)
        if not value:
            continue
        try:
            context.load_verify_locations(**{keyword: value})
        except (ssl.SSLError, OSError) as exc:
            raise OutboundSettingsError(f"environment variable {variable}={value} cannot be loaded: {exc}") from exc
        return


@functools.lru_cache(maxsize=16)
def _cached_ssl_context(
    trust_store: str, ca_bundle: Path | None, cert_file: str | None, cert_dir: str | None
) -> ssl.SSLContext:
    environ = {
        name: value for name, value in (("SSL_CERT_FILE", cert_file), ("SSL_CERT_DIR", cert_dir)) if value is not None
    }
    return build_ssl_context(trust_store, ca_bundle, environ)


def redact_proxy_url(url: str) -> str:
    """去掉代理 URL 里的用户名密码，只留 ``scheme://host[:port]``，给诊断输出用。"""

    candidate = url.strip() if "://" in url else f"http://{url.strip()}"
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        return "<unparseable proxy URL>"
    hostname = parsed.hostname or ""
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parsed.scheme}://{host}" + (f":{port}" if port is not None else "")


def _validate_proxy_url(value: str) -> None:
    """配置层已保证是字符串；这里只判 URL 的形状。"""

    if not value:
        raise OutboundSettingsError("proxy must not be empty; omit it or use null for no configured proxy")
    if value != value.strip() or any(character.isspace() for character in value):
        raise OutboundSettingsError("proxy must not contain whitespace")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:  # 方括号不配对的 IPv6 主机等
        raise OutboundSettingsError(f"proxy is not a valid URL ({exc})") from exc
    if parsed.scheme not in {"http", "https"} or "://" not in value:
        raise OutboundSettingsError("proxy must start with http:// or https:// (SOCKS and PAC are not supported)")
    if parsed.username is not None or parsed.password is not None:
        raise OutboundSettingsError(f"proxy {_PROXY_CREDENTIALS_HINT}")
    if not parsed.hostname:
        raise OutboundSettingsError("proxy must name a host")
    try:
        parsed.port  # noqa: B018 - 取一次端口就是校验（越界或不是数字时抛 ValueError）
    except ValueError as exc:
        raise OutboundSettingsError("proxy port must be a number between 0 and 65535") from exc
    if parsed.path not in {"", "/"}:
        raise OutboundSettingsError("proxy must not contain a path")
    if parsed.query or "?" in value:
        raise OutboundSettingsError("proxy must not contain a query")
    if parsed.fragment or "#" in value:
        raise OutboundSettingsError("proxy must not contain a fragment")


__all__ = [
    "TRUST_STORES",
    "HttpxClientOptions",
    "OutboundSettings",
    "OutboundSettingsError",
    "build_ssl_context",
    "check_ca_bundle",
    "is_loopback_host",
    "redact_proxy_url",
]
