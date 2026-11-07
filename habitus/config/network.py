"""配置里的 ``network`` 一组：调模型服务时的证书库、额外 CA 与代理（W18）。

值对象是 ``foundation.network.OutboundSettings``（各层共用）；这里只负责从 YAML 严格构造它，并在加载时就核实
``ca_bundle`` 指向的文件读得出 PEM 证书。整组可以省略，省略 = 系统证书库、无额外 CA、不配代理。
"""

from __future__ import annotations

from pathlib import Path

from habitus.config.loader import ConfigError, group_fields
from habitus.foundation.network import TRUST_STORES, OutboundSettings, OutboundSettingsError, check_ca_bundle
from habitus.foundation.validation import is_nonblank_str

_PATH = "config.network"


def outbound_settings_from_mapping(value: object) -> OutboundSettings:
    """严格读 ``network`` 一组；任何一项不合法都报 ``ConfigError``，写出字段与原因。"""

    data = group_fields(OutboundSettings, value, _PATH)
    trust_store = data.get("trust_store", "system")
    if not isinstance(trust_store, str) or trust_store not in TRUST_STORES:
        raise ConfigError(
            f"invalid '{_PATH}.trust_store': must be one of {', '.join(TRUST_STORES)}; got {trust_store!r}"
        )
    ca_bundle = _ca_bundle(data.get("ca_bundle"))
    proxy = data.get("proxy")
    if proxy is not None and not isinstance(proxy, str):
        raise ConfigError(f"invalid '{_PATH}.proxy': must be URL text such as http://proxy.example.com:3128")
    try:
        return OutboundSettings(trust_store=trust_store, ca_bundle=ca_bundle, proxy=proxy)
    except OutboundSettingsError as exc:
        raise ConfigError(f"invalid '{_PATH}.proxy': {exc}") from exc


def _ca_bundle(value: object) -> Path | None:
    if value is None:
        return None
    if not is_nonblank_str(value):
        raise ConfigError(f"invalid '{_PATH}.ca_bundle': must be a non-empty path to a PEM file")
    path = Path(value.strip()).expanduser().absolute()
    try:
        check_ca_bundle(path)
    except OutboundSettingsError as exc:
        raise ConfigError(f"invalid '{_PATH}.ca_bundle': {exc}") from exc
    return path


__all__ = ["outbound_settings_from_mapping"]
