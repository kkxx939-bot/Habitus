"""配置里的 ``network`` 一组（W18）：整组可省略，三个字段各自的合法与非法值都在加载时报清楚。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from habitus.config import ConfigError, HabitusConfig
from habitus.foundation.network import OutboundSettings
from habitus.infrastructure.store.filesystem import restrict_file_to_owner
from tests.platform_helpers import deny_all_access
from tests.tls_helpers import TEST_CA

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "habitus" / "config" / "example.yaml"


def _payload(tmp_path: Path, network: object = ...) -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    if network is ...:
        payload.pop("network", None)
    else:
        payload["network"] = network
    return payload


def test_omitting_the_network_group_means_the_defaults(tmp_path: Path) -> None:
    config = HabitusConfig.from_mapping(_payload(tmp_path))
    assert config.network == OutboundSettings()
    assert config.network.trust_store == "system"
    assert config.network.ca_bundle is None
    assert config.network.proxy is None


def test_example_yaml_lists_the_network_group_with_its_defaults(tmp_path: Path) -> None:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert payload["network"] == {"trust_store": "system", "ca_bundle": None, "proxy": None}
    payload["storage"]["root"] = str(tmp_path / "data")
    assert HabitusConfig.from_mapping(payload).network == OutboundSettings()


def test_all_three_fields_load(tmp_path: Path) -> None:
    config = HabitusConfig.from_mapping(
        _payload(
            tmp_path,
            {"trust_store": "bundled", "ca_bundle": str(TEST_CA), "proxy": "http://proxy.corp.example:3128"},
        )
    )
    assert config.network == OutboundSettings(
        trust_store="bundled", ca_bundle=TEST_CA, proxy="http://proxy.corp.example:3128"
    )


def test_ca_bundle_expands_the_home_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "corp-root.pem").write_bytes(TEST_CA.read_bytes())
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    config = HabitusConfig.from_mapping(_payload(tmp_path, {"ca_bundle": "~/corp-root.pem"}))
    assert config.network.ca_bundle == home / "corp-root.pem"


def test_unknown_network_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"unknown config field 'config\.network\.proxy_url'"):
        HabitusConfig.from_mapping(_payload(tmp_path, {"proxy_url": "http://proxy.corp.example:3128"}))


def test_network_must_be_an_object(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"config\.network"):
        HabitusConfig.from_mapping(_payload(tmp_path, ["proxy"]))


@pytest.mark.parametrize("value", ["certifi", "System", 1, None])
def test_trust_store_other_values_are_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises(ConfigError, match=r"config\.network\.trust_store") as raised:
        HabitusConfig.from_mapping(_payload(tmp_path, {"trust_store": value}))
    assert "system" in str(raised.value) and "bundled" in str(raised.value)


def test_ca_bundle_that_does_not_exist_names_field_and_path(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "corp-root.pem"
    with pytest.raises(ConfigError, match=r"config\.network\.ca_bundle") as raised:
        HabitusConfig.from_mapping(_payload(tmp_path, {"ca_bundle": str(missing)}))
    assert str(missing) in str(raised.value)
    assert "does not exist" in str(raised.value)


def test_ca_bundle_that_is_a_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"config\.network\.ca_bundle") as raised:
        HabitusConfig.from_mapping(_payload(tmp_path, {"ca_bundle": str(tmp_path)}))
    assert "regular file" in str(raised.value)
    assert str(tmp_path) in str(raised.value)


def test_ca_bundle_that_is_not_pem_is_rejected(tmp_path: Path) -> None:
    garbage = tmp_path / "corp-root.pem"
    garbage.write_text("-----BEGIN NOTHING-----\nabc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"config\.network\.ca_bundle") as raised:
        HabitusConfig.from_mapping(_payload(tmp_path, {"ca_bundle": str(garbage)}))
    assert "PEM" in str(raised.value)
    assert str(garbage) in str(raised.value)


@pytest.mark.parametrize("value", ["", 3, ["a.pem"]])
def test_ca_bundle_must_be_a_non_empty_path(tmp_path: Path, value: object) -> None:
    with pytest.raises(ConfigError, match=r"config\.network\.ca_bundle"):
        HabitusConfig.from_mapping(_payload(tmp_path, {"ca_bundle": value}))


@pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="POSIX 上 root 读得到一切")
def test_unreadable_ca_bundle_is_rejected(tmp_path: Path) -> None:
    locked = tmp_path / "corp-root.pem"
    locked.write_bytes(TEST_CA.read_bytes())
    deny_all_access(locked)
    try:
        with pytest.raises(ConfigError, match=r"config\.network\.ca_bundle") as raised:
            HabitusConfig.from_mapping(_payload(tmp_path, {"ca_bundle": str(locked)}))
        assert str(locked) in str(raised.value)
    finally:
        restrict_file_to_owner(locked)


@pytest.mark.parametrize(
    ("proxy", "reason"),
    [
        ("http://alice:secret@proxy.corp.example:3128", "HTTPS_PROXY"),
        ("http://proxy.corp.example:3128/corp", "path"),
        ("socks5://proxy.corp.example:1080", "http:// or https://"),
        ("http://:3128", "host"),
        ("http://[::1:3128", "not a valid URL"),
        (3128, "text"),
    ],
)
def test_invalid_proxy_names_field_and_reason(tmp_path: Path, proxy: object, reason: str) -> None:
    with pytest.raises(ConfigError, match=r"config\.network\.proxy") as raised:
        HabitusConfig.from_mapping(_payload(tmp_path, {"proxy": proxy}))
    assert reason in str(raised.value)
    assert "secret" not in str(raised.value)


def test_null_fields_mean_the_defaults(tmp_path: Path) -> None:
    config = HabitusConfig.from_mapping(_payload(tmp_path, {"ca_bundle": None, "proxy": None}))
    assert config.network == OutboundSettings()
