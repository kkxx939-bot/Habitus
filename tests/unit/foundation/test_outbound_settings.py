"""出站设置值对象（W18）：证书库来源、额外 CA、代理 URL 规则、回环直连与 SSL context 缓存。"""

from __future__ import annotations

import ssl
from pathlib import Path

import certifi
import pytest
import truststore

from habitus.foundation.network import (
    OutboundSettings,
    OutboundSettingsError,
    build_ssl_context,
    check_ca_bundle,
    redact_proxy_url,
)
from tests.tls_helpers import TEST_CA, TLS_FIXTURES


def test_defaults_are_system_trust_store_without_extra_ca_or_proxy() -> None:
    settings = OutboundSettings()
    assert settings.trust_store == "system"
    assert settings.ca_bundle is None
    assert settings.proxy is None


def test_system_trust_store_uses_a_truststore_context() -> None:
    context = OutboundSettings().ssl_context()
    assert isinstance(context, truststore.SSLContext)
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_bundled_trust_store_uses_certifi_and_not_truststore() -> None:
    context = OutboundSettings(trust_store="bundled").ssl_context()
    assert not isinstance(context, truststore.SSLContext)
    assert isinstance(context, ssl.SSLContext)
    # certifi 那一份里的根证书都在；系统证书库不读
    certifi_roots = ssl.create_default_context(cafile=certifi.where()).get_ca_certs(binary_form=True)
    assert set(context.get_ca_certs(binary_form=True)) == set(certifi_roots)
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True


@pytest.mark.parametrize("trust_store", ["system", "bundled"])
def test_extra_ca_bundle_is_added_on_top_of_the_trust_store(trust_store: str) -> None:
    context = OutboundSettings(trust_store=trust_store, ca_bundle=TEST_CA).ssl_context()
    test_ca_der = ssl.PEM_cert_to_DER_cert(TEST_CA.read_text(encoding="ascii"))
    # truststore.SSLContext 不实现 get_ca_certs；它把额外 CA 放在内部的 OpenSSL context 里，校验时交给系统当锚点
    inner = context._ctx if isinstance(context, truststore.SSLContext) else context  # type: ignore[attr-defined]
    assert test_ca_der in inner.get_ca_certs(binary_form=True)
    if trust_store == "bundled":
        assert len(context.get_ca_certs()) > 1


def test_ssl_context_is_cached_per_setting_and_not_shared_across_settings() -> None:
    first = OutboundSettings(ca_bundle=TEST_CA)
    assert first.ssl_context() is OutboundSettings(ca_bundle=TEST_CA).ssl_context()
    assert first.ssl_context() is not OutboundSettings().ssl_context()
    assert OutboundSettings(trust_store="bundled").ssl_context() is not OutboundSettings().ssl_context()


@pytest.mark.parametrize("value", ["certifi", "", "SYSTEM", None, 1])
def test_unknown_trust_store_is_rejected(value: object) -> None:
    with pytest.raises(OutboundSettingsError, match="trust_store"):
        OutboundSettings(trust_store=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "proxy",
    [
        "http://proxy.corp.example:3128",
        "https://proxy.corp.example",
        "http://10.0.0.8:8080/",
        "http://[2001:db8::1]:3128",
        "http://127.0.0.1:3128",
    ],
)
def test_valid_proxy_urls_are_accepted(proxy: str) -> None:
    assert OutboundSettings(proxy=proxy).proxy == proxy


@pytest.mark.parametrize(
    ("proxy", "reason"),
    [
        ("http://alice:secret@proxy.corp.example:3128", "HTTPS_PROXY"),
        ("http://alice@proxy.corp.example:3128", "HTTPS_PROXY"),
        ("http://proxy.corp.example:3128/path", "path"),
        ("http://proxy.corp.example:3128?x=1", "query"),
        ("http://proxy.corp.example:3128#frag", "fragment"),
        ("socks5://proxy.corp.example:1080", "http:// or https://"),
        ("proxy.corp.example:3128", "http:// or https://"),
        ("http://", "host"),
        ("http://:3128", "host"),
        ("http://proxy.corp.example:99999", "port"),
        ("http://proxy.corp.example:abc", "port"),
        (" http://proxy.corp.example:3128", "whitespace"),
        ("", "empty"),
    ],
)
def test_invalid_proxy_urls_are_rejected_with_the_reason(proxy: str, reason: str) -> None:
    with pytest.raises(OutboundSettingsError, match="proxy") as raised:
        OutboundSettings(proxy=proxy)
    assert reason in str(raised.value)


def test_proxy_with_credentials_points_to_the_environment_variable_and_hides_the_password() -> None:
    with pytest.raises(OutboundSettingsError) as raised:
        OutboundSettings(proxy="http://alice:secret@proxy.corp.example:3128")
    assert "HTTPS_PROXY" in str(raised.value)
    assert "secret" not in str(raised.value)


def test_without_proxy_httpx_options_only_carry_verify_and_leave_environment_proxies_on() -> None:
    options = OutboundSettings().httpx_options("https://api.example.com/v1")
    assert options.verify is OutboundSettings().ssl_context()
    assert options.proxy is None
    assert options.mounts is None


def test_configured_proxy_applies_to_remote_targets() -> None:
    options = OutboundSettings(proxy="http://proxy.corp.example:3128").httpx_options("https://api.example.com/v1")
    assert options.proxy == "http://proxy.corp.example:3128"
    assert options.mounts is None


@pytest.mark.parametrize(
    ("target", "pattern"),
    [
        ("https://127.0.0.1:8443/v1", "all://127.0.0.1"),
        ("http://127.0.0.5:8000/v1", "all://127.0.0.5"),
        ("http://localhost:8000/v1", "all://localhost"),
        ("http://LocalHost:8000/v1", "all://localhost"),
        ("http://[::1]:8000/v1", "all://[::1]"),
    ],
)
def test_loopback_targets_always_bypass_the_configured_proxy(target: str, pattern: str) -> None:
    options = OutboundSettings(proxy="http://proxy.corp.example:3128").httpx_options(target)
    assert options.mounts == {pattern: None}


@pytest.mark.parametrize(
    ("target", "pattern"),
    [
        ("https://127.0.0.1:8443/v1", "all://127.0.0.1"),
        ("http://localhost:11434/v1", "all://localhost"),
        ("http://[::1]:8000/v1", "all://[::1]"),
    ],
)
def test_loopback_targets_bypass_environment_proxies_too(target: str, pattern: str) -> None:
    """没配 ``network.proxy`` 时环境变量代理照常生效，但本机模型地址同样直连：公司代理到不了用户电脑上的回环地址。"""

    options = OutboundSettings().httpx_options(target)
    assert options.proxy is None
    assert options.mounts == {pattern: None}


def test_check_ca_bundle_rejects_missing_directory_and_non_pem_files(tmp_path: Path) -> None:
    check_ca_bundle(TEST_CA)
    with pytest.raises(OutboundSettingsError, match="does not exist") as missing:
        check_ca_bundle(tmp_path / "missing.pem")
    assert str(tmp_path / "missing.pem") in str(missing.value)
    with pytest.raises(OutboundSettingsError, match="regular file"):
        check_ca_bundle(tmp_path)
    garbage = tmp_path / "garbage.pem"
    garbage.write_text("not a certificate\n", encoding="utf-8")
    with pytest.raises(OutboundSettingsError, match="PEM") as unreadable:
        check_ca_bundle(garbage)
    assert str(garbage) in str(unreadable.value)


def test_build_ssl_context_is_uncached_and_reports_a_vanished_ca_file(tmp_path: Path) -> None:
    copied = tmp_path / "ca.pem"
    copied.write_bytes(TEST_CA.read_bytes())
    assert build_ssl_context("system", copied) is not build_ssl_context("system", copied)
    copied.unlink()
    with pytest.raises(OutboundSettingsError, match="does not exist"):
        build_ssl_context("system", copied)


def test_ssl_cert_file_from_the_environment_is_trusted_and_keys_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    without = OutboundSettings(trust_store="bundled").ssl_context()
    monkeypatch.setenv("SSL_CERT_FILE", str(TEST_CA))
    with_variable = OutboundSettings(trust_store="bundled").ssl_context()
    assert with_variable is not without
    test_ca_der = ssl.PEM_cert_to_DER_cert(TEST_CA.read_text(encoding="utf-8"))
    assert test_ca_der in with_variable.get_ca_certs(binary_form=True)
    assert test_ca_der not in without.get_ca_certs(binary_form=True)


def test_unloadable_ssl_cert_file_names_the_variable(tmp_path: Path) -> None:
    missing = tmp_path / "corp.pem"
    with pytest.raises(OutboundSettingsError, match="SSL_CERT_FILE") as raised:
        build_ssl_context("bundled", None, {"SSL_CERT_FILE": str(missing), "SSL_CERT_DIR": str(tmp_path)})
    assert str(missing) in str(raised.value)


def test_redact_proxy_url_drops_user_and_password() -> None:
    assert redact_proxy_url("http://alice:secret@proxy.corp.example:3128") == "http://proxy.corp.example:3128"
    assert redact_proxy_url("http://proxy.corp.example:3128") == "http://proxy.corp.example:3128"
    assert redact_proxy_url("proxy.corp.example:3128") == "http://proxy.corp.example:3128"
    assert "secret" not in redact_proxy_url("http://alice:secret@[2001:db8::1]:3128")


def test_fixture_directory_is_present() -> None:
    assert (TLS_FIXTURES / "README.md").is_file()
