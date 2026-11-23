"""doctor 的出站网络诊断（W18）：常规 ``network`` 检查与 deep 探测里证书 / 代理认证失败的人话提示。"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.foundation.network import OutboundSettings
from habitus.integrations.local_service import DoctorStatus, run_doctor
from habitus.integrations.local_service import doctor as doctor_module
from habitus.model_client import (
    ChatModelConfig,
    ProviderConfig,
)
from habitus.model_client.adapters import register_builtin_adapters
from habitus.model_client.factory import ProviderFactory
from tests.tls_helpers import FAKE_MODEL_HOST, TEST_CA, ConnectProxy, HTTPSModelServer

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


@pytest.fixture(autouse=True)
def isolated_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "habitus-isolation.invalid")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    yield


def _config(tmp_path: Path, network: dict[str, object] | None = None) -> HabitusConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["chat"]["route"]["credential_ref"] = ""
    payload["network"] = network or {}
    return HabitusConfig.from_mapping(payload)


def _network_check(config: HabitusConfig) -> doctor_module.DoctorCheck:
    report = run_doctor(config, check_port=False)
    return next(check for check in report.checks if check.name == "network")


# ---- 常规 network 检查（不连网） ---------------------------------------------------------------------


def test_network_check_passes_with_the_effective_defaults(tmp_path: Path) -> None:
    check = _network_check(_config(tmp_path))
    assert check.status is DoctorStatus.PASS
    assert "trust_store=system" in check.detail
    assert "ca_bundle=none" in check.detail
    assert "proxy=none" in check.detail


def test_network_check_shows_the_ssl_cert_file_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", str(TEST_CA))
    check = _network_check(_config(tmp_path))
    assert check.status is DoctorStatus.PASS
    assert f"SSL_CERT_FILE={TEST_CA}" in check.detail
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    failed = _network_check(_config(tmp_path))
    assert failed.status is DoctorStatus.FAIL
    assert "SSL_CERT_FILE" in failed.detail


def test_network_check_shows_configured_ca_bundle_and_proxy(tmp_path: Path) -> None:
    check = _network_check(
        _config(
            tmp_path,
            {"trust_store": "bundled", "ca_bundle": str(TEST_CA), "proxy": "http://proxy.corp.example:3128"},
        )
    )
    assert check.status is DoctorStatus.PASS
    assert "trust_store=bundled" in check.detail
    assert f"ca_bundle={TEST_CA}" in check.detail
    assert "proxy=http://proxy.corp.example:3128 (network.proxy;" in check.detail


def test_network_check_shows_the_environment_proxy_without_its_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://alice:s3cret-pass@proxy.corp.example:3128")
    check = _network_check(_config(tmp_path))
    assert check.status is DoctorStatus.PASS
    assert "proxy=http://proxy.corp.example:3128 (environment HTTPS_PROXY;" in check.detail
    assert "loopback model addresses connect directly" in check.detail
    assert "s3cret-pass" not in check.detail
    assert "alice" not in check.detail


def test_network_check_fails_when_the_ca_bundle_vanished_after_loading(tmp_path: Path) -> None:
    ca = tmp_path / "corp-root.pem"
    ca.write_bytes(TEST_CA.read_bytes())
    config = _config(tmp_path, {"ca_bundle": str(ca)})
    ca.unlink()
    check = _network_check(config)
    assert check.status is DoctorStatus.FAIL
    assert str(ca) in check.detail
    assert "network.ca_bundle" in check.detail


def test_network_check_fails_when_truststore_cannot_be_imported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "truststore", None)
    check = _network_check(_config(tmp_path))
    assert check.status is DoctorStatus.FAIL
    assert "truststore" in check.detail
    assert "trust_store: bundled" in check.detail


def test_network_failure_blocks_startup_preflight(tmp_path: Path) -> None:
    ca = tmp_path / "corp-root.pem"
    ca.write_bytes(TEST_CA.read_bytes())
    config = _config(tmp_path, {"ca_bundle": str(ca)})
    ca.unlink()
    with pytest.raises(doctor_module.StartupPreflightError, match="network"):
        doctor_module.run_startup_preflight(config)


# ---- deep 探测：证书 / 代理认证失败给出怎么配 ------------------------------------------------------------


def _deep(
    monkeypatch: pytest.MonkeyPatch,
    config: HabitusConfig,
    *,
    chat: object,
) -> dict[str, doctor_module.DoctorCheck]:
    runtime = SimpleNamespace(
        components=SimpleNamespace(models=SimpleNamespace(chat=chat)),
        close=AsyncMock(),
    )
    monkeypatch.setattr("habitus.runtime.build_runtime", Mock(return_value=runtime))
    checks = doctor_module._deep_checks(config, timeout_seconds=10)
    return {check.name: check for check in checks}


def _chat_client(base_url: str, outbound: OutboundSettings):
    factory = ProviderFactory()
    register_builtin_adapters(factory)
    route = ProviderConfig(
        provider="test-provider",
        adapter="openai_compatible_chat",
        model="test-model",
        base_url=base_url,
        max_retries=2,
        retry_base_delay_seconds=0.01,
        retry_max_delay_seconds=0.02,
    )
    return factory.create_chat_client(ChatModelConfig(route), outbound=outbound)


def test_deep_chat_probe_against_an_untrusted_certificate_says_how_to_configure_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with HTTPSModelServer() as server:
        checks = _deep(
            monkeypatch,
            _config(tmp_path),
            chat=_chat_client(server.base_url, OutboundSettings()),
        )
        assert server.connections == 1
    detail = checks["chat_probe"].detail
    assert checks["chat_probe"].status is DoctorStatus.FAIL
    assert "MODEL_TLS" in detail
    assert "not trusted" in detail
    assert "trust_store=system" in detail
    assert "network.ca_bundle" in detail
    assert "PEM" in detail


def test_deep_chat_probe_through_a_407_proxy_suggests_a_local_forwarding_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ConnectProxy(mode="auth") as proxy:
        checks = _deep(
            monkeypatch,
            _config(tmp_path),
            chat=_chat_client(f"https://{FAKE_MODEL_HOST}/v1", OutboundSettings(proxy=proxy.url)),
        )
        assert len(proxy.connects) == 1
    detail = checks["chat_probe"].detail
    assert checks["chat_probe"].status is DoctorStatus.FAIL
    assert "MODEL_PROXY_AUTH" in detail
    for expected in ("HTTPS_PROXY", "NTLM", "Kerberos", "PAC", "cntlm", "px", "network.proxy", "127.0.0.1:3128"):
        assert expected in detail


def test_other_probe_failures_keep_their_plain_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = SimpleNamespace(health_check_async=AsyncMock(return_value={"ok": False, "error_code": "MODEL_TRANSPORT"}))
    checks = _deep(monkeypatch, _config(tmp_path), chat=chat)
    assert checks["chat_probe"].detail == "MODEL_TRANSPORT"
