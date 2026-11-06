"""W18 端到端：真实 TLS 握手与真实 CONNECT 代理，全部在本机，不打外网。

模型服务用测试 CA 签发的证书（``tests/fixtures/tls``）。证书不被信任时必须一次就失败、归成 ``ModelTLSError``
（数服务端收到的连接）；配上测试 CA 就成功。代理测试的模型地址写成假主机名 ``model.habitus.test``：客户端不自己
解析，而是把 ``CONNECT model.habitus.test:443`` 交给测试代理，由它接到本机服务，这样既证明请求经过了代理，
也证明 TLS 在隧道里照样按 CA 校验。
"""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import Iterator

import httpx
import pytest

from habitus.foundation.network import HttpxClientOptions, OutboundSettings
from habitus.memory.workflow.failure import MemoryJobFailureKind, memory_job_failure_kind
from habitus.model_client import (
    MODEL_UNAVAILABLE,
    ChatModelConfig,
    EmbeddingModelConfig,
    ModelClientError,
    ModelProxyAuthError,
    ModelTLSError,
    ModelTransportError,
    ProviderConfig,
    RerankModelConfig,
)
from habitus.model_client.adapters import register_builtin_adapters
from habitus.model_client.factory import ProviderFactory
from habitus.model_client.retry import normalize_provider_error
from tests.tls_helpers import EMBEDDING_DIMENSION, FAKE_MODEL_HOST, TEST_CA, ConnectProxy, HTTPSModelServer

_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_CERTIFICATE_VARIABLES = ("SSL_CERT_FILE", "SSL_CERT_DIR")
TRUSTED = OutboundSettings(ca_bundle=TEST_CA)
FAKE_BASE_URL = f"https://{FAKE_MODEL_HOST}/v1"


@pytest.fixture(autouse=True)
def isolated_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """清掉进程里的代理环境变量；留一个无害的 NO_PROXY，让 urllib 不去读 macOS / Windows 的系统代理设置。"""

    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "habitus-isolation.invalid")
    for name in _CERTIFICATE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    yield


def _factory() -> ProviderFactory:
    factory = ProviderFactory()
    register_builtin_adapters(factory)
    return factory


def _route(adapter: str, base_url: str, *, credential_ref: str = "") -> ProviderConfig:
    return ProviderConfig(
        provider="test-provider",
        adapter=adapter,
        model="test-model",
        base_url=base_url,
        credential_ref=credential_ref,
        timeout_seconds=10.0,
        # 允许重试 2 次：证书 / 代理认证错误若被当成可重试，服务端会看到 3 条连接
        max_retries=2,
        retry_base_delay_seconds=0.01,
        retry_max_delay_seconds=0.02,
    )


def _chat(base_url: str, outbound: OutboundSettings):
    config = ChatModelConfig(_route("openai_compatible_chat", base_url))
    return _factory().create_chat_client(config, outbound=outbound)


def _embedder(base_url: str, outbound: OutboundSettings):
    config = EmbeddingModelConfig(
        _route("ark_multimodal", base_url, credential_ref="ark"),
        dimension=EMBEDDING_DIMENSION,
        input_mode="multimodal",
    )
    return _factory().create_embedder(config, credentials={"api_key": "test-key"}, outbound=outbound)


def _reranker(base_url: str, outbound: OutboundSettings):
    config = RerankModelConfig(_route("openai_compatible_rerank", base_url, credential_ref="rerank"))
    return _factory().create_reranker(config, credentials={"api_key": "test-key"}, outbound=outbound)


def _close(component: object) -> None:
    asyncio.run(component.aclose())  # type: ignore[attr-defined]


# ---- 真实 TLS：不配 CA 一次就失败，配上就通 -------------------------------------------------------------


def test_untrusted_certificate_fails_once_as_model_tls_error() -> None:
    with HTTPSModelServer() as server:
        chat = _chat(server.base_url, OutboundSettings())
        try:
            with pytest.raises(ModelTLSError) as raised:
                chat.complete("ping")
        finally:
            _close(chat)
        assert server.connections == 1
        assert server.requests == []
    error = raised.value
    assert error.code == "MODEL_TLS"
    assert error.retryable is False
    assert error.failover_eligible is True
    assert "network.ca_bundle" in str(error)


def test_configured_ca_bundle_makes_the_same_chat_call_succeed() -> None:
    with HTTPSModelServer() as server:
        chat = _chat(server.base_url, TRUSTED)
        try:
            response = chat.complete("ping")
        finally:
            _close(chat)
        assert response.content == "pong"
        assert server.requests == [("POST", "/v1/chat/completions")]


def test_async_chat_and_health_probe_use_the_same_trust_settings() -> None:
    async def scenario(server: HTTPSModelServer) -> tuple[str, dict[str, object], dict[str, object]]:
        trusted = _chat(server.base_url, TRUSTED)
        untrusted = _chat(server.base_url, OutboundSettings())
        try:
            response = await trusted.complete_async("ping")
            healthy = await trusted.health_check_async()
            before = server.connections
            failing = await untrusted.health_check_async()
            assert server.connections == before + 1
        finally:
            await trusted.aclose()
            await untrusted.aclose()
        return str(response.content), healthy, failing

    with HTTPSModelServer() as server:
        content, healthy, failing = asyncio.run(scenario(server))
    assert content == "pong"
    assert healthy["ok"] is True
    assert failing["ok"] is False
    assert failing["error_code"] == "MODEL_TLS"


def test_hostname_mismatch_is_a_tls_error_too() -> None:
    # 证书的 SAN 只有 model.habitus.test 与 127.0.0.1，没有 localhost
    with HTTPSModelServer() as server:
        chat = _chat(f"https://localhost:{server.port}/v1", TRUSTED)
        try:
            with pytest.raises(ModelTLSError):
                chat.complete("ping")
        finally:
            _close(chat)
        assert server.connections == 1


def test_embedding_adapter_trusts_the_configured_ca_and_fails_once_without_it() -> None:
    async def scenario(server: HTTPSModelServer) -> None:
        trusted = _embedder(server.base_url, TRUSTED)
        untrusted = _embedder(server.base_url, OutboundSettings())
        try:
            vector = await trusted.embed_query("ping")
            assert len(vector.values) == EMBEDDING_DIMENSION
            before = server.connections
            with pytest.raises(ModelTLSError):
                await untrusted.embed_query("ping")
            assert server.connections == before + 1
        finally:
            await trusted.aclose()
            await untrusted.aclose()

    with HTTPSModelServer() as server:
        asyncio.run(scenario(server))
        assert ("POST", "/v1/embeddings/multimodal") in server.requests


def test_rerank_adapter_trusts_the_configured_ca_and_fails_once_without_it() -> None:
    async def scenario(server: HTTPSModelServer) -> None:
        trusted = _reranker(server.base_url, TRUSTED)
        untrusted = _reranker(server.base_url, OutboundSettings())
        try:
            assert await trusted.rerank("ping", ["pong"]) == (0.5,)
            before = server.connections
            with pytest.raises(ModelTLSError):
                await untrusted.rerank("ping", ["pong"])
            assert server.connections == before + 1
        finally:
            await trusted.aclose()
            await untrusted.aclose()

    with HTTPSModelServer() as server:
        asyncio.run(scenario(server))
        assert ("POST", "/v1/reranks") in server.requests


def test_factory_default_outbound_is_the_system_trust_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # 记下适配器实际拿去建 httpx 客户端的 verify。不按报错文字判：Linux 上 truststore 走 OpenSSL，
    # 报错与 certifi 一样是 "unable to get local issuer certificate"
    import truststore

    used: list[ssl.SSLContext] = []
    original = OutboundSettings.httpx_options

    def recording(self: OutboundSettings, target_url: str) -> HttpxClientOptions:
        options = original(self, target_url)
        used.append(options.verify)
        return options

    monkeypatch.setattr(OutboundSettings, "httpx_options", recording)
    with HTTPSModelServer() as server:
        config = ChatModelConfig(_route("openai_compatible_chat", server.base_url))
        chat = _factory().create_chat_client(config)
        try:
            with pytest.raises(ModelTLSError):
                chat.complete("ping")
        finally:
            _close(chat)
    assert used and all(isinstance(context, truststore.SSLContext) for context in used)


@pytest.mark.parametrize("trust_store", ["system", "bundled"])
def test_ssl_cert_file_environment_variable_still_adds_trust(monkeypatch: pytest.MonkeyPatch, trust_store: str) -> None:
    # W18 之前 httpx 默认 verify=True 会读 SSL_CERT_FILE；靠它让公司根证书生效的做法升级后不能失效
    monkeypatch.setenv("SSL_CERT_FILE", str(TEST_CA))
    with HTTPSModelServer() as server:
        config = ChatModelConfig(_route("openai_compatible_chat", server.base_url))
        chat = _factory().create_chat_client(config, outbound=OutboundSettings(trust_store=trust_store))
        try:
            assert chat.complete("ping").content == "pong"
        finally:
            _close(chat)


# ---- 代理：经过代理、TLS 在隧道里照样校验；配置覆盖环境变量；回环直连 --------------------------------


def _tunnel(server: HTTPSModelServer) -> ConnectProxy:
    return ConnectProxy(routes={f"{FAKE_MODEL_HOST}:443": ("127.0.0.1", server.port)})


def test_configured_proxy_carries_the_request_and_tls_is_still_verified_in_the_tunnel() -> None:
    with HTTPSModelServer() as server, _tunnel(server) as proxy:
        chat = _chat(FAKE_BASE_URL, OutboundSettings(ca_bundle=TEST_CA, proxy=proxy.url))
        untrusted = _chat(FAKE_BASE_URL, OutboundSettings(proxy=proxy.url))
        try:
            assert chat.complete("ping").content == "pong"
            with pytest.raises(ModelTLSError):
                untrusted.complete("ping")
        finally:
            _close(chat)
            _close(untrusted)
        assert proxy.connects == [f"{FAKE_MODEL_HOST}:443", f"{FAKE_MODEL_HOST}:443"]
        assert server.requests == [("POST", "/v1/chat/completions")]


def test_configured_proxy_overrides_the_https_proxy_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    with HTTPSModelServer() as server, _tunnel(server) as configured, _tunnel(server) as environment:
        monkeypatch.setenv("HTTPS_PROXY", environment.url)
        chat = _chat(FAKE_BASE_URL, OutboundSettings(ca_bundle=TEST_CA, proxy=configured.url))
        try:
            assert chat.complete("ping").content == "pong"
        finally:
            _close(chat)
        assert configured.connects == [f"{FAKE_MODEL_HOST}:443"]
        assert environment.connects == []


def test_without_configured_proxy_the_environment_variable_still_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    with HTTPSModelServer() as server, _tunnel(server) as environment:
        monkeypatch.setenv("HTTPS_PROXY", environment.url)
        chat = _chat(FAKE_BASE_URL, TRUSTED)
        try:
            assert chat.complete("ping").content == "pong"
        finally:
            _close(chat)
        assert environment.connects == [f"{FAKE_MODEL_HOST}:443"]


@pytest.mark.parametrize("adapter", ["chat", "embedding", "rerank"])
def test_loopback_model_address_never_goes_through_the_configured_proxy(adapter: str) -> None:
    with HTTPSModelServer() as server, ConnectProxy() as proxy:
        outbound = OutboundSettings(ca_bundle=TEST_CA, proxy=proxy.url)
        if adapter == "chat":
            chat = _chat(server.base_url, outbound)
            try:
                assert chat.complete("ping").content == "pong"
            finally:
                _close(chat)
        elif adapter == "embedding":
            embedder = _embedder(server.base_url, outbound)

            async def embed() -> None:
                try:
                    await embedder.embed_query("ping")
                finally:
                    await embedder.aclose()

            asyncio.run(embed())
        else:
            reranker = _reranker(server.base_url, outbound)

            async def rerank() -> None:
                try:
                    await reranker.rerank("ping", ["pong"])
                finally:
                    await reranker.aclose()

            asyncio.run(rerank())
        assert proxy.connects == []
        assert len(server.requests) == 1


@pytest.mark.parametrize("variable", ["HTTPS_PROXY", "ALL_PROXY", "https_proxy"])
@pytest.mark.parametrize("adapter", ["chat", "embedding", "rerank"])
def test_loopback_model_address_never_goes_through_an_environment_proxy(
    monkeypatch: pytest.MonkeyPatch,
    adapter: str,
    variable: str,
) -> None:
    """公司电脑统一设了代理环境变量、没设 NO_PROXY，本机跑的模型服务照样直连（用户 2026-11-06 定）。"""

    with HTTPSModelServer() as server, ConnectProxy() as proxy:
        monkeypatch.setenv(variable, proxy.url)
        if adapter == "chat":
            chat = _chat(server.base_url, TRUSTED)
            try:
                assert chat.complete("ping").content == "pong"
            finally:
                _close(chat)
        elif adapter == "embedding":
            embedder = _embedder(server.base_url, TRUSTED)

            async def embed() -> None:
                try:
                    await embedder.embed_query("ping")
                finally:
                    await embedder.aclose()

            asyncio.run(embed())
        else:
            reranker = _reranker(server.base_url, TRUSTED)

            async def rerank() -> None:
                try:
                    await reranker.rerank("ping", ["pong"])
                finally:
                    await reranker.aclose()

            asyncio.run(rerank())
        assert proxy.connects == []
        assert len(server.requests) == 1


# ---- 代理要求认证（407）：一次就失败，提示 cntlm / px ---------------------------------------------------


def test_proxy_authentication_required_fails_once_with_the_forwarding_proxy_hint() -> None:
    with ConnectProxy(mode="auth") as proxy:
        chat = _chat(FAKE_BASE_URL, OutboundSettings(proxy=proxy.url))
        try:
            with pytest.raises(ModelProxyAuthError) as raised:
                chat.complete("ping")
        finally:
            _close(chat)
        assert proxy.connects == [f"{FAKE_MODEL_HOST}:443"]
    error = raised.value
    assert error.code == "MODEL_PROXY_AUTH"
    assert error.retryable is False
    assert error.failover_eligible is True
    message = str(error)
    for expected in ("407", "HTTPS_PROXY", "NTLM", "Kerberos", "PAC", "cntlm", "px", "network.proxy"):
        assert expected in message


def test_proxy_authentication_required_from_the_environment_proxy_is_classified_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ConnectProxy(mode="auth") as proxy:
        monkeypatch.setenv("HTTPS_PROXY", proxy.url)
        embedder = _embedder(FAKE_BASE_URL, OutboundSettings())

        async def embed() -> None:
            try:
                await embedder.embed_query("ping")
            finally:
                await embedder.aclose()

        with pytest.raises(ModelProxyAuthError):
            asyncio.run(embed())
        assert len(proxy.connects) == 1


# ---- 错误分类 -----------------------------------------------------------------------------------------


def _wrapped(outer: Exception, inner: BaseException) -> Exception:
    try:
        raise outer from inner
    except Exception as caught:
        return caught


def test_certificate_verification_errors_are_tls_errors_even_when_wrapped() -> None:
    bare = ssl.SSLCertVerificationError(1, "certificate verify failed: unable to get local issuer certificate")
    mismatch = ssl.SSLCertVerificationError(1, "certificate verify failed: Hostname mismatch")
    wrapped = _wrapped(httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]"), bare)
    for error in (bare, mismatch, wrapped):
        failure = normalize_provider_error(error)
        assert isinstance(failure, ModelTLSError)
        assert failure.retryable is False


def test_certificate_errors_win_over_message_keywords() -> None:
    # 证书报错里可能带主机名；主机名碰上内容安全 / 限流这类关键词时仍是证书问题
    error = httpx.ConnectError("certificate is not valid for 'moderation-gw.corp.example'")
    error.__cause__ = ssl.SSLCertVerificationError("certificate is not valid for 'moderation-gw.corp.example'")
    assert isinstance(normalize_provider_error(error), ModelTLSError)
    proxy = httpx.ProxyError("407 Proxy Authentication Required (rate limit gateway)")
    assert isinstance(normalize_provider_error(proxy), ModelProxyAuthError)


def test_plain_connection_errors_stay_retryable_transport_errors() -> None:
    for error in (ConnectionError("connection refused"), ssl.SSLError(1, "handshake failure"), OSError("reset")):
        failure = normalize_provider_error(error)
        assert type(failure) is ModelTransportError
        assert failure.retryable is True


def test_only_a_407_proxy_rejection_is_a_proxy_auth_error() -> None:
    rejected = _wrapped(httpx.ProxyError("407 Proxy Authentication Required"), RuntimeError("tunnel"))
    assert isinstance(normalize_provider_error(rejected), ModelProxyAuthError)
    assert not isinstance(normalize_provider_error(httpx.ProxyError("502 Bad Gateway")), ModelProxyAuthError)

    class StatusError(RuntimeError):
        status_code = 407

    assert isinstance(normalize_provider_error(StatusError("proxy says no")), ModelProxyAuthError)


@pytest.mark.parametrize("error_type", [ModelTLSError, ModelProxyAuthError])
def test_tls_and_proxy_auth_failures_wait_for_the_model_instead_of_failing_the_memory_job(
    error_type: type[ModelClientError],
) -> None:
    assert error_type in MODEL_UNAVAILABLE
    assert memory_job_failure_kind(error_type("boom")) is MemoryJobFailureKind.WAITS_FOR_MODEL
    # 抽取把模型失败包进自己的异常时，按里面的根因照样等模型恢复
    wrapped = _wrapped(RuntimeError("extraction failed"), error_type("boom"))
    assert memory_job_failure_kind(wrapped) is MemoryJobFailureKind.WAITS_FOR_MODEL
