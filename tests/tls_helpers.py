"""W18 测试用的本机 HTTPS 模型服务与 CONNECT 代理（不连外网）。

- ``HTTPSModelServer``：用 ``tests/fixtures/tls`` 里测试 CA 签发的证书，在 ``127.0.0.1`` 上模拟 OpenAI 兼容的
  chat（``/chat/completions``、``/models``）、OpenAI 兼容 embedding（``/embeddings``）与 rerank（``/reranks``）。
  ``connections`` 数的是服务端 accept 到的 TCP 连接（握手之前就计数），用来证明证书不被信任时只发了一次请求。
- ``ConnectProxy``：只认 ``CONNECT``。``tunnel`` 模式把 ``model.habitus.test:443`` 接到上面的本机服务（客户端不自己
  解析假主机名，交给代理），``auth`` 模式对每条 CONNECT 回 ``407`` 并带 ``Proxy-Authenticate: NTLM``。
  ``connects`` 记下收到的每一条 CONNECT 目标。
"""

from __future__ import annotations

import json
import socket
import socketserver
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType

TLS_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tls"
TEST_CA = TLS_FIXTURES / "ca.pem"
TEST_SERVER_CERT = TLS_FIXTURES / "server.pem"
TEST_SERVER_KEY = TLS_FIXTURES / "server.key"
FAKE_MODEL_HOST = "model.habitus.test"
EMBEDDING_DIMENSION = 2


class _ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _CountingTLSServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - 覆盖标准库签名
        return

    def do_GET(self) -> None:
        self.server.requests.append(("GET", self.path))
        if self.path.endswith("/models"):
            self._json({"object": "list", "data": [{"id": "test-model"}]})
        else:
            self._json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.server.requests.append(("POST", self.path))
        if self.path.endswith("/chat/completions"):
            self._json({"model": "test-model", "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}]})
        elif self.path.endswith("/embeddings"):
            self._json(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.6, 0.8][:EMBEDDING_DIMENSION]}],
                }
            )
        elif self.path.endswith("/reranks"):
            self._json({"results": [{"index": 0, "relevance_score": 0.5}]})
        else:
            self._json({"error": {"message": "not found"}}, status=404)

    def _json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _CountingTLSServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ModelHandler)
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(TEST_SERVER_CERT, TEST_SERVER_KEY)
        self.connections = 0
        self.requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def get_request(self) -> tuple[socket.socket, object]:
        raw, address = self.socket.accept()
        with self._lock:
            self.connections += 1
        try:
            return self.context.wrap_socket(raw, server_side=True), address
        except (OSError, ssl.SSLError):
            raw.close()
            raise


class HTTPSModelServer:
    """本机 HTTPS 模型服务：``with HTTPSModelServer() as server: server.base_url``。"""

    def __init__(self) -> None:
        self._server = _CountingTLSServer()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"https://127.0.0.1:{self.port}/v1"

    @property
    def connections(self) -> int:
        return self._server.connections

    @property
    def requests(self) -> list[tuple[str, str]]:
        return list(self._server.requests)

    def __enter__(self) -> HTTPSModelServer:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()


class _ProxyHandler(socketserver.BaseRequestHandler):
    server: _ProxyServer

    def handle(self) -> None:
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            header += chunk
        request_line = header.split(b"\r\n", 1)[0].decode("latin-1")
        method, target, _version = request_line.split(" ", 2)
        self.server.connects.append(target if method == "CONNECT" else f"{method} {target}")
        if method != "CONNECT" or self.server.mode == "auth":
            self.request.sendall(
                b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                b"Proxy-Authenticate: NTLM\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            return
        upstream_address = self.server.routes.get(target)
        if upstream_address is None:
            self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        with socket.create_connection(upstream_address, timeout=10) as upstream:
            self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            _pipe(self.request, upstream)


def _pipe(client: socket.socket, upstream: socket.socket) -> None:
    """双向转发到任一侧关闭；隧道里的 TLS 字节原样经过，代理看不到明文。"""

    def forward(source: socket.socket, target: socket.socket) -> None:
        try:
            while data := source.recv(65536):
                target.sendall(data)
        except OSError:
            pass
        finally:
            try:
                target.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    backward = threading.Thread(target=forward, args=(upstream, client), daemon=True)
    backward.start()
    forward(client, upstream)
    backward.join(timeout=10)


class _ProxyServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, mode: str, routes: dict[str, tuple[str, int]]) -> None:
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.mode = mode
        self.routes = routes
        self.connects: list[str] = []


class ConnectProxy:
    """``tunnel``：把 ``host:port`` 的 CONNECT 接到本机地址；``auth``：一律回 407（NTLM）。"""

    def __init__(self, *, mode: str = "tunnel", routes: dict[str, tuple[str, int]] | None = None) -> None:
        if mode not in {"tunnel", "auth"}:
            raise ValueError("mode must be tunnel or auth")
        self._server = _ProxyServer(mode, dict(routes or {}))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    @property
    def connects(self) -> list[str]:
        return list(self._server.connects)

    def __enter__(self) -> ConnectProxy:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()


__all__ = [
    "EMBEDDING_DIMENSION",
    "FAKE_MODEL_HOST",
    "TEST_CA",
    "TLS_FIXTURES",
    "ConnectProxy",
    "HTTPSModelServer",
]
