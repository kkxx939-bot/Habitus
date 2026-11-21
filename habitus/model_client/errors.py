"""模型调用失败的唯一一套类别、三条处理规则和给人看的话（裁定 108）。

借 OpenViking 的思路（错误先归成固定类别、每类处理定死；只重试等一等可能好的；连着失败就熔断、定时试探），类别按
Habitus 自己的协议定。归类只在 ``model_client.failures.classify_failure`` 一处做；这里只放类别、规则与异常类型。

三条规则：

- **等一等可能好**（服务故障、网络故障、限流、认不出的）：客户端有界重试，仍失败交回调用方；连着失败到阈值熔断。
- **要人处理**（认证、权限、额度、模型或地址不存在、不支持、网络设置）：不重试，熔断直接打开、降到试探的节奏，
  健康检查显示原因；服务那边恢复（额度恢复、权限开通）下一次试探成功就自己接上；改了配置文件要重启服务（配置只在启动时
  读一次），给人看的话里写明。
- **这一条输入有问题**（请求不合法、太长、内容安全）：同一份输入不重发，不计入熔断；调用方按"这一条做不成"处理。
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar


class ModelErrorRule(str, Enum):
    """一类失败怎么处理（三条，定死）。"""

    WAIT = "wait"  # 等一等可能好
    OPERATOR = "operator"  # 要人处理
    INPUT = "input"  # 这一条输入有问题


class ModelErrorKind(str, Enum):
    """模型调用失败的类别（一次定全，不再加）。"""

    SERVICE = "service_error"
    NETWORK = "network_error"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    QUOTA = "quota_exhausted"
    MODEL_NOT_FOUND = "model_not_found"
    UNSUPPORTED = "unsupported"
    NETWORK_SETUP = "network_setup"
    INVALID_REQUEST = "invalid_request"
    INPUT_TOO_LARGE = "input_too_large"
    CONTENT_SAFETY = "content_safety"

    @property
    def rule(self) -> ModelErrorRule:
        return RULES[self]

    @property
    def brief(self) -> str:
        """给人看的一句话。"""

        return BRIEFS[self]


RULES: dict[ModelErrorKind, ModelErrorRule] = {
    ModelErrorKind.SERVICE: ModelErrorRule.WAIT,
    ModelErrorKind.NETWORK: ModelErrorRule.WAIT,
    ModelErrorKind.RATE_LIMIT: ModelErrorRule.WAIT,
    ModelErrorKind.UNKNOWN: ModelErrorRule.WAIT,
    ModelErrorKind.AUTHENTICATION: ModelErrorRule.OPERATOR,
    ModelErrorKind.PERMISSION: ModelErrorRule.OPERATOR,
    ModelErrorKind.QUOTA: ModelErrorRule.OPERATOR,
    ModelErrorKind.MODEL_NOT_FOUND: ModelErrorRule.OPERATOR,
    ModelErrorKind.UNSUPPORTED: ModelErrorRule.OPERATOR,
    ModelErrorKind.NETWORK_SETUP: ModelErrorRule.OPERATOR,
    ModelErrorKind.INVALID_REQUEST: ModelErrorRule.INPUT,
    ModelErrorKind.INPUT_TOO_LARGE: ModelErrorRule.INPUT,
    ModelErrorKind.CONTENT_SAFETY: ModelErrorRule.INPUT,
}

BRIEFS: dict[ModelErrorKind, str] = {
    ModelErrorKind.SERVICE: "模型服务那边暂时出了故障，过一会儿会自动再试。",
    ModelErrorKind.NETWORK: "暂时连不上模型服务（网络超时或断开），过一会儿会自动再试。",
    ModelErrorKind.RATE_LIMIT: "模型服务说请求太频繁，过一会儿会自动再试。",
    ModelErrorKind.UNKNOWN: "调用模型时出了一个没认出来的错误，过一会儿会自动再试。",
    ModelErrorKind.AUTHENTICATION: "模型服务不认这个密钥，请检查配置文件里填的密钥，改好后重启服务。",
    ModelErrorKind.PERMISSION: "这个密钥没有使用该模型的权限，请检查账号权限；要换密钥的话，改好配置后重启服务。",
    ModelErrorKind.QUOTA: "模型服务的额度或余额用完了，请充值或等额度恢复。",
    ModelErrorKind.MODEL_NOT_FOUND: "找不到配置的模型或服务地址，请检查配置里的模型名和地址，改好后重启服务。",
    ModelErrorKind.UNSUPPORTED: "模型服务不支持现在这样调用，请检查模型配置和所选的模型是否对得上，改好后重启服务。",
    ModelErrorKind.NETWORK_SETUP: "网络设置挡住了连接（证书不被信任，或代理要求登录），请检查网络配置，改好后重启服务。",
    ModelErrorKind.INVALID_REQUEST: "模型服务拒收了这一条内容，这一条先跳过。",
    ModelErrorKind.INPUT_TOO_LARGE: "这一条内容太长，超出了模型能处理的长度，这一条先跳过。",
    ModelErrorKind.CONTENT_SAFETY: "这一条内容被模型服务的内容安全拦下了，这一条先跳过。",
}


class ModelClientError(RuntimeError):
    """模型客户端的异常；模型调用失败的那几类带 ``kind``（结构化输出不合格这类不是调用失败，``kind`` 为空）。

    ``status_code`` / ``provider_code`` 是服务返回的 HTTP 状态码与协议里的错误代码（没有就是空），只给观测与排查用。"""

    kind: ClassVar[ModelErrorKind | None] = None
    code = "MODEL_CLIENT_ERROR"

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        status_code: int | None = None,
        provider_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.status_code = status_code
        self.provider_code = provider_code

    @property
    def rule(self) -> ModelErrorRule | None:
        kind = self.kind
        return None if kind is None else kind.rule


class ModelServiceError(ModelClientError):
    kind = ModelErrorKind.SERVICE
    code = "MODEL_SERVICE"


class ModelResponseError(ModelServiceError):
    """服务答回来的东西不合协议（不是 JSON、缺字段、流断了）：算服务故障。"""

    code = "MODEL_RESPONSE"


class ModelTransportError(ModelClientError):
    """连不上、超时。"""

    kind = ModelErrorKind.NETWORK
    code = "MODEL_TRANSPORT"


class ModelRateLimitError(ModelClientError):
    kind = ModelErrorKind.RATE_LIMIT
    code = "MODEL_RATE_LIMIT"


class ModelUnknownError(ModelClientError):
    kind = ModelErrorKind.UNKNOWN
    code = "MODEL_UNKNOWN"


class ModelAuthenticationError(ModelClientError):
    kind = ModelErrorKind.AUTHENTICATION
    code = "MODEL_AUTHENTICATION"


class ModelPermissionError(ModelClientError):
    kind = ModelErrorKind.PERMISSION
    code = "MODEL_PERMISSION"


class ModelQuotaError(ModelClientError):
    kind = ModelErrorKind.QUOTA
    code = "MODEL_QUOTA"


class ModelNotFoundError(ModelClientError):
    """配置的模型或服务地址不存在（404 或协议的"模型不存在"代码）。"""

    kind = ModelErrorKind.MODEL_NOT_FOUND
    code = "MODEL_NOT_FOUND"


class ModelConfigurationError(ModelClientError):
    """模型配置与这次调用对不上（本地发现的，或服务回 405/501/"不支持的参数"）：要人改配置。"""

    kind = ModelErrorKind.UNSUPPORTED
    code = "MODEL_CONFIGURATION"


class ModelDependencyError(ModelConfigurationError):
    code = "MODEL_DEPENDENCY"


#: 证书不被信任时告诉用户配哪个字段（报错与 doctor 共用这一句）。
TLS_UNTRUSTED_HINT = (
    "the model server's TLS certificate is not trusted; if Habitus runs behind a corporate proxy that inspects TLS, "
    "set network.ca_bundle to the corporate root certificate (PEM file)"
)
#: 代理对 CONNECT 回 407 时的去处（用户 11-06 定：NTLM / Kerberos / PAC 不实现，只提示本机转发代理）。
PROXY_AUTH_HINT = (
    "the proxy requires authentication (HTTP 407). Basic authentication (user name and password) can go into the "
    "HTTPS_PROXY environment variable; Windows domain sign-in (NTLM / Kerberos) and PAC auto-configuration are not "
    "supported directly: run a local forwarding proxy such as cntlm or px on this machine and point network.proxy "
    "at it (for example http://127.0.0.1:3128)"
)


class ModelTLSError(ModelClientError):
    """模型服务的证书不被信任（含主机名不匹配）：改了证书配置才能恢复。"""

    kind = ModelErrorKind.NETWORK_SETUP
    code = "MODEL_TLS"


class ModelProxyAuthError(ModelClientError):
    """代理要求认证（407）：改了代理配置才能恢复。"""

    kind = ModelErrorKind.NETWORK_SETUP
    code = "MODEL_PROXY_AUTH"


class ModelInvalidRequestError(ModelClientError):
    kind = ModelErrorKind.INVALID_REQUEST
    code = "MODEL_INVALID_REQUEST"


class ModelInputTooLargeError(ModelClientError):
    kind = ModelErrorKind.INPUT_TOO_LARGE
    code = "MODEL_INPUT_TOO_LARGE"


class ModelContentSafetyError(ModelClientError):
    kind = ModelErrorKind.CONTENT_SAFETY
    code = "MODEL_CONTENT_SAFETY"


class ModelCircuitOpenError(ModelClientError):
    """熔断开着、还没到试探的时候：这次不发请求。``kind`` 是打开熔断的那次失败的类别，``retry_after_seconds`` 是离下一次
    试探还有多久。"""

    code = "MODEL_CIRCUIT_OPEN"

    def __init__(self, message: str, *, opened_by: ModelErrorKind, retry_after_seconds: float) -> None:
        super().__init__(message, retry_after_seconds=retry_after_seconds)
        self.opened_by = opened_by

    @property
    def rule(self) -> ModelErrorRule:
        return self.opened_by.rule


class ModelStructuredOutputError(ModelClientError):
    """模型答了但结构化校验重问用尽仍不合格：不是调用失败，不归上面的类别。"""

    code = "MODEL_STRUCTURED_OUTPUT"


#: 模型此刻用不了（"等一等可能好"与"要人处理"两条规则，含熔断开着）：换一份输入也一样。各处"等模型恢复"的分类只从这一组取。
MODEL_UNAVAILABLE: tuple[type[ModelClientError], ...] = (
    ModelServiceError,
    ModelTransportError,
    ModelRateLimitError,
    ModelUnknownError,
    ModelAuthenticationError,
    ModelPermissionError,
    ModelQuotaError,
    ModelNotFoundError,
    ModelConfigurationError,
    ModelTLSError,
    ModelProxyAuthError,
    ModelCircuitOpenError,
)
#: 这一条输入有问题（请求不合法、太长、内容安全）：同一份输入不重发。
MODEL_DETERMINISTIC: tuple[type[ModelClientError], ...] = (
    ModelInvalidRequestError,
    ModelInputTooLargeError,
    ModelContentSafetyError,
)
#: ``MODEL_DETERMINISTIC`` 里"这一条触发了内容安全"的那一种：行为侧单列（裁定 94 错误分类）。
MODEL_REFUSED: tuple[type[ModelClientError], ...] = (ModelContentSafetyError,)


def model_failure(error: BaseException) -> ModelClientError | None:
    """全流程统一的判定入口：顺着 ``__cause__`` / ``__context__`` 找到第一个归过类的模型调用失败（带规则的那种）。
    各调用方（会话 lane、行为 lane、记忆任务、模型队列……）只按它的 ``rule`` 分派，不各自认异常类型。找不到是 None：
    那不是模型调用失败（结构化输出不合格、我们自己的代码出错等），由调用方原来的路处理。"""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelClientError) and current.rule is not None:
            return current
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return None


def model_failure_rule(error: BaseException) -> ModelErrorRule | None:
    """``model_failure`` 的规则；不是模型调用失败返回 None。"""

    found = model_failure(error)
    return None if found is None else found.rule


class ModelUnreachable(RuntimeError):
    """这一处调模型的结论是"模型此刻用不了"（``MODEL_UNAVAILABLE`` 那一组）：由各触点（``model_client.asking.ask``）抛出，
    换代与预测层据它走退路。不是 ``ModelClientError``：它不是某一次调用的失败分类。"""


__all__ = [
    "BRIEFS",
    "MODEL_DETERMINISTIC",
    "MODEL_REFUSED",
    "MODEL_UNAVAILABLE",
    "PROXY_AUTH_HINT",
    "RULES",
    "TLS_UNTRUSTED_HINT",
    "ModelAuthenticationError",
    "ModelCircuitOpenError",
    "ModelClientError",
    "ModelConfigurationError",
    "ModelContentSafetyError",
    "ModelDependencyError",
    "ModelErrorKind",
    "ModelErrorRule",
    "ModelInputTooLargeError",
    "ModelInvalidRequestError",
    "ModelNotFoundError",
    "ModelPermissionError",
    "ModelProxyAuthError",
    "ModelQuotaError",
    "ModelRateLimitError",
    "ModelResponseError",
    "ModelServiceError",
    "ModelStructuredOutputError",
    "ModelTLSError",
    "ModelTransportError",
    "ModelUnknownError",
    "ModelUnreachable",
    "model_failure",
    "model_failure_rule",
]
