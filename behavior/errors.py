"""Behavior 语义入口、证据与声明层的稳定错误分类。"""


class BehaviorError(Exception):
    """Behavior 第一层所有可预期领域错误的基类。"""


class BehaviorOwnerError(BehaviorError, ValueError):
    """上游没有提供合法的单 Owner 绑定。"""


class BehaviorOwnerConflictError(BehaviorOwnerError):
    """当前 Store 已经永久绑定到另一个 Owner 身份摘要。"""


class SemanticIngressError(BehaviorError, ValueError):
    """语义入口协议、Adapter 或能力绑定不合法。"""


class SemanticRecordError(SemanticIngressError):
    """Owner-scoped 语义记录不满足严格结构或容量边界。"""


class SemanticRecordConflictError(SemanticRecordError):
    """相同语义记录身份对应了不同的规范内容。"""


class SemanticRecordLateError(SemanticRecordError):
    """语义记录的可信事件时间已经越过提交 Watermark。"""


class SemanticClockError(SemanticRecordError):
    """事件时间不满足注入 Clock 的保守边界。"""


class EvidenceBundleError(BehaviorError, ValueError):
    """语义证据 Bundle 无法按硬边界处理。"""


class EvidenceBundleStateError(EvidenceBundleError):
    """语义证据 Bundle 状态不允许当前操作。"""


class EvidenceManifestError(BehaviorError, ValueError):
    """EvidenceManifest 不满足不可变规范。"""


class ClaimSchemaError(BehaviorError, ValueError):
    """Claim 语义提案、Claim 或批次不满足严格 Schema。"""


class ClaimProductionError(BehaviorError):
    """Claim Normalizer 调用或输出处理失败。"""


class ClaimModelTransportError(ClaimProductionError):
    """模型传输或限流失败。"""


class ClaimModelSchemaError(ClaimProductionError):
    """模型响应或结构化输出不合法。"""


class ClaimModelInputError(ClaimProductionError):
    """模型输入超过字符或 Token 边界。"""


class ClaimModelAuthenticationError(ClaimProductionError):
    """模型认证失败。"""


class ClaimModelPermissionError(ClaimProductionError):
    """模型权限不足。"""


class ClaimModelConfigurationError(ClaimProductionError):
    """模型配置无效。"""


class ClaimModelQuotaError(ClaimProductionError):
    """模型配额不足。"""


class ClaimModelContentSafetyError(ClaimProductionError):
    """模型内容安全策略拒绝本次规范化。"""


class ClaimBindingError(BehaviorError, ValueError):
    """语义提案无法忠实绑定当前 Manifest 和语义记录。"""


class ClaimAdmissionError(BehaviorError, ValueError):
    """Claim 准入过程无法形成确定性决定。"""


class ClaimStoreError(BehaviorError):
    """Behavior SQLite Store 初始化或读写失败。"""


class ClaimStoreCapacityError(ClaimStoreError):
    """Behavior Store 的显式容量边界已达到。"""


class ClaimProcessingConflictError(ClaimStoreError):
    """同一处理身份存在不同内容或不完整发布。"""


__all__ = [
    "BehaviorError",
    "BehaviorOwnerConflictError",
    "BehaviorOwnerError",
    "ClaimAdmissionError",
    "ClaimBindingError",
    "ClaimModelAuthenticationError",
    "ClaimModelConfigurationError",
    "ClaimModelContentSafetyError",
    "ClaimModelInputError",
    "ClaimModelPermissionError",
    "ClaimModelQuotaError",
    "ClaimModelSchemaError",
    "ClaimModelTransportError",
    "ClaimProcessingConflictError",
    "ClaimProductionError",
    "ClaimSchemaError",
    "ClaimStoreCapacityError",
    "ClaimStoreError",
    "EvidenceBundleError",
    "EvidenceBundleStateError",
    "EvidenceManifestError",
    "SemanticClockError",
    "SemanticIngressError",
    "SemanticRecordConflictError",
    "SemanticRecordError",
    "SemanticRecordLateError",
]
