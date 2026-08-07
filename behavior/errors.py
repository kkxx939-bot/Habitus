"""Behavior 第一层可预期错误的稳定分类。"""


class BehaviorError(Exception):
    """Behavior 第一层错误基类。"""


class BehaviorEvidenceError(BehaviorError):
    """Evidence 输入或耐久化错误。"""


class BehaviorEvidenceSchemaError(BehaviorEvidenceError, ValueError):
    """Evidence 输入不满足严格结构。"""


class BehaviorEvidenceConflictError(BehaviorEvidenceError):
    """相同输入身份对应不同内容。"""


class BehaviorEvidenceCapacityError(BehaviorEvidenceError):
    """Evidence 或 Receipt 容量耗尽。"""


class BehaviorEvidenceClockError(BehaviorEvidenceError):
    """Evidence 事件时间不满足保守时间策略。"""


class BehaviorAdapterError(BehaviorError):
    """Adapter 注册、查找或执行失败。"""


class BehaviorAdapterCapabilityError(BehaviorAdapterError, ValueError):
    """Adapter 输出超出其已绑定能力。"""


class BehaviorClaimError(BehaviorError):
    """Claim 结构或耐久化错误。"""


class BehaviorClaimSchemaError(BehaviorClaimError, ValueError):
    """Claim 或 Proposal 不满足严格结构。"""


class ClaimCompatibilityError(BehaviorClaimSchemaError):
    """Normalizer Proposal 与当前 Evidence 的兼容策略冲突。"""


class BehaviorClaimConflictError(BehaviorClaimError):
    """相同 Claim 身份对应不同内容。"""


class BehaviorClaimCapacityError(BehaviorClaimError):
    """Claim、Attempt 或 Receipt 容量耗尽。"""


class ClaimNormalizationError(BehaviorClaimError):
    """Claim 规范化失败。"""


class ClaimNormalizationConflictError(ClaimNormalizationError):
    """同一处理身份出现不一致发布。"""


class ClaimModelTransportError(ClaimNormalizationError):
    """模型传输或限流失败。"""


class ClaimModelSchemaError(ClaimNormalizationError):
    """模型响应或结构化输出失败。"""


class ClaimModelInputError(ClaimNormalizationError):
    """模型输入超过字符或 Token 边界。"""


class ClaimModelContentSafetyError(ClaimNormalizationError):
    """模型内容安全策略拒绝本次调用。"""


class ClaimModelAuthenticationError(ClaimNormalizationError):
    """模型认证失败。"""


class ClaimModelPermissionError(ClaimNormalizationError):
    """模型权限不足。"""


class ClaimModelConfigurationError(ClaimNormalizationError):
    """模型配置无效。"""


class ClaimModelQuotaError(ClaimNormalizationError):
    """模型配额不足。"""


class BehaviorStoreError(BehaviorError):
    """Behavior SQLite 初始化、完整性或读写错误。"""


class LegacyBehaviorStoreError(BehaviorStoreError):
    """检测到不兼容的旧 Behavior 数据库。"""


__all__ = [
    "BehaviorAdapterCapabilityError",
    "BehaviorAdapterError",
    "BehaviorClaimCapacityError",
    "BehaviorClaimConflictError",
    "BehaviorClaimError",
    "BehaviorClaimSchemaError",
    "BehaviorError",
    "BehaviorEvidenceCapacityError",
    "BehaviorEvidenceClockError",
    "BehaviorEvidenceConflictError",
    "BehaviorEvidenceError",
    "BehaviorEvidenceSchemaError",
    "BehaviorStoreError",
    "ClaimModelAuthenticationError",
    "ClaimModelConfigurationError",
    "ClaimModelContentSafetyError",
    "ClaimModelInputError",
    "ClaimModelPermissionError",
    "ClaimModelQuotaError",
    "ClaimModelSchemaError",
    "ClaimModelTransportError",
    "ClaimCompatibilityError",
    "ClaimNormalizationConflictError",
    "ClaimNormalizationError",
    "LegacyBehaviorStoreError",
]
