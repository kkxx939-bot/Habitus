"""Behavior 证据与声明层的稳定错误分类。"""


class BehaviorError(Exception):
    """Behavior 第一层所有可预期领域错误的基类。"""


class BehaviorOwnerError(BehaviorError, ValueError):
    """上游 Owner 路由尚未形成可接受的单 Owner 绑定。"""


class BehaviorOwnerConflictError(BehaviorOwnerError):
    """当前 Store 已绑定到另一个 Owner 规范指纹。"""


class SourceRecordError(BehaviorError, ValueError):
    """来源记录不满足严格结构或容量边界。"""


class SourceRecordConflictError(SourceRecordError):
    """相同来源身份对应不同的规范内容。"""


class SourceRecordLateError(SourceRecordError):
    """来源事件时间已经越过提交的 watermark。"""


class EvidenceWindowError(BehaviorError, ValueError):
    """证据窗口无法按硬边界处理。"""


class EvidenceWindowStateError(EvidenceWindowError):
    """证据窗口状态不允许当前操作。"""


class EvidenceManifestError(BehaviorError, ValueError):
    """EvidenceManifest 不满足不可变规范。"""


class ClaimSchemaError(BehaviorError, ValueError):
    """ClaimProposal 或 ClaimBatch 不满足严格 Schema。"""


class ClaimProductionError(BehaviorError):
    """Producer 调用或输出处理失败。"""


class ClaimModelNetworkError(ClaimProductionError):
    """模型网络或共享客户端调用失败。"""


class ClaimModelSchemaError(ClaimProductionError):
    """模型响应未通过严格 ClaimProposal Schema。"""


class ClaimValidationError(BehaviorError, ValueError):
    """ClaimProposal 无法忠实绑定指定 Manifest。"""


class ClaimAdmissionError(BehaviorError, ValueError):
    """Claim 准入过程无法形成确定性决定。"""


class ClaimStoreError(BehaviorError):
    """Evidence/Claim SQLite Store 初始化或读写失败。"""


class ClaimProcessingConflictError(ClaimStoreError):
    """同一处理身份存在不同内容或不完整发布。"""


__all__ = [
    "BehaviorError",
    "BehaviorOwnerConflictError",
    "BehaviorOwnerError",
    "ClaimAdmissionError",
    "ClaimModelNetworkError",
    "ClaimModelSchemaError",
    "ClaimProcessingConflictError",
    "ClaimProductionError",
    "ClaimSchemaError",
    "ClaimStoreError",
    "ClaimValidationError",
    "EvidenceManifestError",
    "EvidenceWindowError",
    "EvidenceWindowStateError",
    "SourceRecordConflictError",
    "SourceRecordError",
    "SourceRecordLateError",
]
