"""持久化向量存储公共协议和显式 Adapter 工厂。"""

from infrastructure.vector.config import (
    VectorStoreConfig,
    VectorStoreRequirements,
    VectorStoreRouteConfig,
)
from infrastructure.vector.contracts import RawVectorBackend, VectorStore
from infrastructure.vector.factory import (
    VectorBackendBuilder,
    VectorStoreBuildContext,
    VectorStoreFactory,
)
from infrastructure.vector.model import (
    VectorPublicationSnapshot,
    VectorScalar,
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
    VectorStoreUnsupportedTopologyError,
    VectorValue,
)
from infrastructure.vector.publication import PublishedVectorStore

__all__ = [
    "VectorScalar",
    "VectorValue",
    "PublishedVectorStore",
    "RawVectorBackend",
    "VectorBackendBuilder",
    "VectorPublicationSnapshot",
    "VectorStore",
    "VectorStoreBuildContext",
    "VectorStoreBusyError",
    "VectorStoreConfig",
    "VectorStoreRequirements",
    "VectorStoreRouteConfig",
    "VectorStoreConflictError",
    "VectorStoreError",
    "VectorStoreFactory",
    "VectorStoreFilter",
    "VectorStoreIntegrityError",
    "VectorStoreMatch",
    "VectorStoreRecord",
    "VectorStoreState",
    "VectorStoreUnsupportedTopologyError",
]
