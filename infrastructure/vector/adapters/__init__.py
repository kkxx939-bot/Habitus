"""远程向量数据库 Adapter；当前首个正式协议是 VikingDB。"""

from infrastructure.vector.adapters.vikingdb import (
    VikingDBBackend,
    build_vikingdb_backend,
    register_builtin_vector_adapters,
)
from infrastructure.vector.adapters.vikingdb_config import (
    VikingDBAuthMode,
    VikingDBSchemaMode,
    VikingDBVectorStoreConfig,
)

__all__ = [
    "VikingDBAuthMode",
    "VikingDBSchemaMode",
    "VikingDBBackend",
    "VikingDBVectorStoreConfig",
    "build_vikingdb_backend",
    "register_builtin_vector_adapters",
]
