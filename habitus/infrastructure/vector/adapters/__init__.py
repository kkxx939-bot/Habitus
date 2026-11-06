"""内置向量数据库 Adapter；当前唯一内置的是进程内本地 SQLite+numpy 后端。"""

from habitus.infrastructure.vector.adapters.local_sqlite import (
    LocalSQLiteVectorBackend,
    LocalSQLiteVectorStoreConfig,
    build_local_sqlite_backend,
    register_builtin_vector_adapters,
)

__all__ = [
    "LocalSQLiteVectorBackend",
    "LocalSQLiteVectorStoreConfig",
    "build_local_sqlite_backend",
    "register_builtin_vector_adapters",
]
