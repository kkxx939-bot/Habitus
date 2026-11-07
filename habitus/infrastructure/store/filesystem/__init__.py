"""安全文件路径与耐久原子字节操作。"""

from habitus.infrastructure.store.filesystem.durable_io import (
    DurableDirectoryEntry,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_rmdir,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
    regular_file_exists,
)
from habitus.infrastructure.store.filesystem.owner_access import (
    describe_access,
    ensure_owner_directory,
    grants_group_or_other_access,
    make_owner_directories,
    restrict_directory_to_owner,
    restrict_file_to_owner,
)
from habitus.infrastructure.store.filesystem.path_safety import (
    DurablePathIntegrityError,
    is_path_alias,
    require_safe_artifact_path,
)

__all__ = [
    "DurableDirectoryEntry",
    "DurablePathIntegrityError",
    "ImmutableArtifactConflictError",
    "atomic_create_bytes",
    "atomic_replace_bytes",
    "atomic_temporary_destination",
    "describe_access",
    "durable_rmdir",
    "durable_unlink",
    "ensure_owner_directory",
    "ensure_real_directory",
    "grants_group_or_other_access",
    "is_path_alias",
    "list_real_directory",
    "make_owner_directories",
    "real_directory_exists",
    "read_regular_bytes",
    "regular_file_exists",
    "require_safe_artifact_path",
    "restrict_directory_to_owner",
    "restrict_file_to_owner",
]
