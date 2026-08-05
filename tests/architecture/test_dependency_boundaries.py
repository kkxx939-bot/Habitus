"""已经确认的新架构边界不得被兼容层或反向依赖破坏。"""

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    "Config",
    "Runtime",
    "ModelClient",
    "pre",
    "memory",
    "behavior",
    "infrastructure",
    "foundation",
)
RETIRED_NAMES = (
    "EvidenceSlice",
    "MemoryEditSource",
    "MemoryEditBatch",
    "SessionArchive",
    "ActionPolicy",
    "ContextURI",
    "memoryos://",
    "viking://",
)


def production_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for root in PRODUCTION_ROOTS
            for path in (REPOSITORY_ROOT / root).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_retired_memory_contracts_and_uri_schemes_do_not_reappear() -> None:
    violations = []
    for path in production_files():
        source = path.read_text(encoding="utf-8")
        for retired in RETIRED_NAMES:
            if retired in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {retired}")
    assert violations == []


@pytest.mark.parametrize(
    "package",
    ("ModelClient", "pre", "memory", "behavior", "infrastructure", "foundation"),
)
def test_domain_packages_never_import_top_level_runtime(package: str) -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / package).rglob("*.py")
        if "Runtime" in imported_roots(path)
    ]
    assert violations == []


def test_pre_contains_only_conversation_schema_and_no_storage_dependency() -> None:
    python_files = tuple((REPOSITORY_ROOT / "pre").rglob("*.py"))
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in python_files
        if imported_roots(path) & {"memory", "infrastructure", "Runtime", "Config"}
    ]
    assert violations == []
    assert not (REPOSITORY_ROOT / "pre" / "session").exists()


def test_config_root_does_not_branch_on_specific_model_or_vector_adapters() -> None:
    source = (REPOSITORY_ROOT / "Config" / "root.py").read_text(encoding="utf-8")
    for adapter_name in ("vikingdb", "ark_multimodal", "openai_compatible_chat"):
        assert adapter_name not in source


def test_local_setup_cli_renders_registry_without_vendor_branches() -> None:
    source = (REPOSITORY_ROOT / "integrations" / "local_service" / "cli.py").read_text(
        encoding="utf-8"
    )
    for vendor_identifier in (
        "deepseek",
        "volcengine",
        "aliyun",
        "qwen3-rerank",
        "vikingdb",
        "openai_compatible_chat",
        "ark_multimodal",
    ):
        assert vendor_identifier not in source.casefold()


def test_memory_kernel_never_imports_the_local_product_shell() -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "memory").rglob("*.py")
        if "integrations" in imported_roots(path)
    ]
    assert violations == []


def test_memory_schema_contains_exactly_the_six_confirmed_l2_kinds() -> None:
    definitions = REPOSITORY_ROOT / "memory" / "schema" / "definitions"
    assert {path.stem for path in definitions.glob("*.yaml")} == {
        "profile",
        "preferences",
        "entities",
        "tools",
        "events",
        "intentions",
    }


def test_behavior_and_memory_keep_strict_peer_dependency_boundaries() -> None:
    behavior_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "behavior").rglob("*.py")
        if imported_roots(path) & {"memory", "integrations", "Runtime", "Config"}
    ]
    memory_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "memory").rglob("*.py")
        if "behavior" in imported_roots(path)
    ]
    integration_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "integrations").rglob("*.py")
        if "behavior" in imported_roots(path)
    ]
    assert behavior_violations == []
    assert memory_violations == []
    assert integration_violations == []


def test_behavior_has_no_internal_multi_owner_partition_model() -> None:
    disallowed_partition_terms = ("user_id", "tenant_id", "account_id")
    violations = []
    for path in (REPOSITORY_ROOT / "behavior").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for term in disallowed_partition_terms:
            if term in source:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {term}")
    assert violations == []


def test_behavior_does_not_extend_memory_kind() -> None:
    source = (REPOSITORY_ROOT / "memory" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    memory_kind = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MemoryKind"
    )
    member_names = {
        target.id
        for node in memory_kind.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "BEHAVIOR" not in member_names
    memory_tree_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY_ROOT / "memory" / "tree").rglob("*.py")
    )
    assert "behaviors" not in memory_tree_source
    assert not (REPOSITORY_ROOT / "memory" / "schema" / "definitions" / "behavior.yaml").exists()
