"""已经确认的新架构边界不得被兼容层或反向依赖破坏。"""

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    "Config",
    "Runtime",
    "ModelClient",
    "behavior",
    "pre",
    "conversation",
    "memory",
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
    "EvidenceLedger",
    "ClaimLedger",
    "RecordSpec",
    "route_executor",
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
    ("ModelClient", "behavior", "pre", "memory", "infrastructure", "foundation"),
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


def test_behavior_semantic_tree_does_not_restore_retired_first_layer() -> None:
    behavior_root = REPOSITORY_ROOT / "behavior"
    assert behavior_root.is_dir()
    assert not (REPOSITORY_ROOT / "Config" / "behavior.py").exists()
    assert not (REPOSITORY_ROOT / "infrastructure" / "store" / "processing_lock.py").exists()

    reverse_dependency_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in production_files()
        if path.relative_to(REPOSITORY_ROOT).parts[0] != "behavior"
        and "behavior" in imported_roots(path)
    ]
    assert reverse_dependency_violations == []

    behavior_dependency_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in behavior_root.rglob("*.py")
        if imported_roots(path)
        & {
            "Config",
            "ModelClient",
            "Runtime",
            "conversation",
            "integrations",
            "memory",
            "pre",
        }
    ]
    assert behavior_dependency_violations == []

    assembly_tree = ast.parse(
        (REPOSITORY_ROOT / "Runtime" / "assembly.py").read_text(encoding="utf-8")
    )
    build_runtime = next(
        node
        for node in assembly_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_runtime"
    )
    build_runtime_parameters = {
        argument.arg
        for argument in (*build_runtime.args.posonlyargs, *build_runtime.args.args, *build_runtime.args.kwonlyargs)
    }
    assert "behavior_adapters" not in build_runtime_parameters

    runtime_tree = ast.parse(
        (REPOSITORY_ROOT / "Runtime" / "runtime.py").read_text(encoding="utf-8")
    )
    runtime_class = next(
        node
        for node in runtime_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Runtime"
    )
    runtime_methods = {
        node.name
        for node in runtime_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert runtime_methods.isdisjoint(
        {
            "ingest_behavior_semantic",
            "normalize_behavior_evidence",
            "retry_behavior_enhancement",
        }
    )

    retired_storage_markers = {"behavior.sqlite3", "evidence_claims.sqlite3"}
    storage_violations = [
        f"{path.relative_to(REPOSITORY_ROOT)}: {marker}"
        for path in production_files()
        for marker in retired_storage_markers
        if marker in path.read_text(encoding="utf-8")
    ]
    assert storage_violations == []


def test_conversation_source_and_projection_do_not_depend_on_memory_or_behavior() -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "conversation").rglob("*.py")
        if imported_roots(path) & {"memory", "behavior", "Runtime", "Config", "integrations"}
    ]
    assert violations == []


def test_behavior_projection_reads_only_source_envelope_batch() -> None:
    projection_path = REPOSITORY_ROOT / "conversation" / "projection" / "behavior.py"
    imported = imported_roots(projection_path)
    source = projection_path.read_text(encoding="utf-8")
    assert imported.isdisjoint({"memory", "behavior", "Runtime", "Config"})
    assert "envelope.batch.messages" in source
    assert "ConversationMessageChunker" not in source
    assert "ConversationSegment" not in source
    assert "ConversationSummary" not in source
    assert "MemoryEditor" not in source


def test_memory_conversation_consumer_wraps_the_single_existing_enqueuer_chain() -> None:
    consumer_source = (
        REPOSITORY_ROOT / "memory" / "workflow" / "conversation_consumer.py"
    ).read_text(encoding="utf-8")
    assert "self.enqueuer.append" in consumer_source
    assert "self.enqueuer.enqueue_ready_segments" in consumer_source
    assert "ConversationToolResultReducer(" not in consumer_source
    assert "ConversationMessageChunker(" not in consumer_source


def test_retired_behavior_does_not_extend_memory_kind() -> None:
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
