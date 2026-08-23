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
    "prediction",
    "pre",
    "conversation",
    "memory",
    "infrastructure",
    "foundation",
)
RETIRED_NAMES = (
    "BehaviorSnapshotReader",
    "BehaviorCASConflictError",
    "BehaviorMergeStrategy",
    "append_outcomes",
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


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


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
    ("ModelClient", "behavior", "prediction", "pre", "memory", "infrastructure", "foundation"),
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
    # Config/behavior.py 曾随旧第一层设计退役；BHV-RUNTIME-001 以**纯标量配置组**的身份重建
    # 它。守卫从"不得存在"改为口径检查：它不得 import behavior——上下文窗口等数值默认的唯一
    # 出处仍在 behavior/fusion/config.py，由组合根解析注入，配置层只有标量。
    assert "behavior" not in imported_roots(REPOSITORY_ROOT / "Config" / "behavior.py")
    assert not (REPOSITORY_ROOT / "infrastructure" / "store" / "processing_lock.py").exists()

    # BHV-RUNTIME-001 接线后，**只有 Runtime**（组合根，跨域组装的唯一合法位置，与 memory
    # 同理）允许 import behavior；其余包对 behavior 的反向依赖仍然禁止——Config 也不例外
    # （BehaviorConfig 是纯标量组，不 import behavior，上下文窗口默认值仍由组合根从
    # behavior/fusion/config.py 的唯一出处解析）。
    reverse_dependency_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in production_files()
        if path.relative_to(REPOSITORY_ROOT).parts[0] not in {"behavior", "Runtime"}
        and "behavior" in imported_roots(path)
    ]
    assert reverse_dependency_violations == []

    # ``ModelClient`` 不在禁止之列：它是供应商无关的能力契约层，本就供领域模块直接使用，
    # ``memory`` 的检索、抽取与语义生成同样直接依赖它。behavior 里只有事件融合调用模型，
    # 观测清洗、确定性派生与行为树写入都不经过它。
    behavior_dependency_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in behavior_root.rglob("*.py")
        if imported_roots(path)
        & {
            "Config",
            "Runtime",
            "conversation",
            "integrations",
            "memory",
            "pre",
        }
    ]
    assert behavior_dependency_violations == []

    # 模型调用收敛在两处受控触点：融合判断（语义生成）与 kinds 归一（身份归属——
    # 写入层唯一的 LLM 触点，只对齐名字不发明判断，见 TODO(BHV-TREE-REBUILD-001)）。
    # 存储、派生与落盘路径仍不得渗入。
    model_callers = sorted(
        str(path.relative_to(REPOSITORY_ROOT))
        for path in behavior_root.rglob("*.py")
        if "ModelClient" in imported_roots(path)
    )
    assert model_callers == [
        "behavior/fusion/service.py",
        "behavior/kinds/resolver.py",
        "behavior/semantic/generator.py",
    ]

    # 直接依赖收敛了还不够：只查一跳的话，``derivation → result → service → ModelClient``
    # 这样的两跳链会完整通过（已用变异测试证伪过一次）。所以这里算**传递闭包**，并且对
    # ``behavior/fusion/`` 下**除白名单外的每一个模块**成立——按名单列举会漏掉后来新增的模块，
    # 判断存储就是这么漏掉的。
    fusion_root = behavior_root / "fusion"
    fusion_modules = {
        f"behavior.fusion.{path.relative_to(fusion_root).with_suffix('').as_posix().replace('/', '.')}"
        .removesuffix(".__init__")
        : path
        for path in fusion_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    allowed_model_callers = {
        "behavior.fusion",
        "behavior.fusion.service",
        "behavior.fusion.runner",
    }

    def reaches_model_client(module: str, seen: set[str]) -> bool:
        if module in seen:
            return False
        seen.add(module)
        path = fusion_modules.get(module)
        if path is None:
            return False
        imported = imported_modules(path)
        if any(name == "ModelClient" or name.startswith("ModelClient.") for name in imported):
            return True
        return any(
            reaches_model_client(name, seen)
            for name in imported
            if name in fusion_modules
        )

    leaking = sorted(
        module
        for module in fusion_modules
        if module not in allowed_model_callers and reaches_model_client(module, set())
    )
    assert leaking == [], f"这些确定性模块传递性地依赖了 ModelClient: {leaking}"

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


def test_reduction_modules_reach_the_model_only_through_the_runner() -> None:
    """归约层唯一的 LLM 触点是 runner 里的 kinds 归一；纯函数模块连传递依赖都不许有。

    与融合层同一形状的传递闭包守卫：只查一跳会放过 ``chains → runner → resolver →
    ModelClient`` 这类多跳链；闭包图必须跨到 kinds 与 fusion，否则经它们中转的泄漏不可见。
    """

    behavior_root = REPOSITORY_ROOT / "behavior"
    graph: dict[str, Path] = {}
    # 图必须纳入 semantic：runner → semantic.refresher → semantic.generator → ModelClient
    # 是真实两跳链，图外的中转会让泄漏不可见（"判断存储就是这么漏掉的"的同构盲区）。
    for package in ("reduction", "kinds", "fusion", "semantic"):
        package_root = behavior_root / package
        for path in package_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            module = (
                f"behavior.{package}."
                f"{path.relative_to(package_root).with_suffix('').as_posix().replace('/', '.')}"
            ).removesuffix(".__init__")
            graph[module] = path

    def reaches_model_client(module: str, seen: set[str]) -> bool:
        if module in seen:
            return False
        seen.add(module)
        path = graph.get(module)
        if path is None:
            return False
        imported = imported_modules(path)
        if any(name == "ModelClient" or name.startswith("ModelClient.") for name in imported):
            return True
        return any(
            reaches_model_client(name, seen) for name in imported if name in graph
        )

    allowed = {"behavior.reduction", "behavior.reduction.runner"}
    leaking = sorted(
        module
        for module in graph
        if module.startswith("behavior.reduction")
        and module not in allowed
        and reaches_model_client(module, set())
    )
    assert leaking == [], f"这些归约确定性模块传递性地依赖了 ModelClient: {leaking}"

    # semantic 包自身的确定性模块（model/config/refresher 的纯逻辑面）同样不许直接碰模型；
    # refresher 经 generator 协议触达是设计路径，generator 与包 __init__ 是仅有的白名单。
    semantic_allowed = {
        "behavior.semantic",
        "behavior.semantic.generator",
        "behavior.semantic.refresher",
    }
    semantic_leaking = sorted(
        module
        for module in graph
        if module.startswith("behavior.semantic")
        and module not in semantic_allowed
        and reaches_model_client(module, set())
    )
    assert semantic_leaking == [], f"这些语义层确定性模块传递性地依赖了 ModelClient: {semantic_leaking}"


def test_prediction_tree_does_not_depend_on_behavior_yet() -> None:
    """旧 Behavior→Prediction 投影包随行为树重建整体退役（TODO(BHV-TREE-REBUILD-001)）。

    新的时间预测树落地前，prediction 不得依赖 behavior；届时唯一的读取入口再以显式包恢复，
    并恢复"只有该包可 import behavior"的边界断言。
    """

    prediction_root = REPOSITORY_ROOT / "prediction"
    assert prediction_root.is_dir()

    behavior_importers = {
        path.relative_to(REPOSITORY_ROOT)
        for path in prediction_root.rglob("*.py")
        if "behavior" in imported_roots(path)
    }
    assert behavior_importers == set()

    forbidden_prediction_dependencies = {
        "Config",
        "ModelClient",
        "Runtime",
        "conversation",
        "integrations",
        "memory",
        "pre",
    }
    prediction_dependency_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in prediction_root.rglob("*.py")
        if imported_roots(path) & forbidden_prediction_dependencies
    ]
    assert prediction_dependency_violations == []

    behavior_reverse_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "behavior").rglob("*.py")
        if "prediction" in imported_roots(path)
    ]
    assert behavior_reverse_violations == []


def test_conversation_source_and_projection_do_not_depend_on_memory_or_behavior() -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "conversation").rglob("*.py")
        if imported_roots(path) & {"memory", "behavior", "Runtime", "Config", "integrations"}
    ]
    assert violations == []


def test_behavior_projection_reads_only_source_envelope_batch() -> None:
    projection_root = REPOSITORY_ROOT / "conversation" / "projection" / "behavior"
    modules = sorted(projection_root.glob("*.py"))
    assert {path.name for path in modules} == {
        "__init__.py",
        "consumer.py",
        "model.py",
        "projector.py",
        "store.py",
    }
    source = "\n".join(path.read_text(encoding="utf-8") for path in modules)
    for path in modules:
        assert imported_roots(path).isdisjoint({"memory", "behavior", "Runtime", "Config"})
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
