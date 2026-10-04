"""已经确认的新架构边界不得被兼容层或反向依赖破坏。"""

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC = REPOSITORY_ROOT / "habitus"
PRODUCTION_ROOTS = (
    "config",
    "runtime",
    "model_client",
    "behavior",
    "scene",
    "prediction",
    "foresight",
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
            for path in (SRC / root).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )


def _root(module: str) -> str:
    """把 ``habitus.memory.editor`` 归到 ``memory``；第三方或标准库保持首段。"""

    parts = module.split(".")
    if parts[0] == "habitus" and len(parts) > 1:
        return parts[1]
    return parts[0]


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(_root(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(_root(node.module))
    return roots


def imported_modules(path: Path) -> set[str]:
    """本文件 import 的模块全名。**相对 import 要先解析成绝对名**。

    只取 ``node.module`` 会让 ``from ..views.stats import Interval`` 变成 ``views.stats``，任何按
    ``habitus.`` 前缀判断的边界断言都放它过去（变异实测：在 ``ledger/opening.py`` 加这一行，scene 的
    边界测试照样 PASS）。相对 import 在仓库里几乎不用，但闸不能是漏的。
    """

    modules: set[str] = set()
    try:
        package = path.relative_to(SRC).parent.as_posix().replace("/", ".")
    except ValueError:
        package = ""
    base = f"habitus.{package}".rstrip(".") if package else "habitus"
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if not node.level:
                if node.module:
                    modules.add(node.module)
                continue
            # level=1 是本包，level=2 是上一层，以此类推。
            parts = base.split(".")
            anchor = ".".join(parts[: len(parts) - (node.level - 1)]) if node.level > 1 else base
            modules.add(f"{anchor}.{node.module}" if node.module else anchor)
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
    ("model_client", "behavior", "scene", "prediction", "pre", "memory", "infrastructure", "foundation"),
)
def test_domain_packages_never_import_top_level_runtime(package: str) -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (SRC / package).rglob("*.py")
        if "runtime" in imported_roots(path)
    ]
    assert violations == []


def test_pre_contains_only_conversation_schema_and_no_storage_dependency() -> None:
    python_files = tuple((SRC / "pre").rglob("*.py"))
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in python_files
        if imported_roots(path) & {"memory", "infrastructure", "runtime", "config"}
    ]
    assert violations == []
    assert not (SRC / "pre" / "session").exists()


def test_config_root_does_not_branch_on_specific_model_or_vector_adapters() -> None:
    source = (SRC / "config" / "root.py").read_text(encoding="utf-8")
    for adapter_name in ("vikingdb", "ark_multimodal", "openai_compatible_chat"):
        assert adapter_name not in source


def test_local_setup_cli_renders_registry_without_vendor_branches() -> None:
    source = (SRC / "integrations" / "local_service" / "cli.py").read_text(
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
        for path in (SRC / "memory").rglob("*.py")
        if "integrations" in imported_roots(path)
    ]
    assert violations == []


def test_memory_schema_contains_exactly_the_six_confirmed_l2_kinds() -> None:
    definitions = SRC / "memory" / "schema" / "definitions"
    assert {path.stem for path in definitions.glob("*.yaml")} == {
        "profile",
        "preferences",
        "entities",
        "tools",
        "events",
        "intentions",
    }


def test_behavior_semantic_tree_does_not_restore_retired_first_layer() -> None:
    behavior_root = SRC / "behavior"
    assert behavior_root.is_dir()
    # Config/behavior.py 曾随旧第一层设计退役；BHV-RUNTIME-001 以**纯标量配置组**的身份重建
    # 它。守卫从"不得存在"改为口径检查：它不得 import behavior——上下文窗口等数值默认的唯一
    # 出处仍在 behavior/fusion/config.py，由组合根解析注入，配置层只有标量。
    assert "behavior" not in imported_roots(SRC / "config" / "behavior.py")
    assert not (SRC / "infrastructure" / "store" / "processing_lock.py").exists()

    # BHV-RUNTIME-001 接线后，**只有 Runtime**（组合根，跨域组装的唯一合法位置，与 memory
    # 同理）允许 import behavior；其余包对 behavior 的反向依赖仍然禁止——Config 也不例外
    # （BehaviorConfig 是纯标量组，不 import behavior，上下文窗口默认值仍由组合根从
    # behavior/fusion/config.py 的唯一出处解析）。唯一的例外是时间预测树的读取入口
    # ``prediction/source.py``：整棵树每夜从行为树重建，那是设计路径而不是泄漏，
    # 收在单个模块里由 test_prediction_reads_the_behaviour_tree_through_exactly_one_module 守住。
    # ``scene``（语义关联层）是行为树的解释层，从行为树派生，允许 import behavior。
    reverse_dependency_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in production_files()
        if path.relative_to(SRC).parts[0] not in {"behavior", "runtime", "scene"}
        and path.relative_to(SRC).as_posix() != "prediction/source.py"
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
            "config",
            "runtime",
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
        str(path.relative_to(SRC))
        for path in behavior_root.rglob("*.py")
        if "model_client" in imported_roots(path)
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
        f"habitus.behavior.fusion.{path.relative_to(fusion_root).with_suffix('').as_posix().replace('/', '.')}"
        .removesuffix(".__init__")
        : path
        for path in fusion_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    allowed_model_callers = {
        "habitus.behavior.fusion",
        "habitus.behavior.fusion.service",
        "habitus.behavior.fusion.runner",
    }

    def reaches_model_client(module: str, seen: set[str]) -> bool:
        if module in seen:
            return False
        seen.add(module)
        path = fusion_modules.get(module)
        if path is None:
            return False
        imported = imported_modules(path)
        if any(name == "habitus.model_client" or name.startswith("habitus.model_client.") for name in imported):
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
        (SRC / "runtime" / "assembly.py").read_text(encoding="utf-8")
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
        (SRC / "runtime" / "runtime.py").read_text(encoding="utf-8")
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

    behavior_root = SRC / "behavior"
    graph: dict[str, Path] = {}
    # 图必须纳入 semantic：runner → semantic.refresher → semantic.generator → ModelClient
    # 是真实两跳链，图外的中转会让泄漏不可见（"判断存储就是这么漏掉的"的同构盲区）。
    for package in ("reduction", "kinds", "fusion", "semantic"):
        package_root = behavior_root / package
        for path in package_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            module = (
                f"habitus.behavior.{package}."
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
        if any(name == "habitus.model_client" or name.startswith("habitus.model_client.") for name in imported):
            return True
        return any(
            reaches_model_client(name, seen) for name in imported if name in graph
        )

    allowed = {"habitus.behavior.reduction", "habitus.behavior.reduction.runner"}
    leaking = sorted(
        module
        for module in graph
        if module.startswith("habitus.behavior.reduction")
        and module not in allowed
        and reaches_model_client(module, set())
    )
    assert leaking == [], f"这些归约确定性模块传递性地依赖了 ModelClient: {leaking}"

    # semantic 包自身的确定性模块（model/config/refresher 的纯逻辑面）同样不许直接碰模型；
    # refresher 经 generator 协议触达是设计路径，generator 与包 __init__ 是仅有的白名单。
    semantic_allowed = {
        "habitus.behavior.semantic",
        "habitus.behavior.semantic.generator",
        "habitus.behavior.semantic.refresher",
    }
    semantic_leaking = sorted(
        module
        for module in graph
        if module.startswith("habitus.behavior.semantic")
        and module not in semantic_allowed
        and reaches_model_client(module, set())
    )
    assert semantic_leaking == [], f"这些语义层确定性模块传递性地依赖了 ModelClient: {semantic_leaking}"


def _prediction_module_graph() -> dict[str, Path]:
    prediction_root = SRC / "prediction"
    graph: dict[str, Path] = {}
    for path in prediction_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(prediction_root).with_suffix("").as_posix().replace("/", ".")
        graph[f"habitus.prediction.{relative}".removesuffix(".__init__")] = path
    return graph


def _prediction_modules_reaching(root: str) -> set[str]:
    """本包内**传递性**触达某个外部根包的模块集合。

    只查一跳不够：``builder → source → behavior`` 这条转发链下，写着 import 的只有 source，
    但任何 import builder 的人都会把 behavior 拖进来（真实发生过：门面 import builder，
    于是 ``import prediction`` 拉进 21 个 behavior 模块，而一跳检查全绿）。
    """

    graph = _prediction_module_graph()
    cache: dict[str, bool] = {}

    def reaches(module: str, stack: set[str]) -> bool:
        if module in cache:
            return cache[module]
        if module in stack:
            return False
        stack.add(module)
        imported = imported_modules(graph[module])
        result = any(name == root or name.startswith(f"{root}.") for name in imported) or any(
            reaches(name, stack) for name in imported if name in graph
        )
        stack.discard(module)
        cache[module] = result
        return result

    return {module for module in graph if reaches(module, set())}


def test_prediction_reads_the_behaviour_tree_through_exactly_one_module() -> None:
    """时间预测树只有一个模块**传递性**触达行为树：``prediction.source``。

    整棵树每夜从行为树重建，所以这条依赖是设计路径而不是泄漏；但它必须收在一个模块里，
    其余模块保持零依赖的纯计算，才能让估计器的正确性在纯函数层穷举验证
    （见 ``TODO(PRED-TREE-001)``）。查传递闭包而不是"谁写了 import"：共享一个住在
    source 里的数据类型就足以让全包连坐，而字面检查看不见。
    """

    assert (SRC / "prediction").is_dir()
    assert _prediction_modules_reaching("habitus.behavior") == {"habitus.prediction.source"}
    # prediction 不读 scene（原来的 ``prediction/scene_source.py`` 已删——它零调用方，留着只会让两个包互指）。
    # 新语义树要对照预测树的期望时，由组合根注入 callable，scene 也不直接读 prediction。
    assert _prediction_modules_reaching("habitus.scene") == set()


def test_prediction_stays_out_of_the_semantic_and_composition_layers() -> None:
    """本层零语义、零 LLM：模型编排与记忆桥接一律住在组合根。

    ``prediction`` 不得触达 ``memory``（两者的桥接是组合根的事），也不得触达 ``ModelClient``
    （在线只有组合根那两个受控调用点）。见 ``TODO(PRED-DOWNSTREAM-001)`` 的"组合根的两条边界"。

    查的是**传递闭包**：一跳检查放不过经 source 或 builder 中转的链路。
    """

    forbidden = (
        "habitus.config",
        "habitus.model_client",
        "habitus.runtime",
        "habitus.conversation",
        "habitus.integrations",
        "habitus.memory",
        "habitus.pre",
    )
    reachable = {root: sorted(_prediction_modules_reaching(root)) for root in forbidden}
    assert {root: modules for root, modules in reachable.items() if modules} == {}


def test_behaviour_never_depends_on_prediction() -> None:
    """依赖是单向的：行为树不知道有人在统计它，否则预测的聚合键会反过来决定行为树的字段。"""

    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (SRC / "behavior").rglob("*.py")
        if imported_roots(path) & {"prediction", "scene"}
    ]
    assert violations == []


def _scene_module_graph() -> dict[str, Path]:
    scene_root = SRC / "scene"
    graph: dict[str, Path] = {}
    for path in scene_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(scene_root).with_suffix("").as_posix().replace("/", ".")
        graph[f"habitus.scene.{relative}".removesuffix(".__init__")] = path
    return graph


def _scene_closure(module: str, graph: dict[str, Path]) -> set[str]:
    """从 ``module`` 出发**传递性**触达的全部模块名（scene 内的按图走，外部的只记名字）。

    一跳检查放不过"经包根转手"这条路：``occurrences/__init__`` 一旦 import 映射器，任何
    ``from habitus.scene.occurrences import ConceptHitStore`` 都把 ``model_client`` 拖进来，而写着 import 的
    只有 ``__init__``。同一份文件里 prediction / fusion / foresight 三处都算了闭包，这里同理。
    """

    reached: set[str] = set()
    stack = [module]
    while stack:
        current = stack.pop()
        if current not in graph:
            continue
        for name in imported_modules(graph[current]):
            if name in reached:
                continue
            reached.add(name)
            if name in graph:
                stack.append(name)
    return reached


def _scene_modules_reaching(graph: dict[str, Path], *targets: str, exact: bool = False) -> list[str]:
    """scene 内传递性触达 ``targets``（模块名或前缀）的模块，按名排序。"""

    def hits(name: str) -> bool:
        return any(name == target or (not exact and name.startswith(f"{target}.")) for target in targets)

    return sorted(module for module in graph if any(hits(name) for name in _scene_closure(module, graph)))


def test_scene_tree_is_a_pure_derivation_of_the_behaviour_tree() -> None:
    """语义关联层只读行为树、只用基础设施与模型契约；不知道预测树、记忆树与组合根。

    旧语义树（关联、规律级、待关联清单、gloss 读口）2026-09-26 整块删掉；新树按《语义树重构》分
    ``concepts / hypotheses / occurrences / ledger / views`` 五支重建。**全部按传递闭包判**：

    - scene 里只有四个模块触达 ``model_client``（基准生成的两个作者 + 闭环的触点③ + 映射器；
      旁册按结构协议注入嵌入器），而且它们都不从包根导出；
    - 没有任何模块触达 prediction——对照期望、常态、复发间隔都由组合根注入的 callable 给；
    - 引用单向：``concepts`` 不引用别支；``hypotheses`` 只引用 ``concepts``（基准要和观测独立，B8）；
      ``occurrences`` 不触达 ``hypotheses`` / ``ledger``（映射者看不到假设与账，七d-7）；四支不触达 ``views``
      也不触达包根（包根导出 views）——投影可整个删掉重建，底层引用它就删不掉了；
    - 只读行为树：不触达写侧（编辑器、归约、融合、词表、观测、语义层）——"行为树一个字不动"要由边界保证。
    """

    scene_root = SRC / "scene"
    assert scene_root.is_dir()
    graph = _scene_module_graph()
    forbidden_roots = ("habitus.memory", "habitus.config", "habitus.runtime", "habitus.conversation", "habitus.integrations", "habitus.pre")
    assert _scene_modules_reaching(graph, *forbidden_roots) == []
    assert _scene_modules_reaching(graph, "habitus.prediction") == []
    # 三个 LLM 触点，一个都不许多：写概念（①）、写假设（②）、映射（③）。**都不从包根导出**——
    # 包根一 import 它们，任何 ``from habitus.scene.concepts import ConceptSet`` 都把 model_client
    # 拖进来，这条边界就只剩一句注释了。
    assert _scene_modules_reaching(graph, "habitus.model_client") == [
        "habitus.scene.concepts.author",
        "habitus.scene.hypotheses.author",
        "habitus.scene.hypotheses.closure",
        "habitus.scene.occurrences.mapper",
    ]
    # 读侧只读行为树。views 里除 ``__init__`` 外不得 import 包根或 ``views`` 包根——投影不能倒灌。
    detours = sorted(
        f"{path.relative_to(SRC)}: {module}"
        for path in (scene_root / "views").rglob("*.py")
        if path.name != "__init__.py"
        for module in imported_modules(path)
        if module in {"habitus.scene", "habitus.scene.views"}
    )
    assert detours == []
    branches = ("concepts", "hypotheses", "occurrences", "ledger")
    shared = {"habitus.scene.codec", "habitus.scene.storage"}

    def branch_modules(branch: str) -> list[str]:
        return sorted(module for module in graph if module == f"habitus.scene.{branch}" or module.startswith(f"habitus.scene.{branch}."))

    def scene_reach(branch: str) -> set[str]:
        return {
            name
            for module in branch_modules(branch)
            for name in _scene_closure(module, graph)
            if name == "habitus.scene" or name.startswith("habitus.scene.")
        } - shared

    def branch_names(*allowed: str) -> set[str]:
        return {name for branch in allowed for name in branch_modules(branch)}

    assert scene_reach("concepts") <= branch_names("concepts"), scene_reach("concepts")
    assert scene_reach("hypotheses") <= branch_names("concepts", "hypotheses"), scene_reach("hypotheses")
    assert scene_reach("occurrences") <= branch_names("concepts", "occurrences"), scene_reach("occurrences")
    assert scene_reach("ledger") <= branch_names("concepts", "hypotheses", "occurrences", "ledger"), scene_reach("ledger")
    for branch in branches:
        assert "habitus.scene" not in scene_reach(branch), branch
        assert not any(name.startswith("habitus.scene.views") for name in scene_reach(branch)), branch
    behavior_write_side = (
        "habitus.behavior.editor",
        "habitus.behavior.reduction",
        "habitus.behavior.fusion",
        "habitus.behavior.kinds",
        "habitus.behavior.observation",
        "habitus.behavior.semantic",
    )
    assert _scene_modules_reaching(graph, *behavior_write_side) == []
    # 旧树不许回来：目录名与模块名一起钉。
    for gone in ("association", "regularity", "backlog.py", "model.py", "uri.py"):
        assert not (scene_root / gone).exists(), gone
    assert not (scene_root / "views" / "gloss.py").exists()


def test_foresight_only_reads_the_derived_trees() -> None:
    """预测层读派生树、不写任何树，也不认识组合根、记忆与配置。

    ``foresight`` 站在 prediction 与 scene 之上：它可以读两棵派生树（数字与背景各取一半），
    但**不得**触达 behavior（行为树只有观测→融合→归约那一个写入口，而读它是 prediction 与
    scene 的职责，不该再多一个消费者）、memory（人物层桥接住组合根）、runtime / config /
    conversation / integrations / pre。反向依赖同样禁止：下层永远不知道预测层存在。
    """

    root = SRC / "foresight"
    assert root.is_dir()
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in root.rglob("*.py")
        if imported_roots(path)
        & {"behavior", "memory", "config", "runtime", "conversation", "integrations", "pre", "infrastructure"}
    ]
    assert violations == []
    # ``infrastructure`` 也在禁止之列：本包只读两棵派生树给出的对象，自己不开文件、不拿锁、
    # 不碰向量库——"只读派生树"这句话要是允许它直接落盘，就等于没说。
    # 模型触点只有判断这一处（提示词渲染 + 服务），客户端从构造器注入。证据装配、渲染、判断的产物
    # 形状、schema 与装配校验都不得碰模型——它们一碰，"证据是纯函数装出来的"就只剩自觉了。
    model_callers = sorted(str(path.relative_to(SRC)) for path in root.rglob("*.py") if "model_client" in imported_roots(path))
    assert model_callers == ["foresight/judge/prompt.py", "foresight/judge/service.py"]
    # 直接依赖收敛了还不够，算传递闭包：包里除判断包的入口、提示词、服务之外，任何模块都不得经
    # 两跳够到模型客户端（融合包那条已经用变异测试证伪过一次"只查一跳"）。
    foresight_modules = {
        f"habitus.foresight.{path.relative_to(root).with_suffix('').as_posix().replace('/', '.')}".removesuffix(".__init__"): path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    allowed_model_callers = {
        "habitus.foresight.judge",
        "habitus.foresight.judge.prompt",
        "habitus.foresight.judge.service",
    }

    def reaches_model_client(module: str, seen: set[str]) -> bool:
        if module in seen:
            return False
        seen.add(module)
        path = foresight_modules.get(module)
        if path is None:
            return False
        imported = imported_modules(path)
        if any(name == "habitus.model_client" or name.startswith("habitus.model_client.") for name in imported):
            return True
        return any(reaches_model_client(name, seen) for name in imported if name in foresight_modules)

    leaking = sorted(
        module for module in foresight_modules if module not in allowed_model_callers and reaches_model_client(module, set())
    )
    assert leaking == [], f"这些确定性模块传递性地依赖了 ModelClient: {leaking}"
    # 认识 scene 读侧的只有取背景的 context / cards、编排的 assemble、渲染与结算；**数字那一侧对 scene 零知识**——
    # 出处日从预测树来、不从别处按覆盖日重推，这条要由边界钉着，不靠自觉。清单是穷举的：
    # 新文件要碰 scene 必须先改这一行，改的时候就得想清楚它属于哪一侧。
    scene_readers = sorted(
        str(path.relative_to(SRC)) for path in root.rglob("*.py") if "scene" in imported_roots(path)
    )
    assert scene_readers == [
        "foresight/assemble.py",
        "foresight/cards.py",
        "foresight/context.py",
        "foresight/ledger/settle.py",
        "foresight/render.py",
    ]
    # 而且只经读口进：``habitus.scene.views``（或包根）。新语义树的存储与写侧都不许直接碰——绕过读口就等于
    # 在预测层里重新长出一条读语义树的路。
    beyond_the_views = sorted(
        f"{path.relative_to(SRC)}: {module}"
        for path in root.rglob("*.py")
        for module in imported_modules(path)
        if module.startswith("habitus.scene") and module not in {"habitus.scene", "habitus.scene.views"}
    )
    assert beyond_the_views == []
    assert "scene" not in imported_roots(root / "numbers.py")
    assert "scene" not in imported_roots(root / "model.py")
    # 判断的产物形状对 scene 零知识：``basis`` 里只有 URI 字符串，不带视图、不带序列。
    assert "scene" not in imported_roots(root / "judge" / "model.py")
    assert "scene" not in imported_roots(root / "judge" / "schema.py")
    # 账本的形状与编解码同样对 scene 零知识；落盘只在组合根的 ``runtime/foresight_ledger.py``——
    # foresight 整包不许 import infrastructure（上面已钉），这里再钉"写 foresight_root 的只有它一个"。
    for name in ("model.py", "codec.py", "gate.py", "claims.py"):
        assert "scene" not in imported_roots(root / "ledger" / name)
    ledger_writers = sorted(
        str(path.relative_to(SRC))
        for path in (SRC / "runtime").rglob("*.py")
        if "habitus.foresight.ledger" in imported_modules(path) and "infrastructure" in imported_roots(path)
    )
    assert ledger_writers == ["runtime/foresight_ledger.py"]
    upstream = [
        str(path.relative_to(REPOSITORY_ROOT))
        for name in ("behavior", "scene", "prediction", "memory")
        for path in (SRC / name).rglob("*.py")
        if "foresight" in imported_roots(path)
    ]
    assert upstream == []
    # 组合根伸进行为侧写侧内部（融合存储、归约）的只有三处：行为管线的组装、预测层的组装（拿三份存储的类型）、
    # 未封口那座桥。桥只许拿"哪些判断还没归约"这一个事实（``reduction.pending``）与账本、词表，不许自己
    # 并链、解析记录——那是在组合根里重抄归约的算法。
    runtime_root = SRC / "runtime"
    inside_behavior = sorted(
        str(path.relative_to(SRC))
        for path in runtime_root.rglob("*.py")
        if any(
            module.startswith(("habitus.behavior.reduction.", "habitus.behavior.fusion."))
            for module in imported_modules(path)
        )
    )
    assert inside_behavior == ["runtime/behavior.py", "runtime/foresight.py", "runtime/unsealed.py"]
    bridge_imports = {
        module for module in imported_modules(runtime_root / "unsealed.py") if module.startswith("habitus.behavior")
    }
    assert bridge_imports <= {
        "habitus.behavior.fusion.store",
        "habitus.behavior.kinds.model",
        "habitus.behavior.kinds.store",
        "habitus.behavior.reduction.ledger",
        "habitus.behavior.reduction.pending",
    }


def test_the_facts_door_only_knows_about_time() -> None:
    """外部条件的事实门是一个只认时刻的可替换接缝：它不认识树、不认识视图，也不认识自己所在的包。

    与 ``calendar`` 同一条纪律：一旦某个提供者从门里读行为树或预测树，条件就不再是"外面的事实"，
    而变成又一个要跟着树一起重算的派生物——而它会被冻结进承诺，事后分不清哪个是哪个。
    """

    facts = SRC / "scene" / "facts.py"
    assert facts.is_file()
    assert imported_modules(facts) == {"__future__", "collections.abc", "dataclasses", "datetime", "typing"}


def test_foundation_depends_on_no_other_package() -> None:
    """``foundation`` 是各层共用的底：它一旦 import 任何领域包，那个包就成了所有人的传递依赖。

    ``foundation.text`` 现在被关联与判断两个面向模型的装配层共用，这条要钉住。
    """

    violations = sorted(
        f"{path.relative_to(SRC)}: {module}"
        for path in (SRC / "foundation").rglob("*.py")
        for module in imported_modules(path)
        if module.startswith("habitus.") and not module.startswith("habitus.foundation")
    )
    assert violations == []


def test_the_calendar_only_knows_about_dates() -> None:
    """当地日历是一个只认日期的可插拔零件：它不认识树、不认识视图，也不认识自己所在的包。

    钉住这条，是因为"这一天在当地是什么日子"一旦开始读行为树或情景树，它就不再是可替换的
    数据源，而变成又一个要跟着树一起重算的派生物。
    """

    calendar = SRC / "scene" / "calendar.py"
    assert calendar.is_file()
    assert imported_modules(calendar) == {"__future__", "datetime", "typing"}


def test_conversation_source_and_projection_do_not_depend_on_memory_or_behavior() -> None:
    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (SRC / "conversation").rglob("*.py")
        if imported_roots(path) & {"memory", "behavior", "runtime", "config", "integrations"}
    ]
    assert violations == []


def test_behavior_projection_reads_only_source_envelope_batch() -> None:
    projection_root = SRC / "conversation" / "projection" / "behavior"
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
        assert imported_roots(path).isdisjoint({"memory", "behavior", "runtime", "config"})
    assert "envelope.batch.messages" in source
    assert "ConversationMessageChunker" not in source
    assert "ConversationSegment" not in source
    assert "ConversationSummary" not in source
    assert "MemoryEditor" not in source


def test_memory_conversation_consumer_wraps_the_single_existing_enqueuer_chain() -> None:
    consumer_source = (
        SRC / "memory" / "workflow" / "conversation_consumer.py"
    ).read_text(encoding="utf-8")
    assert "self.enqueuer.append" in consumer_source
    assert "self.enqueuer.enqueue_ready_segments" in consumer_source
    assert "ConversationToolResultReducer(" not in consumer_source
    assert "ConversationMessageChunker(" not in consumer_source


def test_retired_behavior_does_not_extend_memory_kind() -> None:
    source = (SRC / "memory" / "model.py").read_text(encoding="utf-8")
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
        for path in (SRC / "memory" / "tree").rglob("*.py")
    )
    assert "behaviors" not in memory_tree_source
    assert not (SRC / "memory" / "schema" / "definitions" / "behavior.yaml").exists()


def test_every_package_entry_exports_only_things_that_exist() -> None:
    """``__all__`` 是这一层对外说"我给什么"。名字取不到就是幻觉 API。

    删掉一批实现之后最容易漏的就是它——``from habitus.scene import *`` 会当场抛 AttributeError，
    而按 ``__all__`` 生成文档或做 re-export 的工具会拿到一份并不存在的清单。ruff 的 F822 对包
    初始化文件默认豁免，pyright 的同名检查又看不穿 ``__getattr__`` 懒加载（``integrations`` 那
    两个包就是），所以这条只能按**运行时**判：``hasattr`` 会走 ``__getattr__``，懒加载照样过。
    """

    import importlib

    broken: list[str] = []
    for name in (
        "habitus.scene",
        "habitus.scene.views",
        "habitus.scene.concepts",
        "habitus.scene.hypotheses",
        "habitus.scene.occurrences",
        "habitus.scene.occurrences.mapper",
        "habitus.scene.ledger",
        "habitus.foresight",
        "habitus.foresight.judge",
        "habitus.foresight.ledger",
        "habitus.prediction",
        "habitus.integrations.http_api",
        "habitus.integrations.local_service",
    ):
        module = importlib.import_module(name)
        broken.extend(f"{name}.{item}" for item in getattr(module, "__all__", ()) if not hasattr(module, item))
    assert broken == []
