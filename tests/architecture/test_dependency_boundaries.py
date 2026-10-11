"""已经确认的新架构边界不得被兼容层或反向依赖破坏。"""

import ast
import subprocess
import sys
from functools import cache
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
    "series",
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

    只取 ``node.module`` 会让 ``from ..views.index import DayIndex`` 变成 ``views.index``，任何按
    ``habitus.`` 前缀判断的边界断言都放它过去（变异实测：在 scene 某个子包里加一条相对 import，scene 的
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
                    resolved = node.module
                else:
                    continue
            else:
                # level=1 是本包，level=2 是上一层，以此类推。
                parts = base.split(".")
                anchor = ".".join(parts[: len(parts) - (node.level - 1)]) if node.level > 1 else base
                resolved = f"{anchor}.{node.module}" if node.module else anchor
                modules.add(resolved)
            # ``from habitus.scene.relations import store`` 拿到的是子模块：只记包名会让白名单放过它（第四轮评审架构 4）。
            # 名字是仓库里真实存在的子模块的，一并记下。
            for alias in node.names:
                candidate = f"{resolved}.{alias.name}"
                if _is_module(candidate):
                    modules.add(candidate)
    return modules


def _is_module(name: str) -> bool:
    if not name.startswith("habitus."):
        return False
    relative = Path(*name.split(".")[1:])
    return (SRC / relative).with_suffix(".py").is_file() or (SRC / relative / "__init__.py").is_file()


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
    ("model_client", "behavior", "scene", "series", "prediction", "pre", "memory", "infrastructure", "foundation"),
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
    # behavior/fusion/config.py 的唯一出处解析）。唯一的例外是事件序列的读取入口
    # ``series/reader.py``：预测树与语义树每晚读的那一份序列从行为树读出，那是设计路径而不是泄漏，
    # 收在单个模块里由 test_the_behaviour_tree_is_read_into_the_series_in_one_place 守住。
    # ``scene``（语义关联层）是行为树的解释层，从行为树派生，允许 import behavior。
    reverse_dependency_violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in production_files()
        if path.relative_to(SRC).parts[0] not in {"behavior", "runtime", "scene"}
        and path.relative_to(SRC).as_posix() != "series/reader.py"
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

    # 模型调用收敛在受控触点：融合判断（语义生成）、词表（白天归类 / 每晚新增 / 定期拆改都经
    # ``kinds/calls.py`` 这一个文件）、行为树语义面生成。
    # 存储、派生与落盘路径仍不得渗入。
    model_callers = sorted(
        str(path.relative_to(SRC))
        for path in behavior_root.rglob("*.py")
        if "model_client" in imported_roots(path)
    )
    assert model_callers == [
        "behavior/fusion/lanes/session/service.py",
        "behavior/fusion/service.py",
        "behavior/kinds/calls.py",
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
        # 会话 lane：service 是它的模型触点，lane 把它和写判断串起来，包根转手两者。
        "habitus.behavior.fusion.lanes.session",
        "habitus.behavior.fusion.lanes.session.service",
        "habitus.behavior.fusion.lanes.session.lane",
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
    """归约层的 LLM 触点只有 runner 与词表的两步（白天归类 kinds_step、整理活 kinds_jobs）；纯函数模块连传递依赖都不许有。

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

    # 归约的模型触点：runner 编排，词表的白天归类（kinds_step）与整理活（kinds_jobs）；把版本落到树上的
    # kinds_migration 是纯确定性的，不许碰模型。
    allowed = {
        "habitus.behavior.reduction",
        "habitus.behavior.reduction.runner",
        "habitus.behavior.reduction.kinds_step",
        "habitus.behavior.reduction.kinds_jobs",
    }
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


def test_prediction_never_reads_the_behaviour_tree() -> None:
    """时间预测树一个模块都不**传递性**触达行为树：它读的是第 N 晚的事件序列（``habitus.series``），
    行为树怎么读只写在 ``series.reader`` 一处、与语义树共用（语义树新方案第四节）。

    全包保持零依赖的纯计算，估计器的正确性才能在纯函数层穷举验证（见 ``TODO(PRED-TREE-001)``）。
    查传递闭包而不是"谁写了 import"：共享一个数据类型就足以让全包连坐，而字面检查看不见。
    """

    assert (SRC / "prediction").is_dir()
    assert _prediction_modules_reaching("habitus.behavior") == set()
    assert _prediction_modules_reaching("habitus.series.reader") == set()
    # prediction 不读 scene（原来的 ``prediction/scene_source.py`` 已删——它零调用方，留着只会让两个包互指）。
    # 新语义树要对照预测树的期望时，由组合根注入 callable，scene 也不直接读 prediction。
    assert _prediction_modules_reaching("habitus.scene") == set()


def test_the_behaviour_tree_is_read_into_the_series_in_one_place() -> None:
    """事件序列：值在 ``series.model``（零依赖），读行为树在 ``series.reader``，只有组合根用后者。

    两棵树读的必须是同一份记录、同一套读法（第二轮评审 D5：「待定」两处各写一遍写反了）。所以：
    - ``series`` 包里只有 ``reader`` import behavior，包根不转手它（两棵树 import 包根不会连带行为树与词表）；
    - 生产代码里只有 ``runtime`` import ``series.reader``——预测树、语义树、预测层都只认纯值。
    """

    series_root = SRC / "series"
    readers = sorted(
        path.relative_to(SRC).as_posix() for path in series_root.rglob("*.py") if "behavior" in imported_roots(path)
    )
    assert readers == ["series/reader.py"]
    # 只有标准库：空白作废的规则（``disproves``）用到 bisect 与 collections.abc，仍是纯函数
    assert imported_roots(series_root / "model.py") <= {"__future__", "bisect", "collections", "dataclasses", "datetime", "enum"}
    assert "habitus.series.reader" not in imported_modules(series_root / "__init__.py")
    importers = sorted(
        path.relative_to(SRC).parts[0]
        for path in production_files()
        if "habitus.series.reader" in imported_modules(path) and path.relative_to(SRC).parts[0] != "series"
    )
    assert set(importers) == {"runtime"}


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


@cache
def _repository_graph() -> dict[str, Path]:
    """全仓模块名 → 文件。包写成包名（``__init__`` 去掉）。"""

    graph: dict[str, Path] = {}
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
        graph[f"habitus.{relative}".removesuffix(".__init__")] = path
    return graph


def _with_parents(name: str) -> set[str]:
    """导入 ``a.b.c`` 时 Python 先执行 ``a``、``a.b`` 的 ``__init__``：这些父包也算被触达。"""

    parts = name.split(".")
    return {".".join(parts[:index]) for index in range(1, len(parts) + 1)}


def _repository_closure(module: str) -> set[str]:
    """从 ``module`` 出发**全仓**传递闭包，含父包 ``__init__`` 的边。

    只在一个包里走、又不算父包的闭包看不见这条路：``behavior/schema/fields.py`` import ``behavior.kinds.ids``，
    导入它会先执行 ``behavior/kinds/__init__``——那里曾经转手模型调用模块，于是行为树、预测树、语义树一导入就
    把模型客户端带进来，包内闭包全绿（第二轮评审 C1）。
    """

    graph = _repository_graph()
    reached: set[str] = set()
    stack = [module, *_with_parents(module)]
    while stack:
        current = stack.pop()
        if current in reached:
            continue
        reached.add(current)
        if current in graph:
            stack.extend(parent for name in imported_modules(graph[current]) for parent in _with_parents(name))
    return reached


#: 这些读侧入口无论经哪条路都不许触达模型调用（语义树里调模型的触点不从包根导出，这里不列它们）。
MODEL_FREE_ENTRIES = (
    "habitus.behavior.tree",
    "habitus.behavior.schema",
    "habitus.behavior.kinds.ids",
    "habitus.prediction.source",
    "habitus.prediction.builder",
    "habitus.scene.concepts",
    "habitus.scene.occurrences.store",
    "habitus.series",
    "habitus.series.reader",
)
MODEL_MODULES = ("habitus.model_client", "habitus.behavior.kinds.calls")


@pytest.mark.parametrize("entry", MODEL_FREE_ENTRIES)
def test_read_side_entries_never_reach_the_model_even_through_parent_packages(entry: str) -> None:
    reached = _repository_closure(entry)
    leaks = sorted(name for name in reached if any(name == root or name.startswith(f"{root}.") for root in MODEL_MODULES))
    assert leaks == [], f"{entry} reaches {leaks}"


def test_importing_read_side_entries_does_not_load_the_model_client() -> None:
    """静态闭包之外再实测一次：子进程里真导入，看 ``sys.modules``（静态分析漏掉的动态 import 也逃不掉）。"""

    script = (
        "import sys\n"
        + "".join(f"import {entry}\n" for entry in MODEL_FREE_ENTRIES)
        + "print('\\n'.join(sorted(m for m in sys.modules if m.startswith(('habitus.model_client', 'habitus.behavior.kinds.calls')))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


def scene_reach_of(graph: dict[str, Path], branch: str) -> set[str]:
    """scene 里一支（传递性）触达的 scene 模块。"""

    prefix = f"habitus.scene.{branch}"
    return {
        name
        for module in graph
        if module == prefix or module.startswith(f"{prefix}.")
        for name in _scene_closure(module, graph)
        if name.startswith("habitus.scene.")
    }


def test_scene_tree_is_a_pure_derivation_of_the_behaviour_tree() -> None:
    """语义关联层只读行为树、只用基础设施与模型契约；不知道预测树、记忆树与组合根。

    语义树按新方案（``13-语义树新方案``，裁定 22）重建：旧的假设、账本、关系投影、闭环、安慰剂、继承已删，
    现有 ``concepts / occurrences`` 两支加行为树读侧 ``views``。**全部按传递闭包判**：

    - scene 里触达 ``model_client`` 的只有模型触点（写概念、映射器、先验、候选调节条件）与它们共用的调模型那一段
      （``scene.llm``），而且它们都不从包根导出；
    - 没有任何模块触达 prediction——节律、复发间隔都由组合根注入；
    - 引用单向：``concepts`` 不引用别支；``occurrences`` 只引用 ``concepts``；``relations``（关系检验与状态）只引用前两支，
      不调模型；只有 ``relations.store`` 碰存储，其余（检验、折叠、编码）是纯算法，包根也不转手存储；三支不触达 ``views``
      也不触达包根（包根导出 views）——读侧可整个删掉重建，底层引用它就删不掉了；
    - 只读行为树：不触达写侧（编辑器、归约、融合、词表、观测、语义层）——"行为树一个字不动"要由边界保证。
    """

    scene_root = SRC / "scene"
    assert scene_root.is_dir()
    graph = _scene_module_graph()
    forbidden_roots = ("habitus.memory", "habitus.config", "habitus.runtime", "habitus.conversation", "habitus.integrations", "habitus.pre")
    assert _scene_modules_reaching(graph, *forbidden_roots) == []
    assert _scene_modules_reaching(graph, "habitus.prediction") == []
    # 语义树读事件序列的纯值（``habitus.series``），不读行为树的那一半（``series.reader`` 只给组合根）。
    assert _scene_modules_reaching(graph, "habitus.series.reader") == []
    # 模型触点一个都不许多：写概念（①）、映射器、先验、候选调节条件，加上它们共用的调模型那一段。
    # **都不从包根导出**——包根一 import 它们，任何 ``from habitus.scene.concepts import ConceptSet`` 都把
    # model_client 拖进来，这条边界就只剩一句注释了。
    assert _scene_modules_reaching(graph, "habitus.model_client") == [
        "habitus.scene.advice.conditions",
        "habitus.scene.advice.prior",
        "habitus.scene.concepts.author",
        "habitus.scene.llm",
        "habitus.scene.occurrences.mapper",
    ]
    # 建议（先验、候选条件）只引用概念与关系检验里纯的那几块：看不到命中、看不到关系表，也碰不到行为树
    assert scene_reach_of(graph, "advice") <= {
        name
        for name in graph
        if name.startswith(("habitus.scene.concepts", "habitus.scene.advice", "habitus.scene.relations"))
        or name in {"habitus.scene.llm", "habitus.scene.storage", "habitus.scene.codec", "habitus.scene.occurrences.model"}
    } - {"habitus.scene.relations.store"}
    assert "habitus.scene.advice.prior" not in imported_modules(scene_root / "advice" / "__init__.py")
    assert "habitus.scene.advice.conditions" not in imported_modules(scene_root / "advice" / "__init__.py")
    # 同步词表、词表口是纯算法（裁定 20）：它们在图里、却不在上面之列。
    for pure in ("habitus.scene.concepts.sync", "habitus.scene.concepts.catalog"):
        assert pure in graph, pure
    # 读侧只读行为树。views 里除 ``__init__`` 外不得 import 包根或 ``views`` 包根——投影不能倒灌。
    detours = sorted(
        f"{path.relative_to(SRC)}: {module}"
        for path in (scene_root / "views").rglob("*.py")
        if path.name != "__init__.py"
        for module in imported_modules(path)
        if module in {"habitus.scene", "habitus.scene.views"}
    )
    assert detours == []
    branches = ("concepts", "occurrences", "relations")
    shared = {"habitus.scene.codec", "habitus.scene.storage", "habitus.scene.llm"}

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
    assert scene_reach("occurrences") <= branch_names("concepts", "occurrences"), scene_reach("occurrences")
    assert scene_reach("relations") <= branch_names("concepts", "occurrences", "relations"), scene_reach("relations")
    impure = {
        name
        for module in branch_modules("relations")
        if module != "habitus.scene.relations.store"
        for name in _scene_closure(module, graph)
        if name.startswith(("habitus.model_client", "habitus.infrastructure", "habitus.scene.storage", "habitus.scene.llm"))
        or name in {"habitus.scene.occurrences.store", "habitus.scene.concepts.store", "habitus.scene.occurrences.mapper"}
    }
    assert impure == set(), impure
    assert not any(
        name.startswith(("habitus.model_client", "habitus.scene.llm"))
        for name in _scene_closure("habitus.scene.relations.store", graph)
    )
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
    # 旧树不许回来：目录名与模块名一起钉（2026-09-26 删的旧语义树，2026-10-07 删的假设 / 账本 / 关系投影）。
    for gone in ("association", "regularity", "backlog.py", "model.py", "uri.py", "hypotheses", "ledger"):
        assert not (scene_root / gone).exists(), gone
    for gone in ("gloss", "relations", "accounts", "materialize", "placebo", "people", "strength", "fulfilment"):
        assert not (scene_root / "views" / f"{gone}.py").exists(), gone
    # views 只剩行为树读侧：一个模块都不许触达 concepts / occurrences（关系表另建，不住在这里）。
    assert [
        module
        for module in graph
        if module.startswith("habitus.scene.views")
        and any(name.startswith(("habitus.scene.concepts", "habitus.scene.occurrences")) for name in _scene_closure(module, graph))
    ] == []


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
        "foresight/relations.py",
        "foresight/render.py",
    ]
    # 而且只经读口进：``habitus.scene.views``（或包根），外加关系与概念的**对象**（``scene.relations`` 包根、
    # ``scene.concepts.model``，都是纯的）——关系表由组合根从存储读好交进来（``runtime.scene_relations``）。
    # 新语义树的存储与写侧都不许直接碰：绕过读口就等于在预测层里重新长出一条读语义树的路。
    beyond_the_views = sorted(
        f"{path.relative_to(SRC)}: {module}"
        for path in root.rglob("*.py")
        for module in imported_modules(path)
        if module.startswith("habitus.scene")
        and module
        not in {"habitus.scene", "habitus.scene.views", "habitus.scene.relations", "habitus.scene.concepts.model"}
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
    # 未封口那座桥。桥只许拿"哪些判断还没归约"这一个事实（``reduction.pending``）与账本、归约的白天归类，不许自己
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
    assert inside_behavior == [
        "runtime/behavior.py",
        "runtime/foresight.py",
        "runtime/session_lane.py",
        "runtime/unsealed.py",
    ]
    bridge_imports = {
        module for module in imported_modules(runtime_root / "unsealed.py") if module.startswith("habitus.behavior")
    }
    # 归类用归约自己的白天归类（``KindStamping`` + 它的请求形状与链内容），不另起一套口径。
    assert bridge_imports <= {
        "habitus.behavior.fusion.store",
        "habitus.behavior.kinds.classify",
        "habitus.behavior.reduction.kinds_step",
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


def test_conversation_never_depends_on_memory_or_behavior() -> None:
    """会话源连同各消费者留在它这边的回执，都不认识记忆与行为。"""

    violations = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (SRC / "conversation").rglob("*.py")
        if imported_roots(path) & {"memory", "behavior", "runtime", "config", "integrations"}
    ]
    assert violations == []
    # 行为投影已被会话 lane 取代（裁定 31），不许再长回来。
    assert not (SRC / "conversation" / "projection").exists()


def test_the_session_lane_meets_the_conversation_source_only_in_the_composition_root() -> None:
    """会话 lane 住在行为侧、不认识会话源；会话源也不认识它。两边只在 ``runtime/session_lane.py`` 这一座桥上相遇。"""

    lane_root = SRC / "behavior" / "fusion" / "lanes" / "session"
    assert {path.name for path in lane_root.glob("*.py")} == {
        "__init__.py",
        "config.py",
        "lane.py",
        "material.py",
        "model.py",
        "prompt.py",
        "protocol.py",
        "recorder.py",
        "redaction.py",
        "service.py",
    }
    for path in lane_root.glob("*.py"):
        assert imported_roots(path).isdisjoint({"conversation", "pre", "memory", "runtime", "config"})
    # 本条 lane 只有 service 调模型；取材料、抹密钥、校验、写判断都是确定性的。
    assert sorted(path.name for path in lane_root.glob("*.py") if "model_client" in imported_roots(path)) == [
        "service.py"
    ]
    bridges = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "__pycache__" not in path.parts
        and any(module.startswith("habitus.behavior.fusion.lanes.session") for module in imported_modules(path))
        and path.relative_to(SRC).parts[0] != "behavior"
    )
    assert bridges == ["runtime/behavior.py", "runtime/session_lane.py"]
    # 逐帧融合只认得会话 lane 的协议名（入队时跳过它的凭据），不碰它的其余部分。
    assert {
        module
        for module in imported_modules(SRC / "behavior" / "fusion" / "enqueue.py")
        if module.startswith("habitus.behavior.fusion.lanes")
    } == {"habitus.behavior.fusion.lanes.session.protocol"}


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
        "habitus.scene.occurrences",
        "habitus.scene.occurrences.mapper",
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


def test_the_vocabulary_reaches_the_model_only_through_its_one_caller() -> None:
    """基础词表包里只有 ``calls.py`` 直接 import 模型客户端；值对象、变更日志、存储、待定池、只读口、定时这些
    确定性模块连传递依赖都不许碰模型（裁定 6 的"词表只管词表"落到依赖上）。"""

    kinds_root = SRC / "behavior" / "kinds"
    graph = {
        (
            "habitus.behavior.kinds."
            + path.relative_to(kinds_root).with_suffix("").as_posix().replace("/", ".")
        ).removesuffix(".__init__"): path
        for path in kinds_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    direct = sorted(
        module
        for module, path in graph.items()
        if any(name == "habitus.model_client" or name.startswith("habitus.model_client.") for name in imported_modules(path))
    )
    assert direct == ["habitus.behavior.kinds.calls"]

    def reaches(module: str, seen: set[str]) -> bool:
        if module in seen or module not in graph:
            return False
        seen.add(module)
        imported = imported_modules(graph[module])
        if "habitus.behavior.kinds.calls" in imported:
            return True
        return any(reaches(name, seen) for name in imported if name in graph)

    deterministic = (
        "ids", "model", "changes", "codec", "store", "pending", "render", "schedule", "reader",
        "config", "revision.facts", "revision.proposals", "revision.evidence",
    )
    missing = [name for name in deterministic if f"habitus.behavior.kinds.{name}" not in graph]
    assert missing == [], f"清单里登记的模块不存在（删改模块后要同步这里）: {missing}"
    leaking = [name for name in deterministic if reaches(f"habitus.behavior.kinds.{name}", set())]
    assert leaking == [], f"这些词表确定性模块传递性地依赖了模型: {leaking}"
    # 词表不读树、不认识归约：落树由归约侧执行，词表只交"下一版"这份事实。
    upward = sorted(
        f"{module} -> {name}"
        for module, path in graph.items()
        for name in imported_modules(path)
        if name.startswith(("habitus.behavior.tree", "habitus.behavior.reduction", "habitus.behavior.editor"))
    )
    assert upward == []
