"""代码尺寸闸与非边界模块的运行时类型检查闸（W15 第 1 步、W16 防复发）。

为什么有这两道闸：函数约 60 行、类约 250 行就要拆的规范早就有，"运行时类型检查只放在边界"也已确认，
但架构测试原先只管依赖方向，不管尺寸和类型检查，所以一直没有闸。这里只装闸、不改代码：现有超标与
内部模块现有的类型检查写进基线，基线只许跟着代码收紧，新写的代码自动受约束。真正的拆分与删检查
是后续阶段的事。

口径（与共享说明 P0-2 第 1 段一致，实现与基线用同一套）：

- 范围：``habitus/`` 下全部 ``.py``，包括 ``integrations`` 和 ``benchmark``，不含路径中任一段是
  ``__pycache__`` 或 ``node_modules`` 的文件（``habitus/integrations/openclaw/node_modules/`` 是本地 npm
  装出来、被 ``.gitignore`` 忽略的目录，CI 的检出里没有；不排除的话本地装了带 ``.py`` 的 npm 包会让本地
  和 CI 结果不一样）。不复用 ``test_dependency_boundaries.PRODUCTION_ROOTS``，它不含 ``integrations``
  和 ``benchmark``。
- 函数长度：``FunctionDef`` / ``AsyncFunctionDef`` 的 ``end_lineno - lineno + 1``（``lineno`` 是 ``def``
  那一行，装饰器不算）；嵌套函数单独再算一次。
- 类长度：``ClassDef`` 的 ``end_lineno - lineno + 1``。
- 文件长度：文本行数。
- 超标：函数 > 60、类 > 250、文件 > 600。
- 身份：函数和类用"相对仓库根的路径 + ``::`` + 嵌套限定名"，例如
  ``habitus/runtime/assembly.py::build_runtime``、``habitus/runtime/runtime.py::Runtime.start``；嵌套在
  函数里的函数用 ``.`` 直接连接、不写 ``<locals>``，例如
  ``habitus/integrations/local_service/doctor.py::_deep_checks.probe``；文件用路径。
- 类型检查：一个 ``if`` 语句，条件里有对 ``isinstance`` 的调用（``ast.Call``，函数名是 ``isinstance``），
  并且 ``if`` 的主体只有一条语句、是 ``raise``（抛什么异常都算），没有 ``else`` 也照算；``elif`` 分支同样
  算一条；函数名写成属性形式（如 ``builtins.isinstance(...)``，即 ``ast.Attribute`` 且属性名为
  ``isinstance``）也算。``habitus.foundation.validation`` 的类型判定（``TYPE_PREDICATES``，如 ``is_int``）与
  ``isinstance`` 同等计数：内部模块不能靠把 ``isinstance`` 换成判定函数躲过这道闸（W16）。
- 边界模块：登记在 ``tests/architecture/boundary_modules.txt`` 里的模块（路径相对 ``habitus/``）。没登记的
  模块一律按内部算。登记了不存在的路径直接失败，不允许留过期条目。

基线：``baselines/code_shape.json`` 按 ``functions`` / ``classes`` / ``files`` 三组记录现有超标项的当前长度；
``baselines/runtime_type_checks.json`` 记录每个内部模块现有的类型检查条数。两份都只许减少：新增超标或
变大失败；变小但仍有记录必须把基线改成新的数；降到上限内或降到 0 必须删掉那一条；基线里的身份不存在
也必须删掉。报错直接给出可以替换进基线的 JSON 行，不另写生成脚本。
"""

import ast
import json
from collections import Counter
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

from habitus.foundation.validation import TYPE_PREDICATES

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC = REPOSITORY_ROOT / "habitus"
ARCHITECTURE_DIR = Path(__file__).resolve().parent
BOUNDARY_REGISTRY = ARCHITECTURE_DIR / "boundary_modules.txt"
SHAPE_BASELINE = ARCHITECTURE_DIR / "baselines" / "code_shape.json"
TYPE_CHECK_BASELINE = ARCHITECTURE_DIR / "baselines" / "runtime_type_checks.json"
EXCLUDED_PATH_PARTS = frozenset({"__pycache__", "node_modules"})
SHAPE_LIMITS = {"functions": 60, "classes": 250, "files": 600}
SHAPE_NOUNS = {"functions": "函数", "classes": "类", "files": "文件"}


def source_files() -> tuple[Path, ...]:
    """``habitus/`` 下全部 ``.py``，排除 ``__pycache__`` 与 ``node_modules``。"""

    return tuple(
        sorted(
            path
            for path in SRC.rglob("*.py")
            if not EXCLUDED_PATH_PARTS.intersection(path.relative_to(REPOSITORY_ROOT).parts)
        )
    )


def _identity(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


@cache
def _parse(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


def _definitions(node: ast.AST, prefix: str) -> Iterator[tuple[str, str, int]]:
    """递归给出 ``(组, 限定名, 长度)``；控制流不进限定名，但会往里找嵌套定义。"""

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualified = f"{prefix}.{child.name}" if prefix else child.name
            group = "classes" if isinstance(child, ast.ClassDef) else "functions"
            assert child.end_lineno is not None
            yield group, qualified, child.end_lineno - child.lineno + 1
            yield from _definitions(child, qualified)
        else:
            yield from _definitions(child, prefix)


@cache
def measured_shapes() -> dict[str, list[tuple[str, int]]]:
    """全部函数、类、文件的 ``(身份, 长度)``，不论是否超标。"""

    shapes: dict[str, list[tuple[str, int]]] = {"functions": [], "classes": [], "files": []}
    for path in source_files():
        text, tree = _parse(path)
        file_identity = _identity(path)
        shapes["files"].append((file_identity, len(text.splitlines())))
        for group, qualified, length in _definitions(tree, ""):
            shapes[group].append((f"{file_identity}::{qualified}", length))
    return shapes


def _check_names(tree: ast.AST) -> frozenset[str]:
    """``isinstance``、类型判定的原名，以及本模块里给判定起的别名（``import is_int as check``）。"""

    aliases = {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "habitus.foundation.validation"
        for alias in node.names
        if alias.name in TYPE_PREDICATES and alias.asname
    }
    return frozenset({"isinstance", *TYPE_PREDICATES, *aliases})


def _is_isinstance_call(node: ast.AST, names: frozenset[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    return (isinstance(function, ast.Name) and function.id in names) or (
        isinstance(function, ast.Attribute) and function.attr in names
    )


def count_type_checks(tree: ast.AST) -> int:
    """按口径数类型检查：条件里有 isinstance（或类型判定）调用、主体只有一条 raise 的 ``if``（``elif`` 也是 ``ast.If``）。"""

    names = _check_names(tree)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Raise)
        and any(_is_isinstance_call(part, names) for part in ast.walk(node.test))
    )


def boundary_entries() -> list[str]:
    """``boundary_modules.txt`` 去掉注释行和空行后的条目（相对 ``habitus/``）。"""

    return [
        line.strip()
        for line in BOUNDARY_REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def measured_internal_type_checks() -> dict[str, int]:
    """每个内部（未登记为边界）模块的类型检查条数，只列条数大于 0 的。"""

    boundary = {f"habitus/{entry}" for entry in boundary_entries()}
    counts: dict[str, int] = {}
    for path in source_files():
        identity = _identity(path)
        if identity in boundary:
            continue
        count = count_type_checks(_parse(path)[1])
        if count:
            counts[identity] = count
    return counts


def _load(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.relative_to(REPOSITORY_ROOT)} 顶层必须是 JSON 对象"
    return data


def _line(identity: str, value: int) -> str:
    """基线文件里这一行的内容（不含行尾逗号，最后一项没有逗号也能原样查找替换）。"""

    return f"{json.dumps(identity, ensure_ascii=False)}: {value}"


def compare_with_baseline(
    current: dict[str, int],
    baseline: dict[str, int],
    *,
    limit: int,
    noun: str,
    unit: str,
    baseline_name: str,
    exists: set[str],
    new_label: str,
    new_advice: str,
    cleared: str,
) -> list[str]:
    """把当前超标集合与基线逐条比对，返回排序后的违规说明。

    ``current`` 只含超过上限的项；``exists`` 是当前存在的全部身份，用来区分"降到上限内"和"已不存在"。
    """

    violations: list[str] = []
    for identity, value in current.items():
        if identity not in baseline:
            violations.append(f"{identity}: {new_label}，当前 {value} {unit}，上限 {limit}。{new_advice}")
        elif value > baseline[identity]:
            violations.append(
                f"{identity}: {noun}比基线变大，当前 {value} {unit}，基线 {baseline[identity]}，上限 {limit}。"
                f"基线只许减少：把它改回不大于 {baseline[identity]}。"
            )
        elif value < baseline[identity]:
            violations.append(
                f"{identity}: {noun}比基线变小，当前 {value} {unit}，基线 {baseline[identity]}，上限 {limit}。"
                f"把 {baseline_name} 里的 {_line(identity, baseline[identity])} 替换为 {_line(identity, value)}"
            )
    for identity, value in baseline.items():
        if identity in current:
            continue
        if identity in exists:
            violations.append(
                f"{identity}: {cleared}（基线 {value}）。从 {baseline_name} 里删掉这一条：{_line(identity, value)}"
            )
        else:
            violations.append(
                f"{identity}: 基线里的身份已不存在（被删除或改名）。从 {baseline_name} 里删掉这一条：{_line(identity, value)}"
            )
    return sorted(violations)


def _int_mapping(data: object, where: str) -> dict[str, int]:
    assert isinstance(data, dict), f"{where} 必须是 JSON 对象"
    assert all(isinstance(key, str) and type(value) is int for key, value in data.items()), (
        f"{where} 的每一项必须是 身份 → 整数"
    )
    return dict(data)


@pytest.mark.parametrize("group", ("functions", "classes", "files"))
def test_code_shape_never_grows_beyond_the_baseline(group: str) -> None:
    """函数 > 60 行、类 > 250 行、文件 > 600 行：存量进基线只许减少，新增与变大一律失败。"""

    baseline_file = _load(SHAPE_BASELINE)
    assert set(baseline_file) == set(SHAPE_LIMITS), "code_shape.json 顶层必须正好是 functions/classes/files 三组"
    baseline = _int_mapping(baseline_file[group], f"code_shape.json 的 {group}")
    limit = SHAPE_LIMITS[group]
    measured = measured_shapes()[group]

    # 同一身份出现两次（例如同名重定义）时基线无法区分两者，超标的必须先改成唯一身份。
    duplicated = Counter(identity for identity, _ in measured)
    ambiguous = sorted(
        {
            f"{identity}: 超标{SHAPE_NOUNS[group]}的身份重复 {duplicated[identity]} 次，基线无法区分，先改名使其唯一"
            for identity, length in measured
            if length > limit and duplicated[identity] > 1
        }
    )
    assert ambiguous == [], "\n".join(ambiguous)

    current = {identity: length for identity, length in measured if length > limit}
    violations = compare_with_baseline(
        current,
        baseline,
        limit=limit,
        noun=SHAPE_NOUNS[group],
        unit="行",
        baseline_name=f"tests/architecture/baselines/code_shape.json 的 {group} 组",
        exists={identity for identity, _ in measured},
        new_label=f"新增超标{SHAPE_NOUNS[group]}",
        new_advice=f"拆到 {limit} 行以内；基线只记录存量，不接受新增条目。",
        cleared=f"已降到上限 {limit} 行以内",
    )
    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("if not isinstance(x, int):\n    raise TypeError", 1),
        ("if not is_int(x) or x < 0:\n    raise ValueError", 1),
        ("if not validation.is_nonblank_str(x):\n    raise ValueError", 1),
        ("if x < 0:\n    raise ValueError", 0),
        ("if not is_int(x):\n    x = 0", 0),
        ("from habitus.foundation.validation import is_int as check\nif not check(x):\n    raise ValueError", 1),
    ],
)
def test_type_predicates_count_like_isinstance(source: str, expected: int) -> None:
    """内部模块把 ``isinstance`` 换成 ``foundation.validation`` 的判定函数，闸照样数到。"""

    assert count_type_checks(ast.parse(source)) == expected


def test_every_validation_predicate_is_counted_by_the_gate() -> None:
    """``foundation.validation`` 里每个 ``is_*`` 判定都在 ``TYPE_PREDICATES`` 里：新增判定忘了登记，闸就数不到它。"""

    _, tree = _parse(SRC / "foundation" / "validation.py")
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("is_")}
    assert defined == set(TYPE_PREDICATES)


def test_internal_modules_do_not_add_runtime_type_checks() -> None:
    """运行时类型检查只放在边界（W16）：内部模块的存量进基线只许减少，边界模块不受这道闸约束。

    没登记在 ``boundary_modules.txt`` 的模块一律按内部算；一个模块登记为边界后，它在基线里的条目也要删掉。
    """

    baseline = _int_mapping(_load(TYPE_CHECK_BASELINE), "runtime_type_checks.json")
    current = measured_internal_type_checks()
    violations = compare_with_baseline(
        current,
        baseline,
        limit=0,
        noun="内部模块的类型检查",
        unit="条",
        baseline_name="tests/architecture/baselines/runtime_type_checks.json",
        exists={_identity(path) for path in source_files()},
        new_label="内部模块新增类型检查",
        new_advice=(
            "内部模块不许新增运行时类型检查：类型在边界解码时校验；"
            "若该模块确实直接接收外部数据，登记进 boundary_modules.txt 并在提交说明里写清。基线不接受新增或调大。"
        ),
        cleared="当前没有计入内部模块的类型检查（已删光或已登记为边界模块）",
    )
    assert violations == [], "\n".join(violations)


def test_boundary_registry_lists_only_existing_unique_modules() -> None:
    """边界登记不允许留过期条目：登记了不存在的路径或重复登记都失败。"""

    entries = boundary_entries()
    missing = sorted(
        f"{entry}: 登记的边界模块不存在（habitus/{entry}），从 boundary_modules.txt 删掉"
        for entry in entries
        if not (SRC / entry).is_file()
    )
    duplicated = sorted(f"{entry}: 重复登记 {count} 次" for entry, count in Counter(entries).items() if count > 1)
    assert missing + duplicated == [], "\n".join(missing + duplicated)


def canonical_json(data: object) -> str:
    """基线文件的规范写法：键排序、两格缩进、每条一行、非 ASCII 原样、末尾一个换行。"""

    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("baseline", (SHAPE_BASELINE, TYPE_CHECK_BASELINE), ids=lambda path: path.name)
def test_baselines_keep_the_canonical_one_entry_per_line_format(baseline: Path) -> None:
    """基线靠 diff 审：手改成一行或打乱顺序会让每次收紧的改动看不清，所以文件必须保持规范写法。"""

    text = baseline.read_text(encoding="utf-8")
    expected = canonical_json(json.loads(text))
    assert text == expected, (
        f"{baseline.relative_to(REPOSITORY_ROOT)} 不是规范格式（键排序、两格缩进、每条一行、末尾一个换行）。"
        f"把整个文件替换为：\n{expected}"
    )
