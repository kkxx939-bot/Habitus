"""超大工具结果的确定性关键信息抽取（不调用模型）。

按内容分类、不按工具名：``CONTENT_KINDS`` 一张表依次认空、类二进制、JSON、XML、TSV / CSV、YAML、代码、纯文本，
认成哪一类就按那一类抽取；工具名只写进说明行。全文按位置分窗流式扫描，不把每行物化成对象：内存里只留开头、
结尾、诊断行（带上下文）与少量计数；单次正则调用只扫一个窗口，在工作线程里跑时不会长时间占住 GIL。
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any

import yaml

from habitus.foundation.integrity import canonical_json

#: 行号前缀最多去几层：读到的文件本身又是带行号的转储时有两层。
_MAX_GUTTER_LEVELS = 2
#: 一层读文件工具输出的行号前缀（``12\t`` 或 ``12→``）。
_GUTTER_PREFIX = r"[ \t]*\d+(?:\t|→)"
#: 至多 ``_MAX_GUTTER_LEVELS`` 层行号前缀，与分类时去前缀同一口径。
_GUTTERS = rf"(?:{_GUTTER_PREFIX}){{0,{_MAX_GUTTER_LEVELS}}}"
#: 行首：可带行号前缀。
_LINE_START = rf"^{_GUTTERS}"


@dataclass(frozen=True)
class DiagnosticRule:
    """一类诊断行：名称与按整行匹配的行模式（多行模式编译，大小写写在模式里）。"""

    kind: str
    pattern: re.Pattern[str]


def _rule(kind: str, pattern: str) -> DiagnosticRule:
    return DiagnosticRule(kind, re.compile(pattern, re.MULTILINE))


#: 诊断行规则表（只用于纯文本类）。命中任一条的行连同前后少量上下文优先保留。
DIAGNOSTIC_RULES: tuple[DiagnosticRule, ...] = (
    _rule("错误", r"(?i:\berrors?\b|\berr!)|错误|报错"),
    # 小写 warn 常是样式名或标识符（--warn、.warn），只认全大写的日志级别 WARN 与 warning。
    _rule("警告", r"(?i:\bwarnings?\b)|\bWARN\b|警告"),
    # 驼峰异常类名独占行首（KeyError: …、java.lang.NullPointerException: …、pytest 的 E   AssertionError）。
    _rule(
        "异常",
        r"(?i:\bexceptions?\b)|异常"
        rf"|{_LINE_START}[ \t]*(?:E[ \t]+)?(?i:caused by:[ \t]*)?"
        r"(?:[A-Za-z_$][\w$]*\.)*[A-Z][\w$]*(?:Error|Exception)(?=:|[ \t\r]*$)",
    ),
    _rule(
        "栈",
        rf"{_LINE_START}[ \t]*(?i:traceback)\b"
        rf"|{_LINE_START}[ \t]*(?i:file) \"[^\"\n]+\", (?i:line) \d+"
        rf"|{_LINE_START}[ \t]+(?i:at) \S[^\n]*:\d+(?::\d+)?\)?[ \t\r]*$"
        rf"|{_LINE_START}[ \t]*goroutine \d+ \["
        rf"|{_LINE_START}[ \t]*#\d+[ \t]+0x[0-9a-fA-F]+",
    ),
    _rule("失败", r"(?i:\bfail(?:s|ed|ure|ures|ing)?\b)|失败"),
    _rule("致命", r"(?i:\b(?:fatal|panic(?:ked)?|abort(?:ed)?|segmentation fault)\b)|致命|崩溃"),
    _rule(
        "非零退出码",
        r"(?:(?i:\bexit(?:ed)?(?: with)?[ _]?(?:code|status)|\breturn ?code|\breturncode)|退出码|返回码)"
        r"[^\d\n]{0,3}(?:-\d+|[1-9]\d*)",
    ),
)
#: 诊断候选：上面各条规则的超集，一遍扫全文找候选行，再在候选行上逐条规则核对（比各规则合成一条快数倍）。
#: 平时在转成小写的窗口上用不带词界、不分大小写的 ``_LOWERED_CANDIDATE`` 找（最快）；小写会改变长度的窗口退回这一条。
_LOWERED_CANDIDATE = re.compile(
    r"err|warn|exception|fail|fatal|panic|abort|segmentation|exit|return ?code|traceback"
    r"|错误|报错|警告|异常|失败|致命|崩溃|退出码|返回码"
    rf"|{_LINE_START}[ \t]*(?:file \"|at |goroutine |#\d)",
    re.MULTILINE,
)
_DIAGNOSTIC_CANDIDATE = re.compile(
    r"\b(?i:err|warn|exception|fail|fatal|panic|abort|segmentation|exit|return ?code|returncode|traceback)"
    r"|错误|报错|警告|异常|失败|致命|崩溃|退出码|返回码"
    rf"|{_LINE_START}[ \t]*(?:(?i:file) \"|(?i:at) |goroutine |#\d|(?:E[ \t]+)?(?i:caused by:[ \t]*)?[\w$.]*(?:Error|Exception))",
    re.MULTILINE,
)

#: 编码样长串（base64 等）：只从串的开头匹配（负向后顾），整体对输入长度线性。
_ENCODED_RUN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{512,}")
_RUN_BREAK = re.compile(r"[^A-Za-z0-9+/=_-]")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_GUTTER = re.compile(r"^[ \t]*(\d+)(?:\t|→)", re.MULTILINE)
# 段标题：# 标题，或至少 3 个大写字母、没有小写字母的整行。以换行开头写，正则能直接跳到换行处找（超长行不逐字试）。
_SECTION_BODY = _GUTTERS + r"[ \t]*(#{1,6}[ \t]+\S[^\n]{0,200}?|[A-Z=-][A-Z0-9 :_=/().-]{5,160}?)[ \t\r]*$"
_SECTION_HEADER = re.compile(rf"\n{_SECTION_BODY}", re.MULTILINE)
_FIRST_SECTION_HEADER = re.compile(rf"\A{_SECTION_BODY}", re.MULTILINE)
_CODE_START = re.compile(
    r"(?:from[ \t]+\S+[ \t]+import\b|import[ \t]+\S|package[ \t]+[\w.]+|#include\b|use[ \t]+[\w:]+|"
    r"(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?(?:def|class|function|fn|func|interface|struct|enum)"
    r"[ \t]+[A-Za-z_$]|(?:const|let|var)[ \t]+\w+[ \t]*=[ \t]*require\()"
)
_CODE_IMPORT = re.compile(
    rf"{_LINE_START}[ \t]*((?:from[ \t]+\S+[ \t]+import\b|import[ \t]+\S|#include\b|use[ \t]+[\w:]+|"
    r"(?:const|let|var)[ \t]+\w+[ \t]*=[ \t]*require\()[^\n]*)",
    re.MULTILINE,
)
_CODE_SYMBOL = re.compile(
    rf"{_LINE_START}[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:pub[ \t]+)?(?:async[ \t]+)?"
    r"(def|class|function|fn|func|interface|struct|enum|trait)[ \t]+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_YAML_KEY = re.compile(r"""[A-Za-z0-9_."'/-]+:(?:[ \t]|$)""")
_YAML_OPEN_KEY = re.compile(r"""[A-Za-z0-9_."'/-]+:[ \t]*(?:#.*)?$""")

_PIECE_CHARS = 200
_WINDOW_CHARS = 256 * 1024
_PREFIX_CHARS = 64 * 1024
#: 超过这个字数的行算超长行：规则只在候选附近 ``_LONG_LINE_MARGIN`` 字内核对，也不折叠。
_LONG_LINE_CHARS = 4_096
_LONG_LINE_MARGIN = 256
_CANDIDATE_OVERLAP = 512
#: JSON 文本 / XML / 表格 / YAML 只在不超过这个字数时做结构识别（整份解析的保护闸）。
_STRUCTURED_PARSE_CHARS = 2 * 1024 * 1024
_CONTEXT_BEFORE = 1
_CONTEXT_AFTER = 2
_DIAGNOSTIC_SHARE = 0.5
_TAIL_SHARE = 0.6
_CODE_HEAD_SHARE = 2 / 3
_MAX_LISTED_REPEATS = 5
_MAX_LISTED_VALUES = 8
_MAX_DIAGNOSTIC_GROUPS = 1_000
_MAX_SECTION_HEADERS = 18
_MAX_IMPORTS = 12
_MAX_SYMBOLS = 40
_MAX_LISTED_KEYS = 32
_MAX_SCALAR_SAMPLES = 5
_MAX_SAMPLE_NODES = 200
_FACT_CHARS = 160
_FACT_LIST_CHARS = 600
_BINARY_SAMPLE_CHARS = 60
_CLASSIFY_LINES = 60


# ---------------------------------------------------------------- 内容分类


@dataclass(frozen=True)
class _Content:
    """分类用的内容视图：原值、全文、去掉行号前缀后的开头一段与（不超过解析闸时的）全文。"""

    value: Any
    text: str
    gutter: bool
    prefix: str
    full: str | None


@dataclass(frozen=True)
class ContentKindRule:
    """一类内容：名称、判据说明、识别函数（认出时返回写进说明的形状事实）与抽取做法。"""

    kind: str
    criterion: str
    recognize: Callable[[_Content], tuple[str, ...] | None]
    plan: str


def _recognize_empty(content: _Content) -> tuple[str, ...] | None:
    return ("内容只有空白字符。",) if not content.text or content.text.isspace() else None


def _recognize_binary(content: _Content) -> tuple[str, ...] | None:
    text = content.text
    sample = text[:1_000]
    controls = sum(1 for character in sample if ord(character) < 32 and character not in "\n\r\t")
    nul = text.find("\x00")
    if nul < 0 and controls / max(1, len(sample)) <= 0.05:
        return None
    reason = (
        f"含 NUL（{text.count(chr(0)):,} 个，第一个在第 {nul + 1:,} 字）"
        if nul >= 0
        else f"开头 {len(sample):,} 字里换行 / 回车 / 制表以外的控制字符有 {controls:,} 个（超过 5%）"
    )
    return (
        f"{reason}，按类二进制只留头尾极短样例。",
        f"开头样例：{_escape(text[:_BINARY_SAMPLE_CHARS])}",
        f"结尾样例：{_escape(text[-_BINARY_SAMPLE_CHARS:])}",
    )


def _recognize_json(content: _Content) -> tuple[str, ...] | None:
    if isinstance(content.value, Mapping | list | tuple):
        return _shape_facts("JSON", content.value)
    if content.full is None:
        return None
    stripped = content.full.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except (ValueError, RecursionError):
        return None
    trailing = len(stripped[end:].strip())
    facts = _shape_facts("JSON", value)
    return (*facts, f"首个 JSON 值之后还有 {trailing:,} 个字符。") if trailing > 0 else facts


def _recognize_xml(content: _Content) -> tuple[str, ...] | None:
    if content.full is None:
        return None
    stripped = content.full.strip()
    # 带 DTD 的不解析（不展开实体）；HTML 多半不是良构 XML，解析失败就落到后面几类。
    if not stripped.startswith("<") or re.search(r"<!(?:doctype|entity)", stripped, re.IGNORECASE):
        return None
    try:
        root = ElementTree.fromstring(stripped)
    except (ElementTree.ParseError, ValueError, RecursionError):
        return None
    children = Counter(child.tag for child in root)
    listed = "、".join(f"{tag} {count:,}" for tag, count in children.most_common(10))
    return (
        f"XML：根元素 {_clip(root.tag)}，{len(root.attrib):,} 个属性，共 {sum(1 for _ in root.iter()):,} 个元素。",
        f"直接子元素：{listed or '无'}。",
    )


def _table_recognizer(delimiter: str, label: str) -> Callable[[_Content], tuple[str, ...] | None]:
    def recognize(content: _Content) -> tuple[str, ...] | None:
        if content.full is None:
            return None
        first = next((line for line in content.full.split("\n", _CLASSIFY_LINES) if line.strip()), "")
        if delimiter not in first:
            return None
        columns: list[str] | None = None
        first_row: list[str] | None = None
        rows = 0
        try:
            for row in csv.reader(_lines_with_ends(content.full), delimiter=delimiter):
                if not any(cell.strip() for cell in row):
                    continue
                if columns is None:
                    columns = row
                elif len(row) != len(columns):
                    return None
                else:
                    first_row = first_row or row
                    rows += 1
        except csv.Error:
            return None
        if columns is None or len(columns) < 2 or rows < 1:
            return None
        names = "、".join(
            _clip(cell.strip() or f"第 {index} 列") for index, cell in enumerate(columns[:_MAX_LISTED_KEYS], 1)
        )
        return (
            f"{label}：{rows:,} 行数据、{len(columns):,} 列（各行列数一致）。",
            f"列：{names}{' 等' if len(columns) > _MAX_LISTED_KEYS else ''}。",
            f"首行数据：{_clip(delimiter.join(first_row or []))}",
        )

    return recognize


def _recognize_yaml(content: _Content) -> tuple[str, ...] | None:
    if content.full is None or not _looks_yaml(content.prefix):
        return None
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        value = yaml.load(content.full, Loader=loader)
    except (yaml.YAMLError, ValueError, RecursionError):
        return None
    return _shape_facts("YAML", value) if isinstance(value, Mapping | list) else None


def _recognize_code(content: _Content) -> tuple[str, ...] | None:
    starts = sum(1 for line in _first_lines(content.prefix) if _CODE_START.match(line))
    return () if starts >= 2 else None


def _recognize_text(_content: _Content) -> tuple[str, ...] | None:
    return ()


#: 内容分类规则表：依次判断，第一条认出的决定怎么抽取。
CONTENT_KINDS: tuple[ContentKindRule, ...] = (
    ContentKindRule("空", "只有空白字符", _recognize_empty, "description"),
    ContentKindRule(
        "类二进制", "含 NUL，或开头 1,000 字里换行 / 回车 / 制表以外的控制字符超过 5%", _recognize_binary, "description"
    ),
    ContentKindRule(
        "JSON",
        "本身是结构化值，或去掉开头空白后以 { / [ 开头且首个 JSON 值能解析（其后的尾随字符计数）",
        _recognize_json,
        "structured",
    ),
    ContentKindRule("XML", "以 < 开头、不带 DTD、能解析成良构 XML", _recognize_xml, "structured"),
    ContentKindRule(
        "TSV",
        "首个非空行含制表符，各非空行按制表符分出的列数一致（至少 2 列、1 行数据）",
        _table_recognizer("\t", "TSV 表格"),
        "structured",
    ),
    ContentKindRule(
        "CSV",
        "首个非空行含逗号，各非空行按逗号分出的列数一致（至少 2 列、1 行数据）",
        _table_recognizer(",", "CSV 表格"),
        "structured",
    ),
    ContentKindRule(
        "YAML",
        "以 --- 开头或有无空格的键独占一行且下一行缩进，开头的顶格行都是键或列表项，且能安全解析成映射或列表",
        _recognize_yaml,
        "structured",
    ),
    ContentKindRule(
        "代码",
        f"开头 {_CLASSIFY_LINES} 个非空行里至少 2 行顶格是导入或定义（import / from … import / def / class / function 等）",
        _recognize_code,
        "code",
    ),
    ContentKindRule("纯文本", "以上都不是（日志、命令输出、文档）", _recognize_text, "text"),
)


def _classify(value: Any, text: str) -> tuple[ContentKindRule, tuple[str, ...], bool]:
    head = text[:_PREFIX_CHARS]
    full: str | None = text if len(text) <= _STRUCTURED_PARSE_CHARS else None
    prefix = head
    # 读到的文件本身又是带行号的转储时有两层前缀，一并去掉。
    levels = 0
    while levels < _MAX_GUTTER_LEVELS and _has_gutter(prefix):
        prefix = _GUTTER.sub("", prefix)
        full = _GUTTER.sub("", full) if full is not None else None
        levels += 1
    gutter = levels > 0
    content = _Content(value, text, gutter, prefix, full)
    for rule in CONTENT_KINDS:
        facts = rule.recognize(content)
        if facts is not None:
            return rule, facts, gutter
    raise AssertionError("纯文本规则总能认出")


def _has_gutter(head: str) -> bool:
    numbers: list[int] = []
    for line in islice(head.split("\n", 20), 20):
        match = _GUTTER.match(line)
        if match is None:
            break
        numbers.append(int(match.group(1)))
    # 行号只要求递增：grep -v 去掉空行、只截取一部分行的转储，行号有缺口。
    return len(numbers) >= 3 and all(after > before for before, after in zip(numbers, numbers[1:], strict=False))


def _looks_yaml(prefix: str) -> bool:
    lines = [line for line in _first_lines(prefix) if not line.lstrip().startswith("#")]
    if not lines:
        return False
    if lines[0].rstrip() in {"---", "%YAML"} or lines[0].startswith("%YAML"):
        return True
    top = [line for line in lines if not line[:1].isspace()]
    if any(not (_YAML_KEY.match(line) or line.startswith("- ") or line.rstrip() in {"-", "---"}) for line in top):
        return False
    return any(
        _YAML_OPEN_KEY.match(line) and (after[:1].isspace() or after.startswith("- "))
        for line, after in zip(lines, lines[1:], strict=False)
    )


def _first_lines(text: str) -> Iterator[str]:
    count = 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        yield line.rstrip("\r")
        count += 1
        if count >= _CLASSIFY_LINES:
            return


def _lines_with_ends(text: str) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = text.find("\n", start)
        end = len(text) if end < 0 else end + 1
        yield text[start:end]
        start = end


def _shape_facts(label: str, value: Any) -> tuple[str, ...]:
    facts: list[str] = []
    if isinstance(value, Mapping):
        listed = "；".join(
            f"{_clip(str(key), 60)}（{_type_label(child)}）" for key, child in islice(value.items(), _MAX_LISTED_KEYS)
        )
        more = " 等" if len(value) > _MAX_LISTED_KEYS else ""
        facts.append(f"{label}：顶层是对象，{len(value):,} 个字段。")
        facts.append(f"字段：{listed}{more}。")
    elif isinstance(value, list | tuple):
        facts.append(f"{label}：顶层是数组，{len(value):,} 项。")
        if value:
            types = Counter(_type_name(item) for item in islice(value, 100))
            facts.append(
                f"前 {min(len(value), 100):,} 项的类型：{'、'.join(f'{name} {count:,}' for name, count in types.items())}。"
            )
            if isinstance(value[0], Mapping):
                keys = "、".join(_clip(str(key), 60) for key in islice(value[0], _MAX_LISTED_KEYS))
                facts.append(f"第 1 项的字段：{keys}。")
    else:
        facts.append(f"{label}：顶层是{_type_name(value)}。")
    samples = _scalar_samples(value)
    if samples:
        facts.append(f"标量样例：{'；'.join(samples)}。")
    return tuple(facts)


def _scalar_samples(value: Any) -> list[str]:
    """按先序取前几个标量（数组只看第 1 项）；访问的节点数有上限，共享引用的深层结构也不会展开。"""

    samples: list[str] = []
    stack: list[tuple[str, Any]] = [("", value)]
    visited = 0
    while stack and len(samples) < _MAX_SCALAR_SAMPLES and visited < _MAX_SAMPLE_NODES:
        path, node = stack.pop()
        visited += 1
        if isinstance(node, Mapping):
            children = [
                (f"{path}.{key}" if path else str(key), child) for key, child in islice(node.items(), _MAX_LISTED_KEYS)
            ]
            stack.extend(reversed(children))
        elif isinstance(node, list | tuple):
            if node:
                stack.append((f"{path}[0]", node[0]))
        else:
            samples.append(
                f"{_clip(path or '值', 60)} = {_clip(json.dumps(node, ensure_ascii=False, default=str), 80)}"
            )
    return samples


def _type_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return "对象"
    if isinstance(value, list | tuple):
        return "数组"
    if isinstance(value, bool):
        return "布尔"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return "数"
    return "字符串"


def _type_label(value: Any) -> str:
    if isinstance(value, Mapping):
        return f"对象，{len(value):,} 个字段"
    if isinstance(value, list | tuple):
        return f"数组，{len(value):,} 项"
    return _type_name(value)


def _clip(value: str, limit: int = _FACT_CHARS) -> str:
    value = _SPACES.sub(" ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _escape(value: str) -> str:
    return "".join(
        character
        if character.isprintable()
        else f"\\x{ord(character):02x}"
        if ord(character) < 256
        else f"\\u{ord(character):04x}"
        for character in value
    )


# ---------------------------------------------------------------- 按位置扫描的全文


def _windows(text: str) -> Iterator[tuple[int, int]]:
    """把全文切成约 ``_WINDOW_CHARS`` 字、止于换行之后的窗口。"""

    start = 0
    while start < len(text):
        cut = text.find("\n", start + _WINDOW_CHARS)
        end = len(text) if cut < 0 else cut + 1
        yield start, end
        start = end


def _run_safe_windows(text: str) -> Iterator[tuple[int, int]]:
    """切成约 ``_WINDOW_CHARS`` 字的窗口，切点落在编码样长串之外的字符之后（长串不会被切断）。"""

    start = 0
    while start < len(text):
        cut = _RUN_BREAK.search(text, min(len(text), start + _WINDOW_CHARS))
        end = len(text) if cut is None else cut.end()
        yield start, end
        start = end


@dataclass(frozen=True)
class _Piece:
    """抽取的最小单位：原文一行里按 200 字网格切出的一段（短行就是整行），只记位置与这一段的文字。"""

    start: int
    end: int
    line: int
    line_start: int
    line_end: int
    content_end: int
    text: str

    @property
    def starts_line(self) -> bool:
        return self.start == self.line_start

    @property
    def ends_line(self) -> bool:
        return self.end >= self.content_end


class _Document:
    """编码样长串换成说明后的全文；省略说明的字数按原文实际字数换算。"""

    def __init__(self, text: str) -> None:
        parts: list[str] = []
        self.run_starts: list[int] = []
        self.extra: list[int] = [0]
        cursor = 0
        produced = 0
        for start, end in _run_safe_windows(text):
            for match in _ENCODED_RUN.finditer(text, start, end):
                parts.append(text[cursor : match.start()])
                produced += match.start() - cursor
                note = f"〔省略 {match.end() - match.start():,} 字符的编码样长串〕"
                parts.append(note)
                self.run_starts.append(produced)
                self.extra.append(self.extra[-1] + (match.end() - match.start()) - len(note))
                produced += len(note)
                cursor = match.end()
        self.text = text if not parts else "".join((*parts, text[cursor:]))
        self.length = len(self.text)
        self.lines = self.text.count("\n") + 1

    def original_chars(self, start: int, end: int) -> int:
        if end <= start:
            return 0
        before = self.extra[bisect.bisect_left(self.run_starts, start)]
        after = self.extra[bisect.bisect_left(self.run_starts, end)]
        return end - start + after - before

    def line_bounds(self, position: int) -> tuple[int, int]:
        start = self.text.rfind("\n", 0, position) + 1
        end = self.text.find("\n", position)
        return start, self.length if end < 0 else end

    def piece(self, line: int, line_start: int, line_end: int, index: int) -> _Piece:
        content_end = self._content_end(line_start, line_end)
        start = min(line_start + index * _PIECE_CHARS, content_end)
        end = min(content_end, start + _PIECE_CHARS)
        return _Piece(start, end, line, line_start, line_end, content_end, self.text[start:end])

    def _content_end(self, line_start: int, line_end: int) -> int:
        """行尾的回车不算进段里（省略说明的字数仍算它）。"""

        return line_end - 1 if line_end > line_start and self.text[line_end - 1] == "\r" else line_end

    def piece_at(self, line: int, line_start: int, line_end: int, position: int) -> _Piece:
        return self.piece(line, line_start, line_end, max(0, position - line_start) // _PIECE_CHARS)

    def first(self) -> _Piece:
        return self.piece(1, 0, self.line_bounds(0)[1], 0)

    def last(self) -> _Piece:
        start, end = self.line_bounds(self.length)
        return self.last_of_line(self.lines, start, end)

    def last_of_line(self, line: int, line_start: int, line_end: int) -> _Piece:
        content_end = self._content_end(line_start, line_end)
        return self.piece(line, line_start, line_end, max(0, content_end - line_start - 1) // _PIECE_CHARS)

    def after(self, piece: _Piece) -> _Piece | None:
        if not piece.ends_line:
            return self.piece(
                piece.line, piece.line_start, piece.line_end, (piece.start - piece.line_start) // _PIECE_CHARS + 1
            )
        if piece.line_end >= self.length:
            return None
        start = piece.line_end + 1
        return self.piece(piece.line + 1, start, self.line_bounds(start)[1], 0)

    def before(self, piece: _Piece) -> _Piece | None:
        if not piece.starts_line:
            return self.piece(
                piece.line, piece.line_start, piece.line_end, (piece.start - piece.line_start) // _PIECE_CHARS - 1
            )
        if piece.line_start == 0:
            return None
        end = piece.line_start - 1
        return self.last_of_line(piece.line - 1, self.text.rfind("\n", 0, end) + 1, end)


# ---------------------------------------------------------------- 诊断行


@dataclass
class _Diagnostic:
    """一组去重后的诊断行：首次出现的行与位置、类别，及只差数字的其余行的计数、行号与不同的值。"""

    line: int
    line_start: int
    line_end: int
    match_start: int
    match_end: int
    kinds: tuple[str, ...]
    numbers: tuple[str, ...]
    repeats: int = 0
    repeat_lines: tuple[int, ...] = ()
    values: tuple[str, ...] = ()
    more_values: bool = False


@dataclass(frozen=True)
class _DiagnosticScan:
    groups: tuple[_Diagnostic, ...]
    counts: Mapping[str, int]
    overflow: int
    folded: frozenset[str]


def _signature(line: str) -> str:
    normalized = _SPACES.sub(" ", _DIGITS.sub("#", line.casefold())).strip()
    return hashlib.sha256(normalized.encode("utf-8", "surrogatepass")).hexdigest()


def _check_region(candidate: tuple[int, int], line_start: int, line_end: int) -> tuple[int, int]:
    """候选行上核对规则的范围：普通行是整行；超长行只看候选附近，免得一次正则调用扫完整个超长行。"""

    if line_end - line_start <= _LONG_LINE_CHARS:
        return line_start, line_end
    return max(line_start, candidate[0] - _LONG_LINE_MARGIN), min(line_end, candidate[1] + _LONG_LINE_MARGIN)


class _CandidateFinder:
    """按窗口找诊断候选：每次正则调用只扫一个窗口（多看一小段重叠，候选只认起点在窗口内的），超长行也不会一次扫完。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.window_start = 0
        self.window_end = 0
        self.stop = 0
        self.lowered: str | None = None

    def find(self, position: int) -> tuple[int, int] | None:
        text = self.text
        while position < len(text):
            if not self.window_start <= position < self.window_end:
                self.window_start = position
                self.window_end = min(len(text), position + _WINDOW_CHARS)
                self.stop = min(len(text), self.window_end + _CANDIDATE_OVERLAP)
                chunk = text[position : self.stop]
                lowered = chunk.lower()
                self.lowered = lowered if len(lowered) == len(chunk) else None
            if self.lowered is not None:
                match = _LOWERED_CANDIDATE.search(self.lowered, position - self.window_start)
                found = None if match is None else (self.window_start + match.start(), self.window_start + match.end())
            else:
                fallback = _DIAGNOSTIC_CANDIDATE.search(text, position, self.stop)
                found = None if fallback is None else fallback.span()
            if found is not None and found[0] < self.window_end:
                return found
            position = self.window_end
        return None


def _scan_diagnostics(document: _Document) -> _DiagnosticScan:
    text = document.text
    groups: dict[str, _Diagnostic] = {}
    counts: Counter[str] = Counter()
    overflow = 0
    line = 1
    counted_to = 0
    finder = _CandidateFinder(text)
    position = 0
    while (candidate := finder.find(position)) is not None:
        line_start, line_end = document.line_bounds(candidate[0])
        low, high = _check_region(candidate, line_start, line_end)
        found = [(rule.kind, rule.pattern.search(text, low, high)) for rule in DIAGNOSTIC_RULES]
        hits = [(kind, match) for kind, match in found if match is not None]
        if not hits:
            # 普通行已整行核对过；超长行从候选之后接着找。
            position = line_end + 1 if (low, high) == (line_start, line_end) else candidate[1]
            continue
        line += text.count("\n", counted_to, line_start)
        counted_to = line_start
        kinds = tuple(kind for kind, _ in hits)
        counts.update(kinds)
        first = min((match for _, match in hits), key=lambda match: match.start())
        if line_end - line_start > _LONG_LINE_CHARS:
            # 超长行不折叠（只差数字的判断要看整行）。
            signature, numbers = f"@{line_start}", ()
        else:
            body = text[line_start:line_end]
            signature, numbers = _signature(body), tuple(_DIGITS.findall(body))
        group = groups.get(signature)
        if group is not None:
            _fold(group, line, numbers)
        elif len(groups) < _MAX_DIAGNOSTIC_GROUPS:
            groups[signature] = _Diagnostic(line, line_start, line_end, first.start(), first.end(), kinds, numbers)
        else:
            overflow += 1
        position = line_end + 1
    ordered = tuple(sorted(groups.values(), key=lambda group: group.line_start))
    folded = frozenset(signature for signature, group in groups.items() if group.repeats)
    return _DiagnosticScan(ordered, counts, overflow, folded)


def _fold(group: _Diagnostic, line: int, numbers: tuple[str, ...]) -> None:
    group.repeats += 1
    if len(group.repeat_lines) < _MAX_LISTED_REPEATS:
        group.repeat_lines = (*group.repeat_lines, line)
    for first, value in zip(group.numbers, numbers, strict=False):
        if value == first or value in group.values:
            continue
        if len(group.values) < _MAX_LISTED_VALUES:
            group.values = (*group.values, value)
        else:
            group.more_values = True


def _repeat_note(group: _Diagnostic) -> str | None:
    if not group.repeats:
        return None
    listed = "、".join(str(line) for line in group.repeat_lines)
    more = " 等" if group.repeats > len(group.repeat_lines) else ""
    values = ""
    if group.values:
        values = f"；与本行不同的值：{'、'.join(group.values)}{' 等' if group.more_values else ''}"
    return f"〔只差数字的行另有 {group.repeats:,} 处：第 {listed} 行{more}{values}〕"


# ---------------------------------------------------------------- 往上限里装段


class _Layout:
    """按优先级往上限里装段；渲染长度增量维护，每次尝试都按真实渲染长度判断。"""

    def __init__(self, maximum: int, description: str, document: _Document) -> None:
        self.maximum = maximum
        self.description = description
        self.document = document
        self.starts: list[int] = []
        self.pieces: dict[int, _Piece] = {}
        self.notes: dict[int, str] = {}
        self.trailer = ""
        self.length = len(description) + self._link(None, None)
        self._head: _Piece | None = document.first()
        self._tail: _Piece | None = document.last()

    def share(self, fraction: float) -> int:
        return self.length + int((self.maximum - self.length) * fraction)

    def kept(self, piece: _Piece) -> bool:
        return piece.start in self.pieces and self.pieces[piece.start].end == piece.end

    def add_head(self, limit: int) -> None:
        while self._head is not None:
            if not self.kept(self._head) and not self.add((self._head,), None, None, limit):
                return
            self._head = self.document.after(self._head)

    def add_tail(self, limit: int) -> None:
        while self._tail is not None:
            if not self.kept(self._tail) and not self.add((self._tail,), None, None, limit):
                return
            self._tail = self.document.before(self._tail)

    def set_trailer(self, trailer: str) -> None:
        old = len(self.trailer) + 1 if self.trailer else 0
        self.trailer = trailer
        self.length += (len(trailer) + 1 if trailer else 0) - old

    def add(self, pieces: Sequence[_Piece], anchor: _Piece | None, note: str | None, limit: int) -> bool:
        """把一组段连同锚点上的说明一起装入；装不下就整组撤回。"""

        added: list[_Piece] = []
        for piece in sorted(pieces, key=lambda item: item.start):
            if piece.start not in self.pieces:
                self._insert(piece)
                added.append(piece)
        noted = anchor is not None and note is not None and anchor.start not in self.notes
        if noted:
            assert anchor is not None and note is not None
            self._set_note(anchor, note)
        if self.length <= min(limit, self.maximum):
            return True
        if noted:
            assert anchor is not None
            self._set_note(anchor, None)
        for piece in reversed(added):
            self._remove(piece)
        return False

    def _neighbours(self, start: int) -> tuple[_Piece | None, _Piece | None]:
        index = bisect.bisect_left(self.starts, start)
        before = self.pieces[self.starts[index - 1]] if index > 0 else None
        after = self.pieces[self.starts[index]] if index < len(self.starts) else None
        return before, after

    def _insert(self, piece: _Piece) -> None:
        before, after = self._neighbours(piece.start)
        self.length += (
            self._link(before, piece) + len(piece.text) + self._link(piece, after) - self._link(before, after)
        )
        bisect.insort(self.starts, piece.start)
        self.pieces[piece.start] = piece

    def _remove(self, piece: _Piece) -> None:
        self.starts.remove(piece.start)
        del self.pieces[piece.start]
        before, after = self._neighbours(piece.start)
        self.length -= (
            self._link(before, piece) + len(piece.text) + self._link(piece, after) - self._link(before, after)
        )

    def _set_note(self, piece: _Piece, note: str | None) -> None:
        # 说明挂在段后，会改变它与下一段能否接回同一行。
        _, after = self._neighbours(piece.start + 1)
        self.length -= self._link(piece, after) + (len(self.notes[piece.start]) + 1 if piece.start in self.notes else 0)
        if note is None:
            self.notes.pop(piece.start, None)
        else:
            self.notes[piece.start] = note
        self.length += self._link(piece, after) + (len(note) + 1 if note is not None else 0)

    def _merged(self, before: _Piece | None, after: _Piece) -> bool:
        return (
            before is not None
            and before.line == after.line
            and before.end == after.start
            and before.start not in self.notes
        )

    def _link(self, before: _Piece | None, after: _Piece | None) -> int:
        gap = self._gap(before, after)
        length = len(gap) + 1 if gap else 0
        if after is not None and not self._merged(before, after):
            length += 1
        return length

    def _gap(self, before: _Piece | None, after: _Piece | None) -> str | None:
        document = self.document
        if before is not None and after is not None and before.line == after.line:
            if before.end >= after.start:
                return None
            return f"……省略了第 {before.line} 行中的 {document.original_chars(before.end, after.start):,} 字……"
        if before is None:
            first, start = 1, 0
        elif before.ends_line:
            first, start = before.line + 1, before.line_end + 1
        else:
            first, start = before.line, before.end
        if after is None:
            last, end = document.lines, document.length
        elif after.starts_line:
            last, end = after.line - 1, after.line_start - 1
        else:
            last, end = after.line, after.start
        if first > last:
            return None
        lines = f"第 {first} 行" if first == last else f"第 {first}–{last} 行"
        return f"……省略了{lines}（{document.original_chars(start, end):,} 字）……"

    def render(self) -> str:
        parts = [self.description]
        if self.trailer:
            parts.append(self.trailer)
        before: _Piece | None = None
        for start in self.starts:
            piece = self.pieces[start]
            gap = self._gap(before, piece)
            if gap:
                parts.append(gap)
            if self._merged(before, piece):
                parts[-1] += piece.text
            else:
                parts.append(piece.text)
            if start in self.notes:
                parts.append(self.notes[start])
            before = piece
        gap = self._gap(before, None)
        if gap:
            parts.append(gap)
        return "\n".join(parts)


# ---------------------------------------------------------------- 抽取


class ToolResultExtractor:
    """把超大工具结果抽成不超过 ``maximum`` 字的确定性摘录。

    先按 ``CONTENT_KINDS`` 认内容类别，说明行写明认成哪一类；再按类别装入：
    纯文本 → 段标题、诊断行（各带上下文，只差数字的行折叠并列出不同的值）、结尾、开头；
    代码 → 不扫报错词，留行数、导入、定义的符号，再装开头与结尾；JSON / XML / 表格 / YAML → 形状与头部样例；
    类二进制与空 → 只有描述。被省略的部分写明行号与原文字数。
    """

    def __init__(self, maximum: int) -> None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("maximum must be a positive integer")
        self.maximum = maximum

    def extract(
        self,
        value: Any,
        *,
        tool_name: str | None = None,
        tool_status: str | None = None,
        size_bytes: int | None = None,
        call_argument: str | None = None,
    ) -> str:
        """``call_argument`` 不为空时抽的是工具调用的参数（值是参数名，空串表示整份参数），说明行照此写。"""

        text = value if isinstance(value, str) else canonical_json(value)
        rule, facts, gutter = _classify(value, text)
        document = _Document(text)
        scan = _scan_diagnostics(document) if rule.plan == "text" else None
        lines = [
            self._header(text, document.lines, rule.kind, tool_name, tool_status, size_bytes, call_argument),
        ]
        if gutter:
            lines.append("每行带行号前缀（读文件工具的输出形状），分类时去掉前缀判断。")
        lines.extend(facts)
        if rule.plan == "code":
            lines.extend(_code_facts(document))
        elif scan is not None:
            lines.extend(_text_facts(document, scan))
        layout = _Layout(self.maximum, "\n".join(lines)[: self.maximum // 2], document)
        if rule.plan == "text":
            assert scan is not None
            self._add_diagnostics(layout, scan, int(self.maximum * _DIAGNOSTIC_SHARE))
            layout.add_tail(layout.share(_TAIL_SHARE))
            layout.add_head(self.maximum)
        elif rule.plan == "code":
            layout.add_head(layout.share(_CODE_HEAD_SHARE))
        elif rule.plan == "structured":
            layout.add_head(self.maximum)
        # 前面各段停在装不下的段上时，剩下的位置留给结尾。
        if rule.plan != "description":
            layout.add_tail(self.maximum)
        return layout.render()[: self.maximum]

    @staticmethod
    def _header(
        text: str,
        line_count: int,
        kind: str,
        tool_name: str | None,
        tool_status: str | None,
        size_bytes: int | None,
        call_argument: str | None,
    ) -> str:
        facts = [f"共 {len(text):,} 字、{line_count:,} 行"]
        if size_bytes is not None:
            facts.append(f"{size_bytes:,} 字节")
        if call_argument is not None:
            argument = f"参数 {call_argument}" if call_argument else "整份参数"
            return (
                f"工具调用参数已压缩（关键信息抽取，未调用模型）：{tool_name or '未知工具'} 调用的{argument}；"
                f"按内容认作{kind}，{'、'.join(facts)}。"
            )
        return (
            f"工具结果已压缩（关键信息抽取，未调用模型）：{tool_name or '未知工具'} 的结果，"
            f"状态 {tool_status or 'unknown'}；按内容认作{kind}，{'、'.join(facts)}。"
        )

    @staticmethod
    def _add_diagnostics(layout: _Layout, scan: _DiagnosticScan, limit: int) -> None:
        document = layout.document
        skipped: list[int] = []
        for group in scan.groups:
            anchor = document.piece_at(group.line, group.line_start, group.line_end, group.match_start)
            pieces = [anchor]
            # 诊断词跨过 200 字网格时连同下一段一起留。
            if group.match_end > anchor.end and not anchor.ends_line:
                follow = document.after(anchor)
                assert follow is not None
                pieces.append(follow)
            context = list(pieces)
            neighbour: _Piece | None = pieces[0]
            for _ in range(_CONTEXT_BEFORE):
                neighbour = document.before(neighbour) if neighbour is not None else None
                if neighbour is not None and not _is_folded(document, neighbour, scan.folded):
                    context.append(neighbour)
            neighbour = pieces[-1]
            for _ in range(_CONTEXT_AFTER):
                neighbour = document.after(neighbour) if neighbour is not None else None
                if neighbour is not None and not _is_folded(document, neighbour, scan.folded):
                    context.append(neighbour)
            last = pieces[-1]
            note = _repeat_note(group)
            if layout.add(context, last, note, limit):
                continue
            if not layout.add(pieces, last, note, limit):
                skipped.append(group.line)
        total = len(skipped) + scan.overflow
        if total:
            listed = "、".join(str(line) for line in skipped[:_MAX_LISTED_REPEATS])
            more = " 等" if total > min(len(skipped), _MAX_LISTED_REPEATS) else ""
            layout.set_trailer(
                f"〔另有 {total:,} 处诊断行因上限未装下：第 {listed} 行{more}〕"
                if listed
                else f"〔另有 {total:,} 处诊断行因上限未装下〕"
            )
            if layout.length > limit:
                layout.set_trailer(f"〔另有 {total:,} 处诊断行因上限未装下〕")


def _is_folded(document: _Document, piece: _Piece, folded: frozenset[str]) -> bool:
    """被折叠进别处说明的诊断行不再当上下文抄一遍。"""

    if not folded or piece.line_end - piece.line_start > _LONG_LINE_CHARS:
        return False
    if not any(rule.pattern.search(document.text, piece.line_start, piece.line_end) for rule in DIAGNOSTIC_RULES):
        return False
    return _signature(document.text[piece.line_start : piece.line_end]) in folded


def _text_facts(document: _Document, scan: _DiagnosticScan) -> list[str]:
    facts: list[str] = []
    headers: list[str] = []
    seen: set[str] = set()
    line = 1
    counted_to = 0
    first = _FIRST_SECTION_HEADER.match(document.text)
    for window_start, window_end in _windows(document.text):
        # 窗口往前多看一个字，止于上一窗口末尾换行处的标题也认得出。
        found = _SECTION_HEADER.finditer(document.text, max(0, window_start - 1), window_end)
        for match in chain((first,) if first is not None and window_start == 0 else (), found):
            title = _clip(match.group(1), 80)
            if not title.startswith("#") and sum(1 for character in title if "A" <= character <= "Z") < 3:
                continue
            start = match.start() + (0 if match.re is _FIRST_SECTION_HEADER else 1)
            line += document.text.count("\n", counted_to, start)
            counted_to = start
            if title in seen:
                continue
            seen.add(title)
            headers.append(f"第 {line} 行 {title}")
            if len(headers) >= _MAX_SECTION_HEADERS:
                break
        if len(headers) >= _MAX_SECTION_HEADERS:
            break
    if headers:
        facts.append(f"段标题：{' | '.join(headers)}")
    if scan.counts:
        summary = "、".join(
            f"{rule.kind} {scan.counts[rule.kind]:,}" for rule in DIAGNOSTIC_RULES if scan.counts[rule.kind]
        )
        groups = f"{len(scan.groups):,}" if not scan.overflow else f"超过 {len(scan.groups):,}"
        facts.append(f"诊断行：{summary}（行数，同一行可属多类）；去重后 {groups} 处。")
    return facts


def _code_facts(document: _Document) -> list[str]:
    imports: dict[str, None] = {}
    import_count = 0
    symbols: dict[tuple[str, str], int] = {}
    symbol_count = 0
    line = 1
    counted_to = 0
    for window_start, window_end in _windows(document.text):
        for match in _CODE_IMPORT.finditer(document.text, window_start, window_end):
            import_count += 1
            if len(imports) < _MAX_IMPORTS:
                imports.setdefault(_clip(match.group(1), 100))
        for match in _CODE_SYMBOL.finditer(document.text, window_start, window_end):
            symbol_count += 1
            line += document.text.count("\n", counted_to, match.start())
            counted_to = match.start()
            if len(symbols) < _MAX_SYMBOLS:
                symbols.setdefault((match.group(1), match.group(2)), line)
    facts = [f"代码：{document.lines:,} 行，导入 {import_count:,} 处、定义 {symbol_count:,} 个（不扫报错词）。"]
    if imports:
        facts.append(f"导入：{_listed(list(imports), import_count)}")
    if symbols:
        listed = [f"{number} {kind} {name}" for (kind, name), number in symbols.items()]
        facts.append(f"定义（行号 种类 名称，同名只列第一处）：{_listed(listed, symbol_count)}")
    return facts


def _listed(items: list[str], total: int) -> str:
    """把一组条目连成一句，超过 ``_FACT_LIST_CHARS`` 字就停在完整的条目上，没列全的写"等"。"""

    kept: list[str] = []
    used = 0
    for item in items:
        if kept and used + len(item) + 1 > _FACT_LIST_CHARS:
            break
        kept.append(item)
        used += len(item) + 1
    return "；".join(kept) + (" 等" if total > len(kept) else "")


__all__ = [
    "CONTENT_KINDS",
    "DIAGNOSTIC_RULES",
    "ContentKindRule",
    "DiagnosticRule",
    "ToolResultExtractor",
]
