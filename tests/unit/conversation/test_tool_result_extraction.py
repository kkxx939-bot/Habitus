"""超大工具结果的确定性关键信息抽取与纯媒体结果描述（裁定 72 第 3 条、裁定 73）。"""

import hashlib
import json
import re
import time
import tracemalloc
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
    ConversationToolResultReducer,
    tool_result_extraction,
)
from habitus.memory.conversation.tool_result_extraction import (
    CONTENT_KINDS,
    ToolResultExtractor,
)
from habitus.memory.workflow import ConversationMemoryEnqueuer
from habitus.memory.workflow.jobs import MemoryJobStore
from habitus.pre.conversation import (
    ConversationAdapterContext,
    ConversationAdapterRegistry,
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.pre.conversation.media import describe_media_only
from tests.helpers import message
from tests.lock_store import ProcessLocalLockStore

MAXIMUM = 4_000


def _extract(text: object, *, tool_name: str | None = None, tool_status: str | None = None, maximum: int = MAXIMUM) -> str:
    output = ToolResultExtractor(maximum).extract(text, tool_name=tool_name, tool_status=tool_status)
    assert len(output) <= maximum
    return output


def _normal_lines(count: int, *, start: int = 1) -> list[str]:
    return [f"step {number:05d}: processed batch of records and moved on" for number in range(start, start + count)]


def _kind(output: str) -> str:
    match = re.search(r"按内容认作(\S+?)，", output)
    assert match, output[:200]
    return match.group(1)


def _diagnostic_summary(output: str) -> str | None:
    return next((line for line in output.splitlines() if line.startswith("诊断行：")), None)


def _tool_result(content: object, *, tool_name: str = "Bash") -> ConversationMessage:
    return message(
        2,
        ConversationMessageRole.TOOL_RESULT,
        content,
        tool_call_id="call-1",
        tool_name=tool_name,
        tool_status=ConversationToolResultStatus.COMPLETED,
        content_mode=ConversationToolResultContentMode.INLINE,
    )


# ---------------------------------------------------------------- 纯文本：诊断行、结尾、开头


def test_an_error_in_the_middle_of_a_long_output_is_kept_with_its_context() -> None:
    lines = _normal_lines(3_000)
    lines[1_499:1_499] = [
        "Traceback (most recent call last):",
        '  File "/srv/app/jobs.py", line 88, in run',
        "sqlite3.OperationalError: database is locked",
    ]
    output = _extract("\n".join(lines), tool_name="Bash", tool_status="error")

    assert _kind(output) == "纯文本"
    assert "sqlite3.OperationalError: database is locked" in output
    assert "Traceback (most recent call last):" in output
    assert lines[1_498] in output  # 报错前一行上下文
    assert lines[-1] in output  # 结尾照样留
    assert re.search(r"……省略了第 \d+–\d+ 行（[\d,]+ 字）……", output)


def test_lines_that_differ_only_in_numbers_fold_and_name_the_values_that_differ() -> None:
    lines = _normal_lines(400)
    for index in range(50, 350):
        lines[index] = f"WARN retry {index} of 3 for upstream at 10.0.0.{index % 250}"
    output = _extract("\n".join(lines))

    assert output.count("WARN retry") == 1
    assert "〔只差数字的行另有 299 处：第 52、53、54、55、56 行 等；与本行不同的值：51、52、53、54、55、56、57、58 等〕" in output
    assert "同样的行" not in output


def test_r6_two_errors_that_differ_only_in_status_code_both_keep_their_values() -> None:
    """R6：只差状态码 / 退出码的两行折叠后，不同的那个值仍在输出里，说明不说"同样的行"。"""

    lines = _normal_lines(3_000)
    lines[1_000] = "GET /api/user failed with HTTP 404"
    lines[2_000] = "GET /api/user failed with HTTP 500"
    lines[2_500] = "process exited with code 1"
    lines[2_600] = "process exited with code 137"
    output = _extract("\n".join(lines), maximum=1_500)

    assert "GET /api/user failed with HTTP 404" in output
    assert "〔只差数字的行另有 1 处：第 2001 行；与本行不同的值：500〕" in output
    assert "process exited with code 1" in output
    assert "137" in output
    assert "同样的行" not in output


@pytest.mark.parametrize(
    "line",
    [
        "KeyError: 'user_id'",
        "ValueError: bad value",
        "java.lang.NullPointerException: name is null",
        "Caused by: java.io.IOException: closed",
        "E   AssertionError",
        "E       AssertionError: assert 1 == 2",
    ],
)
def test_r7_camel_case_exception_class_lines_are_diagnostics(line: str) -> None:
    lines = _normal_lines(600)
    lines[300] = line
    output = _extract("\n".join(lines))

    assert _diagnostic_summary(output) == "诊断行：异常 1（行数，同一行可属多类）；去重后 1 处。"
    assert line in output


def test_camel_case_words_inside_a_sentence_are_not_exception_lines() -> None:
    lines = _normal_lines(600)
    lines[300] = "the MirrorError column stays empty"
    assert _diagnostic_summary(_extract("\n".join(lines))) is None


def test_r7_a_mid_output_traceback_with_folded_frames_keeps_the_exception_line() -> None:
    """R7：同一文件、只差行号的栈帧被折叠，不再给末行提供上下文；KeyError 那行本身是诊断行，照样留。"""

    pad = [f"ok line {number}" for number in range(3_000)]
    frames: list[str] = []
    for depth in range(40):
        frames += [f'  File "/srv/app/handlers.py", line {100 + depth}, in handle', f"    return dispatch(req, depth={depth})"]
    lines = [*pad[:1_500], "Traceback (most recent call last):", *frames, "    lookup(row)", "KeyError: 'user_id'", *pad[1_500:]]
    output = _extract("\n".join(lines), tool_name="bash", tool_status="error", maximum=1_500)

    assert "Traceback (most recent call last):" in output
    assert "KeyError: 'user_id'" in output


@pytest.mark.parametrize("offset", [196, 197, 198, 199, 200])
def test_r8_a_diagnostic_word_across_the_200_character_grid_is_still_found(offset: int) -> None:
    """R8：规则在整行上匹配再定位到段；error 从第 197–199 列开始（跨过 200 字切段边界）也认得出、留得下。"""

    lines = _normal_lines(600)
    lines[300] = "x" * offset + " error happened " + "y" * 300
    output = _extract("\n".join(lines))

    assert _diagnostic_summary(output) == "诊断行：错误 1（行数，同一行可属多类）；去重后 1 处。"
    assert "error" in output.replace("诊断行：错误", "")


def test_r9_omission_notes_count_the_characters_actually_omitted() -> None:
    """R9：省略说明与总长都按原文实际字数（一行 1,000 字切成 5 段也只算 1,000）。"""

    text = "head\n" + "ab." * 333 + "a" + "\ntail"
    output = _extract(text, tool_name="bash", maximum=200)

    assert "共 1,010 字、3 行" in output
    assert output.splitlines()[1:] == ["head", "……省略了第 2 行（1,000 字）……", "tail"]
    nothing_kept = _extract("x" * 300 + "\n" + "y" * 300, maximum=128)
    assert nothing_kept.endswith("……省略了第 1–2 行（601 字）……")


def test_an_error_at_the_start_keeps_the_exit_code_and_the_ending() -> None:
    lines = ["Exit code 2", "error: could not compile `habitus` due to previous error", *_normal_lines(2_000)]
    lines.append("summary: 0 passed, 2000 skipped")
    output = _extract("\n".join(lines))

    assert output.splitlines()[2:4] == lines[:2]
    assert "非零退出码 1" in output
    assert output.splitlines()[-1] == "summary: 0 passed, 2000 skipped"
    assert "……省略了第" in output


def test_plain_output_keeps_the_ending_then_the_beginning_and_names_the_omitted_lines() -> None:
    lines = _normal_lines(2_000)
    output = _extract("\n".join(lines), tool_name="Bash")

    assert "诊断行" not in output
    kept = output.splitlines()
    assert kept[1] == lines[0]
    assert kept[-1] == lines[-1]
    gap = next(line for line in kept if line.startswith("……省略了"))
    first, last = map(int, re.findall(r"第 (\d+)–(\d+) 行", gap)[0])
    assert kept[kept.index(gap) - 1] == lines[first - 2]
    assert kept[kept.index(gap) + 1] == lines[last]
    tail_lines = len(kept) - kept.index(gap) - 1
    assert tail_lines > kept.index(gap) - 1  # 结尾优先于开头


def test_text_lists_its_section_headers() -> None:
    lines = ["# Report", "", *_normal_lines(800), "## Details", *_normal_lines(800), "SHORT TEST SUMMARY INFO", "2026-10-26 12:00:00"]
    output = _extract("\n".join(lines))

    assert "段标题：第 1 行 # Report | 第 803 行 ## Details | 第 1604 行 SHORT TEST SUMMARY INFO" in output


def test_binary_like_runs_are_described_instead_of_copied() -> None:
    blob = "iVBORw0KGgo" + "QUJDRA" * 20_000
    output = _extract("\n".join(["header: dump follows", blob, "trailer: done"]))

    assert "〔省略 120,011 字符的编码样长串〕" in output
    assert not re.search(r"[A-Za-z0-9+/=]{200,}", output)
    assert "trailer: done" in output


# 测试工程师第二轮存活变异（TR04 / TR05 / TR06 / TR08 / TR16）对应规则各一条


def test_tr04_lowercase_warn_is_not_a_warning_but_upper_case_warn_is() -> None:
    lines = _normal_lines(600)
    lines[100] = "eslint --warn-unused .warn-text"
    assert _diagnostic_summary(_extract("\n".join(lines))) is None
    lines[200] = "WARN disk almost full"
    assert _diagnostic_summary(_extract("\n".join(lines))) == "诊断行：警告 1（行数，同一行可属多类）；去重后 1 处。"


def test_tr05_exit_code_zero_is_not_a_non_zero_exit_code() -> None:
    lines = _normal_lines(600)
    lines[100] = "process finished with exit code 0"
    assert _diagnostic_summary(_extract("\n".join(lines))) is None
    lines[200] = "process finished with exit code 3"
    assert _diagnostic_summary(_extract("\n".join(lines))) == "诊断行：非零退出码 1（行数，同一行可属多类）；去重后 1 处。"


def test_tr06_an_encoded_run_of_600_characters_is_described() -> None:
    lines = _normal_lines(600)
    lines[599] = "Q" * 600
    output = _extract("\n".join(lines))
    assert output.splitlines()[-1] == "〔省略 600 字符的编码样长串〕"
    lines[599] = "Q" * 511
    assert _extract("\n".join(lines)).splitlines()[-1] == "Q" * 200 + "Q" * 200 + "Q" * 111


def test_tr08_the_description_is_capped_at_half_the_bound_so_samples_still_fit() -> None:
    lines = _normal_lines(600)
    output = _extract("\n".join(lines), tool_name="T" * 10_000, maximum=1_000)

    description = output.split("\n", 1)[0]
    assert len(description) == 500
    assert output.splitlines()[-1] == lines[-1]


def test_tr16_panic_abort_and_segfault_are_fatal() -> None:
    lines = _normal_lines(900)
    lines[100] = "thread 'main' panicked at src/main.rs:2:5"
    lines[300] = "Aborted (core dumped)"
    lines[500] = "Segmentation fault"
    assert _diagnostic_summary(_extract("\n".join(lines))) == "诊断行：致命 3（行数，同一行可属多类）；去重后 3 处。"


# ---------------------------------------------------------------- 按内容分类（裁定 73 第 1 条）


def test_the_content_kind_table_is_ordered_as_the_ruling_lists_it() -> None:
    assert [rule.kind for rule in CONTENT_KINDS] == ["空", "类二进制", "JSON", "XML", "TSV", "CSV", "YAML", "代码", "纯文本"]
    assert all(rule.criterion for rule in CONTENT_KINDS)


def test_code_keeps_imports_symbols_and_both_ends_and_is_not_scanned_for_error_words() -> None:
    source = ["import os", "from pathlib import Path", "", "class Loader:", "    def load(self):", "        raise ValueError('bad value')"]
    source += [f"    def step_{number}(self):\n        raise RuntimeError('error {number}')" for number in range(1_500)]
    text = "\n".join(f"{number}\t{line}" for number, line in enumerate("\n".join(source).split("\n"), 1))
    output = _extract(text, tool_name="Read")

    assert _kind(output) == "代码"
    assert "每行带行号前缀" in output
    assert "导入：import os；from pathlib import Path" in output
    assert "定义（行号 种类 名称，同名只列第一处）：4 class Loader；5 def load；7 def step_0" in output
    assert "诊断行" not in output
    kept = output.splitlines()
    assert "1\timport os" in kept
    assert kept[-1] == text.splitlines()[-1]


def test_a_tool_named_read_whose_output_is_a_log_still_keeps_its_diagnostics() -> None:
    """Q5：不按工具名认读文件——自定义叫 read 的工具输出日志，中段报错照样留。"""

    lines = _normal_lines(3_000)
    lines[1_500] = "ERROR worker 7 lost its lease"
    output = _extract("\n".join(lines), tool_name="read")

    assert _kind(output) == "纯文本"
    assert "ERROR worker 7 lost its lease" in output


def test_json_values_keep_their_shape_samples_and_head() -> None:
    content = {"rows": [{"id": index, "status": "ok"} for index in range(3_000)], "total": 3_000}
    reduced = ConversationToolResultReducer().reduce(_tool_result(content))

    assert reduced.content_mode is ConversationToolResultContentMode.SUMMARIZED
    assert _kind(reduced.content) == "JSON"
    assert "JSON：顶层是对象，2 个字段。" in reduced.content
    assert "字段：rows（数组，3,000 项）；total（数）。" in reduced.content
    assert '标量样例：rows[0].id = 0；rows[0].status = "ok"；total = 3000。' in reduced.content
    assert '{"rows":[{"id":0,"status":"ok"}' in reduced.content


def test_json_text_with_trailing_characters_is_still_json() -> None:
    text = "[" + ",".join(f'{{"n": {index}}}' for index in range(5_000)) + "]\n\nexit status 0\n"
    output = _extract(text)

    assert _kind(output) == "JSON"
    assert "JSON：顶层是数组，5,000 项。" in output
    assert "首个 JSON 值之后还有 13 个字符。" in output


def test_csv_needs_the_same_column_count_on_every_row() -> None:
    rows = ["name,age,city", *(f"n{index},{index},c{index}" for index in range(3_000))]
    output = _extract("\n".join(rows))
    assert _kind(output) == "CSV"
    assert "CSV 表格：3,000 行数据、3 列（各行列数一致）。" in output
    assert "列：name、age、city。" in output
    assert output.splitlines()[4] == "name,age,city"

    rows[2_000] = "this row, unlike the others, has four columns, really"
    assert _kind(_extract("\n".join(rows))) == "纯文本"


def test_tsv_is_recognised() -> None:
    rows = ["id\tstate", *(f"{index}\tok" for index in range(5_000))]
    output = _extract("\n".join(rows))
    assert _kind(output) == "TSV"
    assert "TSV 表格：5,000 行数据、2 列（各行列数一致）。" in output


def test_yaml_keeps_its_top_level_keys() -> None:
    items = "\n".join(f"  - name: item-{index}\n    weight: {index}" for index in range(2_000))
    output = _extract(f"server:\n  host: 127.0.0.1\n  port: 8787\nitems:\n{items}\n")

    assert _kind(output) == "YAML"
    assert "字段：server（对象，2 个字段）；items（数组，2,000 项）。" in output
    assert output.splitlines()[4:6] == ["server:", "  host: 127.0.0.1"]


def test_a_log_made_of_key_value_lines_is_not_yaml() -> None:
    lines = ["Traceback (most recent call last):", '  File "a.py", line 1, in <module>', "KeyError: 'a'", *_normal_lines(3_000)]
    output = _extract("\n".join(lines))
    assert _kind(output) == "纯文本"
    assert "KeyError: 'a'" in output


def test_xml_keeps_root_and_child_counts() -> None:
    text = "<feed version='2'>" + "".join(f"<entry id='{index}'><title>t{index}</title></entry>" for index in range(3_000)) + "</feed>"
    output = _extract(text)

    assert _kind(output) == "XML"
    assert "XML：根元素 feed，1 个属性，共 6,001 个元素。" in output
    assert "直接子元素：entry 3,000。" in output


def test_xml_with_a_dtd_is_not_parsed() -> None:
    text = '<?xml version="1.0"?><!DOCTYPE lol [<!ENTITY lol "lol">]><r>' + "&lol;" * 20_000 + "</r>"
    assert _kind(_extract(text)) != "XML"


def test_binary_like_content_only_gets_a_description_and_tiny_samples() -> None:
    text = "\x00\x01PK\x03\x04" + "\x07data" * 20_000
    output = _extract(text)

    assert _kind(output) == "类二进制"
    assert "开头样例：\\x00\\x01PK\\x03\\x04" in output
    assert output.splitlines()[-1] == "……省略了第 1 行（100,006 字）……"
    assert len(output) < 600


def test_whitespace_only_content_is_empty() -> None:
    output = _extract("\n" * 100_000)
    assert _kind(output) == "空"
    assert output.splitlines()[1:] == ["内容只有空白字符。", "……省略了第 1–100001 行（100,000 字）……"]


@pytest.mark.parametrize("maximum", [128, 500, 4_000])
def test_extraction_never_exceeds_the_configured_bound(maximum: int) -> None:
    lines = _normal_lines(1_000)
    lines[500] = "FATAL: " + "x" * 3_000
    output = ToolResultExtractor(maximum).extract("\n".join(lines), tool_name="Bash", tool_status="error")
    assert 0 < len(output) <= maximum


def test_the_incrementally_tracked_length_is_the_rendered_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """往上限里装段时增量维护的长度必须与真实渲染逐字相等，否则上限判断会装多或装少。"""

    render = tool_result_extraction._Layout.render

    def checked(layout: tool_result_extraction._Layout) -> str:
        rendered = render(layout)
        assert len(rendered) == layout.length
        return rendered

    monkeypatch.setattr(tool_result_extraction._Layout, "render", checked)
    pool = ["ok", "", "ERROR boom 1", "WARN retry 2", "x" * 450, "KeyError: 'k'", "A" * 700, "y" * 199 + " error", "中文失败"]
    for maximum in (128, 333, 1_000, 4_000):
        for separator in ("\n", "\r\n"):
            lines = [pool[(index * 7 + maximum) % len(pool)] for index in range(400)]
            ToolResultExtractor(maximum).extract(separator.join(lines))


# ---------------------------------------------------------------- 行号前缀与分类判据的边界


def _numbered(lines: list[str], *, arrow: bool = False) -> str:
    return "\n".join(f"{number:6}{'→' if arrow else chr(9)}{line}" for number, line in enumerate(lines, 1))


def test_code_read_back_with_two_layers_of_line_numbers_keeps_its_code_facts() -> None:
    """``cat -n`` 的转储再被读文件工具读一次：分类与代码事实都去两层前缀。"""

    source = ["import os", "import sys", "from pathlib import Path", "", "class Foo:", "    def bar(self):"]
    source += ["        return 1", "", "def main():", "    print(Path(os.getcwd()))"]
    twice = _numbered(_numbered(source * 40).split("\n"), arrow=True)
    output = _extract(twice, tool_name="Read")

    assert _kind(output) == "代码"
    assert "代码：400 行，导入 120 处、定义 120 个（不扫报错词）。" in output
    assert "定义（行号 种类 名称，同名只列第一处）：5 class Foo；6 def bar；9 def main" in output


def test_a_log_read_back_with_two_layers_of_line_numbers_keeps_its_diagnostics() -> None:
    log = [f"step {number} ok" for number in range(300)]
    log += ["Traceback (most recent call last):", '  File "app.py", line 3, in <module>', "KeyError: 'user'"]
    log += [f"step {number} ok" for number in range(300, 600)]
    output = _extract(_numbered(_numbered(log).split("\n"), arrow=True), tool_name="Read", maximum=1_500)

    assert _kind(output) == "纯文本"
    assert _diagnostic_summary(output) == "诊断行：异常 1、栈 2（行数，同一行可属多类）；去重后 3 处。"
    assert "KeyError: 'user'" in output


def test_a_numbered_dump_with_gaps_in_the_numbers_is_still_code() -> None:
    """``cat -n | grep -v`` 去掉空行后行号有缺口：仍按行号前缀去掉再分类，不认成两列的 TSV。"""

    source = ["import os", "", "from pathlib import Path", "", "class Foo:", "    def bar(self):", "        return 1", ""]
    source += ["def main():", "    print(Path(os.getcwd()))", ""]
    dump = "\n".join(f"{number:6}\t{line}" for number, line in enumerate(source * 30, 1) if line.strip())
    output = _extract(dump, tool_name="Bash")

    assert _kind(output) == "代码"
    assert "代码：210 行，导入 60 处、定义 90 个（不扫报错词）。" in output


def test_one_nul_far_into_the_text_makes_it_binary_like_and_the_description_names_it() -> None:
    prose = "all good here\n" * 3_000
    output = _extract(prose[:20_000] + "\x00" + prose[20_000:])

    assert _kind(output) == "类二进制"
    assert "含 NUL（1 个，第一个在第 20,001 字），按类二进制只留头尾极短样例。" in output
    assert "里 0 个" not in output


def test_control_characters_above_five_percent_of_the_first_thousand_make_it_binary_like() -> None:
    text = "".join("\x01" if index % 10 == 0 else "a" for index in range(1_000)) + "\nrest\n"
    output = _extract(text)

    assert _kind(output) == "类二进制"
    assert "开头 1,000 字里换行 / 回车 / 制表以外的控制字符有 100 个（超过 5%）" in output


def test_a_header_row_without_data_rows_is_not_a_table() -> None:
    assert _kind(_extract("name\tvalue\n")) == "纯文本"


def test_one_import_line_does_not_make_a_log_code() -> None:
    log = "from app import main\n" + "".join(f"step {index} ok\n" for index in range(300))
    log += "ERROR: database locked\n" + "".join(f"step {index} ok\n" for index in range(300))
    output = _extract(log, maximum=1_500)

    assert _kind(output) == "纯文本"
    assert "ERROR: database locked" in output


def test_json_text_beyond_the_structured_parse_gate_is_plain_text() -> None:
    text = json.dumps({"items": [{"id": index, "name": "n" * 20} for index in range(100_000)]})
    assert len(text) > 2 * 1024 * 1024

    assert _kind(_extract(text)) == "纯文本"


def test_a_log_whose_top_level_lines_are_not_all_keys_is_not_yaml() -> None:
    output = _extract("Build summary: failed\nsteps:\n  - compile\n  - link\nERROR count: 2\n")

    assert _kind(output) == "纯文本"
    assert _diagnostic_summary(output) is not None


# ---------------------------------------------------------------- 诊断扫描的窗口、分组上限、折叠、上下文


def _filler(characters: int) -> str:
    return ("x" * 99 + "\n") * (characters // 100)


def test_two_overlong_error_lines_that_differ_only_in_numbers_are_not_folded() -> None:
    first = "ERROR " + "q " * 2_500 + " code 1"
    second = "ERROR " + "q " * 2_500 + " code 2"
    output = _extract(f"start\n{first}\n{second}\nend\n", maximum=20_000)

    assert _diagnostic_summary(output) == "诊断行：错误 2（行数，同一行可属多类）；去重后 2 处。"


def test_diagnostic_groups_beyond_one_thousand_are_counted_as_overflow() -> None:
    def word(index: int) -> str:
        letters = ""
        index += 1
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters

    lines = [f"ERROR {word(index)} refused the connection" for index in range(1_005)]
    output = _extract("\n".join(lines), maximum=20_000)

    assert _diagnostic_summary(output) == "诊断行：错误 1,005（行数，同一行可属多类）；去重后 超过 1,000 处。"


def test_a_diagnostic_starting_just_before_a_scan_window_edge_is_found() -> None:
    body = _filler(262_100) + "x" * 41 + "\n"
    text = body + "error: disk full\n" + _filler(5_000)
    assert text.index("error") == 256 * 1024 - 2

    assert "error: disk full" in _extract(text, maximum=3_000)


def test_a_window_whose_lowercase_changes_length_still_finds_diagnostics() -> None:
    text = "İ" * 10 + "\n" + _filler(100_000) + "ERROR: upstream unavailable\n" + _filler(100_000)

    assert "ERROR: upstream unavailable" in _extract(text, maximum=3_000)


def test_a_section_header_right_at_the_start_of_a_scan_window_is_listed() -> None:
    pad = _filler(300_000)
    cut = pad.find("\n", 256 * 1024) + 1
    output = _extract(pad[:cut] + "BUILD RESULTS SECTION\n" + pad[cut:], maximum=3_000)

    assert "BUILD RESULTS SECTION" in next(line for line in output.splitlines() if line.startswith("段标题："))


def test_the_line_after_a_diagnostic_is_kept_as_its_context() -> None:
    text = _filler(50_000) + "ERROR: request failed\nreason: certificate expired for api.example\n" + _filler(50_000)

    assert "reason: certificate expired for api.example" in _extract(text, maximum=3_000)


def test_the_overflow_trailer_drops_the_line_list_when_it_does_not_fit() -> None:
    def word(index: int) -> str:
        return chr(ord("a") + index % 26) + chr(ord("a") + index // 26)

    lines = [f"line {index} fine" for index in range(2_000)]
    for index in range(60):
        lines[100 + index * 30] = f"ERROR {word(index)} {word(index + 7)} refused the connection"
    output = _extract("\n".join(lines), maximum=1_500)

    assert _diagnostic_summary(output) == "诊断行：错误 60（行数，同一行可属多类）；去重后 60 处。"
    assert "〔另有 54 处诊断行因上限未装下〕" in output
    assert "因上限未装下：第" not in output


# ---------------------------------------------------------------- R3 / R4：时间与内存有上界


def test_r3_runs_just_under_the_encoded_threshold_are_scanned_in_linear_time() -> None:
    """R3：16 MiB 的"511 字长串 + 空格"重复输入，编码长串正则只从串开头匹配；修前要约 3.7 s。"""

    text = ("A" * 511 + " ") * (16 * 1024 * 1024 // 512)
    started = time.perf_counter()
    tool_result_extraction._ENCODED_RUN.sub("", text)
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("unit", ["\n", "x\n"])
def test_r4_extraction_streams_instead_of_materialising_every_line(unit: str) -> None:
    """R4：4 MiB 全换行 / 两字一行的输入，峰值内存与耗时有上界；修前按行建对象约 0.7 GiB、6 s。"""

    text = unit * (4 * 1024 * 1024 // len(unit))
    tracemalloc.start()
    started = time.perf_counter()
    try:
        output = ToolResultExtractor(8_000).extract(text, tool_name="bash", tool_status="error")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert time.perf_counter() - started < 10
    assert peak < 64 * 1024 * 1024
    assert len(output) <= 8_000


# ---------------------------------------------------------------- 只有媒体的工具结果


def _claude_code_turn(tool_content: object) -> list[dict[str, object]]:
    return [
        {"type": "user", "message": {"role": "user", "content": "看看这张截图"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/tmp/shot.png"}}],
            },
        },
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": tool_content}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "看到了"}]}},
    ]


@pytest.mark.parametrize("payload_chars", [370_096, 2_000])
def test_image_only_tool_results_are_stored_as_a_description(payload_chars: int) -> None:
    """B28：只有图片、没有文字的工具结果，适配器原样给出段列表；降载曾对它 json.dumps 抛 TypeError，整轮存不进去。"""

    payload = "A" * payload_chars
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": payload}}
    adapted = ConversationAdapterRegistry.with_builtins().adapt(
        "claude_code",
        _claude_code_turn([image]),
        ConversationAdapterContext("conversation-1", 0, datetime(2026, 10, 23, tzinfo=UTC)),
    )
    result = next(item for item in adapted.batch.messages if item.role is ConversationMessageRole.TOOL_RESULT)

    reduced = ConversationToolResultReducer().reduce(result)

    assert reduced.content_mode is ConversationToolResultContentMode.OMITTED
    assert reduced.content == f"工具结果只有媒体、没有文字，原始载荷未保存：image（image/png，载荷 {payload_chars:,} 字节）。"
    assert reduced.original_size_bytes == payload_chars
    assert reduced.original_sha256 == hashlib.sha256(payload.encode()).hexdigest()


def test_plugin_media_placeholders_carry_the_original_payload_size_and_digest() -> None:
    """Q6：插件占位带原载荷的字节数与 sha256，本地服务按它记；经不经插件同一条记录同一种解释。"""

    payload = "iVBORw0KGgo" + "A" * 4_085
    raw = ({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": payload}},)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    placeholder = ({"type": "image", "media_type": "image/jpeg", "omitted_bytes": 4_096, "omitted_sha256": digest},)

    via_plugin = ConversationToolResultReducer().reduce(_tool_result(placeholder))
    direct = ConversationToolResultReducer().reduce(_tool_result(raw))

    assert via_plugin.content_mode is ConversationToolResultContentMode.OMITTED
    assert "image（image/jpeg，载荷 4,096 字节）" in via_plugin.content
    assert (via_plugin.original_size_bytes, via_plugin.original_sha256) == (4_096, digest)
    assert via_plugin.to_dict() == direct.to_dict()


def test_a_placeholder_without_a_digest_is_not_media() -> None:
    assert describe_media_only(({"type": "image", "media_type": "image/png", "omitted_bytes": 10},)) is None


def test_several_media_parts_add_their_sizes_and_chain_their_digests() -> None:
    first, second = "A" * 100, "data:image/png;base64," + "B" * 50
    described = describe_media_only(
        (
            {"type": "image", "data": first, "mimeType": "image/png"},
            {"type": "input_image", "image_url": second},
        )
    )
    digests = [hashlib.sha256(value.encode()).hexdigest() for value in (first, second)]

    assert described is not None
    assert described.original_size_bytes == 100 + len(second)
    assert described.original_sha256 == hashlib.sha256("\n".join(digests).encode()).hexdigest()


def test_the_write_path_summarizes_a_read_result_by_its_content(tmp_path: Path) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    jobs = MemoryJobStore(tmp_path / "workflow", path_lock, memory_root=tmp_path / "memory")
    planner = ConversationRetentionPlanner(ConversationSegmentationConfig(), token_estimator=lambda _message: 1)
    enqueuer = ConversationMemoryEnqueuer(journal, jobs, planner)
    address = ConversationAddress("conversation-1", date(2026, 10, 23))
    call = message(
        1,
        ConversationMessageRole.TOOL_CALL,
        {"file_path": "/repo/big.log"},
        tool_call_id="call-1",
        tool_name="Read",
    )
    lines = _normal_lines(3_000)
    lines[1_500] = "ERROR: replica 3 refused the write"
    result = _tool_result("\n".join(lines), tool_name="Read")
    batch = ConversationBatch(
        "conversation-1",
        (
            message(0, ConversationMessageRole.PROMPT, "读一下这个文件"),
            call,
            result,
            message(3, ConversationMessageRole.COMPLETION, "读完了"),
        ),
    )

    enqueuer.append(address, batch)
    live = journal.read_live(address)

    assert live is not None
    stored = live.messages[2]
    assert stored.content_mode is ConversationToolResultContentMode.SUMMARIZED
    assert _kind(stored.content) == "纯文本"
    assert "ERROR: replica 3 refused the write" in stored.content
    assert live.messages[1].content == {"file_path": "/repo/big.log"}  # 调用本身（含路径）全量在记忆里
