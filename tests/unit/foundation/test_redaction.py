from __future__ import annotations

import pytest

from habitus.foundation.redaction import redact_secrets, redact_sensitive_text


def test_redaction_keeps_cause_after_path() -> None:
    assert redact_sensitive_text("failed to open /tmp/x.json: permission denied") == (
        "failed to open [PATH]: permission denied"
    )


def test_redaction_keeps_text_between_multiple_paths() -> None:
    redacted = redact_sensitive_text('copy /Users/Alice Smith/a.txt: denied; moved "/srv/b c/d.txt" ok')

    assert redacted == 'copy [PATH]: denied; moved "[PATH]" ok'


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\Alice Smith\secret.txt",
        "C:/Users/Alice Smith/secret.txt",
        r"\\server\Private Share\Alice Smith\secret.txt",
        r"\Users\Alice Smith\secret.txt",
        r"C:Users\Alice Smith\secret.txt",
        "//server/share/Alice Smith/secret.txt",
        "file://server/share/Alice Smith/secret.txt",
        "smb://server/Alice Smith/secret.txt",
    ],
)
def test_redaction_keeps_cause_after_windows_and_network_paths(path: str) -> None:
    redacted = redact_sensitive_text(f"failed at {path}: access denied")

    assert redacted == "failed at [PATH]: access denied"


def test_redaction_keeps_last_line_of_multiline_traceback() -> None:
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "/Users/alice/project/habitus/app.py", line 10, in lifespan\n'
        "    await runtime.start()\n"
        '  File "/Users/alice/project/habitus/client.py", line 42, in start\n'
        '    raise RuntimeError("provider check failed: 401 Unauthorized")\n'
        "RuntimeError: provider check failed: 401 Unauthorized"
    )

    redacted = redact_sensitive_text(traceback_text)

    assert "/Users/alice" not in redacted
    assert 'File "[PATH]", line 42, in start' in redacted
    assert redacted.endswith("RuntimeError: provider check failed: 401 Unauthorized")


def test_redaction_still_hides_secrets_split_across_lines() -> None:
    redacted = redact_sensitive_text("Authorization: Bearer\n  secret-value\napi_key:\n  sk-private")

    assert "secret-value" not in redacted
    assert "sk-private" not in redacted


def test_redaction_ends_an_unquoted_path_at_the_end_of_its_line() -> None:
    """路径逐行识别：行尾的路径不吞下一行（下一行常是真正的报错）。"""

    redacted = redact_sensitive_text("cannot open /Users/alice/notes.txt\nPermissionError denied")

    assert redacted == "cannot open [PATH] PermissionError denied"


@pytest.mark.parametrize("name", ["api_key", "api-key", "apikey", "authorization", "password", "secret", "token"])
def test_redaction_hides_named_secret_values(name: str) -> None:
    redacted = redact_sensitive_text(f"connect failed {name}=hunter2-value; retry")

    assert redacted == f"connect failed {name}=[REDACTED]; retry"


@pytest.mark.parametrize(
    "echoed",
    [
        "The security token[AKLTabc123XYZ] included in the request is invalid.",
        "The AccessKey[AKLTabc123XYZ] is not valid",
        "access_key_id [AKLTabc123XYZ] was rejected",
        "secret key[AKLTabc123XYZ] mismatch",
    ],
)
def test_a_credential_echoed_in_brackets_by_a_cloud_error_is_hidden(echoed: str) -> None:
    """云服务 401 回包把被拒的凭据夹在方括号里（例如 ``security token[…]``）：日志与启动失败那一行都去掉它。"""

    for redact in (redact_secrets, redact_sensitive_text):
        redacted = redact(f"provider request failed: 401 InvalidAccessKey: {echoed}")
        assert "AKLTabc123XYZ" not in redacted
        assert "[REDACTED]" in redacted
        assert redacted.startswith("provider request failed: 401 InvalidAccessKey: ")


def test_redact_secrets_keeps_paths_and_layout() -> None:
    """只清凭据的那一种给本机终端看：路径与换行原样留着。"""

    assert redact_secrets("lock /tmp/a b/http.lock held; api_key=sk-x\nnext") == "lock /tmp/a b/http.lock held; api_key=[REDACTED]\nnext"
