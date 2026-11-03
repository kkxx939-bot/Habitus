"""跨公开边界复用的敏感文本清理。"""

from __future__ import annotations

import re

# 路径允许含空格与逗号，到行尾、双引号或"冒号加空白"为止；其后是报错原因，不属于路径。
_PATH_CHAR = r'(?!:(?:\s|$))[^"]'
_PATH_TAIL = f"(?:{_PATH_CHAR})*"

_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_NAMED_SECRET = re.compile(r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*[^\s,;]+")
_PREFIXED_SECRET = re.compile(r"(?i)\b(?:ghp|github_pat|sk|xox[baprs])[-_][A-Za-z0-9._-]+")
# 云服务回包把被拒的凭据原样夹在方括号里（VikingDB / 火山引擎 401：``The security token[AK…] included in the request is invalid``）
_BRACKETED_SECRET = re.compile(
    r"(?i)\b(access[\s_-]?key(?:[\s_-]?id)?|secret(?:[\s_-]?key)?|token|password|credential)\s*\[[^\]\s]+\]"
)
_FILE_URL = re.compile(rf"(?i)\bfile://(?:[^/\s]+)?/{_PATH_TAIL}")
_NETWORK_PATH_URL = re.compile(rf"(?i)\b(?:smb|nfs|afp)://[^/\s]+/{_PATH_TAIL}")
_FORWARD_UNC_PATH = re.compile(rf"(?<![A-Za-z0-9:])//[^/\s]+/{_PATH_TAIL}")
_UNC_PATH = re.compile(rf"(?<![A-Za-z0-9])\\\\(?:\?\\)?{_PATH_TAIL}")
_WINDOWS_ROOTED_PATH = re.compile(rf"(?<![A-Za-z0-9\\])\\(?![\\?])(?:(?!\\){_PATH_CHAR})+(?:\\{_PATH_TAIL})?")
_WINDOWS_PATH = re.compile(rf"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]{_PATH_TAIL}")
_WINDOWS_DRIVE_RELATIVE_PATH = re.compile(rf"(?i)(?<![A-Za-z0-9])[A-Z]:(?![\\/])[^\\/:]+[\\/]{_PATH_TAIL}")
_POSIX_PATH = re.compile(rf"(?<![A-Za-z0-9/])/(?!/)(?:(?!/){_PATH_CHAR})+(?:/{_PATH_TAIL})?")
_PATHS = (
    _FILE_URL,
    _NETWORK_PATH_URL,
    _FORWARD_UNC_PATH,
    _UNC_PATH,
    _WINDOWS_PATH,
    _WINDOWS_DRIVE_RELATIVE_PATH,
    _WINDOWS_ROOTED_PATH,
    _POSIX_PATH,
)


def redact_sensitive_text(value: str) -> str:
    """清理凭据与各平台绝对路径后折叠空白。

    凭据在整段文本上识别（可跨行）；路径逐行识别，止于行尾、双引号或"冒号加空白"，
    所以路径后的报错原因和多行栈的末行异常保留下来。
    """

    text = redact_secrets(value)
    return " ".join(" ".join(_redact_paths(line) for line in text.splitlines()).split())


def redact_secrets(value: str) -> str:
    """只清理凭据（Bearer、``key=…``、带前缀的 token、回包里夹在方括号里的凭据），路径与空白原样保留：
    给本机终端看的一行原因用（路径要留着他才知道是哪儿）。"""

    if not isinstance(value, str):
        raise TypeError("redacted value must be text")
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", value)
    text = _NAMED_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _BRACKETED_SECRET.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
    return _PREFIXED_SECRET.sub("[REDACTED]", text)


def _redact_paths(line: str) -> str:
    for pattern in _PATHS:
        line = pattern.sub("[PATH]", line)
    return line


__all__ = ["redact_secrets", "redact_sensitive_text"]
