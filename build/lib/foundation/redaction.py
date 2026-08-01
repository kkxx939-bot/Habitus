"""跨公开边界复用的敏感文本清理。"""

from __future__ import annotations

import re

_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_NAMED_SECRET = re.compile(r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*[^\s,;]+")
_PREFIXED_SECRET = re.compile(r"(?i)\b(?:ghp|github_pat|sk|xox[baprs])[-_][A-Za-z0-9._-]+")
_FILE_URL = re.compile(r"(?i)\bfile://(?:[^/\s]+)?/.*")
_NETWORK_PATH_URL = re.compile(r"(?i)\b(?:smb|nfs|afp)://[^/\s]+/.*")
_FORWARD_UNC_PATH = re.compile(r"(?<![A-Za-z0-9:])//[^/\s]+/.*")
_UNC_PATH = re.compile(r"(?<![A-Za-z0-9])\\\\(?:\?\\)?.*")
_WINDOWS_ROOTED_PATH = re.compile(r"(?<![A-Za-z0-9\\])\\(?![\\?])[^\\]+(?:\\.*)?")
_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/].*")
_WINDOWS_DRIVE_RELATIVE_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:(?![\\/])[^\\/:]+[\\/].*")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9/])/(?!/)[^/]+(?:/.*)?")


def redact_sensitive_text(value: str) -> str:
    """折叠空白并清理凭据与各平台绝对路径。"""

    if not isinstance(value, str):
        raise TypeError("redacted value must be text")
    normalized = " ".join(value.split())
    normalized = _BEARER_SECRET.sub("Bearer [REDACTED]", normalized)
    normalized = _NAMED_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", normalized)
    normalized = _PREFIXED_SECRET.sub("[REDACTED]", normalized)
    normalized = _FILE_URL.sub("[PATH]", normalized)
    normalized = _NETWORK_PATH_URL.sub("[PATH]", normalized)
    normalized = _FORWARD_UNC_PATH.sub("[PATH]", normalized)
    normalized = _UNC_PATH.sub("[PATH]", normalized)
    normalized = _WINDOWS_ROOTED_PATH.sub("[PATH]", normalized)
    normalized = _WINDOWS_PATH.sub("[PATH]", normalized)
    normalized = _WINDOWS_DRIVE_RELATIVE_PATH.sub("[PATH]", normalized)
    return _POSIX_PATH.sub("[PATH]", normalized)


__all__ = ["redact_sensitive_text"]
