"""密钥抹除（确定性规则，不调模型）。

行为树只增不改，密钥一旦被写进概要就是永久的，所以抹两道（裁定 37）：材料给模型之前一道，
模型写出的原话、概要、步骤落盘之前再一道。规则只认形状明确的密钥；认不出的不在这里兜底。
"""

from __future__ import annotations

import re

REDACTED = "［已抹去的密钥］"

# 密钥前面不能紧挨着拉丁字母或数字（免得从一个长词中间认出"前缀"）；中文、标点、下划线后面的照认。
# 不用 ``\b``：中文字符也算"词字符"，「密钥是sk-…」里 ``\b`` 不成立，会整条漏掉。
_START = r"(?<![A-Za-z0-9])"

_PATTERNS = (
    # 带固定前缀的厂商密钥：sk-…、sk-ant-…、ark-…、ghp_…、github_pat_…、xox?-…、AKIA…、AIza…、hf_…、glpat-…
    re.compile(_START + r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(_START + r"ark-[A-Za-z0-9\-]{16,}"),
    re.compile(_START + r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(_START + r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(_START + r"xox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(_START + r"AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"),
    re.compile(_START + r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(_START + r"hf_[A-Za-z0-9]{20,}"),
    re.compile(_START + r"glpat-[A-Za-z0-9_\-]{16,}"),
    # JWT：三段 base64url
    re.compile(_START + r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    # 请求头里的令牌
    re.compile(r"(?i)" + _START + r"Bearer\s+[A-Za-z0-9._~+/\-]{16,}=*"),
    # PEM 私钥块
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
)
# "名字 = 值"形状的：api_key / apikey / secret / token / password 后面跟着一串。只抹值，名字留着。
# 名字可以是更长的变量名的结尾（``ARK_API_KEY=…``），也可以带引号（JSON 的 ``"api_key": "…"``，
# 以及它被再转义一层之后的 ``\\"api_key\\": \\"…``）。
# 值是一条路径的不抹（``token: /Users/…``、``~/.config/…``、``./…``）——那是放密钥的地方，不是密钥。
_ASSIGNMENT = re.compile(
    r"(?i)" + _START + r"((?:api[_\- ]?key|secret|access[_\- ]?token|token|password|passwd)"
    r"\\{0,2}[\"']?\s*[:=：]\s*\\{0,2}[\"']?)"
    r"(?![/~.])([A-Za-z0-9._~+/\-]{12,}=*)"
)


def redact(text: str) -> str:
    """把文字里形状明确的密钥换成占位。"""

    cleaned = text
    for pattern in _PATTERNS:
        cleaned = pattern.sub(REDACTED, cleaned)
    return _ASSIGNMENT.sub(lambda match: match.group(1) + REDACTED, cleaned)


__all__ = ["REDACTED", "redact"]
