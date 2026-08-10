"""预测样本 Schema 的受控词表与格式常量。"""

import re

RECORD_ID = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
URI = re.compile(r"^[a-z][a-z0-9+.-]*://[^\s]+$")
URI_HEX = frozenset("0123456789abcdefABCDEF")
URI_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
FACT_CATEGORIES = frozenset({"semantic", "environment", "object_state", "device_state", "human_state"})
KNOWLEDGE_STATES = frozenset({"observed", "reported", "inferred", "corrected", "unknown"})
STEP_KINDS = frozenset({"event", "action", "phase"})
ATTRIBUTIONS = frozenset({"direct", "supported", "temporal_only", "unknown"})


__all__ = [
    "ATTRIBUTIONS",
    "FACT_CATEGORIES",
    "KNOWLEDGE_STATES",
    "RECORD_ID",
    "SHA256",
    "STEP_KINDS",
    "URI",
    "URI_HEX",
    "URI_UNRESERVED",
]
