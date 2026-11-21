"""摘要压缩被模型以"这一条输入有问题"拒收过的来源组（裁定 108）：同一组来源不再重问，来源变了（多了、少了、换了）才再试。

一个 Conversation 一个文件 ``<root>/summary_refusals/<地址摘要>.json``，只存来源组的指纹（阶段 + 来源 id 与摘要值的
规范 JSON 的 SHA-256）、类别和时间，不存摘要正文。这是可重建的旁路：文件丢了只会把那组来源再问一次。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes, regular_file_exists
from habitus.memory.conversation.layout import ConversationAddress, ConversationLayout
from habitus.model_client import ModelErrorRule, model_failure
from habitus.pre.conversation import ConversationRangeSummaryStage, ConversationSummarySourceRef

#: 一个 Conversation 最多记多少组（保护闸；满了丢最早的，最多让那组再问一次）。
_MAX_ENTRIES = 256
_MAX_BYTES = 256 * 1024


def input_refusal(error: BaseException) -> str | None:
    """压缩调模型失败是"这一条输入有问题"时给出类别名，否则 None（按全流程统一的判定入口）。"""

    failure = model_failure(error)
    if failure is None or failure.rule is not ModelErrorRule.INPUT or failure.kind is None:
        return None
    return failure.kind.value


def refusal_key(stage: ConversationRangeSummaryStage, sources: Sequence[ConversationSummarySourceRef]) -> str:
    """来源组的指纹：阶段 + 每个来源的 id 与摘要值。"""

    material = canonical_json({"stage": stage.value, "sources": [source.to_dict() for source in sources]})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ConversationCompactionRefusals:
    def __init__(self, layout: ConversationLayout) -> None:
        self.layout = layout

    def path(self, address: ConversationAddress) -> Path:
        name = hashlib.sha256(self.layout.lock_key(address).encode("utf-8")).hexdigest()
        return self.layout.root / "summary_refusals" / f"{name}.json"

    def keys(self, address: ConversationAddress) -> frozenset[str]:
        return frozenset(entry["key"] for entry in self._read(address))

    def record(self, address: ConversationAddress, key: str, *, kind: str, at: datetime) -> None:
        entries = [entry for entry in self._read(address) if entry["key"] != key]
        entries.append({"key": key, "kind": kind, "refused_at": at.isoformat()})
        encoded = (canonical_json({"refusals": entries[-_MAX_ENTRIES:]}) + "\n").encode("utf-8")
        atomic_replace_bytes(self.path(address), encoded, artifact_root=self.layout.root)

    def _read(self, address: ConversationAddress) -> list[dict[str, str]]:
        path = self.path(address)
        if not regular_file_exists(path, artifact_root=self.layout.root):
            return []
        try:
            payload = json.loads(read_regular_bytes(path, artifact_root=self.layout.root, max_bytes=_MAX_BYTES))
            entries = payload["refusals"]
            return [
                {"key": str(entry["key"]), "kind": str(entry["kind"]), "refused_at": str(entry["refused_at"])}
                for entry in entries
            ]
        except (ValueError, KeyError, TypeError):
            return []  # 旁路坏了只会让那组来源再问一次，不挡压缩


__all__ = ["ConversationCompactionRefusals", "input_refusal", "refusal_key"]
