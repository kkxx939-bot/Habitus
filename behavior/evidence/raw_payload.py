"""外部 Raw Payload 的唯一 JSON 信任边界。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from behavior._validation import json_value_snapshot
from behavior.config import BehaviorConfig
from behavior.errors import BehaviorEvidenceSchemaError
from foundation.integrity import canonical_digest, canonical_json


@dataclass(frozen=True, slots=True)
class RawPayloadSnapshot:
    canonical_text: str
    digest: str

    def detached_copy(self) -> Any:
        return json.loads(self.canonical_text)


class RawPayloadCodec:
    """只处理 JSON 结构，不解释任何 Behavior 领域字段。"""

    def __init__(self, config: BehaviorConfig) -> None:
        self.config = config

    def snapshot(self, value: object) -> RawPayloadSnapshot:
        try:
            normalized = json_value_snapshot(
                value,
                "raw_payload",
                maximum_chars=self.config.store.max_json_bytes,
                maximum_items=max(
                    self.config.evidence.max_payload_items,
                    self.config.evidence.max_batch_size,
                ),
                maximum_depth=self.config.evidence.max_payload_depth,
                reject_inline_data=False,
            )
            text = canonical_json(normalized)
        except (TypeError, ValueError) as exc:
            raise BehaviorEvidenceSchemaError(
                "raw payload is not canonical bounded JSON"
            ) from exc
        return RawPayloadSnapshot(text, canonical_digest(normalized))
