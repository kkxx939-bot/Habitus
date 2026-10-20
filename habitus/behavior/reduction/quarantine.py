"""归约里确定性失败的东西进隔离区（26 号清单 2.4.2、2.4.3）。

以前它们既不删也不进账本，归类又排在 stage 之前，于是每轮都重新调一次模型归类、每轮报同一句。现在：

- **stage 里确定性失败的链**（没有可读名、物化失败）：原判断的副本进隔离区，从判断存储里撤下，换成一条覆盖同一段时间、
  同一批观测的「没读懂」判断。行为树只有"融合 → 归约"一扇写入门，所以不直接写树：下一轮这条判断照常归约成「没读懂」
  空白，下游看到的是"这段看了没读懂"，不是"没做"（缺失必须记录）。它引用的观测随空白发布照常释放。
- **解析失败的判断**：读不出时间，没法落成空白；副本进隔离区、从存储撤下，留信号。

顺序是崩溃安全的依据：先存副本 → 再写「没读懂」判断（身份由链决定，重来幂等）→ 最后撤原判断。中途崩溃，下一轮
那条链还在、又会失败一次、再走一遍，每一步都是幂等的。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.reduction.chains import BehaviorChain
from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.lifecycle import Quarantine

_UNREADABLE_SCHEMA = "behavior_quarantine_unreadable_v1"


class ChainQuarantine:
    def __init__(self, judgements: BehaviorJudgementStore, quarantine: Quarantine) -> None:
        if not isinstance(judgements, BehaviorJudgementStore):
            raise TypeError("judgements must be BehaviorJudgementStore")
        if not isinstance(quarantine, Quarantine):
            raise TypeError("quarantine must be Quarantine")
        self.judgements = judgements
        self.quarantine = quarantine

    def isolate_chain(self, chain: BehaviorChain, reason: str) -> str:
        """返回顶替它的「没读懂」判断的身份。"""

        identities = sorted({item.judgement_id for item in chain.consumed})
        records = [record for identity in identities if (record := self.judgements.read(identity)) is not None]
        if not records:
            raise BehaviorReductionError(f"chain {chain.chain_digest} has no stored judgement left to isolate")
        self.quarantine.put(
            f"chain:{chain.chain_digest}",
            _encode(records),
            reason=reason,
            meta={"kind": "chain", "judgements": ",".join(identities)},
        )
        unreadable = unreadable_record(chain.chain_digest, records)
        self.judgements.put_payload(unreadable)
        for identity in identities:
            self.judgements.discard(identity)
        return str(unreadable["judgement_id"])

    def isolate_record(self, record: Mapping[str, Any], reason: str) -> None:
        identity = record.get("judgement_id")
        if not isinstance(identity, str) or not identity:
            raise BehaviorReductionError("a judgement record without an identity cannot be isolated")
        self.quarantine.put(f"judgement:{identity}", _encode([record]), reason=reason, meta={"kind": "judgement"})
        self.judgements.discard(identity)


def unreadable_record(chain_digest: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """覆盖这几条判断那一段的一条「没读懂」判断：时间取最早开始到最晚看到，观测与来源取并集，其余留空。"""

    earliest = min(records, key=lambda item: _instant(item["started_at"]))
    latest = max(records, key=lambda item: _instant(item["last_observed_at"]))
    head = min(records, key=lambda item: (_instant(item["started_at"]), str(item["judgement_id"])))
    return {
        "judgement_id": canonical_digest({"schema_version": _UNREADABLE_SCHEMA, "chain_digest": chain_digest}),
        "judged_at": max((str(item["judged_at"]) for item in records), key=_instant),
        "evidence_ready_at": max((str(item["evidence_ready_at"]) for item in records), key=_instant),
        "started_at": earliest["started_at"],
        "last_observed_at": latest["last_observed_at"],
        "observation_ids": sorted({str(value) for item in records for value in item["observation_ids"]}),
        "source_refs": sorted({str(value) for item in records for value in item["source_refs"]}),
        "subjects": [],
        "behavior": None,
        "goal": None,
        "summary": None,
        "basis": [],
        "status": None,
        "status_basis": None,
        "relations": [],
        "fusion_version": head["fusion_version"],
        "prompt_version": head["prompt_version"],
    }


def _instant(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


def _encode(records: Sequence[Mapping[str, Any]]) -> bytes:
    return json.dumps(list(records), ensure_ascii=False, sort_keys=True).encode("utf-8")


__all__ = ["ChainQuarantine", "unreadable_record"]
