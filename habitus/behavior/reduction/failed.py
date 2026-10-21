"""归约里确定性失败的东西（26 号清单 2.4.2、2.4.3；裁定 67）。

以前它们既不删也不进账本，归类又排在 stage 之前，于是每轮都重新调一次模型归类、每轮报同一句。现在：

- **stage 里确定性失败的链**（没有可读名、物化失败）：原判断放进每晚重试的堆（当晚跑批前放回判断存储、再归约一次），
  从判断存储里撤下，换成一条覆盖同一段时间、同一批观测的「没读懂」判断。行为树只有"融合 → 归约"一扇写入门，所以不直接
  写树：下一轮这条判断照常归约成「没读懂」空白，下游看到的是"这段看了没读懂"，不是"没做"（缺失必须记录）。它引用的观测随
  空白发布照常释放。同一份代码再归约几乎必败：这一类的重试只是"推迟一晚再删"。
- **当晚重试又失败的链**：它的「没读懂」早已归约过（消费账本里有）——不再造第二条（同一条链身份的第二段空白会和账本撞车），
  也不再进堆，原判断撤下就完。
- **解析失败的判断**：读不出时间，没法落成空白，也没法重试；从存储撤下，由调用方留信号计数。

顺序是崩溃安全的依据：先进堆 → 再写「没读懂」判断（身份由链决定，重来幂等）→ 最后撤原判断。中途崩溃，下一轮
那条链还在、又会失败一次、再走一遍，每一步都是幂等的。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.reduction.chains import BehaviorChain
from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.retry import Admission, NoRetryBacklog, RetryBacklog, RetryCause, RetryItem, RetryKind, retry_id
from habitus.foundation.integrity import canonical_digest

_UNREADABLE_SCHEMA = "behavior_failed_chain_unreadable_v1"


class FailedChains:
    def __init__(
        self,
        judgements: BehaviorJudgementStore,
        *,
        ledger: BehaviorReductionLedger,
        clock: Callable[[], datetime],
        backlog: RetryBacklog | None = None,
    ) -> None:
        if not isinstance(judgements, BehaviorJudgementStore):
            raise TypeError("judgements must be BehaviorJudgementStore")
        if not isinstance(ledger, BehaviorReductionLedger):
            raise TypeError("ledger must be BehaviorReductionLedger")
        if backlog is not None and not isinstance(backlog, RetryBacklog):
            raise TypeError("backlog must implement RetryBacklog")
        self.judgements = judgements
        self.ledger = ledger
        self.backlog: RetryBacklog = backlog if backlog is not None else NoRetryBacklog()
        self._clock = clock

    def fail_chain(self, chain: BehaviorChain, *, consumed: frozenset[str]) -> tuple[str | None, Admission | None]:
        """返回顶替它的「没读懂」判断的身份与进堆的结果；当晚重试又失败（那条「没读懂」已归约过，``consumed`` 里有，
        是这一轮归约读好的消费账本）时两样都是 None。"""

        identities = sorted({item.judgement_id for item in chain.consumed})
        records = [record for identity in identities if (record := self.judgements.read(identity)) is not None]
        if not records:
            raise BehaviorReductionError(f"chain {chain.chain_digest} has no stored judgement left to set aside")
        unreadable = unreadable_record(chain.chain_digest, records)
        if unreadable["judgement_id"] in consumed:
            for identity in identities:
                self.judgements.discard(identity)
            return None, None
        admission = self.backlog.put(
            RetryItem(
                retry_id(RetryKind.CHAIN, chain.chain_digest),
                RetryKind.CHAIN,
                RetryCause.DETERMINISTIC,
                _instant(unreadable["started_at"]).date(),
                self._clock(),
                {"records": [dict(record) for record in records]},
            )
        )
        self.judgements.put_payload(unreadable)
        for identity in identities:
            self.judgements.discard(identity)
        return str(unreadable["judgement_id"]), admission

    def restore(self, item: RetryItem) -> bool:
        """当晚重试：链的原判断放回判断存储（身份不变，消费账本里没有它，下一轮归约照常捡起）；载荷坏了放不回为假。"""

        records = item.payload.get("records")
        if not isinstance(records, list) or not records:
            return False
        try:
            for record in records:
                self.judgements.put_payload(record)
        except (BehaviorFusionError, TypeError, ValueError, KeyError):
            return False
        return True

    def withdraw(self, item: RetryItem) -> None:
        """放回了、还没归约就断了的那几条原判断撤下（不让它们之后在白天落进已经跑过批的那一天，裁定 68 第 3 条）。"""

        records = item.payload.get("records")
        if not isinstance(records, list):
            return
        consumed = self.ledger.consumed_judgement_ids()
        for record in records:
            identity = str(record.get("judgement_id"))
            if identity not in consumed:
                self.judgements.discard(identity)

    def recovered(self, item: RetryItem) -> bool:
        """放回的那几条判断这一轮被归约消费了（落成了行为）。"""

        records = item.payload.get("records")
        if not isinstance(records, list):
            return False
        return {str(record.get("judgement_id")) for record in records} <= self.ledger.consumed_judgement_ids()

    def drop_record(self, record: Mapping[str, Any]) -> None:
        identity = record.get("judgement_id")
        if not isinstance(identity, str) or not identity:
            raise BehaviorReductionError("a judgement record without an identity cannot be dropped")
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


__all__ = ["FailedChains", "unreadable_record"]
