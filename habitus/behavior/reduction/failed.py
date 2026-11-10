"""归约里出了错误的链（26 号清单 2.4.2、2.4.3；裁定 67；裁定 94 第 3 条）。

以前它们既不删也不进账本，归类又排在 stage 之前，于是每轮都重新调一次模型归类、每轮报同一句。现在：

- **stage 里出错的链**：原判断放进夜里待处理的堆，从判断存储里撤下，换成一条覆盖同一段时间、同一批观测的「错误」判断
  （空判断、依据 ``error``）。行为树只有"融合 → 归约"一扇写入门，所以不直接写树：下一轮这条判断照常归约成「错误」空白，
  下游看到的是"这段看不到"，不是"没做"（缺失必须记录）；那一天照常定稿，不等它。它引用的观测随空白发布照常释放。
  进堆的原因定夜里试不试：没有可读名、物化失败是我们自己处理不了的（``DETERMINISTIC``，夜里不白试、报出等人修）；
  归类时模型调不通的（``UNAVAILABLE``）夜里前置处理放回判断存储、再归约一次（裁定 94 第 3 条：不再"下轮再归"拖住那天）。
- **当晚重试又失败的链**：它的「错误」早已归约过（消费账本里有）——不再造第二条（同一条链身份的第二段空白会和账本撞车），
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
from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.reduction.payloads import ERROR_BASIS
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.behavior.retry import (
    Admission,
    Attempt,
    NoRetryBacklog,
    RetryBacklog,
    RetryCause,
    RetryItem,
    RetryKind,
    retry_id,
)
from habitus.foundation.integrity import canonical_digest

_ERROR_SCHEMA = "behavior_failed_chain_unreadable_v1"  # 身份的 schema 名不改：同一条链顶替判断的身份跨版本不变


class FailedChains:
    def __init__(
        self,
        judgements: BehaviorJudgementStore,
        *,
        ledger: BehaviorReductionLedger,
        clock: Callable[[], datetime],
        backlog: RetryBacklog | None = None,
    ) -> None:
        self.judgements = judgements
        self.ledger = ledger
        self.backlog: RetryBacklog = backlog if backlog is not None else NoRetryBacklog()
        self._clock = clock

    def fail_chain(
        self,
        record: ReducibleJudgement,
        *,
        consumed: frozenset[str],
        cause: RetryCause = RetryCause.DETERMINISTIC,
    ) -> tuple[str | None, Admission | None]:
        """返回顶替它的「错误」判断的身份与进堆的结果；当晚重试又失败（那条「错误」已归约过，``consumed`` 里有，
        是这一轮归约读好的消费账本）时两样都是 None。``cause`` 是它为什么出错（定夜里试不试）。"""

        identities = [record.judgement_id]
        records = [stored for identity in identities if (stored := self.judgements.read(identity)) is not None]
        if not records:
            raise BehaviorReductionError(f"chain {record.chain_digest} has no stored judgement left to set aside")
        failed = error_record(record.chain_digest, records)
        if failed["judgement_id"] in consumed:
            for identity in identities:
                self.judgements.discard(identity)
            return None, None
        admission = self.backlog.put(
            RetryItem(
                retry_id(RetryKind.CHAIN, record.chain_digest),
                RetryKind.CHAIN,
                cause,
                _instant(failed["started_at"]).date(),
                self._clock(),
                {"records": [dict(item) for item in records]},
            )
        )
        self.judgements.put_payload(failed)
        for identity in identities:
            self.judgements.discard(identity)
        return str(failed["judgement_id"]), admission

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
        """放回了、还没归约就断了的那几条原判断撤下（不让它们之后在白天落进已经做过重试收尾的那一天，裁定 68 第 3 条）。"""

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


class ChainHandler:
    """重试堆里链的处理器（裁定 67、68）：原判断放回判断存储、等归约；放回与"补回了没有"都归 ``FailedChains``。"""

    def __init__(self, failed: FailedChains) -> None:
        self.failed = failed

    async def attempt(self, item: RetryItem) -> Attempt:
        return Attempt.RESTORED if self.failed.restore(item) else Attempt.DETERMINISTIC

    def recovered(self, item: RetryItem) -> bool:
        return self.failed.recovered(item)

    def abandon(self, item: RetryItem) -> None:
        self.failed.withdraw(item)


def error_record(chain_digest: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """覆盖这几条判断那一段的一条「错误」判断：时间取最早开始到最晚看到，观测与来源取并集，依据 ``error``，其余留空。"""

    earliest = min(records, key=lambda item: _instant(item["started_at"]))
    latest = max(records, key=lambda item: _instant(item["last_observed_at"]))
    head = min(records, key=lambda item: (_instant(item["started_at"]), str(item["judgement_id"])))
    return {
        "judgement_id": canonical_digest({"schema_version": _ERROR_SCHEMA, "chain_digest": chain_digest}),
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
        "status_basis": ERROR_BASIS,
        "fusion_version": head["fusion_version"],
        "prompt_version": head["prompt_version"],
    }


def _instant(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


__all__ = ["ChainHandler", "FailedChains", "error_record"]
