"""两条 lane 共用的落盘序列：凭据 → 判断 → 回执 → 覆盖记录。

先落凭据再落判断：归约落树时要读凭据；顺序反过来的话，中途崩溃会留下一条读不到凭据的判断。回执与覆盖最后落，
它们在就说明这个单位写完了；判断已落、回执没落就是"写到一半"，由各 lane 认回来接着写完。

"写到一半"只按凭据认（这一组凭据下落盘的判断），不看判断记录上的版本：版本里带着提示词版本，升级之后重启，
按版本认就认不出上一版写下的那条，同一个单位会再写一条。接着写完时沿用那条判断落盘时的版本，身份与内容都不变。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.derivation import derive_judgements, judgement_payload
from habitus.behavior.fusion.judgement import BehaviorJudgement, BehaviorJudgementBatch
from habitus.behavior.fusion.receipt import build_fusion_receipt
from habitus.behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.observation import BehaviorObservationEnvelope, BehaviorObservationStore


@dataclass(frozen=True)
class FusionVersions:
    """一条判断是哪条 lane、哪一版规则 / 提示词写的。"""

    fusion_version: str
    prompt_version: str

    @classmethod
    def of(cls, payload: dict[str, Any]) -> FusionVersions:
        """一条已落盘判断的版本。"""

        return cls(fusion_version=payload["fusion_version"], prompt_version=payload["prompt_version"])


def written_judgements(judgements: BehaviorJudgementStore, observation_ids: Sequence[str]) -> list[dict[str, Any]]:
    """这一组凭据下已经落盘的判断（不看版本）。"""

    owned = list(observation_ids)
    return [item for item in judgements.list() if item["observation_ids"] == owned]


def publish_judgement(
    *,
    observations: BehaviorObservationStore,
    judgements: BehaviorJudgementStore,
    receipts: BehaviorFusionReceiptStore,
    coverage: BehaviorCoverageIndex,
    envelope: BehaviorObservationEnvelope,
    judgement: BehaviorJudgement,
    versions: FusionVersions,
    judged_at: datetime,
    validation_attempts: int,
    primary_subject: str,
    sources: tuple[str, ...] = (),
) -> str:
    """按固定顺序写一个单位：凭据、判断、回执、覆盖记录，返回判断的编号。同一单位重写幂等。

    判断与回执先在内存里派生好、校验过，才落凭据（裁定 94 第 8 条）：派生不过的单位一个文件都不留——落了凭据却永远等不来
    回执，那份凭据会把封口视界一直钉在它那一刻。之后的几步只剩写盘，中途崩了由各 lane 的"写到一半"接着写完。"""

    evidence = envelope.batch.observations
    derived = derive_judgements(
        BehaviorJudgementBatch((judgement,)),
        evidence,
        source_refs=(envelope.source_id,),
        judged_at=judged_at,
        fusion_version=versions.fusion_version,
        prompt_version=versions.prompt_version,
    )
    receipt = build_fusion_receipt(
        derived,
        evidence,
        source_refs=(envelope.source_id,),
        prompt_version=versions.prompt_version,
        validation_attempts=validation_attempts,
        primary_subject=primary_subject,
        judged_at=judged_at,
    )
    payloads = [judgement_payload(item) for item in derived]
    observations.put(envelope)
    for payload in payloads:
        judgements.put_payload(payload)
    coverage.record(receipts.put(receipt), sources=sources)
    return derived[0].judgement_id


__all__ = ["FusionVersions", "publish_judgement", "written_judgements"]
