"""一次融合的耐久回执：只记处置，不复制正文。

判断本身在 ``store.py`` 里有自己的耐久记录，所以回执回到了它本来该是的样子——**一份处置清单**：
这批观测被融合成了哪几条判断、用的哪个版本、什么时候。正文一律不复制。

它回答三个问题：

    这段观测处理过没有        排队扫描据此跳过，不必扫判断存储
    产出的判断在哪            按 judgement_id 指过去
    读不懂的比例是多少        上游语义退化最早的信号
    多少帧不属于主体          场景复杂度的信号（家里来客人、保姆在干活），与"读不懂"是两回事

``receipt_id`` 由片段集合与版本共同派生：同一批观测在同一版本下只有一份回执，混版产物因此可
分治。但**融合不是纯函数**（模型输出会变），所以重放撞车时复用既有回执，而不是像观测交付那样
要求内容逐字节相同——那里的内容是输入的纯函数，这里不是。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from behavior.fusion.derivation import FUSION_VERSION, DurableJudgement, persistable_judgements
from behavior.fusion.errors import BehaviorFusionError
from behavior.observation import BehaviorObservation
from foundation.integrity import canonical_digest, canonicalize

RECEIPT_SCHEMA_VERSION = "behavior_fusion_receipt_v4"
_RECEIPT_IDENTITY_SCHEMA = "behavior_fusion_receipt_identity_v1"

_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "segment_digest",
    "observation_ids",
    "source_refs",
    "fusion_version",
    "prompt_version",
    "validation_attempts",
    "judged_at",
    "judgement_ids",
    "unreadable_observation_ids",
    "out_of_scope_observation_ids",
    "unowned_observation_ids",
    "record_digest",
}


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BehaviorFusionError(f"{label} must be lowercase SHA-256 text")
    return value


def _utc(value: object, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BehaviorFusionError(f"{label} must be ISO-8601") from exc
    if not isinstance(parsed, datetime) or parsed.utcoffset() is None:
        raise BehaviorFusionError(f"{label} must be a timezone-aware datetime")
    try:
        return parsed.astimezone(UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise BehaviorFusionError(f"{label} is outside the representable time range") from exc


def _texts(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise BehaviorFusionError(f"{label} must be an array")
    if any(not isinstance(item, str) or not item for item in value):
        raise BehaviorFusionError(f"{label} item must be non-empty text")
    return tuple(value)


@dataclass(frozen=True)
class BehaviorFusionReceipt:
    """一次融合的处置记录。"""

    receipt_id: str
    segment_digest: str
    observation_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    fusion_version: str
    prompt_version: str
    validation_attempts: int
    judged_at: datetime
    judgement_ids: tuple[str, ...]
    unreadable_observation_ids: tuple[str, ...]
    # 读得懂、但做这件事的不是我们跟踪的那个人。与"读不懂"必须分开：路人多不等于上游在退化，
    # 混进同一个比例会让上游质量信号失真。
    out_of_scope_observation_ids: tuple[str, ...]
    # 读得懂、不属于任何判断的观测（无意识小动作、过渡帧）："看到了、不构成任何事"。它是
    # "允许模型不产出"的出口，占比要作为信号——与"没读懂""旁人"三者口径各自干净。
    unowned_observation_ids: tuple[str, ...]
    record_digest: str

    def __post_init__(self) -> None:
        _sha256(self.receipt_id, "receipt_id")
        _sha256(self.record_digest, "record_digest")
        if not self.observation_ids:
            raise BehaviorFusionError("a fusion receipt must cover at least one observation")
        if not self.source_refs:
            raise BehaviorFusionError("a fusion receipt must record its source deliveries")
        # judgement_ids 允许为空：一段观测里全是别人的行为时，"本段没有主体的行为"本身是有效结论。
        if self.validation_attempts < 1:
            raise BehaviorFusionError("validation_attempts must be at least one")
        for group, label in (
            (self.observation_ids, "observation_ids"),
            (self.judgement_ids, "judgement_ids"),
            (self.unreadable_observation_ids, "unreadable_observation_ids"),
            (self.out_of_scope_observation_ids, "out_of_scope_observation_ids"),
            (self.unowned_observation_ids, "unowned_observation_ids"),
        ):
            for item in group:
                _sha256(item, f"{label} item")
        if self.receipt_id != receipt_identity(
            self.segment_digest, self.fusion_version, self.prompt_version
        ):
            raise BehaviorFusionError("receipt_id does not match its segment and versions")
        # 记了别人的账和漏记一样有害：事后审计据此得出的结论会是错的。
        covered = set(self.observation_ids)
        if not set(self.unreadable_observation_ids) <= covered:
            raise BehaviorFusionError("receipt reports unreadable observations outside its segment")
        if not set(self.out_of_scope_observation_ids) <= covered:
            raise BehaviorFusionError("receipt reports out-of-scope observations outside its segment")
        if not set(self.unowned_observation_ids) <= covered:
            raise BehaviorFusionError("receipt reports unowned observations outside its segment")
        if self.record_digest != canonical_digest(_record(self._values())):
            raise BehaviorFusionError("record_digest does not match the receipt record")

    def _values(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "segment_digest": self.segment_digest,
            "observation_ids": self.observation_ids,
            "source_refs": self.source_refs,
            "fusion_version": self.fusion_version,
            "prompt_version": self.prompt_version,
            "validation_attempts": self.validation_attempts,
            "judged_at": self.judged_at,
            "judgement_ids": self.judgement_ids,
            "unreadable_observation_ids": self.unreadable_observation_ids,
            "out_of_scope_observation_ids": self.out_of_scope_observation_ids,
            "unowned_observation_ids": self.unowned_observation_ids,
        }

    @property
    def out_of_scope_ratio(self) -> float:
        """不属于主体的观测占比；这个数上升说明场景里出现了别人，不是上游在退化。"""

        return len(self.out_of_scope_observation_ids) / len(self.observation_ids)

    @property
    def unreadable_ratio(self) -> float:
        """读不懂的观测占比；这个数上升即上游语义在退化。"""

        return len(self.unreadable_observation_ids) / len(self.observation_ids)

    @property
    def unowned_ratio(self) -> float:
        """无归属观测占比——"允许不产出"的出口用得多不多，是压制产出的告警依据。"""

        return len(self.unowned_observation_ids) / len(self.observation_ids)

    def to_dict(self) -> dict[str, Any]:
        payload = canonicalize({**_record(self._values()), "record_digest": self.record_digest})
        assert isinstance(payload, dict)
        return payload

    @classmethod
    def from_dict(cls, value: object) -> BehaviorFusionReceipt:
        if not isinstance(value, Mapping) or set(value) != _RECEIPT_KEYS:
            raise BehaviorFusionError("behavior fusion receipt schema is invalid")
        if value["schema_version"] != RECEIPT_SCHEMA_VERSION:
            raise BehaviorFusionError("behavior fusion receipt schema version is invalid")
        attempts = value["validation_attempts"]
        if isinstance(attempts, bool) or not isinstance(attempts, int):
            raise BehaviorFusionError("validation_attempts must be an integer")
        return cls(
            receipt_id=_sha256(value["receipt_id"], "receipt_id"),
            segment_digest=_sha256(value["segment_digest"], "segment_digest"),
            observation_ids=_texts(value["observation_ids"], "observation_ids"),
            source_refs=_texts(value["source_refs"], "source_refs"),
            fusion_version=str(value["fusion_version"]),
            prompt_version=str(value["prompt_version"]),
            validation_attempts=attempts,
            judged_at=_utc(value["judged_at"], "judged_at"),
            judgement_ids=_texts(value["judgement_ids"], "judgement_ids"),
            unreadable_observation_ids=_texts(
                value["unreadable_observation_ids"], "unreadable_observation_ids"
            ),
            out_of_scope_observation_ids=_texts(
                value["out_of_scope_observation_ids"], "out_of_scope_observation_ids"
            ),
            unowned_observation_ids=_texts(
                value["unowned_observation_ids"], "unowned_observation_ids"
            ),
            record_digest=_sha256(value["record_digest"], "record_digest"),
        )


def segment_identity(fragments: Sequence[BehaviorObservation]) -> str:
    """一批片段的稳定身份：按观测内容身份排序后取摘要，与呈现顺序无关。

    与顺序无关是必需的：交付重叠会让同一批观测以不同顺序进入切段，若身份随顺序变化，同一段观测
    就会被融合两次并留下两份互不相认的回执。
    """

    if isinstance(fragments, (str, bytes)) or not isinstance(fragments, Sequence) or not fragments:
        raise BehaviorFusionError("segment identity requires a non-empty fragment sequence")
    if any(not isinstance(item, BehaviorObservation) for item in fragments):
        raise TypeError("fragments must contain BehaviorObservation values")
    return canonical_digest(
        {
            "schema_version": _RECEIPT_IDENTITY_SCHEMA,
            "observation_ids": sorted({item.observation_id for item in fragments}),
        }
    )


def receipt_identity(segment_digest: str, fusion_version: str, prompt_version: str) -> str:
    """同一批观测在同一实现与提示词版本下只会有一份回执。"""

    return canonical_digest(
        {
            "schema_version": _RECEIPT_IDENTITY_SCHEMA,
            "segment_digest": _sha256(segment_digest, "segment_digest"),
            "fusion_version": fusion_version,
            "prompt_version": prompt_version,
        }
    )


def build_fusion_receipt(
    judgements: Sequence[DurableJudgement],
    fragments: Sequence[BehaviorObservation],
    *,
    source_refs: Sequence[str],
    prompt_version: str,
    validation_attempts: int,
    primary_subject: str,
    judged_at: datetime | None = None,
) -> BehaviorFusionReceipt:
    """从一次融合的产物合成回执。

    ``judgements`` 传**全部**判断（含不属于主体的那些）：主体的行为与没读懂的观测段进
    ``judgement_ids``（与落盘同一口径，见 ``persistable_judgements``），旁人的只留观测身份。
    缺失必须记录——直接丢弃的话，事后没人知道这段里有多少帧属于别人。
    """

    if isinstance(judgements, (str, bytes)) or not isinstance(judgements, Sequence):
        raise BehaviorFusionError("judgements must be a sequence")
    if any(not isinstance(item, DurableJudgement) for item in judgements):
        raise TypeError("judgements must contain DurableJudgement values")
    segment_digest = segment_identity(fragments)
    covered = {item.observation_id for item in fragments}
    in_scope = persistable_judgements(judgements, primary_subject)
    # 与 ``BehaviorJudgementBatch.unreadable_fragment_count`` 必须同一个口径：一条观测若同时
    # 被某条可读判断认领，它已经被读懂了，不该计入。两处口径不一致时，落盘的这一份会永久偏高
    # 且改不了——而它正是"上游语义退化"的告警依据。
    readable = {
        observation_id
        for judgement in judgements
        if judgement.is_readable
        for observation_id in judgement.observation_ids
    }
    unreadable = {
        observation_id
        for judgement in judgements
        if not judgement.is_readable
        for observation_id in judgement.observation_ids
    } - readable
    # tracked 只算**主体的可读判断**认领的观测：in_scope 自 v3 起含没读懂判断，若把它们的
    # 观测也算作"已跟踪"，一条同时被旁人可读判断与没读懂判断认领的帧会从 out_of_scope 里
    # 消失——它明明读懂了、且不属于主体。两个比例各自的语义必须干净。
    tracked = {
        observation_id
        for judgement in in_scope
        if judgement.is_readable
        for observation_id in judgement.observation_ids
    }
    out_of_scope = {
        observation_id
        for judgement in judgements
        if judgement.is_readable and primary_subject not in judgement.subjects
        for observation_id in judgement.observation_ids
    } - tracked
    outside = sorted((unreadable | out_of_scope) - covered)
    if outside:
        raise BehaviorFusionError(f"judgements report observations outside this segment: {outside}")
    # 无归属 = 本段观测里没有任何判断（主体的、旁人的、没读懂的）认领的那些。
    claimed = {observation_id for judgement in judgements for observation_id in judgement.observation_ids}
    unowned = covered - claimed
    values: dict[str, Any] = {
        "receipt_id": receipt_identity(segment_digest, FUSION_VERSION, prompt_version),
        "segment_digest": segment_digest,
        "observation_ids": tuple(sorted(covered)),
        "source_refs": tuple(dict.fromkeys(source_refs)),
        "fusion_version": FUSION_VERSION,
        "prompt_version": prompt_version,
        "validation_attempts": validation_attempts,
        "judged_at": _utc(
            judgements[0].judged_at if judgements else _require_moment(judged_at), "judged_at"
        ),
        "judgement_ids": tuple(item.judgement_id for item in in_scope),
        "unreadable_observation_ids": tuple(sorted(unreadable)),
        "out_of_scope_observation_ids": tuple(sorted(out_of_scope)),
        "unowned_observation_ids": tuple(sorted(unowned)),
    }
    return BehaviorFusionReceipt(**values, record_digest=canonical_digest(_record(values)))


def _require_moment(value: datetime | None) -> datetime:
    if value is None:
        raise BehaviorFusionError("judged_at is required when no judgement was produced")
    return value


def _record(values: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonicalize(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "receipt_id": values["receipt_id"],
            "segment_digest": values["segment_digest"],
            "observation_ids": list(values["observation_ids"]),
            "source_refs": list(values["source_refs"]),
            "fusion_version": values["fusion_version"],
            "prompt_version": values["prompt_version"],
            "validation_attempts": values["validation_attempts"],
            "judged_at": values["judged_at"],
            "judgement_ids": list(values["judgement_ids"]),
            "unreadable_observation_ids": list(values["unreadable_observation_ids"]),
            "out_of_scope_observation_ids": list(values["out_of_scope_observation_ids"]),
            "unowned_observation_ids": list(values["unowned_observation_ids"]),
        }
    )
    assert isinstance(payload, dict)
    return payload


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "BehaviorFusionReceipt",
    "build_fusion_receipt",
    "receipt_identity",
    "segment_identity",
]
