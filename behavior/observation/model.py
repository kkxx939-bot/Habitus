"""云侧行为 agent 送入的规范行为观测；清洗层的唯一数据契约。

上游的 VLM 与 ASR 位于 Habitus 之外，它们先把感知结果变成行为语义，再由 agent 送进来；
本层永远不接触原始帧、音频或识别片段，只保存对它们的引用。

## 本层做什么、不做什么

清洗层是确定性的、无模型的、可重放的：协议归一、结构与枚举校验、时间归一、内容身份派生、
交付级幂等和容量边界。它**可以拒绝，但不能解释**——能因为结构不合法、时间不合法、超出容量或
重复交付而拒绝一条输入，但不能因为"这条看起来没意义"而丢弃。低置信度观测照常收下，是否采信
属于后续融合层的语义判断。

这里的幂等只到**交付**粒度：同一 ``delivery_id`` 重投复用既有记录。**跨交付的观测级重复由
融合层负责**（见 ``store`` 模块的 ``TODO(BHV-FUSION-003)``），本层不做，也不应声称做了。

## ``participants`` 必须非空：无主体即非行为观测

一条观测必须至少有一个主体。这是**结构判断**而不是语义判断——行为 Event 的定义是"某个主体
的一个有意识的目标"，没有主体就无从谈起，因此无主体的输入在定义上不是行为观测，与"这条看起来
没价值"那种语义过滤完全不同，不违反本层"可以拒绝但不能解释"的纪律。

它同时挡掉两类输入：上游在无人时段发出的**无活动心跳**（本通道只接受行为观测，liveness 信号
应走运维通道），以及**无主体的环境变化**（如"门被风吹开"）。后者曾被考虑作为 ``trigger`` 的
证据来源而单独设一类，最终决定一并挡掉：规则保持干净，环境证据的价值等有真实数据后再评估。

## 两条明确不做的事

以下两项曾被提出并否决，理由都是 Habitus 与上游的职责边界；再次提出前请先确认该边界已改变。

1. **不记录上游模型版本。** 云侧 VLM/ASR 换代会让语义分布迁移，但上游换不换模型不是 Habitus
   的问题——Habitus 负责处理记忆与行为，不管理上游版本。这与 coding agent 那条 lane 的混版
   契约不矛盾：那里的 ``projector_version`` 记录的是 Habitus **自己的**投影实现版本。
2. **观测不携带时段。** 观测只有 ``occurred_at`` 一个时间点，不设 ``ended_at``。事件的时间
   跨度需要 LLM 在事件融合时判断；让观测携带时间范围，等于让上游提前替融合层切定事件边界。

## ``available_at`` 是本层最重要的字段

``occurred_at`` 是"这件事什么时候发生"，``available_at`` 是"这条语义什么时候第一次可被下游
使用"。摄像头 20:00:00.0 拍到人开门，云侧推理与送达耗时 1.5 秒，则 ``occurred_at`` 是
20:00:00.0 而 ``available_at`` 是 20:00:01.5——在这 1.5 秒里"有人回家"这条语义在世界上还
不存在，任何下游都不可能拿它做决策。

这个延迟只有上游知道，因此它是**必填**字段，缺失即拒绝，绝不用 ``occurred_at`` 顶替：整条
预测链的防标签泄漏都建立在它上面（预测侧的每个切点都以它为可见性 cutoff——落进行为树后
即 occurrence 的 ``onset_available_at`` 与 basis 各步的 ``available_at``）。把它填成观测时间，模型就会学到它在线上
永远拿不到的信息，而这种失真在离线指标上完全看不出来。

注意本层只能拦住"字段缺失"这一种误用；``available_at == occurred_at`` 是合法的零延迟取值，
与"上游偷懒直接填了发生时间"在数据上无法区分。下游不应假设本层提供了比"字段存在且不早于
发生时间"更强的保证。

## 时间一律以 UTC 表示，本地偏移单独保存

内存与磁盘上的 ``occurred_at`` / ``available_at`` 恒为 UTC，本地偏移由 ``utc_offset_minutes``
显式承载。两个原因：

1. **比较必须落在绝对时刻上。** CPython 的 aware datetime 在 ``self.tzinfo is other.tzinfo``
   时按墙钟裸比而不按瞬时比；夏令时回拨当天同一墙钟出现两次（``fold=0/1``），真实相差一小时
   却被判为相等。任何用 ``ZoneInfo`` 把 epoch 转本地时间的上游桥接层都会自然产出这种值，于是
   ``available_at`` 早于 ``occurred_at`` 的核心不变量会被静默绕过。全部先归一到 UTC 再比较，
   这条缝就不存在。
2. **本地日历日必须可还原。** 行为树地址要求 ``occurred_on`` 是行为发生地的**本地**日历日期，
   而规范序列化会把 datetime 一律折成 UTC。若不单独保存偏移，东八区凌晨的观测回读后会落到
   前一天，融合层再也无法还原正确日期。``local_occurred_at`` 提供确定性的还原。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any

from behavior.observation.config import BehaviorObservationConfig
from behavior.observation.errors import BehaviorObservationError, BehaviorObservationLimitError
from foundation.integrity import canonical_digest, canonicalize

OBSERVATION_SCHEMA_VERSION = "behavior_observation_v1"
BATCH_SCHEMA_VERSION = "behavior_observation_batch_v1"
ENVELOPE_SCHEMA_VERSION = "behavior_observation_envelope_v1"
_ENVELOPE_IDENTITY_SCHEMA = "behavior_observation_envelope_identity_v1"

# 观测只能记录被直接感知、被主体自述或由上游推断的事实。这组词表归观测契约自己所有
#（旧行为树词表随 TODO(BHV-TREE-REBUILD-001) 退役后，这里是它唯一的定义处）；事后修正
# 语义不属于一条原始观测。
OBSERVATION_KNOWLEDGE_STATES = frozenset({"observed", "reported", "inferred"})

_MAX_UTC_OFFSET_MINUTES = 24 * 60


class BehaviorObservationModality(str, Enum):
    """产生该观测语义的上游感知模态。"""

    VISION = "vision"
    AUDIO = "audio"


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BehaviorObservationError(f"{label} must be lowercase SHA-256 text")
    return value


def normalized_identifier(value: object, label: str, *, max_chars: int) -> str:
    """校验一个参与身份派生的标识符。

    ``observer_id`` 与 ``protocol`` 都直接进入身份公式，因此必须先归一：多一个换行就会变成
    另一个观测者，重投随即不再幂等。这里的严格程度与其它文本字段保持一致，不再是只判非空。
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise BehaviorObservationError(f"{label} must be non-empty normalized text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BehaviorObservationError(f"{label} contains control characters")
    if len(value) > max_chars:
        raise BehaviorObservationLimitError(f"{label} exceeds its configured character limit")
    return value


def aware_timestamp(value: object, label: str) -> datetime:
    """把输入解析为带时区、整分钟偏移的时刻，仍保留其原始偏移。"""

    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BehaviorObservationError(f"{label} must be ISO-8601") from exc
    if not isinstance(parsed, datetime):
        raise BehaviorObservationError(f"{label} must be a datetime")
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None:
        raise BehaviorObservationError(f"{label} must be timezone-aware")
    if offset.microseconds or offset.total_seconds() % 60:
        raise BehaviorObservationError(f"{label} UTC offset must use whole minutes")
    return parsed


def to_utc(value: datetime, label: str) -> datetime:
    """归一到 UTC；极端年份的偏移换算会溢出，收敛成本模块的契约异常。"""

    try:
        return value.astimezone(UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise BehaviorObservationError(f"{label} is outside the representable time range") from exc


def offset_minutes(value: datetime) -> int:
    offset = value.utcoffset()
    assert offset is not None
    return int(offset.total_seconds() // 60)


def bounded_text(value: object, label: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BehaviorObservationError(f"{label} must be non-empty text")
    normalized = value.strip()
    if any(not character.isprintable() and character not in "\n\t" for character in normalized):
        raise BehaviorObservationError(f"{label} contains control characters")
    if len(normalized) > max_chars:
        raise BehaviorObservationLimitError(f"{label} exceeds its configured character limit")
    return normalized


def text_tuple(value: object, label: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    """把文本序列归一为元组；容量上限由调用方按 Config 另行检查。

    这里必须落成元组：``frozen`` 只冻结属性绑定，塞进来的 list 事后仍可被修改，那会造出
    "内容已变而内容身份未变"的对象。
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise BehaviorObservationError(f"{label} must be an array")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise BehaviorObservationError(f"{label} item must be non-empty text")
        normalized = item.strip()
        if any(not character.isprintable() and character not in "\n\t" for character in normalized):
            raise BehaviorObservationError(f"{label} item contains control characters")
        items.append(normalized)
    resolved = tuple(items)
    if not allow_empty and not resolved:
        raise BehaviorObservationError(f"{label} must not be empty")
    if len(resolved) != len(set(resolved)):
        raise BehaviorObservationError(f"{label} must not contain duplicates")
    return resolved


def require_record(
    value: object,
    *,
    expected: set[str],
    label: str,
    schema_version: str | None = None,
) -> Mapping[str, Any]:
    """解码耐久记录前的统一守卫：键集合必须完全相等，Schema 版本必须吻合。"""

    if not isinstance(value, Mapping) or set(value) != expected:
        raise BehaviorObservationError(f"{label} schema is invalid")
    if schema_version is not None and value.get("schema_version") != schema_version:
        raise BehaviorObservationError(f"{label} schema version is invalid")
    return value


@dataclass(frozen=True)
class BehaviorObservation:
    """一条已经由上游成为语义、且可独立追溯到证据的行为观测。

    ``__post_init__`` 执行与配置无关的全部不变量校验并归一字段类型，因此"摘要自洽"能够
    推出"内容合法"。只比对摘要是不够的：那样可以绕过 ``create`` 造出非法对象，写进存储后
    再也读不回来。
    """

    observation_id: str
    observer_id: str
    occurred_at: datetime
    available_at: datetime
    utc_offset_minutes: int
    modality: BehaviorObservationModality
    semantics: str
    participants: tuple[str, ...]
    knowledge_state: str
    confidence: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.observation_id, "observation_id")
        object.__setattr__(
            self,
            "observer_id",
            normalized_identifier(self.observer_id, "observer_id", max_chars=_UNBOUNDED_IDENTIFIER),
        )
        for name in ("occurred_at", "available_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
                raise BehaviorObservationError(f"{name} must already be normalized to UTC")
        if self.available_at < self.occurred_at:
            raise BehaviorObservationError("available_at cannot precede occurred_at")
        if (
            isinstance(self.utc_offset_minutes, bool)
            or not isinstance(self.utc_offset_minutes, int)
            or abs(self.utc_offset_minutes) > _MAX_UTC_OFFSET_MINUTES
        ):
            raise BehaviorObservationError("utc_offset_minutes must be a whole-minute UTC offset")
        try:
            object.__setattr__(self, "modality", BehaviorObservationModality(self.modality))
        except ValueError as exc:
            raise BehaviorObservationError("modality is invalid") from exc
        object.__setattr__(self, "semantics", bounded_text(self.semantics, "semantics", max_chars=_UNBOUNDED_TEXT))
        object.__setattr__(
            self, "participants", text_tuple(self.participants, "participants", allow_empty=False)
        )
        if self.knowledge_state not in OBSERVATION_KNOWLEDGE_STATES:
            raise BehaviorObservationError(
                f"knowledge_state must be one of {sorted(OBSERVATION_KNOWLEDGE_STATES)}"
            )
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise BehaviorObservationError("confidence must be numeric")
        # ``-0.0`` 与 ``0.0`` 数值相等却序列化成不同文本，会让同一条内容得到两个身份。
        normalized_confidence = float(self.confidence) + 0.0
        if not 0.0 <= normalized_confidence <= 1.0:
            raise BehaviorObservationError("confidence must be between zero and one")
        object.__setattr__(self, "confidence", normalized_confidence)
        object.__setattr__(
            self, "evidence_refs", text_tuple(self.evidence_refs, "evidence_refs", allow_empty=False)
        )
        if self.observation_id != canonical_digest(self._identity_payload()):
            raise BehaviorObservationError("observation_id does not match observation content")

    @property
    def local_occurred_at(self) -> datetime:
        """按保存的本地偏移还原发生时刻；融合层据此确定行为树的本地日历日。"""

        return self.occurred_at.astimezone(timezone(timedelta(minutes=self.utc_offset_minutes)))

    @classmethod
    def create(
        cls,
        *,
        observer_id: object,
        occurred_at: object,
        available_at: object,
        modality: object,
        semantics: object,
        participants: object,
        knowledge_state: object,
        confidence: object,
        evidence_refs: object,
        config: BehaviorObservationConfig,
        utc_offset_minutes: object | None = None,
    ) -> BehaviorObservation:
        """校验并归一一条观测；``observation_id`` 始终由内容派生，不接受上游给定。

        ``utc_offset_minutes`` 只在读路径显式给出：落盘时间恒为 UTC，本地偏移无法再从
        时间戳自身推出，必须由记录里的该字段提供。写路径留空，由 ``occurred_at`` 自带的
        偏移派生。
        """

        if not isinstance(config, BehaviorObservationConfig):
            raise TypeError("config must be BehaviorObservationConfig")
        resolved_observer = normalized_identifier(
            observer_id, "observer_id", max_chars=config.max_identifier_chars
        )
        # 枚举与数值必须在算摘要之前归一：``canonical_digest`` 遇到非法枚举或 NaN 会抛出
        # 本模块契约之外的异常，调用方的 except 接不住。
        try:
            resolved_modality = BehaviorObservationModality(modality)
        except ValueError as exc:
            raise BehaviorObservationError("modality is invalid") from exc
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise BehaviorObservationError("confidence must be numeric")
        resolved_confidence = float(confidence) + 0.0
        if not 0.0 <= resolved_confidence <= 1.0:
            raise BehaviorObservationError("confidence must be between zero and one")
        aware_occurred = aware_timestamp(occurred_at, "occurred_at")
        aware_available = aware_timestamp(available_at, "available_at")
        occurred_utc = to_utc(aware_occurred, "occurred_at")
        available_utc = to_utc(aware_available, "available_at")
        semantics_text = bounded_text(semantics, "semantics", max_chars=config.max_semantics_chars)
        participant_values = text_tuple(participants, "participants", allow_empty=False)
        if len(participant_values) > config.max_participants:
            raise BehaviorObservationLimitError("participants exceeds its configured item limit")
        evidence_values = text_tuple(evidence_refs, "evidence_refs", allow_empty=False)
        if len(evidence_values) > config.max_evidence_refs:
            raise BehaviorObservationLimitError("evidence_refs exceeds its configured item limit")
        for label, values in (("participants", participant_values), ("evidence_refs", evidence_values)):
            for item in values:
                if len(item) > config.max_reference_chars:
                    raise BehaviorObservationLimitError(f"{label} item exceeds its configured character limit")
        if utc_offset_minutes is None:
            resolved_offset = offset_minutes(aware_occurred)
        elif isinstance(utc_offset_minutes, bool) or not isinstance(utc_offset_minutes, int):
            raise BehaviorObservationError("utc_offset_minutes must be a whole-minute UTC offset")
        else:
            resolved_offset = utc_offset_minutes
        values_map: dict[str, Any] = {
            "observer_id": resolved_observer,
            "occurred_at": occurred_utc,
            "available_at": available_utc,
            "utc_offset_minutes": resolved_offset,
            "modality": resolved_modality,
            "semantics": semantics_text,
            "participants": participant_values,
            "knowledge_state": knowledge_state,
            "confidence": resolved_confidence,
            "evidence_refs": evidence_values,
        }
        return cls(observation_id=_identity_digest(values_map), **values_map)

    def _identity_payload(self) -> dict[str, Any]:
        return _identity(
            {
                "observer_id": self.observer_id,
                "occurred_at": self.occurred_at,
                "available_at": self.available_at,
                "utc_offset_minutes": self.utc_offset_minutes,
                "modality": self.modality,
                "semantics": self.semantics,
                "participants": self.participants,
                "knowledge_state": self.knowledge_state,
                "confidence": self.confidence,
                "evidence_refs": self.evidence_refs,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = canonicalize({**self._identity_payload(), "observation_id": self.observation_id})
        assert isinstance(payload, dict)
        return payload

    @classmethod
    def from_dict(cls, value: object, *, config: BehaviorObservationConfig) -> BehaviorObservation:
        record = require_record(
            value,
            expected={
                "schema_version",
                "observation_id",
                "observer_id",
                "occurred_at",
                "available_at",
                "utc_offset_minutes",
                "modality",
                "semantics",
                "participants",
                "knowledge_state",
                "confidence",
                "evidence_refs",
            },
            label="behavior observation",
            schema_version=OBSERVATION_SCHEMA_VERSION,
        )
        observation = cls.create(
            observer_id=record["observer_id"],
            occurred_at=record["occurred_at"],
            available_at=record["available_at"],
            modality=record["modality"],
            semantics=record["semantics"],
            participants=record["participants"],
            knowledge_state=record["knowledge_state"],
            confidence=record["confidence"],
            evidence_refs=record["evidence_refs"],
            utc_offset_minutes=record["utc_offset_minutes"],
            config=config,
        )
        if observation.observation_id != record["observation_id"]:
            raise BehaviorObservationError("observation_id does not match observation content")
        return observation


# ``__post_init__`` 只负责与配置无关的不变量；长度上限由 ``create`` 按 Config 检查，
# 直接构造时以这两个宽松上界兜底，避免把运维参数硬编码进领域不变量。
_UNBOUNDED_TEXT = 1_000_000
_UNBOUNDED_IDENTIFIER = 4_096


def _identity(values: Mapping[str, Any]) -> dict[str, Any]:
    modality = values["modality"]
    payload = canonicalize(
        {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "observer_id": values["observer_id"],
            "occurred_at": values["occurred_at"],
            "available_at": values["available_at"],
            "utc_offset_minutes": values["utc_offset_minutes"],
            "modality": modality.value if isinstance(modality, BehaviorObservationModality) else modality,
            "semantics": values["semantics"],
            "participants": list(values["participants"]),
            "knowledge_state": values["knowledge_state"],
            "confidence": values["confidence"],
            "evidence_refs": list(values["evidence_refs"]),
        }
    )
    assert isinstance(payload, dict)
    return payload


def _identity_digest(values: Mapping[str, Any]) -> str:
    """values 必须已经由调用方归一；本函数不再兜底非法枚举或非有限数值。"""

    return canonical_digest(_identity(values))


@dataclass(frozen=True)
class BehaviorObservationBatch:
    """一次交付携带的观测序列；按 ``occurred_at`` 单调不减。"""

    observer_id: str
    observations: tuple[BehaviorObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observer_id",
            normalized_identifier(self.observer_id, "observer_id", max_chars=_UNBOUNDED_IDENTIFIER),
        )
        if isinstance(self.observations, list):
            object.__setattr__(self, "observations", tuple(self.observations))
        if not isinstance(self.observations, tuple) or not self.observations:
            raise BehaviorObservationError("observation batch must not be empty")
        if any(not isinstance(item, BehaviorObservation) for item in self.observations):
            raise TypeError("observations must contain BehaviorObservation values")
        if any(item.observer_id != self.observer_id for item in self.observations):
            raise BehaviorObservationError("observation batch contains another observer's observation")
        identities = [item.observation_id for item in self.observations]
        if len(set(identities)) != len(identities):
            raise BehaviorObservationError("observation IDs must be unique within a batch")
        if any(
            current.occurred_at > following.occurred_at
            for current, following in zip(self.observations, self.observations[1:], strict=False)
        ):
            raise BehaviorObservationError("observations must be chronological by occurred_at")

    @property
    def started_at(self) -> datetime:
        return self.observations[0].occurred_at

    @property
    def last_available_at(self) -> datetime:
        return max(item.available_at for item in self.observations)

    def to_dict(self) -> dict[str, Any]:
        payload = canonicalize(
            {
                "schema_version": BATCH_SCHEMA_VERSION,
                "observer_id": self.observer_id,
                "observations": [item.to_dict() for item in self.observations],
            }
        )
        assert isinstance(payload, dict)
        return payload

    @classmethod
    def from_dict(cls, value: object, *, config: BehaviorObservationConfig) -> BehaviorObservationBatch:
        record = require_record(
            value,
            expected={"schema_version", "observer_id", "observations"},
            label="behavior observation batch",
            schema_version=BATCH_SCHEMA_VERSION,
        )
        raw = record["observations"]
        if not isinstance(raw, list):
            raise BehaviorObservationError("behavior observation batch observations must be a list")
        if len(raw) > config.max_observations_per_batch:
            raise BehaviorObservationLimitError("observation batch exceeds its configured item limit")
        return cls(
            observer_id=record["observer_id"],
            observations=tuple(BehaviorObservation.from_dict(item, config=config) for item in raw),
        )


@dataclass(frozen=True)
class BehaviorObservationEnvelope:
    """一次不可变的观测交付；``source_id`` 由观测者与 ``delivery_id`` 确定性派生。"""

    source_id: str
    observer_id: str
    protocol: str
    batch: BehaviorObservationBatch
    delivery_id: str
    payload_digest: str
    recorded_at: datetime
    record_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.source_id, "source_id")
        require_sha256(self.delivery_id, "delivery_id")
        require_sha256(self.payload_digest, "payload_digest")
        require_sha256(self.record_digest, "record_digest")
        if not isinstance(self.batch, BehaviorObservationBatch):
            raise TypeError("batch must be BehaviorObservationBatch")
        object.__setattr__(
            self,
            "observer_id",
            normalized_identifier(self.observer_id, "observer_id", max_chars=_UNBOUNDED_IDENTIFIER),
        )
        object.__setattr__(
            self, "protocol", normalized_protocol(self.protocol, max_chars=_UNBOUNDED_IDENTIFIER)
        )
        if self.batch.observer_id != self.observer_id:
            raise BehaviorObservationError("batch belongs to another observer")
        recorded_at = to_utc(aware_timestamp(self.recorded_at, "recorded_at"), "recorded_at")
        object.__setattr__(self, "recorded_at", recorded_at)
        if self.source_id != self.source_identity(self.observer_id, self.delivery_id):
            raise BehaviorObservationError("source_id does not match the observation delivery identity")
        if self.payload_digest != canonical_digest(self._payload()):
            raise BehaviorObservationError("payload_digest does not match the observation payload")
        if self.record_digest != canonical_digest(self._record_without_digest()):
            raise BehaviorObservationError("record_digest does not match the observation record")

    @classmethod
    def create(
        cls,
        *,
        observer_id: str,
        protocol: str,
        batch: BehaviorObservationBatch,
        delivery_id: str,
        recorded_at: object,
        config: BehaviorObservationConfig,
    ) -> BehaviorObservationEnvelope:
        if not isinstance(config, BehaviorObservationConfig):
            raise TypeError("config must be BehaviorObservationConfig")
        if not isinstance(batch, BehaviorObservationBatch):
            raise TypeError("batch must be BehaviorObservationBatch")
        resolved_observer = normalized_identifier(
            observer_id, "observer_id", max_chars=config.max_identifier_chars
        )
        resolved_protocol = normalized_protocol(protocol, max_chars=config.max_identifier_chars)
        resolved_delivery = require_sha256(delivery_id, "delivery_id")
        resolved_recorded_at = to_utc(aware_timestamp(recorded_at, "recorded_at"), "recorded_at")
        # 语义还没可用就已经被送达，是结构性矛盾而不是时钟抖动；容差由 Config 给出。
        skew = (batch.last_available_at - resolved_recorded_at).total_seconds()
        if skew > config.max_clock_skew_seconds:
            raise BehaviorObservationError(
                "observation delivery carries semantics that are not available yet"
            )
        source_id = cls.source_identity(resolved_observer, resolved_delivery)
        payload = _envelope_payload(
            source_id=source_id,
            observer_id=resolved_observer,
            protocol=resolved_protocol,
            batch=batch,
            delivery_id=resolved_delivery,
        )
        payload_digest = canonical_digest(payload)
        record = canonicalize(
            {**payload, "payload_digest": payload_digest, "recorded_at": resolved_recorded_at}
        )
        return cls(
            source_id=source_id,
            observer_id=resolved_observer,
            protocol=resolved_protocol,
            batch=batch,
            delivery_id=resolved_delivery,
            payload_digest=payload_digest,
            recorded_at=resolved_recorded_at,
            record_digest=canonical_digest(record),
        )

    @staticmethod
    def source_identity(observer_id: str, delivery_id: str) -> str:
        """同一观测者用同一 ``delivery_id`` 重投必得同一身份，重复交付因此天然幂等。"""

        return canonical_digest(
            {
                "schema_version": _ENVELOPE_IDENTITY_SCHEMA,
                "observer_id": normalized_identifier(
                    observer_id, "observer_id", max_chars=_UNBOUNDED_IDENTIFIER
                ),
                "delivery_id": require_sha256(delivery_id, "delivery_id"),
            }
        )

    def _payload(self) -> dict[str, Any]:
        return _envelope_payload(
            source_id=self.source_id,
            observer_id=self.observer_id,
            protocol=self.protocol,
            batch=self.batch,
            delivery_id=self.delivery_id,
        )

    def _record_without_digest(self) -> dict[str, Any]:
        payload = canonicalize(
            {**self._payload(), "payload_digest": self.payload_digest, "recorded_at": self.recorded_at}
        )
        assert isinstance(payload, dict)
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = canonicalize({**self._record_without_digest(), "record_digest": self.record_digest})
        assert isinstance(payload, dict)
        return payload

    @classmethod
    def from_dict(cls, value: object, *, config: BehaviorObservationConfig) -> BehaviorObservationEnvelope:
        record = require_record(
            value,
            expected={
                "schema_version",
                "source_id",
                "observer_id",
                "protocol",
                "batch",
                "delivery_id",
                "payload_digest",
                "recorded_at",
                "record_digest",
            },
            label="behavior observation envelope",
            schema_version=ENVELOPE_SCHEMA_VERSION,
        )
        return cls(
            source_id=record["source_id"],
            observer_id=record["observer_id"],
            protocol=record["protocol"],
            batch=BehaviorObservationBatch.from_dict(record["batch"], config=config),
            delivery_id=record["delivery_id"],
            payload_digest=record["payload_digest"],
            recorded_at=record["recorded_at"],
            record_digest=record["record_digest"],
        )


def normalized_protocol(value: object, *, max_chars: int) -> str:
    """协议名一律小写归一：适配器注册表按小写解析，落盘若保留大小写会让同一条交付重投被
    误判为内容冲突。"""

    return normalized_identifier(value, "protocol", max_chars=max_chars).lower()


def _envelope_payload(
    *,
    source_id: str,
    observer_id: str,
    protocol: str,
    batch: BehaviorObservationBatch,
    delivery_id: str,
) -> dict[str, Any]:
    payload = canonicalize(
        {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "source_id": source_id,
            "observer_id": observer_id,
            "protocol": protocol,
            "batch": batch.to_dict(),
            "delivery_id": delivery_id,
        }
    )
    assert isinstance(payload, dict)
    return payload


__all__ = [
    "BATCH_SCHEMA_VERSION",
    "ENVELOPE_SCHEMA_VERSION",
    "OBSERVATION_KNOWLEDGE_STATES",
    "OBSERVATION_SCHEMA_VERSION",
    "BehaviorObservation",
    "BehaviorObservationBatch",
    "BehaviorObservationEnvelope",
    "BehaviorObservationModality",
    "aware_timestamp",
    "normalized_identifier",
    "normalized_protocol",
    "require_record",
    "require_sha256",
    "to_utc",
]
