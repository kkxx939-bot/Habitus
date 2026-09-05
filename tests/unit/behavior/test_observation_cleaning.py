from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from habitus.behavior.observation import (
    BehaviorObservation,
    BehaviorObservationAdapterRegistry,
    BehaviorObservationBatch,
    BehaviorObservationConfig,
    BehaviorObservationEnvelope,
    BehaviorObservationError,
    BehaviorObservationLimitError,
    BehaviorObservationProtocolError,
    BehaviorObservationStore,
)
from habitus.behavior.observation.model import _identity
from habitus.foundation.integrity import canonical_digest

TZ8 = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 13, 20, 0, 0, tzinfo=TZ8)
CONFIG = BehaviorObservationConfig()
PROTOCOL = "habitus_behavior_observation_v1"
OBSERVER = "home-a/hall"


def raw_observation(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "occurred_at": NOW.isoformat(),
        "available_at": (NOW + timedelta(milliseconds=1500)).isoformat(),
        "modality": "vision",
        "semantics": "有人开门进入玄关",
        "participants": ["家庭成员A"],
        "knowledge_state": "observed",
        "confidence": 0.92,
        "evidence_refs": ["cam-front:frame-88213"],
    }
    payload.update(overrides)
    return payload


def adapt(*observations: dict[str, object], observer_id: str = OBSERVER) -> BehaviorObservationBatch:
    registry = BehaviorObservationAdapterRegistry.load_default()
    return registry.adapt(
        PROTOCOL,
        {"observations": list(observations) if observations else [raw_observation()]},
        observer_id=observer_id,
        config=CONFIG,
    )


def envelope(
    batch: BehaviorObservationBatch,
    *,
    delivery_seed: str = "delivery-1",
    protocol: str = PROTOCOL,
) -> BehaviorObservationEnvelope:
    return BehaviorObservationEnvelope.create(
        observer_id=batch.observer_id,
        protocol=protocol,
        batch=batch,
        delivery_id=canonical_digest(delivery_seed),
        recorded_at=NOW + timedelta(seconds=4),
        config=CONFIG,
    )


def observation(**overrides: object) -> BehaviorObservation:
    values: dict[str, object] = {
        "observer_id": OBSERVER,
        "occurred_at": NOW,
        "available_at": NOW,
        "modality": "vision",
        "semantics": "有人开门进入玄关",
        "participants": ["家庭成员A"],
        "knowledge_state": "observed",
        "confidence": 0.92,
        "evidence_refs": ["cam-front:frame-88213"],
    }
    values.update(overrides)
    return BehaviorObservation.create(config=CONFIG, **values)  # type: ignore[arg-type]


# --- available_at 契约 ---------------------------------------------------------------


def test_available_at_is_required_and_cannot_precede_occurred_at() -> None:
    """available_at 是防标签泄漏的地基：它只能等于或晚于事情发生的时间。"""

    with pytest.raises(BehaviorObservationProtocolError):
        adapt({key: value for key, value in raw_observation().items() if key != "available_at"})
    with pytest.raises(BehaviorObservationError):
        adapt(raw_observation(available_at=(NOW - timedelta(milliseconds=1)).isoformat()))
    same_moment = adapt(raw_observation(available_at=NOW.isoformat()))
    assert same_moment.observations[0].available_at == same_moment.observations[0].occurred_at


def test_dst_fold_cannot_bypass_the_available_at_invariant() -> None:
    """同一 tzinfo 对象的 aware datetime 按墙钟比较，夏令时回拨会让两个时刻"相等"。

    比较必须落在绝对时刻上，否则 available_at 可以比 occurred_at 早整整一小时而校验放行，
    且这种值由 ``datetime.fromtimestamp(epoch, tz=ZoneInfo(...))`` 自然产生，无需恶意构造。
    """

    new_york = ZoneInfo("America/New_York")
    later = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=1)
    earlier = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=0)
    assert earlier.astimezone(UTC) < later.astimezone(UTC)
    assert not (earlier < later)  # 墙钟比较看不出先后，这正是缺陷的来源

    with pytest.raises(BehaviorObservationError):
        observation(occurred_at=later, available_at=earlier)


def test_delivery_cannot_carry_semantics_that_are_not_available_yet() -> None:
    future = adapt(raw_observation(available_at="9999-01-01T00:00:00+00:00"))
    with pytest.raises(BehaviorObservationError):
        envelope(future)


# --- 本地时区可还原 -------------------------------------------------------------------


def test_local_calendar_day_survives_the_durable_round_trip() -> None:
    """规范序列化把 datetime 折成 UTC，本地偏移必须单独保存，否则行为树会落错日期目录。"""

    midnight = datetime(2026, 8, 13, 0, 30, tzinfo=TZ8)
    original = observation(occurred_at=midnight, available_at=midnight)
    assert original.utc_offset_minutes == 480

    record = original.to_dict()
    assert record["occurred_at"] == "2026-08-12T16:30:00.000000Z"  # 落盘确实是 UTC
    restored = BehaviorObservation.from_dict(record, config=CONFIG)

    assert restored.local_occurred_at.isoformat() == "2026-08-13T00:30:00+08:00"
    assert restored.local_occurred_at.date().isoformat() == "2026-08-13"
    # 比字典而不是比对象：aware datetime 的 == 比的是绝对时刻，偏移丢失它一概看不出来。
    assert restored.to_dict() == record


# --- 身份 -----------------------------------------------------------------------------


def test_identity_covers_the_observer_so_independent_sensors_corroborate() -> None:
    """两路感知佐证同一事实是融合层算 confidence 与 conflicts 的核心输入。

    若观测者不进身份，两条内容相同的观测会撞同一个 observation_id，在同一批里被当成重复拒收。
    """

    front = observation(observer_id="home-a/cam-front")
    hall = observation(observer_id="home-a/mic-hall")
    assert front.observation_id != hall.observation_id


def test_identity_is_content_derived_and_normalizes_signed_zero() -> None:
    assert observation().observation_id == observation().observation_id
    assert observation(semantics="有人离开玄关").observation_id != observation().observation_id
    # -0.0 与 0.0 数值相等却序列化成不同文本，不归一会让同一条内容得到两个身份。
    assert observation(confidence=-0.0).observation_id == observation(confidence=0.0).observation_id


def test_direct_construction_normalizes_types_and_rejects_illegal_content() -> None:
    """摘要自洽必须能推出内容合法，否则可以绕过 create 造出写得进、读不回的记录。"""

    fields: dict[str, object] = {
        "observer_id": OBSERVER,
        "occurred_at": NOW.astimezone(UTC),
        "available_at": NOW.astimezone(UTC),
        "utc_offset_minutes": 480,
        "modality": "vision",
        "semantics": "有人开门",
        "participants": ["家庭成员A"],
        "knowledge_state": "observed",
        "confidence": 0.9,
        "evidence_refs": ["cam:1"],
    }
    built = BehaviorObservation(observation_id=canonical_digest(_identity(fields)), **fields)  # type: ignore[arg-type]
    # frozen 只冻结属性绑定；list 必须归一成元组，否则构造后还能改内容而身份不变。
    assert isinstance(built.participants, tuple)
    assert isinstance(built.evidence_refs, tuple)

    for illegal in (
        {"knowledge_state": "corrected"},
        {"confidence": 7.5},
        {"modality": "thermal"},
        {"available_at": (NOW - timedelta(hours=1)).astimezone(UTC)},
    ):
        broken = {**fields, **illegal}
        with pytest.raises(BehaviorObservationError):
            BehaviorObservation(observation_id=canonical_digest(_identity(broken)), **broken)  # type: ignore[arg-type]


# --- 清洗层的边界 ---------------------------------------------------------------------


def test_cleaning_keeps_low_confidence_observations() -> None:
    """清洗层只做清洗：低置信度照收，是否采信由后续融合层在语义里裁量。"""

    assert observation(confidence=0.01).confidence == 0.01
    for invalid in (-0.01, 1.01, True, "0.5", float("nan"), float("inf")):
        with pytest.raises(BehaviorObservationError):
            observation(confidence=invalid)


def test_adapter_rejects_unknown_and_missing_fields() -> None:
    with pytest.raises(BehaviorObservationProtocolError):
        adapt({**raw_observation(), "room": "hall"})
    with pytest.raises(BehaviorObservationProtocolError):
        adapt({key: value for key, value in raw_observation().items() if key != "semantics"})
    with pytest.raises(BehaviorObservationProtocolError):
        BehaviorObservationAdapterRegistry.load_default().adapt(
            "some_other_agent_v1", {"observations": []}, observer_id=OBSERVER, config=CONFIG
        )


def test_observation_without_a_subject_is_not_a_behavior_observation() -> None:
    """无主体即非行为观测：这是结构判断，不是"看起来没价值"的语义过滤。

    它同时挡掉上游在无人时段发出的无活动心跳，以及无主体的环境变化。
    """

    with pytest.raises(BehaviorObservationError):
        adapt(raw_observation(participants=[]))
    with pytest.raises(BehaviorObservationError):
        observation(participants=[])


def test_controlled_vocabularies_are_enforced() -> None:
    with pytest.raises(BehaviorObservationError):
        adapt(raw_observation(modality="thermal"))
    with pytest.raises(BehaviorObservationError):
        adapt(raw_observation(knowledge_state="guessed"))
    # corrected 属于 Outcome 的事后修正语义，不能出现在一条原始观测上。
    with pytest.raises(BehaviorObservationError):
        adapt(raw_observation(knowledge_state="corrected"))


def test_identifiers_are_normalized_so_idempotency_cannot_be_split() -> None:
    delivery = canonical_digest("delivery-1")
    baseline = BehaviorObservationEnvelope.source_identity(OBSERVER, delivery)
    for variant in (f" {OBSERVER}", f"{OBSERVER}\n", "home\x00a", ""):
        with pytest.raises(BehaviorObservationError):
            BehaviorObservationEnvelope.source_identity(variant, delivery)
    assert BehaviorObservationEnvelope.source_identity(OBSERVER, delivery) == baseline

    # registry 按小写解析协议名；envelope 若保留大小写，同一交付重投会被误判为内容冲突。
    lower = envelope(adapt())
    upper = envelope(adapt(), protocol=PROTOCOL.upper())
    assert upper.protocol == PROTOCOL
    assert upper.record_digest == lower.record_digest


def test_extreme_timestamps_stay_inside_the_module_error_contract() -> None:
    for value in ("0001-01-01T00:00:00+08:00", "9999-12-31T23:59:59-08:00"):
        with pytest.raises(BehaviorObservationError):
            observation(occurred_at=value, available_at=value)
    for value in (12345, None, datetime(2026, 8, 13, 20, 0)):
        with pytest.raises(BehaviorObservationError):
            observation(occurred_at=value, available_at=value)
    with pytest.raises(BehaviorObservationError):
        odd = NOW.astimezone(timezone(timedelta(seconds=90)))
        observation(occurred_at=odd, available_at=odd)


def test_capacity_bounds_are_operational_not_semantic() -> None:
    with pytest.raises(BehaviorObservationLimitError):
        adapt(raw_observation(semantics="长" * (CONFIG.max_semantics_chars + 1)))
    with pytest.raises(BehaviorObservationLimitError):
        adapt(raw_observation(participants=[f"人{index}" for index in range(CONFIG.max_participants + 1)]))
    with pytest.raises(BehaviorObservationLimitError):
        adapt(raw_observation(evidence_refs=[f"e{index}" for index in range(CONFIG.max_evidence_refs + 1)]))
    with pytest.raises(BehaviorObservationLimitError):
        adapt(*[raw_observation() for _ in range(CONFIG.max_observations_per_batch + 1)])
    with pytest.raises(BehaviorObservationError):
        adapt(raw_observation(evidence_refs=[]))
    # 容量错误必须落在观测错误谱系内，否则调小配置就能让既有数据静默漏出上层的兜底。
    assert issubclass(BehaviorObservationLimitError, BehaviorObservationError)


def test_batch_requires_one_observer_and_chronological_order() -> None:
    later = raw_observation(
        occurred_at=(NOW + timedelta(seconds=3)).isoformat(),
        available_at=(NOW + timedelta(seconds=4)).isoformat(),
        semantics="说了一句「我回来了」",
        modality="audio",
        knowledge_state="reported",
    )
    with pytest.raises(BehaviorObservationError):
        adapt(later, raw_observation())
    assert len(adapt(raw_observation(), later).observations) == 2
    with pytest.raises(BehaviorObservationError):
        BehaviorObservationBatch(observer_id="home-b", observations=(observation(),))
    with pytest.raises(BehaviorObservationError):
        BehaviorObservationBatch(observer_id=OBSERVER, observations=())


# --- 存储 -----------------------------------------------------------------------------


def test_store_reuses_the_existing_record_when_a_delivery_is_replayed(tmp_path) -> None:
    """真实重投是"同 payload、不同 recorded_at"：网络重试会重新盖时间戳。

    字节完全相同的重复写入不会触发底层冲突，因此走不到 put 的复用分支——那正是本用例要覆盖的。
    """

    store = BehaviorObservationStore(tmp_path, config=CONFIG)
    batch = adapt()
    first = store.put(envelope(batch))
    replayed = BehaviorObservationEnvelope.create(
        observer_id=batch.observer_id,
        protocol=PROTOCOL,
        batch=batch,
        delivery_id=canonical_digest("delivery-1"),
        recorded_at=NOW + timedelta(seconds=30),
        config=CONFIG,
    )
    assert replayed.record_digest != first.record_digest
    assert replayed.payload_digest == first.payload_digest
    assert store.put(replayed) == first
    assert store.list() == (first,)


def test_store_rejects_a_reused_delivery_id_carrying_other_content(tmp_path) -> None:
    store = BehaviorObservationStore(tmp_path, config=CONFIG)
    original = envelope(adapt())
    store.put(original)
    conflicting = envelope(adapt(raw_observation(semantics="有人离开玄关")))
    assert conflicting.source_id == original.source_id
    with pytest.raises(BehaviorObservationError):
        store.put(conflicting)


def test_store_enumeration_ignores_entries_it_did_not_write(tmp_path) -> None:
    """macOS 上 Finder 看一眼目录就会生成 .DS_Store；为它整库硬失败等于永久瘫痪枚举。"""

    store = BehaviorObservationStore(tmp_path, config=CONFIG)
    stored = store.put(envelope(adapt()))
    directory = store.envelope_root
    (directory / ".DS_Store").write_bytes(b"\x00")
    (directory / "notes.txt").write_text("scratch", encoding="utf-8")
    (directory / "nested").mkdir()
    assert store.list() == (stored,)


def test_store_refuses_to_persist_bytes_it_cannot_read_back(tmp_path) -> None:
    """写路径若比读路径宽松，毒记录会永久占死身份并让枚举失效；必须在落盘前失败。"""

    store = BehaviorObservationStore(tmp_path, config=CONFIG)
    oversized = BehaviorObservationStore(
        tmp_path, config=BehaviorObservationConfig(max_observations_per_batch=1)
    )
    batch = adapt(
        raw_observation(),
        raw_observation(
            occurred_at=(NOW + timedelta(seconds=1)).isoformat(),
            available_at=(NOW + timedelta(seconds=2)).isoformat(),
            semantics="走向客厅",
        ),
    )
    with pytest.raises(BehaviorObservationError):
        oversized.put(envelope(batch))
    assert store.list() == ()
    assert list(store.envelope_root.glob("*.json")) == []


def test_store_round_trip_and_tampering_detection(tmp_path) -> None:
    store = BehaviorObservationStore(tmp_path, config=CONFIG)
    original = store.put(envelope(adapt()))
    assert store.read(original.source_id).to_dict() == original.to_dict()

    # 只改 observer_id 会先被"批次属于另一个观测者"拦下，三道摘要校验一次都不会执行；
    # 因此必须篡改那些只有摘要能识破的字段。
    for field, value in (
        ("recorded_at", "2026-08-13T13:00:00.000000Z"),
        ("payload_digest", canonical_digest("forged")),
        ("record_digest", canonical_digest("forged")),
        ("delivery_id", canonical_digest("another-delivery")),
    ):
        tampered = {**original.to_dict(), field: value}
        with pytest.raises(BehaviorObservationError):
            BehaviorObservationEnvelope.from_dict(tampered, config=CONFIG)

    inner = original.to_dict()
    inner["batch"]["observations"][0]["semantics"] = "有人离开玄关"
    with pytest.raises(BehaviorObservationError):
        BehaviorObservationEnvelope.from_dict(inner, config=CONFIG)


def test_a_duplicated_observation_in_one_batch_is_deduplicated_not_rejected() -> None:
    """同一批里两条内容完全相同的观测（ASR 重复吐出）只保留一条；此前整批拒收。"""

    raw = raw_observation()
    batch = adapt(raw, dict(raw), raw_observation(semantics="人在洗手"))
    assert len(batch.observations) == 2
    assert len({item.observation_id for item in batch.observations}) == 2
