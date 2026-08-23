"""重建后的行为树：地址/URI、Schema 双面角色、codec 往返与渲染守卫。"""

from __future__ import annotations

from datetime import timedelta, timezone

import pytest

from behavior import (
    BehaviorAddress,
    BehaviorDirectory,
    BehaviorDocumentCodec,
    BehaviorFieldRole,
    BehaviorKind,
    BehaviorSchemaError,
    BehaviorSchemaRegistry,
    BehaviorURI,
    BehaviorURIError,
)
from behavior.document import BehaviorDocumentMetadata
from behavior.model import behavior_static_directories
from tests.unit.behavior.tree_payloads import (
    DAY,
    OBS_A,
    action_segment_payload,
    gap_payload,
    local,
    occurrence_payload,
)

REGISTRY = BehaviorSchemaRegistry.load_default()
CODEC = BehaviorDocumentCodec(REGISTRY)
NOW = local(23, 0).astimezone(timezone.utc)


# --- 地址与 URI ------------------------------------------------------------------------


def test_tree_has_exactly_two_kinds_and_flat_prefixes() -> None:
    assert [kind.value for kind in BehaviorKind] == ["occurrence", "gap"]
    assert behavior_static_directories() == (("gaps",), ("occurrences",))


def test_occurrence_address_uri_roundtrip_preserves_offset() -> None:
    address = BehaviorAddress.occurrence(DAY, "洗了手", local(19, 30, 18))
    uri = BehaviorURI.from_address(address)
    assert uri.segments[:4] == ("occurrences", "2026", "08", "16")
    assert "+0800" in uri.segments[-1]
    parsed = BehaviorURI.parse(str(uri))
    restored = parsed.to_address()
    assert restored == address
    assert restored.started_at.utcoffset() == timedelta(hours=8)


def test_gap_address_uses_gap_kind_as_leaf_name() -> None:
    address = BehaviorAddress.gap(DAY, "没读懂", local(20, 10))
    uri = BehaviorURI.from_address(address)
    assert uri.segments[0] == "gaps"
    assert uri.segments[-1].startswith("没读懂--")
    assert BehaviorURI.parse(str(uri)).to_address() == address


def test_address_rejects_mismatched_local_day() -> None:
    # 东八区凌晨行为：本地日必须按本地时间判——错给前一天要炸。
    with pytest.raises(ValueError, match="local started_at date"):
        BehaviorAddress.occurrence(DAY, "起夜", local(0, 30) - timedelta(days=1))


def test_old_tree_paths_no_longer_parse() -> None:
    with pytest.raises(BehaviorURIError):
        BehaviorURI("behavior://behaviors/events/2026/08/16")
    with pytest.raises(BehaviorURIError):
        BehaviorURI("behavior://episodes/2026/08/16")


def test_directory_factories_map_to_new_prefixes() -> None:
    assert BehaviorDirectory.occurrences(2026, 8, 16).identity_parts == (
        "occurrences",
        "2026",
        "08",
        "16",
    )
    assert BehaviorDirectory.gaps().identity_parts == ("gaps",)


# --- Schema 双面角色 --------------------------------------------------------------------


def test_occurrence_schema_declares_all_faces() -> None:
    schema = REGISTRY.get(BehaviorKind.OCCURRENCE)
    names = {role: [f.name for f in schema.fields_of(role)] for role in BehaviorFieldRole}
    assert names[BehaviorFieldRole.ADDRESS] == ["occurred_on", "name", "started_at"]
    assert names[BehaviorFieldRole.NUMERIC] == [
        "kind_token",
        "status",
        "status_basis",
        "last_observed_at",
        "onset_available_at",
        "reminded",
    ]
    assert set(names[BehaviorFieldRole.SEMANTIC]) == {
        "goal",
        "summary",
        "subjects",
        "place",
        "original_name",
        "basis",
    }
    assert set(names[BehaviorFieldRole.SYSTEM]) == {
        "judgement_ids",
        "observation_ids",
        "source_refs",
        "fusion_version",
        "reduction_version",
    }


def test_validate_accepts_canonical_and_action_segment_payloads() -> None:
    REGISTRY.validate(BehaviorKind.OCCURRENCE, occurrence_payload())
    REGISTRY.validate(BehaviorKind.OCCURRENCE, action_segment_payload())
    REGISTRY.validate(BehaviorKind.GAP, gap_payload())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"status": "failed"}, "status"),  # 任务态词表已退役
        ({"status_basis": "corrected"}, "status_basis"),
        ({"goal": None}, "without a goal"),  # goal 空却带 basis
        ({"basis": ()}, "must record its basis"),  # goal 非空却无 basis
        ({"subjects": ()}, "at least one subject"),
        ({"original_name": "洗了手"}, "must differ"),  # 消歧记录的原始名不能等于地址名
        ({"last_observed_at": local(19, 0)}, "cannot precede"),
        ({"started_at": local(19, 30, 18).astimezone(timezone.utc)}, "non-zero local"),
        ({"onset_available_at": local(19, 0)}, "cannot precede"),
        ({"judgement_ids": ()}, "at least one judgement"),
    ],
)
def test_occurrence_cross_field_rules(overrides: dict, match: str) -> None:
    with pytest.raises(BehaviorSchemaError, match=match):
        REGISTRY.validate(BehaviorKind.OCCURRENCE, occurrence_payload(**overrides))


def test_basis_steps_must_stay_inside_the_occurrence() -> None:
    stray = occurrence_payload()
    steps = list(stray["basis"])
    steps[0] = {**steps[0], "observation_ids": (("f" * 64),)}
    with pytest.raises(BehaviorSchemaError, match="outside this occurrence"):
        REGISTRY.validate(BehaviorKind.OCCURRENCE, {**stray, "basis": tuple(steps)})


def test_basis_steps_may_fall_outside_the_started_at_window() -> None:
    """supersedes 换链头后 started_at 取新链头，早于它的真实步骤必须照常通过（时间窗规则已删）。"""

    late_head = occurrence_payload(
        started_at=local(19, 33, 0),
        occurred_on=DAY,
        onset_available_at=local(19, 33, 5),
        last_observed_at=local(19, 34, 0),  # 全部观测的最大值，仍 ≥ started_at（该规则可证、保留）
    )
    # basis 第一步仍是 19:30:18 起——比 started_at 早，数据全真，不许拒。
    REGISTRY.validate(BehaviorKind.OCCURRENCE, late_head)
    body = REGISTRY.render_markdown(BehaviorKind.OCCURRENCE, late_head)
    assert "19:30:18" in body  # 早期步骤如实渲染


def test_a_disambiguated_duplicate_is_marked_and_not_counted() -> None:
    """撞车消歧的记录：原始名照存语义面，正文明确写出"统计不计入"。"""

    payload = occurrence_payload(name="洗了手-2", original_name="洗了手")
    REGISTRY.validate(BehaviorKind.OCCURRENCE, payload)
    body = REGISTRY.render_markdown(BehaviorKind.OCCURRENCE, payload)
    assert "**原始名** 洗了手" in body
    assert "统计不计入" in body


def test_zero_offset_never_enters_the_tree() -> None:
    """树上不存在 UTC 时间：+00:00 是上游折 UTC 的事故信号，地址与字段两侧都硬拒。"""

    utc_started = local(19, 30, 18).astimezone(timezone.utc)
    with pytest.raises(ValueError, match="non-zero local"):
        BehaviorAddress.occurrence(utc_started.date(), "洗了手", utc_started)


def test_gap_rules() -> None:
    # 零时长（起止同刻）是合法的单观测段；终点早于起点才是自相矛盾。
    REGISTRY.validate(BehaviorKind.GAP, gap_payload(ended_at=local(20, 10)))
    with pytest.raises(BehaviorSchemaError, match="must not end before"):
        REGISTRY.validate(BehaviorKind.GAP, gap_payload(ended_at=local(20, 9)))
    with pytest.raises(BehaviorSchemaError, match="unreadable gap"):
        REGISTRY.validate(BehaviorKind.GAP, gap_payload(judgement_ids=()))
    # 未观测空白允许空溯源（上游契约未接入）。
    REGISTRY.validate(
        BehaviorKind.GAP,
        gap_payload(gap_kind="未观测", judgement_ids=(), observation_ids=()),
    )


def test_unknown_and_missing_fields_are_rejected() -> None:
    with pytest.raises(BehaviorSchemaError, match="unknown fields"):
        REGISTRY.validate(BehaviorKind.OCCURRENCE, occurrence_payload(confidence=0.9))
    incomplete = occurrence_payload()
    incomplete.pop("kind_token")
    with pytest.raises(BehaviorSchemaError, match="missing required field"):
        REGISTRY.validate(BehaviorKind.OCCURRENCE, incomplete)


# --- codec 往返与渲染守卫 ---------------------------------------------------------------


def test_codec_roundtrip_preserves_local_offsets() -> None:
    metadata = BehaviorDocumentMetadata.initial(NOW)
    document = CODEC.build(BehaviorKind.OCCURRENCE, occurrence_payload(), metadata=metadata)
    # 钉死存储字段本身（不是正文子串）：结构块里的时间必须原样带本地偏移
    assert document.fields["started_at"] == "2026-08-16T19:30:18.000000+08:00"
    assert document.fields["onset_available_at"] == "2026-08-16T19:30:20.000000+08:00"
    assert document.fields["basis"][0]["available_at"] == "2026-08-16T19:30:20.000000+08:00"
    raw = CODEC.encode(document)
    restored = CODEC.decode(raw, expected_address=document.address)
    assert restored == document
    assert CODEC.encode(restored) == raw  # 双向逐字节


def _expected_render_fragments(field, payload) -> list[str]:
    """按字段类型给出"渲染结果里必须出现"的片段；新类型未登记时直接失败，逼守卫同步扩展。"""

    from behavior.schema.model import BehaviorFieldType as T

    value = payload[field.name]
    if field.field_type in (T.STRING, T.OPTIONAL_STRING, T.GAP_KIND):
        return [value] if value is not None else []
    if field.field_type is T.DATE:
        return [value.isoformat()]
    if field.field_type is T.DATETIME:
        return [value.isoformat(timespec="seconds")]
    if field.field_type is T.STRING_LIST:
        return list(value)
    if field.field_type is T.OCCURRENCE_STATUS:
        from behavior.schema.renderers import _STATUS_TEXT

        return [_STATUS_TEXT[value]]
    if field.field_type is T.STATUS_BASIS:
        from behavior.schema.renderers import _BASIS_TEXT

        return [_BASIS_TEXT[value]]
    if field.field_type is T.BASIS_LIST:
        fragments: list[str] = []
        for step in value:
            fragments.append(step["semantics"])
            fragments.append(step["started_at"].strftime("%H:%M:%S"))
            fragments.append(step["available_at"].strftime("%H:%M:%S"))
        return fragments
    if field.field_type is T.BOOLEAN:
        return []  # 布尔靠变异检查（见下），子串检查表达不了
    raise AssertionError(f"渲染守卫未登记的字段类型: {field.field_type}")


@pytest.mark.parametrize(
    ("kind", "payload_factory"),
    [
        (BehaviorKind.OCCURRENCE, occurrence_payload),
        (BehaviorKind.GAP, gap_payload),
    ],
)
def test_render_covers_every_non_system_field(kind, payload_factory) -> None:
    """守卫从 schema 派生：yaml 加一个非 system 字段而 renderer 忘了渲染，这里必须变红。"""

    payload = payload_factory()
    body = REGISTRY.render_markdown(kind, payload)
    schema = REGISTRY.get(kind)
    for field in schema.fields:
        if field.role is BehaviorFieldRole.SYSTEM:
            continue
        for fragment in _expected_render_fragments(field, payload):
            assert fragment in body, f"渲染缺失字段 {field.name} 的内容: {fragment}"


def test_render_boolean_fields_change_the_body() -> None:
    """布尔字段的变异检查：翻转 reminded 必须改变正文，且 True 面有明确文案。"""

    plain = REGISTRY.render_markdown(BehaviorKind.OCCURRENCE, occurrence_payload(reminded=False))
    flagged = REGISTRY.render_markdown(BehaviorKind.OCCURRENCE, occurrence_payload(reminded=True))
    assert plain != flagged
    assert "此前被提醒过" in flagged and "此前被提醒过" not in plain
    assert "completed" not in flagged  # 状态以中文呈现，英文 token 不进正文


def test_render_excludes_system_provenance() -> None:
    payload = occurrence_payload()
    body = REGISTRY.render_markdown(BehaviorKind.OCCURRENCE, payload)
    assert OBS_A not in body
    assert payload["fusion_version"] not in body


def test_gap_render_is_honest_about_both_kinds() -> None:
    unreadable = REGISTRY.render_markdown(BehaviorKind.GAP, gap_payload())
    assert "没能读懂" in unreadable
    uncovered = REGISTRY.render_markdown(
        BehaviorKind.GAP,
        gap_payload(gap_kind="未观测", judgement_ids=(), observation_ids=()),
    )
    assert "不知道发生了什么" in uncovered


def test_identity_roundtrip_with_separator_in_name_and_negative_offset() -> None:
    """叶名分隔符与名字重合、以及负偏移，都必须无损往返到规范身份。"""

    from datetime import datetime, timedelta, timezone

    tricky = BehaviorAddress.occurrence(DAY, "热身--拉伸", local(19, 30, 18))
    restored = BehaviorURI.parse(str(BehaviorURI.from_address(tricky))).to_address()
    assert restored == tricky
    assert restored.name == "热身--拉伸"

    caracas = timezone(timedelta(hours=-4, minutes=-30))
    started = datetime(2026, 8, 16, 22, 30, tzinfo=caracas)
    negative = BehaviorAddress.occurrence(started.date(), "洗手", started)
    uri = BehaviorURI.from_address(negative)
    assert "-0430" in uri.segments[-1]
    parsed = BehaviorURI.parse(str(uri)).to_address()
    assert parsed == negative
    assert parsed.started_at.utcoffset() == timedelta(hours=-4, minutes=-30)


def test_kinds_registry_lives_in_the_address_space() -> None:
    """behavior://kinds.md：树根唯一的登记表节点——可解析、可往返、不冒充文档。"""

    from behavior.uri import BehaviorURINodeType

    uri = BehaviorURI.kinds()
    assert str(uri) == "behavior://kinds.md"
    parsed = BehaviorURI.parse("behavior://kinds.md")
    assert parsed == uri
    assert parsed.node_type is BehaviorURINodeType.REGISTRY
    assert parsed.containing_directory == BehaviorDirectory.root()
    with pytest.raises(BehaviorURIError):
        parsed.to_address()
    # 日期目录里的 kinds.md 不是登记表也不是合法文档地址
    with pytest.raises(BehaviorURIError):
        BehaviorURI("behavior://occurrences/2026/08/16/kinds.md")


def test_registry_path_resolves_to_the_tree_root_file(tmp_path) -> None:
    from behavior import BehaviorTree

    tree = BehaviorTree(tmp_path / "behavior-tree")
    assert tree.path_for_uri("behavior://kinds.md") == tree.root / "kinds.md"


def test_pagination_cursor_order_matches_enumeration_order(tmp_path) -> None:
    """游标序与枚举序必须同一口径：叶名内嵌 --YYYYMMDD 时字面序与 casefold 序会反转，
    口径不一致会让分页边界附近的条目被静默漏掉或重复（评审构造的反例）。"""

    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from behavior import BehaviorDocumentWriter, BehaviorTree
    from infrastructure.store.locks import ProcessLocalLockStore

    cst = _tz(_td(hours=8))
    tree = BehaviorTree(tmp_path / "behavior-tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    for name, started in (
        ("x", _dt(2026, 8, 16, 1, 1, 1, tzinfo=cst)),
        ("x--20260816a", _dt(2026, 8, 16, 2, 2, 2, tzinfo=cst)),
    ):
        writer.publish(
            BehaviorKind.OCCURRENCE,
            occurrence_payload(
                name=name,
                kind_token="测试",
                started_at=started,
                last_observed_at=started + _td(minutes=1),
                onset_available_at=started + _td(seconds=2),
                basis=(),
                goal=None,
                summary=name,
            ),
        )
    full = tree.list_addresses(BehaviorKind.OCCURRENCE)
    paged = []
    cursor = None
    while True:
        page = tree.list_addresses(BehaviorKind.OCCURRENCE, limit=1, after=cursor)
        if not page:
            break
        paged.extend(page)
        cursor = page[-1]
    assert [a.identity_name for a in paged] == [a.identity_name for a in full]
    assert len(paged) == 2  # 无遗漏、无重复
