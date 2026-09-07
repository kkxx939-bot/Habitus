"""情景 L2：Schema 校验只守自洽、正文是字段的确定性函数、codec 往返逐字节一致。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from habitus.scene import (
    SceneDocumentCodec,
    SceneDocumentIntegrityError,
    SceneDocumentMetadata,
    SceneFieldRole,
    SceneLinkType,
    SceneSchemaError,
    SceneSchemaRegistry,
    SceneStoredLink,
    SceneURI,
)
from habitus.scene.document.link import link_lag_seconds, parse_link_target
from habitus.scene.model import SceneRole
from tests.unit.scene.scene_payloads import (
    BUY_GROCERIES,
    DAY,
    DISCUSS_DINNER,
    LOOKUP_RECIPE,
    WASH_VEGETABLES,
    local,
    occurrence_uri,
    scene_payload,
    shopping_payload,
)

CREATED = datetime(2026, 8, 17, 1, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def registry() -> SceneSchemaRegistry:
    return SceneSchemaRegistry.load_default()


@pytest.fixture(scope="module")
def codec(registry: SceneSchemaRegistry) -> SceneDocumentCodec:
    return SceneDocumentCodec(registry)


def _mutated(payload: dict[str, Any], name: str) -> dict[str, Any]:
    """给某个字段一个语义上不同、仍然合法的值。"""

    value = payload[name]
    if name == "label":
        return {**payload, name: value + "改"}
    if name == "ended_at":
        return {**payload, name: local(23, 0)}
    if name == "members":
        return {**payload, name: (*value[:-1], {**value[-1], "role": "optional"})}
    if name == "effects":
        return {**payload, name: (*value, "新增的改变")}
    if name == "pending_effects":
        return {**payload, name: (*value, {"text": "新增的前提", "producer_uri": LOOKUP_RECIPE})}
    raise AssertionError(f"no mutation defined for {name}")


def test_every_non_system_field_is_rendered(registry: SceneSchemaRegistry) -> None:
    """守卫：每个非 system 字段变了正文就得变——加字段忘了进正文会在这里失败。"""

    base = scene_payload()
    body = registry.materialize(base).markdown_body
    assert body.startswith("# 准备晚饭\n")
    assert "2026-08-16T19:12:00+08:00 — 2026-08-16T20:30:00+08:00" in body
    # 成员按开始时刻序、显示为「时刻 行为名」，不带百分号编码
    assert body.index("必要 19:12 商量晚餐") < body.index("必要 19:41 查配方") < body.index("无关 20:15 看手机")
    assert "%2B" not in body
    assert "- 晚饭做好了" in body
    assert "查到了今晚做汤的配方（由 19:41 查配方 建立）" in body
    for field in registry.schema.fields_of(SceneFieldRole.SYSTEM):
        assert field.name not in body
    for field in (*registry.schema.fields_of(SceneFieldRole.SEMANTIC), *registry.schema.fields_of(SceneFieldRole.ADDRESS)):
        if field.name in {"occurred_on", "started_at"}:
            continue  # 地址时间与日期由首成员决定，改它等于换成员，另测
        assert registry.materialize(_mutated(base, field.name)).markdown_body != body, field.name


def test_all_roles_render(registry: SceneSchemaRegistry) -> None:
    members = tuple(
        {"uri": uri, "role": role.value}
        for uri, role in zip((DISCUSS_DINNER, LOOKUP_RECIPE, WASH_VEGETABLES), SceneRole, strict=True)
    )
    body = registry.materialize(scene_payload(members=members, pending_effects=())).markdown_body
    assert "必要 19:12" in body and "顺带 19:41" in body and "无关 19:43" in body


def test_members_are_stored_in_start_order(registry: SceneSchemaRegistry) -> None:
    original = scene_payload()["members"]
    stored = registry.materialize(scene_payload(members=tuple(reversed(original)))).storage_fields["members"]
    assert [m["uri"] for m in stored] == [m["uri"] for m in original]


def test_empty_effects_and_pending_render_as_none(registry: SceneSchemaRegistry) -> None:
    body = registry.materialize(scene_payload(effects=(), pending_effects=())).markdown_body
    assert body.count("- （无）") == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"ended_at": local(19, 0)}, "ended_at cannot precede"),
        ({"occurred_on": DAY.replace(day=17)}, "occurred_on must match"),
        ({"started_at": local(19, 41)}, "start of the earliest member"),
        ({"members": ()}, "at least one member"),
        ({"pending_effects": ({"text": "x", "producer_uri": BUY_GROCERIES},)}, "produced by a member"),
        (
            {"pending_effects": ({"text": "x", "producer_uri": LOOKUP_RECIPE}, {"text": "x", "producer_uri": LOOKUP_RECIPE})},
            "must not contain duplicates",
        ),
        ({"members": ({"uri": DISCUSS_DINNER, "role": "essential"}, {"uri": DISCUSS_DINNER, "role": "optional"})}, "same occurrence twice"),
        ({"members": ({"uri": DISCUSS_DINNER, "role": "主要"},)}, "role must be one of"),
        ({"members": ({"uri": "scene://scenes/2026/08/16/x--20260816T191200000000%2B0800.md", "role": "essential"},)}, "behavior occurrence"),
        ({"effects": ("a", "a")}, "must not contain duplicates"),
        ({"effects": ("第一行\n## 待用前提",)}, "single line"),
        ({"scene_version": ""}, "must be non-empty"),
        ({"extra": 1}, "unknown fields"),
    ],
)
def test_schema_rejects_inconsistent_payloads(registry: SceneSchemaRegistry, overrides: dict, message: str) -> None:
    with pytest.raises(SceneSchemaError, match=message):
        registry.validate(scene_payload(**overrides))


def test_schema_accepts_single_member_scene_without_effects(registry: SceneSchemaRegistry) -> None:
    """一条成员、没有 effects 是现实的形状，不是产物矛盾。"""

    normalized = registry.validate(shopping_payload(effects=(), pending_effects=()))
    assert normalized["label"] == "去超市采购"


def test_scene_may_span_midnight(registry: SceneSchemaRegistry) -> None:
    """跨午夜的情景是现实的形状：占用日按首成员的本地日，结束时刻可以是次日。"""

    late = occurrence_uri("看电影", local(23, 30))
    normalized = registry.validate(
        scene_payload(
            started_at=local(23, 30),
            ended_at=local(1, 10, day=DAY.replace(day=17)),
            members=({"uri": late, "role": "essential"},),
            pending_effects=(),
        )
    )
    assert normalized["occurred_on"] == DAY


def test_codec_round_trip_with_links(codec: SceneDocumentCodec) -> None:
    metadata = SceneDocumentMetadata(created_at=CREATED)
    dinner = codec.build(scene_payload(), metadata=metadata)
    shopping = codec.build(shopping_payload(), metadata=metadata)
    links = (
        SceneStoredLink.between(dinner.uri, shopping.uri, SceneLinkType.NEEDS),
        SceneStoredLink.between(dinner.uri, BUY_GROCERIES, SceneLinkType.RESULTS_FROM),
    )
    assert {link.lag_seconds for link in links} == {13920}
    document = codec.build(scene_payload(), metadata=metadata, links=links)
    encoded = codec.encode(document)
    metadata_block = encoded.split("HABITUS_SCENE_FIELDS")[1].removesuffix("\n-->\n")
    assert "--" not in metadata_block  # 正文里的 -- 会提前闭合 HTML 注释，结构块里必须转义
    decoded = codec.decode(encoded, expected_address=document.address)
    assert decoded == document
    # links 按 (from, to, type) 身份定序：behavior:// 排在 scene:// 之前
    assert [(str(link.to_uri), link.link_type) for link in decoded.links] == [
        (BUY_GROCERIES, SceneLinkType.RESULTS_FROM),
        (str(shopping.uri), SceneLinkType.NEEDS),
    ]
    assert codec.encode(decoded) == encoded


def test_codec_rejects_tampered_body_noncanonical_block_and_wrong_address(codec: SceneDocumentCodec) -> None:
    document = codec.build(scene_payload(), metadata=SceneDocumentMetadata(created_at=CREATED))
    encoded = codec.encode(document)
    with pytest.raises(SceneDocumentIntegrityError):
        codec.decode(encoded.replace("# 准备晚饭", "# 准备早饭", 1), expected_address=document.address)
    other = codec.build(shopping_payload(), metadata=SceneDocumentMetadata(created_at=CREATED))
    with pytest.raises(SceneDocumentIntegrityError):
        codec.decode(encoded, expected_address=other.address)
    with pytest.raises(SceneDocumentIntegrityError):
        codec.decode(encoded.replace('"scene_type": "scene"', '"scene_type": "x"'), expected_address=document.address)
    # 结构块换行/缩进与规范序列化不同也不是我们的产物
    head, block = encoded.split("\n<!-- HABITUS_SCENE_FIELDS\n")
    with pytest.raises(SceneDocumentIntegrityError):
        codec.decode(head + "\n<!-- HABITUS_SCENE_FIELDS\n" + block.replace("\n", " ", 1), expected_address=document.address)


def test_links_derive_lag_and_direction_from_endpoints(codec: SceneDocumentCodec) -> None:
    dinner = codec.build(scene_payload(), metadata=SceneDocumentMetadata(created_at=CREATED))
    assert link_lag_seconds(dinner.uri, parse_link_target(BUY_GROCERIES)) == 13920
    with pytest.raises(ValueError, match="start-time difference"):
        SceneStoredLink(dinner.uri, BUY_GROCERIES, SceneLinkType.NEEDS, 5)
    later = occurrence_uri("洗碗", local(21, 0))
    with pytest.raises(ValueError, match="earlier target"):
        SceneStoredLink.between(dinner.uri, later, SceneLinkType.NEEDS)
    with pytest.raises(ValueError):
        SceneStoredLink(dinner.uri, dinner.uri, SceneLinkType.NEEDS, 0)
    with pytest.raises(ValueError):
        SceneStoredLink(dinner.uri, BUY_GROCERIES, SceneLinkType.NEEDS, -1)
    with pytest.raises(ValueError):
        SceneStoredLink.between(dinner.uri, "behavior://gaps/2026/08/16/没读懂--20260816T191200000000%2B0800.md", SceneLinkType.NEEDS)
    with pytest.raises(ValueError):
        SceneStoredLink.between(dinner.uri, "scene://scenes/2026/08/16", SceneLinkType.NEEDS)
    with pytest.raises(ValueError):
        parse_link_target("memory://profile.md")
    # 同一目标可以同时是 needs 与 results_from；同一目标同一类型不能重复
    both = (
        SceneStoredLink.between(dinner.uri, BUY_GROCERIES, SceneLinkType.NEEDS),
        SceneStoredLink.between(dinner.uri, BUY_GROCERIES, SceneLinkType.RESULTS_FROM),
    )
    assert len(codec.build(scene_payload(), metadata=SceneDocumentMetadata(created_at=CREATED), links=both).links) == 2
    with pytest.raises(ValueError):
        codec.build(scene_payload(), metadata=SceneDocumentMetadata(created_at=CREATED), links=(both[0], both[0]))
    # 来源 URI 必须是本文档
    foreign_source = SceneURI.parse(str(codec.build(shopping_payload(), metadata=SceneDocumentMetadata(created_at=CREATED)).uri))
    foreign = SceneStoredLink.between(foreign_source, BUY_GROCERIES, SceneLinkType.NEEDS)
    with pytest.raises(ValueError):
        codec.build(scene_payload(), metadata=SceneDocumentMetadata(created_at=CREATED), links=(foreign,))


def test_registry_loads_the_confirmed_definition(registry: SceneSchemaRegistry) -> None:
    schema = registry.schema
    assert [f.name for f in schema.fields_of(SceneFieldRole.ADDRESS)] == ["occurred_on", "label", "started_at"]
    assert {f.name for f in schema.fields_of(SceneFieldRole.SYSTEM)} == {"source_digest", "scene_version"}
    assert {f.name for f in schema.fields_of(SceneFieldRole.SEMANTIC)} == {"ended_at", "members", "effects", "pending_effects"}
