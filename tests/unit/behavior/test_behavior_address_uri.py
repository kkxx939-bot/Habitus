"""已确认 Behavior 树地址与 URI 的严格映射测试。"""

from datetime import date, datetime, timezone

import pytest

from behavior.model import BehaviorAddress, BehaviorDirectory, BehaviorKind, BehaviorLevel
from behavior.uri import BehaviorURI, BehaviorURIError, BehaviorURINodeType

_STARTED_AT = datetime(2026, 8, 8, 19, 2, 3, 123456, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        (
            BehaviorAddress.event(date(2026, 8, 8), "主人回家后打开空调", _STARTED_AT),
            "behavior://behaviors/events/2026/08/08/主人回家后打开空调--20260808T190203123456%2B0000.md",
        ),
        (
            BehaviorAddress.outcome(date(2026, 8, 8), "主人回家后打开空调", _STARTED_AT),
            "behavior://behaviors/outcomes/2026/08/08/主人回家后打开空调--20260808T190203123456%2B0000.md",
        ),
        (
            BehaviorAddress.episode(date(2026, 8, 8), "主人回家后的晚间生活", _STARTED_AT),
            "behavior://episodes/2026/08/08/主人回家后的晚间生活--20260808T190203123456%2B0000.md",
        ),
    ],
)
def test_address_uri_round_trip(address: BehaviorAddress, expected: str) -> None:
    uri = BehaviorURI.from_address(address)
    assert str(uri) == expected
    assert uri.node_type is BehaviorURINodeType.DOCUMENT
    assert uri.to_address() == address


def test_outcome_mirrors_event_identity_without_behavior_name_directory() -> None:
    event = BehaviorAddress.event(date(2026, 8, 8), "主人回家", _STARTED_AT)
    outcome = BehaviorAddress.outcome(date(2026, 8, 8), "主人回家", _STARTED_AT)
    assert event.identity_name == outcome.identity_name
    assert event.occurred_on == outcome.occurred_on
    assert not BehaviorURI.is_valid(
        "behavior://behaviors/home-arrival/events/2026/08/08/主人回家--20260808T190203123456%2B0000.md"
    )


@pytest.mark.parametrize("name", ["", "../escape", "name.md", ".abstract", " value ", "a/b"])
def test_address_rejects_unsafe_or_reserved_names(name: str) -> None:
    with pytest.raises(ValueError):
        BehaviorAddress.event(date(2026, 8, 8), name, _STARTED_AT)


def test_directory_grammar_and_lineage_are_fixed() -> None:
    directory = BehaviorDirectory.events(2026, 8, 8)
    assert directory.lineage() == (
        BehaviorDirectory(("behaviors", "events", "2026", "08", "08")),
        BehaviorDirectory(("behaviors", "events", "2026", "08")),
        BehaviorDirectory(("behaviors", "events", "2026")),
        BehaviorDirectory.events(),
        BehaviorDirectory.behaviors(),
        BehaviorDirectory.root(),
    )
    with pytest.raises(ValueError, match="outside"):
        BehaviorDirectory(("events",))
    with pytest.raises(ValueError, match="calendar"):
        BehaviorDirectory(("episodes", "2026", "02", "30"))


def test_directory_and_layer_uri_round_trip() -> None:
    directory = BehaviorDirectory.episodes(2026, 8)
    directory_uri = BehaviorURI.from_directory(directory)
    overview_uri = BehaviorURI.from_layer(directory, BehaviorLevel.OVERVIEW)
    assert directory_uri.to_directory() == directory
    assert overview_uri.to_layer() == (directory, BehaviorLevel.OVERVIEW)
    assert overview_uri.parent == directory_uri


@pytest.mark.parametrize(
    "uri",
    [
        "behaviors/events/2026/08/08/name.md",
        "memory://behaviors/events/2026/08/08/name.md",
        "behavior://events/2026/08/08/name.md",
        "behavior://behaviors/events/2026/08/08/name",
        "behavior://behaviors/events/2026/08/08//name.md",
        "behavior://behaviors/events/2026/08/08/name.md/",
        "behavior://behaviors/events/2026/02/30/name.md",
    ],
)
def test_uri_rejects_noncanonical_or_outside_tree_forms(uri: str) -> None:
    assert not BehaviorURI.is_valid(uri)
    with pytest.raises(BehaviorURIError):
        BehaviorURI(uri)


def test_uri_preserves_unicode_and_matches_complete_segments() -> None:
    uri = BehaviorURI("behavior://behaviors/events/2026/08/08/视频%20输出--20260808T190203123456%2B0000.md")
    assert str(uri) == ("behavior://behaviors/events/2026/08/08/视频%20输出--20260808T190203123456%2B0000.md")
    assert uri.decoded_path == "behaviors/events/2026/08/08/视频 输出--20260808T190203123456+0000.md"
    assert uri.matches_prefix("behavior://behaviors/events")
    assert not uri.matches_prefix("behavior://behaviors/outcomes")


def test_address_fields_remain_kind_specific_factories() -> None:
    address = BehaviorAddress(BehaviorKind.EVENT, date(2026, 8, 8), "事件", _STARTED_AT)
    assert address == BehaviorAddress.event(date(2026, 8, 8), "事件", _STARTED_AT)


def test_address_rejects_datetime_to_preserve_uri_round_trip() -> None:
    with pytest.raises(TypeError, match="date without a time"):
        BehaviorAddress.event(
            datetime(2026, 8, 8, 10, tzinfo=timezone.utc),  # type: ignore[arg-type]
            "事件",
            _STARTED_AT,
        )


@pytest.mark.parametrize("name", [".hidden", ".event", "x" * 256])
def test_address_rejects_names_the_physical_tree_cannot_store(name: str) -> None:
    with pytest.raises(ValueError):
        BehaviorAddress.event(date(2026, 8, 8), name, _STARTED_AT)


def test_address_requires_local_date_and_timestamp_identity() -> None:
    local = datetime.fromisoformat("2026-08-09T00:30:00.000000+08:00")
    address = BehaviorAddress.event(local.date(), "主人回家", local)

    assert address.started_at == local
    assert address.identity_name.endswith("--20260809T003000000000+0800")
    assert BehaviorURI.from_address(address).to_address() == address

    with pytest.raises(ValueError, match="local started_at date"):
        BehaviorAddress.event(date(2026, 8, 8), "主人回家", local)


def test_directory_and_uri_dates_require_ascii_digits() -> None:
    with pytest.raises(ValueError, match="format"):
        BehaviorDirectory(("behaviors", "events", "２０２６", "０８", "０８"))
    with pytest.raises(BehaviorURIError, match="confirmed behavior tree"):
        BehaviorURI("behavior://behaviors/events/２０２６/０８/０８")
