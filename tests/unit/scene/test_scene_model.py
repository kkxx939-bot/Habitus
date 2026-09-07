"""情景树的地址、目录与 URI：身份规则沿行为树，逻辑地址不含物理代。"""

from __future__ import annotations

from datetime import date

import pytest

from habitus.scene import SceneAddress, SceneDirectory, SceneURI, SceneURIError, SceneURINodeType
from habitus.scene.model import scene_label
from tests.unit.scene.scene_payloads import DAY, local


def test_address_identity_and_uri_round_trip() -> None:
    address = SceneAddress(DAY, "准备晚饭", local(19, 12))
    uri = SceneURI.from_address(address)
    assert str(uri) == "scene://scenes/2026/08/16/准备晚饭--20260816T191200000000%2B0800.md"
    parsed = SceneURI(str(uri))
    assert parsed.node_type is SceneURINodeType.DOCUMENT
    assert parsed.to_address() == address
    assert parsed.to_address().label == "准备晚饭"
    assert SceneAddress.from_identity(DAY, address.identity_name) == address


def test_address_rejects_date_mismatch_and_bad_labels() -> None:
    with pytest.raises(ValueError):
        SceneAddress(date(2026, 8, 17), "准备晚饭", local(19, 12))
    with pytest.raises(ValueError):
        SceneAddress(DAY, "准备晚饭.md", local(19, 12))
    with pytest.raises(ValueError):
        scene_label("x" * 500, "scene label")
    with pytest.raises(TypeError):
        SceneAddress(local(19, 12), "准备晚饭", local(19, 12))  # datetime 不是日期


def test_directory_shapes() -> None:
    assert SceneDirectory.root().parts == ()
    assert SceneDirectory.scenes().identity_parts == ("scenes",)
    day_directory = SceneDirectory.for_day(DAY)
    assert day_directory.identity_parts == ("scenes", "2026", "08", "16")
    assert day_directory.day() == DAY
    assert day_directory.parent() == SceneDirectory.scenes(2026, 8)
    assert SceneDirectory.scenes(2026).day() is None
    for parts in (("occurrences",), ("scenes", "26"), ("scenes", "2026", "13"), ("scenes", "2026", "02", "30"), ("scenes", "2026", "08", "16", "x")):
        with pytest.raises(ValueError):
            SceneDirectory(parts)


def test_uri_classification_and_rejections() -> None:
    assert SceneURI.root().is_root
    directory = SceneURI("scene://scenes/2026/08/16")
    assert directory.node_type is SceneURINodeType.DIRECTORY
    assert directory.to_directory() == SceneDirectory.for_day(DAY)
    with pytest.raises(SceneURIError):
        directory.to_address()
    for bad in ("behavior://scenes/2026/08/16", "scene://scenes/2026/08/16/", "scene://scenes/2026/08/16/x.txt", "scene://scenes/2026/08/16/准备晚饭.md", "scene://scenes/2026/08/16/准备晚饭--%ZZ.md"):
        with pytest.raises(SceneURIError):
            SceneURI(bad)
    assert not SceneURI.is_valid("scene://nowhere")
    assert SceneURI.is_valid("scene://scenes")
    assert SceneURI("scene://scenes/2026/08/16/准备晚饭--20260816T191200000000%2B0800.md") == (
        "scene://scenes/2026/08/16/准备晚饭--20260816T191200000000%2B0800.md"
    )
