"""规律级的地址与 URI：按候选归档、按行为身份定位。

身份这一套刻意与 ``SceneAddress`` / ``BehaviorAddress`` / ``MemoryAddress`` 一致——三方审查
实测过，自己另写一套的代价是两个候选静默互相覆盖、幸存的那条谁都读不出来。
"""

from __future__ import annotations

import unicodedata
from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.uri import BehaviorURI
from habitus.scene.model import (
    AssociationAddress,
    KindDirectory,
    RegularityLevel,
    SceneLinkType,
)
from habitus.scene.regularity.link import SceneStoredLink
from habitus.scene.uri import SceneURI, SceneURIError, SceneURINodeType

CST = timezone(timedelta(hours=8))
DAY = date(2026, 6, 10)
AT = datetime(2026, 6, 10, 15, 30, tzinfo=CST)


def address(kind: str = "打球", name: str = "去球场打球", at: datetime = AT) -> AssociationAddress:
    return AssociationAddress(kind, at.date(), name, at)


def behaviour(name: str, moment: datetime) -> BehaviorURI:
    return BehaviorURI.from_address(
        BehaviorAddress(kind=BehaviorKind.OCCURRENCE, occurred_on=moment.date(), name=name, started_at=moment)
    )


# --- 身份 -------------------------------------------------------------------------------


def test_the_candidate_identity_is_canonical_not_the_human_spelling() -> None:
    """``semantic_name`` 只**校验**归一后的形式合法，**返回的是原名**。

    直接拿它当目录名的话，``Gym`` 与 ``gym`` 在大小写不敏感的文件系统上是同一个文件，而两个
    地址却不相等——一个候选会把另一个的记录物理覆盖，幸存的那条对两边都读不出来。
    """

    assert address(kind="Gym") == address(kind="gym")
    assert hash(address(kind="Gym")) == hash(address(kind="gym"))
    assert KindDirectory.for_kind("Gym").parts == ("kinds", "gym")
    assert address(kind="Gym").identity_kind == "gym"
    assert address(kind="Gym").kind_token == "Gym"  # 人写法原样留在字段里
    nfc, nfd = unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café")
    assert nfc != nfd
    assert address(kind=nfc) == address(kind=nfd)


def test_the_leaf_carries_the_behaviour_name_because_the_instant_is_not_unique() -> None:
    """同一个 kind、同一时刻、不同名字的两条 occurrence 在行为树上完全合法（撞车按名字消歧）。

    叶名只用时刻的话，它们会互相覆盖——实测能让一条关联被另一条销毁。
    """

    assert address(name="去球场打球") != address(name="跟同事打球")
    assert address().identity_name == "去球场打球--20260610T153000000000+0800"
    assert AssociationAddress.from_identity("打球", DAY, address().identity_name) == address()


def test_a_non_canonical_leaf_is_refused_not_repaired() -> None:
    """走行为树那套解析：非规范的时间戳被**拒**，而不是被"修复"成另一个文件名。

    修复的下场实测过：URI 被静默改写成磁盘上并不存在的路径，``read_day`` 报"记录不存在"，
    而记录明明在那里。
    """

    for bad in (
        "去球场打球--20260610T153000000000+0060",  # 分钟数 ≥ 60
        "去球场打球--20260610T153000000000+0099",
        "20260610T153000000000+0800",  # 少了名字
    ):
        with pytest.raises(ValueError):
            AssociationAddress.from_identity("打球", DAY, bad)


def test_the_directory_date_must_agree_with_the_instant() -> None:
    with pytest.raises(ValueError, match="must match the local started_at date"):
        AssociationAddress("打球", date(2026, 6, 11), "去球场打球", AT)


def test_a_kind_token_is_a_safe_directory_name_and_reserves_the_sidecars() -> None:
    for reserved in (".abstract", ".overview"):
        with pytest.raises(ValueError):
            KindDirectory.for_kind(reserved)
    for bad in ("", "a/b", "名字.md", " 打球", ".."):
        with pytest.raises((ValueError, TypeError)):
            KindDirectory.for_kind(bad)
    # 路径上出现人写法而不是规范身份，说明有人绕过 for_kind 拼了路径——那一刻就埋了一次覆盖。
    with pytest.raises(ValueError, match="canonical kind identity"):
        KindDirectory(("kinds", "Gym"))


def test_the_day_shard_lets_provenance_days_be_found_directly() -> None:
    directory = KindDirectory.for_day("打球", DAY)
    assert directory.parts == ("kinds", "打球", "2026", "06", "10")
    assert directory.day() == DAY and directory.kind_token == "打球"
    assert KindDirectory.for_address(address()) == directory
    assert KindDirectory.for_kind("打球").day() is None
    for bad in (
        ("scenes",),
        ("kinds", "打球", "26", "06", "10"),
        ("kinds", "打球", "2026", "13", "10"),
        ("kinds", "打球", "2026", "02", "30"),
    ):
        with pytest.raises(ValueError):
            KindDirectory(bad)


def test_levels_mirror_the_other_trees() -> None:
    assert RegularityLevel.ABSTRACT.sidecar_filename == ".abstract.md"
    assert RegularityLevel.OVERVIEW.sidecar_filename == ".overview.md"
    assert RegularityLevel.from_sidecar_filename(".overview.md") is RegularityLevel.OVERVIEW
    assert RegularityLevel.from_sidecar_filename("whatever.md") is None
    with pytest.raises(ValueError):
        _ = RegularityLevel.DETAIL.sidecar_filename


# --- URI：一个 scheme 三种节点 -----------------------------------------------------------


def test_one_scheme_carries_both_regions_and_the_sidecars() -> None:
    """一个 scheme 三种节点：记录、目录、侧车。

    侧车也要能被引用：判断者在候选清单上读的就是 L0 那一句，引用不到它这一层就白建了
    （``behavior://`` 早有 LAYER 节点，这里一度漏了）。按天情景那一半已经随日情景树删掉。
    """

    uri = SceneURI.from_association(address(kind="Gym"))
    assert str(uri) == "scene://kinds/gym/2026/06/10/去球场打球--20260610T153000000000%2B0800.md"
    assert uri.is_association and uri.to_association() == address(kind="Gym")
    assert uri.started_at() == AT and SceneURI.parse(str(uri)) == uri

    layer = SceneURI.from_layer(KindDirectory.for_kind("Gym"), RegularityLevel.ABSTRACT)
    assert str(layer) == "scene://kinds/gym/.abstract.md"
    assert layer.node_type is SceneURINodeType.LAYER
    assert layer.to_layer() == (KindDirectory.for_kind("gym"), RegularityLevel.ABSTRACT)
    assert SceneURI.parse(str(layer)) == layer

    directory = SceneURI.from_directory(KindDirectory.for_day("Gym", DAY))
    assert str(directory) == "scene://kinds/gym/2026/06/10"
    assert directory.node_type is SceneURINodeType.DIRECTORY
    for wrong in (directory.started_at, layer.to_association, directory.to_layer):
        with pytest.raises(SceneURIError):
            wrong()


def test_a_uri_that_is_not_already_canonical_is_refused() -> None:
    """非规范的 URI 必须被拒，不能被悄悄改写成另一个文件名。"""

    for bad in (
        "scene://kinds/Gym/2026/06/10/去球场打球--20260610T153000000000%2B0800.md",  # 目录名是人写法
        "scene://kinds/gym/2026/06/10/去球场打球--20260610T153000000000%2B0060.md",  # 时间戳非规范
    ):
        with pytest.raises(SceneURIError):
            SceneURI(bad)


def test_a_cause_edge_points_from_this_occurrence_to_an_earlier_behaviour() -> None:
    """前因边：这次发生 → 更早的那条行为；lag 等于两端时刻差，方向反了当场拒。"""

    earlier = datetime(2026, 6, 8, 20, 0, tzinfo=CST)
    link = SceneStoredLink.between(
        SceneURI.from_association(address()), behaviour("跟朋友通话", earlier), SceneLinkType.RESULTS_FROM
    )
    assert link.lag_seconds == int((AT - earlier).total_seconds())
    with pytest.raises(ValueError, match="point at an earlier target"):
        SceneStoredLink.between(
            SceneURI.from_association(address(name="跟朋友通话", at=earlier)),
            behaviour("去球场打球", AT),
            SceneLinkType.RESULTS_FROM,
        )
