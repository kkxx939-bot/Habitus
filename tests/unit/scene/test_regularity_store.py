"""规律级的存储：记录按地址写、侧车按候选覆写、**完成标记**说这一天做完了没有。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.uri import BehaviorURI
from habitus.scene.model import AssociationAddress, RegularityLevel, SceneLinkType
from habitus.scene.regularity import (
    AssociationDocument,
    AssociationDocumentError,
    RegularityTree,
    RegularityTreeError,
    decode,
    encode,
)
from habitus.scene.regularity.document import premise_identity
from habitus.scene.regularity.link import SceneStoredLink
from habitus.scene.regularity.store import DONE_FILENAME
from habitus.scene.uri import SceneURI

CST = timezone(timedelta(hours=8))
DAY = date(2026, 6, 10)
NOW = datetime(2026, 6, 11, tzinfo=UTC)


def occurrence(name: str, moment: datetime) -> str:
    return str(
        BehaviorURI.from_address(
            BehaviorAddress(kind=BehaviorKind.OCCURRENCE, occurred_on=moment.date(), name=name, started_at=moment)
        )
    )


def record(
    hour: int = 15,
    *,
    kind: str = "打球",
    name: str = "去球场打球",
    minute: int = 30,
    offset: int = 8,
    causes: tuple[datetime, ...] = (),
    left: tuple[tuple[str, str], ...] = (),
    consumed: tuple[tuple[str, str], ...] = (),
) -> AssociationDocument:
    moment = datetime(DAY.year, DAY.month, DAY.day, hour, minute, tzinfo=timezone(timedelta(hours=offset)))
    address = AssociationAddress(kind, DAY, name, moment)
    links = tuple(
        SceneStoredLink.between(
            SceneURI.from_association(address), occurrence("跟朋友通话", cause), SceneLinkType.RESULTS_FROM
        )
        for cause in causes
    )
    return AssociationDocument(
        address=address,
        created_at=NOW,
        occurrence_uri=occurrence(name, moment),
        context="约好和朋友一起打",
        situation="S1",
        left=left,
        consumed=consumed,
        links=links,
    )


def tree(tmp_path) -> RegularityTree:
    store = RegularityTree(tmp_path / "regularity")
    store.initialize()
    return store


# --- 记录 -------------------------------------------------------------------------------


def test_a_record_round_trips_through_disk(tmp_path) -> None:
    store = tree(tmp_path)
    written = store.write(record(causes=(datetime(2026, 6, 8, 20, 0, tzinfo=CST),)))
    assert store.exists(written.address) and store.read(written.address) == written


def test_the_record_must_carry_its_own_occurrence(tmp_path) -> None:
    """关联与 occurrence 一一对应：地址里的时刻必须就是那条行为的开始时刻。"""

    moment = datetime(DAY.year, DAY.month, DAY.day, 15, 30, tzinfo=CST)
    with pytest.raises(ValueError, match="occurrence's own start time"):
        AssociationDocument(
            address=AssociationAddress("打球", DAY, "去球场打球", moment),
            created_at=NOW,
            occurrence_uri=occurrence("去球场打球", moment + timedelta(minutes=1)),
            context="约好和朋友一起打",
            situation="S1",
        )


def test_every_field_is_rendered_so_single_sided_tampering_is_caught(tmp_path) -> None:
    """正文没承载的字段，自洽校验就盖不住。

    三方审查用变异实测过三处漏网：情境编号（L1 的日期归属靠它）、``created_at``、以及每条边的
    ``link_type``（``needs`` 被改成 ``results_from`` 照样通过）。现在逐一钉住。
    """

    document = record(
        causes=(datetime(2026, 6, 8, 20, 0, tzinfo=CST),),
        left=(("买回了明天的球拍胶皮", "打球"),),
        consumed=((occurrence("跟朋友通话", datetime(2026, 6, 8, 20, 0, tzinfo=CST)), "和朋友约好这周打一次球"),),
    )
    raw = encode(document)
    assert decode(raw, expected_address=document.address) == document
    for broken in (
        raw.replace('"situation": "S1"', '"situation": "S2"'),
        raw.replace("- 情境：S1", "- 情境：S2"),
        raw.replace('"created_at": "2026-06-11', '"created_at": "1999-01-01'),
        raw.replace("- 关联于：2026-06-11", "- 关联于：1999-01-01"),
        raw.replace('"link_type": "results_from"', '"link_type": "needs"'),
        raw.replace("- results_from：", "- needs："),
        # 单边：带前缀的只在正文里，带引号的只在 JSON 里。全局替换会把两边改成一致的另一份
        # 内容，那是另一条合法记录、不是损坏。
        raw.replace("- 留下：买回了明天的球拍胶皮", "- 留下：买回了明天的球鞋"),
        raw.replace('"买回了明天的球拍胶皮"', '"买回了明天的球鞋"'),
        raw.replace("  等待：打球", "  等待：跑步"),
        raw.replace("- 留下：", "- 铺下："),
        raw.replace("- 用掉：和朋友约好这周打一次球", "- 用掉：跟人约了球"),
        raw.replace('"和朋友约好这周打一次球"', '"跟人约了球"'),
        raw.replace("- 用掉：", "- 兑现："),
        raw.replace("# 打球 ·", "# 游泳 ·"),
        raw[: raw.rindex("-->")],
    ):
        with pytest.raises(AssociationDocumentError):
            decode(broken, expected_address=document.address)
    # 正文与 JSON 被**同时**改成一致的另一份内容，不是损坏——那只是另一条合法记录。
    assert decode(raw.replace("约好和朋友一起打", "临时起意"), expected_address=document.address).context == "临时起意"


def test_records_are_ordered_by_instant_not_by_file_name(tmp_path) -> None:
    """同一本地日、不同 UTC 偏移时，文件名序与时刻序会分叉——必须按时刻排。"""

    store = tree(tmp_path)
    store.write(record(hour=9, minute=30, offset=1, name="早上打球"))  # 08:30Z
    store.write(record(hour=15, minute=30, offset=9, name="下午打球"))  # 06:30Z
    day = store.read_day("打球", DAY)
    assert [item.address.name for item in day] == ["下午打球", "早上打球"]  # 时刻序
    assert sorted(item.address.identity_name for item in day)[0].startswith("下午打球")  # 名字序相反


def test_two_occurrences_at_the_same_instant_do_not_overwrite_each_other(tmp_path) -> None:
    """同一个 kind、同一时刻、不同名字的两条行为在行为树上合法，这里必须是两条记录。"""

    store = tree(tmp_path)
    store.write(record(name="去球场打球"))
    store.write(record(name="跟同事打球"))
    assert len(store.read_day("打球", DAY)) == 2


def test_a_candidate_spelled_differently_is_the_same_candidate(tmp_path) -> None:
    """``Gym`` 与 ``gym`` 是同一个候选：不能一个把另一个的记录物理覆盖。"""

    store = tree(tmp_path)
    store.write(record(kind="Gym", name="去健身房"))
    assert store.exists(record(kind="gym", name="去健身房").address)
    # 从磁盘还原只拿得到规范身份；人写法的真源在行为树那条 occurrence 上。
    assert store.read(record(kind="gym", name="去健身房").address).address.identity_kind == "gym"
    assert store.list_kinds() == ("gym",)


# --- 完成标记 ---------------------------------------------------------------------------


def test_a_day_counts_as_associated_only_after_the_completion_marker(tmp_path) -> None:
    """**日目录存在不等于这天做完了。**

    目录在第一条记录落盘之前就建好（``atomic_replace_bytes`` 自己也会建父目录），中断后树上
    留下空目录；拿目录当判据的话，``backlog`` 的差集会把这一天永久排除——那条关联再也不会被
    写出来。三方审查用真 SIGKILL 复现过。
    """

    store = tree(tmp_path)
    written = store.write(record())
    assert store.days_for("打球") == frozenset()  # 记录写了，但还没提交
    store.complete_day("打球", DAY, records=1, completed_at=NOW)
    assert store.days_for("打球") == frozenset({DAY})
    assert (store.root / "kinds" / "打球" / "2026" / "06" / "10" / DONE_FILENAME).is_file()
    assert store.read(written.address) == written


def test_the_marker_refuses_to_claim_more_records_than_are_readable(tmp_path) -> None:
    """标记里记下当天应有几条；对不上说明中间掉了东西，这一天就不该被提交。

    一天两次发生只写成一条（崩在中间、或消费者只处理了第一个格子）时，靠这条拦下来。
    """

    store = tree(tmp_path)
    store.write(record(hour=9))
    with pytest.raises(RegularityTreeError, match="2 association records but 1"):
        store.complete_day("打球", DAY, records=2, completed_at=NOW)
    assert store.days_for("打球") == frozenset()
    store.write(record(hour=20, name="晚上打球"))
    store.complete_day("打球", DAY, records=2, completed_at=NOW)
    assert store.days_for("打球") == frozenset({DAY})


def test_a_failed_write_leaves_the_day_unclaimed(tmp_path) -> None:
    """写失败之后这一天必须仍在待办里，否则关联静默丢失。"""

    store = tree(tmp_path)
    document = record()
    store._day_path("打球", DAY).mkdir(parents=True)  # noqa: SLF001 - 模拟"目录已建、记录未写"
    assert store.days_for("打球") == frozenset()
    assert store.read_day("打球", DAY) == ()
    store.write(document)
    store.complete_day("打球", DAY, records=1, completed_at=NOW)
    assert store.days_for("打球") == frozenset({DAY})


# --- 侧车 -------------------------------------------------------------------------------


def test_layers_are_written_overview_first(tmp_path) -> None:
    """先写 L1、再写由它派生的 L0——反过来会留下"摘要比它总结的东西还新"的一瞬。"""

    store = tree(tmp_path)
    abstract_path, overview_path = store.write_layers("Gym", overview="三种情境……", abstract="通常因为约好")
    assert store.read_layer("gym", RegularityLevel.OVERVIEW) == "三种情境……"  # 大小写不影响
    assert store.read_layer("Gym", RegularityLevel.ABSTRACT) == "通常因为约好"
    assert overview_path.stat().st_mtime_ns <= abstract_path.stat().st_mtime_ns
    store.write_layers("Gym", overview="四种情境……", abstract="通常因为约好")
    assert store.read_layer("Gym", RegularityLevel.OVERVIEW) == "四种情境……"
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            store.write_layers("Gym", overview=bad, abstract="x")
    with pytest.raises(RegularityTreeError, match="not a semantic layer"):
        store.read_layer("Gym", RegularityLevel.DETAIL)


def test_a_kind_with_only_layers_has_associated_nothing(tmp_path) -> None:
    store = tree(tmp_path)
    store.write_layers("游泳", overview="x", abstract="y")
    assert store.days_for("游泳") == frozenset()


# --- 损坏与杂物 -------------------------------------------------------------------------


def test_junk_in_a_day_directory_is_refused_not_silently_ignored(tmp_path) -> None:
    """杂物要报成本层的完整性错误，不能静默忽略、也不能漏出裸 ``ValueError``。

    崩溃遗留的原子写临时文件是唯一的例外：按本模块自己的命名规则认出来、跳过。
    """

    store = tree(tmp_path)
    store.write(record())
    directory = store._day_path("打球", DAY)  # noqa: SLF001
    (directory / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RegularityTreeError, match="only Markdown records"):
        store.read_day("打球", DAY)
    (directory / "notes.txt").unlink()
    (directory / "README.md").write_text("x", encoding="utf-8")
    with pytest.raises(RegularityTreeError, match="non-canonical leaf name"):
        store.read_day("打球", DAY)
    (directory / "README.md").unlink()
    leftover = directory / (".去球场打球--20260610T153000000000+0800.md." + "0123456789ab" * 2 + "01234567" + ".tmp")
    leftover.write_text("x", encoding="utf-8")
    assert len(store.read_day("打球", DAY)) == 1  # 临时文件被认出来、跳过
    # 提交时顺手清掉：只跳过不清理的话它会一直堆着，而且计入目录条目上限。
    store.complete_day("打球", DAY, records=1, completed_at=NOW)
    assert not leftover.exists()


def test_corrupt_bytes_become_this_layers_error(tmp_path) -> None:
    store = tree(tmp_path)
    document = store.write(record())
    path = store._record_path(document.address)  # noqa: SLF001
    path.write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(RegularityTreeError, match="not valid UTF-8"):
        store.read(document.address)


def test_a_symlinked_candidate_directory_is_refused(tmp_path) -> None:
    """路径安全必须真的拦得住符号链接。

    ``_inside`` 一度返回 ``resolve()`` 之后的路径，等于替 durable_io 把链接洗白了——写入会
    静默落到链接指向的地方，两个候选的历史被合并（实测）。
    """

    store = tree(tmp_path)
    kinds = store.root / "kinds"
    (kinds / "real").mkdir(parents=True)
    (kinds / "打球").symlink_to(kinds / "real", target_is_directory=True)
    with pytest.raises(RegularityTreeError):
        store.write(record())
    with pytest.raises(RegularityTreeError, match="cannot be a symbolic link"):
        RegularityTree(kinds / "打球")


def test_reading_something_that_is_not_there(tmp_path) -> None:
    store = tree(tmp_path)
    with pytest.raises(RegularityTreeError, match="does not exist"):
        store.read(record().address)
    with pytest.raises(RegularityTreeError, match="does not exist"):
        store.read_layer("打球", RegularityLevel.ABSTRACT)
    assert not store.layer_exists("打球", RegularityLevel.ABSTRACT)
    assert store.read_day("打球", DAY) == ()


# --- 待用前提的两件事实 -------------------------------------------------------------------


def test_the_two_premise_facts_round_trip_through_disk(tmp_path) -> None:
    """产生与兑现都按**前提本身**记：一句话 + 它在等什么 / 产生方 + 那句话。"""

    store = tree(tmp_path)
    earlier = occurrence("去超市买菜", datetime(2026, 6, 8, 15, 0, tzinfo=CST))
    written = store.write(record(left=(("买回了明天的食材", "做饭"),), consumed=((earlier, "家里有今晚要用的食材"),)))
    read = store.read(written.address)
    assert read.left == (("买回了明天的食材", "做饭"),)
    assert read.consumed == ((earlier, "家里有今晚要用的食材"),)


def test_one_behaviour_can_leave_several_independent_premises(tmp_path) -> None:
    """去超市一趟留下"家里有菜"和"买到了灯泡"：用掉其中一条，另一条不受影响。

    这正是旧实现错的地方——它按"整条行为被指过一次"判兑现，于是两条一起消失。
    """

    store = tree(tmp_path)
    written = store.write(record(left=(("家里有菜", "做饭"), ("买到了灯泡", "换灯泡"))))
    assert store.read(written.address).left == (("买到了灯泡", "换灯泡"), ("家里有菜", "做饭"))


def test_a_premise_identity_is_the_producer_and_the_sentence(tmp_path) -> None:
    earlier = occurrence("去超市买菜", datetime(2026, 6, 8, 15, 0, tzinfo=CST))
    other = occurrence("去药店", datetime(2026, 6, 8, 16, 0, tzinfo=CST))
    # 身份走规范形式（NFC + casefold），所以同一句话换个大小写还是同一条前提。
    assert premise_identity(earlier, "Fresh Veg") == premise_identity(earlier, "fresh veg")
    assert premise_identity(earlier, "家里有菜") != premise_identity(earlier, "买到了灯泡")
    assert premise_identity(earlier, "家里有菜") != premise_identity(other, "家里有菜")


def test_a_premise_consumed_before_it_was_produced_is_refused() -> None:
    """用掉一条比这次还晚才建立的前提，是我们自己产物里的矛盾。"""

    later = occurrence("去超市买菜", datetime(DAY.year, DAY.month, DAY.day, 23, 0, tzinfo=CST))
    with pytest.raises(ValueError, match="produced later than this occurrence"):
        record(consumed=((later, "家里有菜"),))


def test_the_same_premise_cannot_be_consumed_twice_by_one_occurrence() -> None:
    earlier = occurrence("去超市买菜", datetime(2026, 6, 8, 15, 0, tzinfo=CST))
    with pytest.raises(ValueError, match="duplicate entry"):
        record(consumed=((earlier, "家里有菜"), (earlier, "家里有菜")))
    # 换个大小写仍是同一条前提——按规范身份判，不按字面。
    with pytest.raises(ValueError, match="repeats a premise"):
        record(consumed=((earlier, "Fresh Veg"), (earlier, "fresh veg")))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left", (("买回了食材", ""),)),
        ("left", (("", "做饭"),)),
        ("left", (("上\n下", "做饭"),)),
        ("left", (("买回了食材",),)),
        ("consumed", (("not-a-uri", "家里有菜"),)),
    ],
)
def test_a_malformed_premise_is_refused(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        record(**{field: value})  # type: ignore[arg-type]


def test_a_record_without_premises_renders_neither_line(tmp_path) -> None:
    body = record().markdown_body
    assert "- 留下：" not in body and "- 用掉：" not in body


def test_a_premise_whose_text_contains_the_separator_cannot_be_rewritten_from_one_side() -> None:
    """两半拼在一行的话，文本里出现同一个分隔符时正文就定不出分割点——只改 JSON 一侧
    把分割点挪走能骗过正文比对，而两条语义不同的前提还会渲染成一模一样的一行。"""

    tricky = record(left=(("家里有菜 → 冰箱", "做饭"),))
    other = record(left=(("家里有菜", "冰箱 → 做饭"),))
    assert tricky.markdown_body != other.markdown_body
    raw = encode(tricky)
    moved = raw.replace('"家里有菜 → 冰箱",\n        "做饭"', '"家里有菜",\n        "冰箱 → 做饭"')
    if moved != raw:
        with pytest.raises(AssociationDocumentError):
            decode(moved, expected_address=tricky.address)


def test_the_record_must_carry_the_occurrence_own_name() -> None:
    """同一微秒上两条不同名的 occurrence 在行为树上完全合法——时刻单独不构成身份。"""

    moment = datetime(DAY.year, DAY.month, DAY.day, 15, 30, tzinfo=CST)
    with pytest.raises(ValueError, match="occurrence's own name"):
        AssociationDocument(
            address=AssociationAddress("打球", DAY, "去球场打球", moment),
            created_at=NOW,
            occurrence_uri=occurrence("洗了手", moment),
            context="约好和朋友一起打",
        )


@pytest.mark.parametrize("bad", ["上\r下", "上\n下"])
def test_a_context_with_a_line_break_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="single line"):
        AssociationDocument(
            address=AssociationAddress("打球", DAY, "去球场打球", datetime(DAY.year, DAY.month, DAY.day, 15, 30, tzinfo=CST)),
            created_at=NOW,
            occurrence_uri=occurrence("去球场打球", datetime(DAY.year, DAY.month, DAY.day, 15, 30, tzinfo=CST)),
            context=bad,
        )
