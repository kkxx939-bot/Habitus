"""接缝测试：真实行为树 → 快照 → 重建 → 发布 → 查询。

纯函数层的单测只能证明实现符合我自己的想法；真正会出问题的地方在模块之间的接缝——
字段名对不对得上、时间偏移有没有被折掉、并行链接读不读得出来、消歧重复有没有被跳过。
所以这里从 ``BehaviorDocumentWriter`` 真写一棵行为树开始，一路走到 ``query`` 出结果。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from behavior import BehaviorDocumentWriter, BehaviorKind, BehaviorLinkType, BehaviorTree, BehaviorURI
from infrastructure.store.locks import ProcessLocalLockStore
from prediction import builder, codec, query, source
from prediction.edges import NO_SUCCESSOR
from prediction.errors import PredictionTreeError, PredictionTreeStoreError
from prediction.model import PredictionTree
from prediction.store import GENERATIONS_DIRECTORY, TREE_FILENAME, PredictionTreeStore
from tests.unit.behavior.tree_payloads import gap_payload, occurrence_payload
from tests.unit.prediction.prediction_fixtures import config

CST = timezone(timedelta(hours=8))
FIRST = date(2026, 6, 1)  # 周一


def moment(day_offset: int, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(FIRST + timedelta(days=day_offset), datetime.min.time(), tzinfo=CST).replace(
        hour=hour, minute=minute
    )


def writer_for(tmp_path, name: str = "behavior-tree") -> tuple[BehaviorDocumentWriter, BehaviorTree]:
    tree = BehaviorTree(tmp_path / name)
    return BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: moment(80, 23)), tree


def publish_occurrence(writer, name: str, started_at: datetime, *, links=(), **overrides):
    payload = occurrence_payload(
        occurred_on=started_at.date(),
        name=name,
        kind_token=overrides.pop("kind_token", name),
        started_at=started_at,
        last_observed_at=started_at + timedelta(minutes=1),
        onset_available_at=started_at + timedelta(seconds=2),
        basis=(),
        goal=None,
        **overrides,
    )
    return writer.publish(BehaviorKind.OCCURRENCE, payload, links=links)


def test_a_real_behaviour_tree_rebuilds_into_a_queryable_generation(tmp_path) -> None:
    """每天早上吃药、每周二晚上打球：两种规律都要能从真实的树里被捞出来。"""

    writer, tree = writer_for(tmp_path)
    for offset in range(70):
        publish_occurrence(writer, "吃药", moment(offset, 7, 30))
        if (FIRST + timedelta(days=offset)).weekday() == 1:
            publish_occurrence(writer, "打球", moment(offset, 19))

    snapshot = source.read(tree)
    assert len(snapshot.actions) == 70 + 10
    assert snapshot.latest_day == FIRST + timedelta(days=69)

    settings = config()
    built = builder.build(
        snapshot, config=settings, reference=snapshot.latest_day, built_at=moment(69, 23)
    )
    morning = query.slot_outlook(built, query.slot_at(built, moment(0, 7, 30)))
    medicine = next(item for item in morning.candidates if item.action == "吃药")
    assert medicine.marginal > 0.9
    assert medicine.lift_all_day > 50  # 一天 96 个槽里只占一个 → 真峰

    tuesday = query.slot_outlook(built, query.slot_at(built, moment(1, 19)))
    ball = next(item for item in tuesday.candidates if item.action == "打球")
    assert ball.marginal > 0.8
    # 日钟面上每周只有 1/7 的天在打球，周维度把它捞了出来
    assert ball.lift_weekday > 3.0


def test_disambiguated_duplicates_never_reach_the_statistics(tmp_path) -> None:
    """撞车消歧的重复记录已经在写入时打好标记，本层机械跳过——不重新判重。"""

    writer, tree = writer_for(tmp_path)
    publish_occurrence(writer, "洗手", moment(0, 12))
    publish_occurrence(writer, "洗手-2", moment(0, 12), original_name="洗手")

    snapshot = source.read(tree)
    assert snapshot.skipped_duplicates == 1
    assert [item.action for item in snapshot.actions] == ["洗手"]


def test_concurrent_links_survive_the_read_and_stay_out_of_transitions(tmp_path) -> None:
    """行为树标了并行的两条，读出来还得是并行——否则会变成假因果。"""

    writer, tree = writer_for(tmp_path)
    for offset in range(12):
        meal = publish_occurrence(writer, "吃饭", moment(offset, 18))
        publish_occurrence(
            writer,
            "看手机",
            moment(offset, 18, 5),
            links=((BehaviorLinkType.CONCURRENT_WITH, BehaviorURI.from_address(meal.address)),),
        )
        publish_occurrence(writer, "洗碗", moment(offset, 18, 40))

    snapshot = source.read(tree)
    assert len(snapshot.concurrent) == 12
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(11, 23)
    )
    targets = {item.target for item in query.successors(built, "吃饭")}
    assert "看手机" not in targets
    assert "洗碗" in targets
    assert "看手机" in {item.target for item in query.parallels(built, "吃饭")}


def test_gaps_read_from_the_tree_reduce_exposure(tmp_path) -> None:
    """没读懂的时段照样扣曝光——观测越差算出的概率越低，方向就反了。"""

    writer, tree = writer_for(tmp_path)
    for offset in range(14):
        publish_occurrence(writer, "吃药", moment(offset, 7, 30))
    for offset in range(14, 28):
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(
                occurred_on=moment(offset, 7).date(),
                started_at=moment(offset, 7),
                ended_at=moment(offset, 8),
            ),
        )

    snapshot = source.read(tree)
    assert len(snapshot.gaps) == 14
    built = builder.build(
        snapshot, config=config(), reference=date(2026, 6, 28), built_at=moment(27, 23)
    )
    outlook = query.slot_outlook(built, query.slot_at(built, moment(0, 7, 30)))
    medicine = next(item for item in outlook.candidates if item.action == "吃药")
    # 后 14 天该时段全被空白盖住，不进分母；概率仍应接近 1 而不是被腰斩。
    assert medicine.marginal > 0.85


def test_local_offset_survives_the_round_trip(tmp_path) -> None:
    """行为树存的是本地时刻加偏移；折成 UTC 会把槽位整体挪走。"""

    writer, tree = writer_for(tmp_path)
    publish_occurrence(writer, "吃药", moment(0, 7, 30))
    action = source.read(tree).actions[0]
    assert action.started_at.utcoffset() == timedelta(hours=8)
    assert action.started_at.hour == 7
    assert action.day == FIRST


# --- 发布 -------------------------------------------------------------------------------


def tree_for(tmp_path, *, name: str = "behavior-tree", hour: int = 23) -> PredictionTree:
    """一棵最小但真实的树：二十天，每天早上吃药。"""

    writer, behavior_tree = writer_for(tmp_path, name)
    for offset in range(20):
        publish_occurrence(writer, "吃药", moment(offset, 7, 30))
    snapshot = source.read(behavior_tree)
    return builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(19, hour)
    )


def test_publish_activates_a_generation_and_load_pins_it(tmp_path) -> None:
    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=3)
    built = tree_for(tmp_path)
    published = store.publish(built)

    assert store.active() == published
    loaded = store.load()
    assert loaded is not None
    assert loaded == built  # 整棵原样回来，不是"差不多"


def test_nothing_is_active_before_the_first_publish(tmp_path) -> None:
    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=2)
    assert store.active() is None
    assert store.load() is None


def test_a_corrupted_generation_is_refused_instead_of_served(tmp_path) -> None:
    """指针带着内容摘要：字节被改过就拒绝服务，而不是把半棵树端上去。"""

    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=2)
    published = store.publish(tree_for(tmp_path))
    path = store.root / GENERATIONS_DIRECTORY / published.generation / TREE_FILENAME
    path.write_text(path.read_text(encoding="utf-8").replace("吃药", "吃饭"), encoding="utf-8")
    with pytest.raises(PredictionTreeStoreError, match="does not match its pointer"):
        store.load()


def test_old_generations_are_pruned_but_the_active_one_survives(tmp_path) -> None:
    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=2)
    for hour in (20, 21, 22):
        published = store.publish(tree_for(tmp_path, hour=hour))

    generations = store.generations()
    assert len(generations) == 2
    assert published.generation in generations
    assert store.load() is not None


def test_pointer_is_untouched_when_materialisation_fails(tmp_path) -> None:
    """两阶段发布的要害：第一阶段失败，旧代必须继续服务。"""

    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=3)
    first = store.publish(tree_for(tmp_path))

    # 代名只由内容与构建时刻决定，与存储根无关——借一个别处的存储把名字问出来，
    # 就能在不动本存储指针的前提下，在目标路径上预埋一段异内容的字节。
    later = tree_for(tmp_path, name="behavior-tree-later", hour=22)
    name = PredictionTreeStore(tmp_path / "scratch", retained_generations=1).publish(later).generation
    planted = store.root / GENERATIONS_DIRECTORY / name
    planted.mkdir(parents=True)
    (planted / TREE_FILENAME).write_text("不是这一代的字节", encoding="utf-8")

    with pytest.raises(PredictionTreeStoreError, match="already bound"):
        store.publish(later)
    assert store.active() == first
    assert store.load() is not None


def test_encoding_round_trips_exactly(tmp_path) -> None:
    built = tree_for(tmp_path)
    assert codec.decode(codec.encode(built)) == built


# --- 组合契约 ---------------------------------------------------------------------------


def test_joint_cell_beats_the_product_of_two_marginals(tmp_path) -> None:
    """周二吃完饭打球：两个边缘相乘会把基线乘两遍，联合格子直接数就没这个问题。"""

    writer, behavior_tree = writer_for(tmp_path)
    for offset in range(70):
        weekday = (FIRST + timedelta(days=offset)).weekday()
        publish_occurrence(writer, "吃饭", moment(offset, 18))
        if weekday == 1:
            publish_occurrence(writer, "打球", moment(offset, 18, 30))
        else:
            publish_occurrence(writer, "洗碗", moment(offset, 18, 30))

    snapshot = source.read(behavior_tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(69, 23)
    )
    tuesday = query.slot_at(built, moment(1, 18))
    thursday = query.slot_at(built, moment(3, 18))

    at_tuesday = {item.target: item for item in query.successors(built, "吃饭", slot=tuesday)}
    at_thursday = {item.target: item for item in query.successors(built, "吃饭", slot=thursday)}
    assert at_tuesday["打球"].probability > 0.9
    assert at_tuesday["打球"].approximate is False
    assert at_thursday["打球"].probability < 0.1

    # 不带槽位的边缘概率把七天混在一起，只有 1/7 上下——它不是"周二打球"的答案。
    marginal = {item.target: item for item in query.successors(built, "吃饭")}
    assert marginal["打球"].probability == pytest.approx(1 / 7, abs=0.05)


def test_no_successor_enters_the_joint_denominator(tmp_path) -> None:
    """"这个点做完 A 通常就收工"必须压低联合概率，否则近似成"必然接着做 B"。

    十个周一里散步之后只有五次接着洗澡；另外五次什么都没做。如果 ∅ 不进联合的分母，
    这一格就只剩下"洗澡"一个去向，概率会算成 1.0。
    """

    writer, behavior_tree = writer_for(tmp_path)
    for offset in range(70):
        publish_occurrence(writer, "散步", moment(offset, 21))
        if (FIRST + timedelta(days=offset)).weekday() == 0 and (offset // 7) % 2 == 0:
            publish_occurrence(writer, "洗澡", moment(offset, 21, 20))

    snapshot = source.read(behavior_tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(69, 23)
    )
    slot = query.slot_at(built, moment(0, 21))  # 周一 21:00
    outgoing = {item.target: item for item in query.successors(built, "散步", slot=slot)}
    assert outgoing["洗澡"].probability == pytest.approx(0.5, abs=0.05)
    assert outgoing[NO_SUCCESSOR].probability == pytest.approx(0.5, abs=0.05)


def test_missing_joint_cell_falls_back_to_a_flagged_approximation(tmp_path) -> None:
    """该槽没有联合样本时给近似值，但**必须**标出来——近似不能冒充实测。"""

    writer, behavior_tree = writer_for(tmp_path)
    for offset in range(20):
        publish_occurrence(writer, "洗手", moment(offset, 12))
        publish_occurrence(writer, "吃饭", moment(offset, 12, 5))

    snapshot = source.read(behavior_tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(19, 23)
    )
    empty_slot = query.slot_at(built, moment(0, 3))
    outgoing = {item.target: item for item in query.successors(built, "洗手", slot=empty_slot)}
    assert outgoing["吃饭"].approximate is True
    assert 0.0 <= outgoing["吃饭"].probability <= 1.0


# --- 留白 -------------------------------------------------------------------------------


def test_a_single_dominant_habit_leaves_little_room_for_silence(tmp_path) -> None:
    writer, behavior_tree = writer_for(tmp_path)
    for offset in range(30):
        publish_occurrence(writer, "吃药", moment(offset, 7, 30))
    snapshot = source.read(behavior_tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(29, 23)
    )
    outlook = query.slot_outlook(built, query.slot_at(built, moment(0, 7, 30)))
    assert outlook.irregularity == pytest.approx(0.0, abs=1e-9)
    assert outlook.escape < 0.2


def test_a_crowded_slot_of_one_offs_says_keep_quiet(tmp_path) -> None:
    """一堆只见过一次的动作挤在同一个格子：熵高、逃逸高，该闭嘴。"""

    writer, behavior_tree = writer_for(tmp_path)
    for offset, name in enumerate(["翻书", "浇花", "找钥匙", "擦桌子", "剪指甲"]):
        publish_occurrence(writer, name, moment(offset * 7, 15))
    snapshot = source.read(behavior_tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(29, 23)
    )
    outlook = query.slot_outlook(built, query.slot_at(built, moment(0, 15)))
    assert outlook.irregularity > 0.9
    assert outlook.escape == pytest.approx(1.0, abs=0.01)


def test_an_empty_slot_is_all_escape(tmp_path) -> None:
    built = tree_for(tmp_path)
    outlook = query.slot_outlook(built, query.slot_at(built, moment(0, 3)))
    assert outlook.candidates == ()
    assert outlook.escape == 1.0
    assert outlook.irregularity == 1.0


# --- 规格点名、此前没有测试的几处 -------------------------------------------------------


def test_config_digest_covers_estimation_parameters_only(tmp_path) -> None:
    """改运维节奏不该让已发布的树看起来"出自另一套统计"；改估计参数必须。"""

    from prediction.config import PredictionTreeConfig

    base = config()
    same = {
        name: builder.config_digest(PredictionTreeConfig(**{**vars(base), name: value}))
        for name, value in (("published_generations", 9), ("rebuild_interval_seconds", 3_600.0))
    }
    assert set(same.values()) == {builder.config_digest(base)}

    changed = builder.config_digest(PredictionTreeConfig(**{**vars(base), "slot_minutes": 30}))
    assert changed != builder.config_digest(base)


def test_a_query_is_pinned_to_one_generation(tmp_path) -> None:
    """跨代混读会让节点与边出自不同批次；查询层只接整棵树，没有"按需取一格"的口子。"""

    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=3)
    first = store.publish(tree_for(tmp_path, hour=20))
    pinned = store.load()
    assert pinned is not None

    # 发布新的一代之后，手里那棵树一个数都不会变。
    store.publish(tree_for(tmp_path, name="behavior-tree-2", hour=21))
    assert store.active() != first
    assert pinned.built_at == first.published_at
    assert pinned == store.load_generation(first.generation, expected_digest=first.digest)


def test_a_tree_larger_than_the_bound_is_refused(tmp_path, monkeypatch) -> None:
    """超限拒绝发布，而不是写出一个读不回来的代。"""

    from prediction import store as store_module

    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=2)
    monkeypatch.setattr(store_module, "MAX_TREE_BYTES", 16)
    with pytest.raises(PredictionTreeStoreError, match="size bound"):
        store.publish(tree_for(tmp_path))
    assert store.active() is None


def test_an_unknown_schema_version_is_refused_rather_than_half_parsed(tmp_path) -> None:
    payload = dict(codec.encode(tree_for(tmp_path)))
    payload["schema_version"] = codec.SCHEMA_VERSION + 1
    with pytest.raises(PredictionTreeError, match="schema version"):
        codec.decode(payload)


def test_both_gap_kinds_reduce_exposure_the_same_way(tmp_path) -> None:
    """"没读懂"和"未观测"对"他做了我们能不能看见"是同一件事，扣减必须一样。"""

    snapshots = []
    for index, gap_kind in enumerate(("没读懂", "未观测")):
        writer, behavior_tree = writer_for(tmp_path, f"behavior-tree-{index}")
        for offset in range(10):
            publish_occurrence(writer, "吃药", moment(offset, 7, 30))
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(
                occurred_on=moment(5, 7).date(),
                gap_kind=gap_kind,
                started_at=moment(5, 7),
                ended_at=moment(5, 8),
            ),
        )
        snapshots.append(source.read(behavior_tree))

    assert len(snapshots[0].gaps) == len(snapshots[1].gaps) == 1
    trees = [
        builder.build(item, config=config(), reference=item.latest_day, built_at=moment(9, 23))
        for item in snapshots
    ]
    assert trees[0].exposure == trees[1].exposure


def test_an_occurrence_marked_as_reminded_is_refused(tmp_path) -> None:
    """干预账本还没建；把被提醒之后的发生数进自然率，事后无法再分开。"""

    writer, behavior_tree = writer_for(tmp_path)
    publish_occurrence(writer, "吃药", moment(0, 7, 30), reminded=True)
    with pytest.raises(PredictionTreeError, match="reminded"):
        source.read(behavior_tree)
