"""接缝测试：真实行为树 → 快照 → 重建 → 发布 → 查询。

纯函数层的单测只能证明实现符合我自己的想法；真正会出问题的地方在模块之间的接缝——
字段名对不对得上、时间偏移有没有被折掉、并行链接读不读得出来、消歧重复有没有被跳过。
所以这里从 ``BehaviorDocumentWriter`` 真写一棵行为树开始，一路走到 ``query`` 出结果。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.behavior import BehaviorDocumentWriter, BehaviorKind, BehaviorLinkType, BehaviorTree, BehaviorURI
from habitus.foundation.integrity import text_digest
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.prediction import builder, codec, query, source
from habitus.prediction.edges import NO_SUCCESSOR
from habitus.prediction.errors import PredictionTreeError, PredictionTreeStoreError
from habitus.prediction.model import PredictionTree, SlotKey
from habitus.prediction.store import (
    GENERATIONS_DIRECTORY,
    TREE_FILENAME,
    PredictionTreeStore,
    PublishedGeneration,
)
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
    # 候选来自曲线而不是格子，所以这个动作照样在列——但它如实说"这一格一次都没见过"
    # （count 0、边际率小到几乎不存在），而两条留白曲线只看真的发生过的格子。
    assert [item.action for item in outlook.candidates] == ["吃药"]
    assert outlook.candidates[0].count == 0.0
    assert outlook.candidates[0].marginal < 1e-6
    assert outlook.escape == 1.0
    assert outlook.irregularity == 1.0


# --- 规格点名、此前没有测试的几处 -------------------------------------------------------


def test_config_digest_covers_estimation_parameters_only(tmp_path) -> None:
    """改运维节奏不该让已发布的树看起来"出自另一套统计"；改估计参数必须。"""

    from habitus.prediction.config import PredictionTreeConfig

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

    from habitus.prediction import store as store_module

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


def _tree_with_gap(tmp_path, name: str, *, gap_kind: str | None, start, end):
    """十天每天 07:30 吃药；可选地在第 5 天加一段空白。"""

    writer, behavior_tree = writer_for(tmp_path, name)
    for offset in range(10):
        publish_occurrence(writer, "吃药", moment(offset, 7, 30))
    if gap_kind is not None:
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(
                occurred_on=start.date(),
                gap_kind=gap_kind,
                started_at=start,
                ended_at=end,
            ),
        )
    snapshot = source.read(behavior_tree)
    return snapshot, builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(9, 23)
    )


def test_both_gap_kinds_reduce_exposure_the_same_way(tmp_path) -> None:
    """没有行为落在里面时，"没读懂"和"未观测"对曝光的扣减必须一样。

    两类空白对"他做了我们能不能看见"是同一件事——差别只在**能不能被一条读出来的行为证伪**
    （见 ``nodes.reconcile_gaps``），而这条空白落在凌晨、里面一条行为都没有。
    """

    trees = [
        _tree_with_gap(
            tmp_path, f"behavior-tree-{index}", gap_kind=kind, start=moment(5, 3), end=moment(5, 4)
        )[1]
        for index, kind in enumerate(("没读懂", "未观测"))
    ]
    assert trees[0].exposure == trees[1].exposure


def test_an_unreadable_gap_is_voided_by_a_behaviour_read_inside_it(tmp_path) -> None:
    """"没读懂"断言这段读不出行为；真读出了一条，这句断言就被证伪——整段作废。

    这条钉住的是"一份数据一个真相"：作废之后曝光与转移删失读的是**同一份**空白账，而不是
    曝光把这一槽记满、边那边照旧把它当洞。
    """

    _, without = _tree_with_gap(tmp_path, "no-gap", gap_kind=None, start=None, end=None)
    _, voided = _tree_with_gap(
        tmp_path, "unreadable", gap_kind="没读懂", start=moment(5, 7), end=moment(5, 8)
    )
    assert voided.exposure == without.exposure


def test_an_unobserved_gap_containing_a_behaviour_is_an_upstream_contradiction(tmp_path) -> None:
    """"未观测"说没在看，却又读出了一条行为——上游自相矛盾，本层不替它圆场。

    这类空白目前树里没有生产者（覆盖信号契约未接入），所以这是一条前瞻护栏：宁可在这里以
    一句说得清的话炸掉，也不要悄悄把它当成"看见了"数进分母。
    """

    with pytest.raises(PredictionTreeError, match="unobserved gap"):
        _tree_with_gap(
            tmp_path, "unobserved", gap_kind="未观测", start=moment(5, 7), end=moment(5, 8)
        )


def test_an_occurrence_marked_as_reminded_is_refused(tmp_path) -> None:
    """干预账本还没建；把被提醒之后的发生数进自然率，事后无法再分开。"""

    writer, behavior_tree = writer_for(tmp_path)
    publish_occurrence(writer, "吃药", moment(0, 7, 30), reminded=True)
    with pytest.raises(PredictionTreeError, match="reminded"):
        source.read(behavior_tree)


# --- 时间画像与并行的对称闭包（本轮新增） --------------------------------------------------


def test_day_outlook_answers_when_and_how_wide(tmp_path) -> None:
    """"这个行为在周二通常几点、范围多宽"——预测层最基本的问题，此前没有接口回答。

    逐槽累乘 h(t)·Π(1−h(s)) 早先只存在于 evaluation 的离线回测函数里。
    """

    writer, tree = writer_for(tmp_path)
    for week in range(10):
        offset = 1 + 7 * week  # FIRST 是周一，+1 即周二
        # 早饭的时刻在 07:00 / 07:15 / 07:30 之间浮动——"时间范围"正是要接住这种抖动。
        publish_occurrence(writer, "吃早饭", moment(offset, 7, (week % 3) * 15))

    snapshot = source.read(tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(70, 23)
    )
    outlook = query.day_outlook(built, 1, "吃早饭")
    assert outlook is not None
    assert outlook.earliest.slot <= outlook.median.slot <= outlook.latest.slot
    # 07:00–07:30 这一段：中位落在窗口里，且窗口不是整天
    assert 7 * 4 - 2 <= outlook.median.slot <= 7 * 4 + 4
    assert outlook.latest.slot - outlook.earliest.slot < 4 * 4  # 不到四小时宽
    assert outlook.mass > 0.5  # 周二基本一定会发生
    assert query.day_outlook(built, 3, "吃早饭") is None  # 周四从没做过 → 没有画像


def test_slot_outlook_answers_cumulative_for_actions_that_never_hit_this_cell(tmp_path) -> None:
    """"到这个点为止今天做了没有"必须覆盖这一格从没发生过的动作——缺失检测正是要问它们。

    并且答案要是**这个周几自己的**：跨周几混读会让"周一早上吃、周二不吃"读成同一个数。
    """

    writer, tree = writer_for(tmp_path)
    for offset in range(28):
        if (FIRST + timedelta(days=offset)).weekday() == 0:  # 只有周一吃
            publish_occurrence(writer, "吃药", moment(offset, 7, 30))
        publish_occurrence(writer, "洗手", moment(offset, 12))

    snapshot = source.read(tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(27, 23)
    )
    monday_evening = _by_action(query.slot_outlook(built, SlotKey(weekday=0, slot=80)))
    assert monday_evening["吃药"].count == 0.0  # 晚上从没吃过
    assert monday_evening["吃药"].cumulative > 0.9  # 但到晚上它今天早就做完了
    monday_dawn = _by_action(query.slot_outlook(built, SlotKey(weekday=0, slot=20)))
    assert monday_dawn["吃药"].cumulative == pytest.approx(0.0)  # 凌晨还没做
    # 周二从来不吃：同一个槽位、不同周几，答案必须不一样
    assert "吃药" not in _by_action(query.slot_outlook(built, SlotKey(weekday=1, slot=80)))


def _by_action(outlook):
    return {item.action: item for item in outlook.candidates}


def test_parallels_read_the_same_evidence_from_either_side(tmp_path) -> None:
    """并行是对称关系：从哪一边问都该看到同一份证据，而不是被劈开的两半。"""

    writer, tree = writer_for(tmp_path)
    for offset in range(10):
        # 一半的日子吃饭先开始，另一半看手机先开始——旧写法会因此落进两个键。
        if offset % 2 == 0:
            first = publish_occurrence(writer, "吃饭", moment(offset, 12))
            publish_occurrence(
                writer,
                "看手机",
                moment(offset, 12, 5),
                links=((BehaviorLinkType.CONCURRENT_WITH, BehaviorURI.from_address(first.address)),),
            )
        else:
            first = publish_occurrence(writer, "看手机", moment(offset, 12))
            publish_occurrence(
                writer,
                "吃饭",
                moment(offset, 12, 5),
                links=((BehaviorLinkType.CONCURRENT_WITH, BehaviorURI.from_address(first.address)),),
            )

    snapshot = source.read(tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(9, 23)
    )
    assert set(built.parallels) == {("吃饭", "看手机")}
    meal = {item.target: item for item in query.parallels(built, "吃饭")}
    phone = {item.target: item for item in query.parallels(built, "看手机")}
    assert meal["看手机"].count == pytest.approx(phone["吃饭"].count)  # 同一份证据
    assert meal["看手机"].probability == pytest.approx(1.0)  # 吃饭时必然在看手机
    assert meal["看手机"].lift is None  # 并行没有 lift，不填一个会被误读的 0.0


# --- 发布形态的自洽（本轮新增） ------------------------------------------------------------


def test_a_previous_schema_version_is_refused_as_a_version_not_as_corruption(tmp_path) -> None:
    """旧代的字节读不了是**版本不对**，不是存储损坏——报错说错了方向，运维就查错地方。

    磁盘上最多留着 ``published_generations`` 代旧字节，回滚路径全靠这条错误信息分辨。
    """

    store = PredictionTreeStore(tmp_path / "prediction", retained_generations=2)
    published = store.publish(tree_for(tmp_path))
    path = store.root / GENERATIONS_DIRECTORY / published.generation / TREE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = codec.SCHEMA_VERSION - 1
    raw = json.dumps(payload, ensure_ascii=False)
    path.write_text(raw, encoding="utf-8")
    store._activate(  # noqa: SLF001 - 指针要跟着改，否则先撞上摘要校验
        PublishedGeneration(
            generation=published.generation,
            digest=text_digest(raw),
            config_digest=published.config_digest,
            published_at=published.published_at,
        )
    )
    with pytest.raises(PredictionTreeStoreError, match="cannot be decoded"):
        store.load()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["curves"][0].__setitem__("marginal", [[0.5, 2.0]]), "clock face"),
        (lambda p: p["curves"][0].__setitem__("weekday", 99), "below 7"),
        (lambda p: p["parallel_totals"].clear(), "participation weight"),
    ],
)
def test_a_malformed_payload_is_refused_instead_of_half_parsed(tmp_path, mutate, message) -> None:
    """槽位键有越界检查，曲线与并行分母也必须有。

    不校验的后果实测过：一条长度 2 的曲线能解码成功，之后每次查询抛**裸 IndexError**——
    从一个把全部错误归一成 PredictionTreeError 的层里漏出 builtin；并行分母缺了则让
    ``query.parallels`` 静默算出"证据 1.0、概率 0.0"。
    """

    payload = codec.encode(_tree_with_parallels(tmp_path))
    mutate(payload)
    with pytest.raises(PredictionTreeError, match=message):
        codec.decode(payload)


def test_encoding_round_trips_parallels_and_a_missing_trend(tmp_path) -> None:
    """往返断言要盖到并行与 ``trend=None``：只有节点和边的树走不到那两段编码。"""

    built = _tree_with_parallels(tmp_path)
    assert built.parallels and built.parallel_totals
    assert any(curve.trend is None for curve in built.curves.values())
    assert codec.decode(codec.encode(built)) == built


def _tree_with_parallels(tmp_path) -> PredictionTree:
    """一棵带并行边、且至少有一条曲线没有趋势的树。"""

    writer, behavior_tree = writer_for(tmp_path, f"parallel-{id(tmp_path)}")
    for offset in range(6):
        meal = publish_occurrence(writer, "吃饭", moment(offset, 12))
        publish_occurrence(
            writer,
            "看手机",
            moment(offset, 12, 5),
            links=((BehaviorLinkType.CONCURRENT_WITH, BehaviorURI.from_address(meal.address)),),
        )
    publish_occurrence(writer, "剪头发", moment(0, 15))  # 只出现一次 → 没有复发证据 → 没有趋势
    snapshot = source.read(behavior_tree)
    return builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(5, 23)
    )


def test_recurrence_status_answers_whether_it_is_overdue(tmp_path) -> None:
    """四类查询之一，此前全仓零测试。"""

    writer, behavior_tree = writer_for(tmp_path)
    for offset in range(10):
        publish_occurrence(writer, "浇花", moment(offset * 3, 9))  # 每三天一次
    snapshot = source.read(behavior_tree)
    built = builder.build(
        snapshot, config=config(), reference=snapshot.latest_day, built_at=moment(27, 23)
    )
    status = query.recurrence_status(built, "浇花", elapsed_seconds=6 * 86_400)
    assert status is not None
    assert status.intervals.p50 == pytest.approx(3 * 86_400, rel=0.1)
    assert status.overdue == pytest.approx(2.0, rel=0.1)  # 拖了两倍中位间隔
    assert query.recurrence_status(built, "从没做过的事", elapsed_seconds=1.0) is None
