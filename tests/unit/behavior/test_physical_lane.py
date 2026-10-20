"""物理 lane 的融合：对上游约定的校验、一段一条判断、去重、改判不收、没在看的时间记成空白、不许和收过的矛盾。

依据是方案第四节与裁定 40；每条用例对应其中一条规则，用例名写的就是那条规则。数据是自己造的，还没有上游的真实样例。
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.fusion.lanes.physical import (
    PHYSICAL_FUSION_VERSION,
    PHYSICAL_PROTOCOL,
    CoverageState,
    PhysicalDeliveryError,
    PhysicalLane,
    PhysicalSegmentLedger,
    PhysicalSegmentRecorder,
    UnitDisposition,
    parse_delivery,
    physical_observer,
)
from tests.unit.behavior.session_fixtures import SUBJECT, Stores

DAY = "2026-10-11"


def moment(clock: str) -> str:
    return f"{DAY}T{clock}+08:00"


def delivery(**changes: Any) -> dict[str, Any]:
    """一天的手表数据：上午洗手、傍晚跑步，中午充电一个多小时。"""

    body: dict[str, Any] = {
        "source": "physical/s12",
        "device": "watch-01",
        "delivery_id": "delivery-0001",
        "sent_at": moment("22:05:00"),
        "segments": [
            {"id": "seg-000812", "behavior": "洗手", "started_at": moment("09:12:03"), "ended_at": moment("09:12:41")},
            {
                "id": "seg-000813",
                "behavior": "跑步",
                "started_at": moment("18:30:10"),
                "ended_at": moment("19:02:44"),
                "facts": ["3.2 公里"],
            },
        ],
        "coverage": [
            {"started_at": moment("07:40:00"), "ended_at": moment("13:05:00"), "state": "observing"},
            {"started_at": moment("13:05:00"), "ended_at": moment("14:20:00"), "state": "charging"},
            {"started_at": moment("14:20:00"), "ended_at": moment("22:00:00"), "state": "observing"},
        ],
    }
    body.update(changes)
    return body


def edited(path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    body = copy.deepcopy(delivery())
    target: Any = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return body


def wired(tmp_path: Path) -> tuple[PhysicalLane, Stores]:
    stores = Stores(tmp_path)
    stores.now = datetime.fromisoformat(moment("22:10:00"))
    recorder = PhysicalSegmentRecorder(
        observations=stores.observations,
        judgements=stores.judgements,
        receipts=stores.receipts,
        coverage=stores.coverage,
        ledger=PhysicalSegmentLedger(tmp_path),
        primary_subject=SUBJECT,
        clock=lambda: stores.now,
    )
    return PhysicalLane(recorder, max_backfill_days=7), stores


def readable(stores: Stores) -> list[dict[str, Any]]:
    return [item for item in stores.judgements.list() if item["behavior"] is not None]


def blanks(stores: Stores) -> list[dict[str, Any]]:
    return [item for item in stores.judgements.list() if item["behavior"] is None]


# ── 对上游的约定 ─────────────────────────────────────────────────────────────────────


def test_a_delivery_that_keeps_the_contract_is_read_as_is() -> None:
    parsed = parse_delivery(delivery())
    assert [item.behavior for item in parsed.segments] == ["洗手", "跑步"]
    assert parsed.segments[1].facts == ("3.2 公里",)
    assert [item.state for item in parsed.gaps] == [CoverageState.CHARGING]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # 来源标识认定 lane：不是物理 lane 的不收
        (delivery(source="session/claude"), "not a physical-lane delivery"),
        # 只带结论，不带原始读数、置信度、分类器的东西
        (edited(("segments", 0, "heart_rate"), 92), "outside the contract"),
        (edited(("segments", 0, "confidence"), 0.83), "outside the contract"),
        (delivery(model_version="rf-3"), "outside the contract"),
        # 时间要带偏移
        (edited(("segments", 0, "started_at"), f"{DAY}T09:12:03"), "must carry its UTC offset"),
        # 只送已经结束的完整一段
        (edited(("segments", 0, "ended_at"), moment("09:00:00")), "ends before it starts"),
        # 同一时刻只有一个行为
        (edited(("segments", 1, "started_at"), moment("09:12:30")), "must not overlap"),
        # 段要落在"在看"的时段里
        (edited(("segments", 0, "started_at"), moment("07:00:00")), "outside the observed coverage"),
        (edited(("segments", 1, "started_at"), moment("13:30:00")), "outside the observed coverage"),
        # 覆盖时段首尾相接、不留洞
        (edited(("coverage", 2, "started_at"), moment("14:30:00")), "must be contiguous"),
        (edited(("coverage", 1, "state"), "asleep"), "not a known coverage state"),
        (edited(("segments", 1, "id"), "seg-000812"), "must be unique"),
        # 段的开始不能正好踩在"在看"结束、"没在看"开始的那一瞬
        (
            edited(
                ("segments", 0),
                {"id": "z", "behavior": "洗手", "started_at": moment("13:05:00"), "ended_at": moment("13:05:00")},
            ),
            "outside the observed coverage",
        ),
        # 只送已经结束的：结束时刻晚于交付送出的时刻
        (delivery(sent_at=moment("18:40:00")), "had not ended"),
        # 设备名、段编号要进凭据的身份，先在入口挡
        (delivery(device="w" * 300), "at most 120 characters"),
        (edited(("segments", 0, "id"), "s" * 600), "at most 200 characters"),
        (delivery(device="watch\x00"), "printable text"),
        (edited(("segments", 0, "behavior"), "  "), "must be non-empty text"),
    ],
)
def test_a_delivery_that_breaks_the_contract_is_refused_whole(body: dict[str, Any], message: str) -> None:
    with pytest.raises(PhysicalDeliveryError, match=message):
        parse_delivery(body)


def test_a_segment_may_run_across_two_adjoining_observed_spans() -> None:
    body = delivery(
        coverage=[
            {"started_at": moment("07:40:00"), "ended_at": moment("09:12:20"), "state": "observing"},
            {"started_at": moment("09:12:20"), "ended_at": moment("22:00:00"), "state": "observing"},
        ]
    )
    assert len(parse_delivery(body).segments) == 2


def test_two_behaviours_may_meet_at_the_same_instant() -> None:
    """吃饭结束的那一瞬正好是接电话的开始：首尾相接不算重叠。"""

    body = delivery(
        segments=[
            {"id": "a", "behavior": "吃饭", "started_at": moment("12:10:00"), "ended_at": moment("12:35:00")},
            {"id": "b", "behavior": "接电话", "started_at": moment("12:35:00"), "ended_at": moment("12:41:00")},
        ]
    )
    assert [item.behavior for item in parse_delivery(body).segments] == ["吃饭", "接电话"]


# ── 一段一条判断，不调模型 ───────────────────────────────────────────────────────────


def test_each_segment_becomes_one_judgement_that_only_restates_what_upstream_said(tmp_path: Path) -> None:
    lane, stores = wired(tmp_path)
    outcomes = lane.consume(parse_delivery(delivery()))

    assert [item.disposition for item in outcomes] == [UnitDisposition.RECORDED] * 3
    wash, run = sorted(readable(stores), key=lambda item: item["started_at"])
    assert wash["behavior"] == "洗手"
    assert wash["summary"] == "洗手，09:12–09:12"  # 固定句式：什么行为、几点到几点
    assert run["summary"] == "跑步，18:30–19:02；3.2 公里"  # 上游带了补充事实就拼上
    assert (wash["goal"], wash["basis"], wash["relations"]) == (None, [], [])  # 手表看不出目标与步骤；不接续
    assert wash["started_at"].startswith(f"{DAY}T09:12:03")
    assert wash["last_observed_at"].startswith(f"{DAY}T09:12:41")
    assert wash["subjects"] == [SUBJECT]
    assert wash["fusion_version"] == PHYSICAL_FUSION_VERSION
    assert (wash["status"], wash["status_basis"]) == ("completed", "observed")  # 上游只送做完了的，进行中的不落树


def test_a_behaviour_keeps_the_local_time_of_where_it_happened(tmp_path: Path) -> None:
    """人在外地：行为算在它发生地的那一天、那个钟点，不折到本机的时区。"""

    lane, stores = wired(tmp_path)
    body = {
        "source": "physical/s12",
        "device": "watch-01",
        "delivery_id": "tokyo",
        "sent_at": "2026-10-12T08:00:00+09:00",
        "segments": [
            {
                "id": "t1",
                "behavior": "洗手",
                "started_at": "2026-10-12T00:30:00+09:00",
                "ended_at": "2026-10-12T00:31:00+09:00",
            }
        ],
        "coverage": [
            {"started_at": "2026-10-12T00:00:00+09:00", "ended_at": "2026-10-12T07:00:00+09:00", "state": "observing"}
        ],
    }
    stores.now = datetime.fromisoformat("2026-10-12T08:05:00+09:00")
    lane.consume(parse_delivery(body))
    (wash,) = readable(stores)
    assert wash["started_at"] == "2026-10-12T00:30:00.000000+09:00"
    assert wash["summary"] == "洗手，00:30–00:31"


def test_the_evidence_names_the_device_and_the_segment_and_nothing_raw(tmp_path: Path) -> None:
    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    envelopes = stores.observations.list()
    assert {item.protocol for item in envelopes} == {PHYSICAL_PROTOCOL}
    assert {item.observer_id for item in envelopes} == {physical_observer("watch-01")}
    refs = {ref for item in envelopes for observation in item.batch.observations for ref in observation.evidence_refs}
    assert "watch-01#segment#seg-000812" in refs


def test_time_nobody_was_watching_becomes_a_blank_not_a_behaviour(tmp_path: Path) -> None:
    """没戴、充电、没数据的时间和"他没做"在行为数据上分不出来，所以单独记成空白。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    (blank,) = blanks(stores)
    assert blank["started_at"].startswith(f"{DAY}T13:05:00")
    assert blank["last_observed_at"].startswith(f"{DAY}T14:20:00")
    assert blank["subjects"] == []
    assert blank["summary"] is None


def test_watched_time_with_no_segment_writes_nothing(tmp_path: Path) -> None:
    """在看、但没有任何段：意思是没发现在做已知的行为，不为它造记录。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery(segments=[])))
    assert readable(stores) == []
    assert len(blanks(stores)) == 1


def test_path_symbols_in_a_behaviour_are_replaced_the_same_way_every_time(tmp_path: Path) -> None:
    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(edited(("segments", 0, "behavior"), "洗手/擦手")))
    assert "洗手 擦手" in {item["behavior"] for item in readable(stores)}


def test_a_behaviour_that_cannot_be_a_name_is_left_out_and_the_rest_still_lands(tmp_path: Path) -> None:
    lane, stores = wired(tmp_path)
    outcomes = lane.consume(parse_delivery(edited(("segments", 0, "behavior"), "..")))
    assert outcomes[0].disposition is UnitDisposition.UNUSABLE_NAME
    assert [item["behavior"] for item in readable(stores)] == ["跑步"]


# ── 重发、改判、迟到 ─────────────────────────────────────────────────────────────────


def test_the_same_segment_sent_again_in_another_delivery_is_not_written_twice(tmp_path: Path) -> None:
    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    stores.now += timedelta(hours=3)
    again = lane.consume(parse_delivery(delivery(delivery_id="delivery-0002", sent_at=moment("23:59:00"))))

    assert {item.disposition for item in again} == {UnitDisposition.ALREADY_RECORDED}
    assert len(readable(stores)) == 2
    assert len(blanks(stores)) == 1


def test_a_segment_changed_after_it_was_sent_is_refused(tmp_path: Path) -> None:
    """行为树只增不改：同一个编号换了说法或换了时间，是上游改判，不收。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    renamed = lane.consume(parse_delivery(edited(("segments", 0, "behavior"), "洗碗")))
    moved = lane.consume(parse_delivery(edited(("segments", 0, "ended_at"), moment("09:13:30"))))

    assert renamed[0].disposition is UnitDisposition.REVISION_REJECTED
    assert moved[0].disposition is UnitDisposition.REVISION_REJECTED
    assert sorted(item["behavior"] for item in readable(stores)) == ["洗手", "跑步"]


def test_what_counts_as_the_same_segment_depends_only_on_what_upstream_sent(tmp_path: Path) -> None:
    """小账上的摘要只由上游给的字段算：本机换了主体名，之前收下的段再送来仍然认得出，不会被当成改判。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    lane.recorder.primary_subject = "改了名字的主体"
    again = lane.consume(parse_delivery(delivery(delivery_id="delivery-0002")))
    assert {item.disposition for item in again} == {UnitDisposition.ALREADY_RECORDED}
    # 补充事实也是上游给的内容：只改它同样是改判
    changed = lane.consume(parse_delivery(edited(("segments", 1, "facts"), ["4.0 公里"])))
    assert changed[1].disposition is UnitDisposition.REVISION_REJECTED


def test_the_same_delivery_arriving_twice_at_once_is_written_once(tmp_path: Path) -> None:
    """上游超时重试、两个入口同时送同一份：一次只处理一份，后到的认出已经收过。"""

    import threading

    lane, stores = wired(tmp_path)
    results: list[set[UnitDisposition]] = []

    def send() -> None:
        results.append({item.disposition for item in lane.consume(parse_delivery(delivery()))})

    threads = [threading.Thread(target=send) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(len(item) for item in results) == [1, 1, 1, 1]
    assert results.count({UnitDisposition.RECORDED}) == 1
    assert len(readable(stores)) == 2 and len(blanks(stores)) == 1


# ── 不许和收过的矛盾 ─────────────────────────────────────────────────────────────────


def later(**changes: Any) -> dict[str, Any]:
    """同一天稍后的另一份交付（编号不同）。"""

    body: dict[str, Any] = {
        "source": "physical/s12",
        "device": "watch-01",
        "delivery_id": "delivery-later",
        "sent_at": moment("23:30:00"),
        "segments": [],
        "coverage": [],
    }
    body.update(changes)
    return body


def test_a_behaviour_inside_time_already_recorded_as_unobserved_is_refused(tmp_path: Path) -> None:
    """先说 13:05–14:20 在充电，后来又说这段在看、13:30 洗了手：两边都落树，下游就读不了这棵树了。整份拒收。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    contradicting = later(
        segments=[{"id": "x1", "behavior": "洗手", "started_at": moment("13:30:00"), "ended_at": moment("13:30:40")}],
        coverage=[{"started_at": moment("13:00:00"), "ended_at": moment("15:00:00"), "state": "observing"}],
    )
    with pytest.raises(PhysicalDeliveryError, match="already recorded as unobserved"):
        lane.consume(parse_delivery(contradicting))
    assert sorted(item["behavior"] for item in readable(stores)) == ["洗手", "跑步"]  # 什么都没多写


def test_a_behaviour_overlapping_one_already_recorded_under_another_id_is_refused(tmp_path: Path) -> None:
    """同一时刻只有一个行为，跨交付也一样：换个编号把同一段再送一遍，不是新的一次。首尾相接的照收。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))  # 洗手 09:12:03–09:12:41
    observed = [{"started_at": moment("07:40:00"), "ended_at": moment("13:05:00"), "state": "observing"}]
    same_again = later(
        segments=[
            {"id": "new-id", "behavior": "洗手", "started_at": moment("09:12:03"), "ended_at": moment("09:12:41")}
        ],
        coverage=observed,
    )
    with pytest.raises(PhysicalDeliveryError, match="overlaps a segment already recorded"):
        lane.consume(parse_delivery(same_again))
    touching = later(
        delivery_id="delivery-touching",
        segments=[{"id": "next", "behavior": "擦手", "started_at": moment("09:12:41"), "ended_at": moment("09:13:00")}],
        coverage=observed,
    )
    assert [item.disposition for item in lane.consume(parse_delivery(touching))] == [UnitDisposition.RECORDED]
    assert sorted(item["behavior"] for item in readable(stores)) == ["擦手", "洗手", "跑步"]


def test_unobserved_time_over_a_behaviour_already_recorded_is_refused(tmp_path: Path) -> None:
    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    contradicting = later(
        coverage=[{"started_at": moment("09:00:00"), "ended_at": moment("09:30:00"), "state": "no_data"}]
    )
    with pytest.raises(PhysicalDeliveryError, match="covers the start of a segment"):
        lane.consume(parse_delivery(contradicting))
    assert len(blanks(stores)) == 1


def test_unobserved_time_reported_twice_is_recorded_as_its_union(tmp_path: Path) -> None:
    """两份都说没在看、起止不完全一样（边界修了两分钟、后一份报得更长）：只落还没记过的那一截，不重复落。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))  # 13:05–14:20 充电
    longer = later(coverage=[{"started_at": moment("13:03:00"), "ended_at": moment("15:00:00"), "state": "charging"}])
    (outcome,) = lane.consume(parse_delivery(longer))
    assert outcome.disposition is UnitDisposition.RECORDED
    spans = sorted((item["started_at"][11:19], item["last_observed_at"][11:19]) for item in blanks(stores))
    assert spans == [("13:03:00", "13:05:00"), ("13:05:00", "14:20:00"), ("14:20:00", "15:00:00")]
    # 同一段没在看的时间换个原因、换个偏移写法再报，不是新的一段
    same = later(
        delivery_id="delivery-again",
        coverage=[
            {"started_at": "2026-10-11T05:05:00+00:00", "ended_at": "2026-10-11T06:20:00+00:00", "state": "not_worn"}
        ],
    )
    (outcome,) = lane.consume(parse_delivery(same))
    assert outcome.disposition is UnitDisposition.ALREADY_RECORDED
    assert len(blanks(stores)) == 3


def test_resends_and_revisions_are_still_recognised_after_the_evidence_is_released(tmp_path: Path) -> None:
    """凭据在落树之后释放、覆盖记录也会清；补发跨度之内晚到几天的重发与改判仍要靠小账认得出来；跨度之外的整份拒收
    （小账只记得这个跨度，26 号清单 6.4）。"""

    import shutil

    from habitus.behavior.fusion.lanes.physical.model import PhysicalDeliveryError

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    for envelope in stores.observations.list():
        stores.observations.discard(envelope.source_id)
    stores.now += timedelta(days=6)
    shutil.rmtree(tmp_path / "fusion" / "coverage")  # 覆盖记录已清：只剩小账

    resent = lane.consume(parse_delivery(delivery(delivery_id="delivery-late")))
    revised = lane.consume(parse_delivery(edited(("segments", 0, "behavior"), "洗脸")))

    assert {item.disposition for item in resent} == {UnitDisposition.ALREADY_RECORDED}
    assert revised[0].disposition is UnitDisposition.REVISION_REJECTED
    assert sorted(item["behavior"] for item in readable(stores)) == ["洗手", "跑步"]
    stores.now += timedelta(days=24)
    with pytest.raises(PhysicalDeliveryError, match="backfill"):
        lane.consume(parse_delivery(delivery(delivery_id="delivery-far-too-late")))


def test_the_ledger_forgets_whole_utc_days_past_the_backfill_span(tmp_path: Path) -> None:
    """26 号清单 6.4：编号与时段两份都按 UTC 日分目录，过了补发跨度再多一个日就整日删。"""

    lane, stores = wired(tmp_path)
    lane.consume(parse_delivery(delivery()))
    ledger = lane.recorder.ledger
    segment_days = [path.name for path in (tmp_path / "fusion" / "physical" / "segments").iterdir() if path.is_dir()]
    assert segment_days == [DAY]
    assert ledger.expire(stores.now + timedelta(days=8), backfill_days=7) == 0  # 跨度 + 一个日之内：都留着
    removed = ledger.expire(stores.now + timedelta(days=9), backfill_days=7)
    assert removed >= 2  # 编号那一日、时段那一日
    assert not any((tmp_path / "fusion" / "physical" / "segments").iterdir())


def test_a_delivery_that_arrives_days_late_lands_on_the_day_it_happened(tmp_path: Path) -> None:
    lane, stores = wired(tmp_path)
    stores.now += timedelta(days=3)
    lane.consume(parse_delivery(delivery()))
    assert all(item["started_at"].startswith(DAY) for item in stores.judgements.list())


@pytest.mark.parametrize("crash_at", ["judgements.put_payload", "receipts.put", "coverage.record"])
def test_a_segment_interrupted_halfway_is_finished_not_written_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: str
) -> None:
    """判断的身份里有判断时刻：写到一半崩了，重做时照原来那条的时刻接着写完，不多出一条。"""

    lane, stores = wired(tmp_path)
    store, method = crash_at.split(".")

    def crash(*_: object) -> None:
        raise RuntimeError("进程崩溃")

    with monkeypatch.context() as patched:
        patched.setattr(getattr(stores, store), method, crash)
        with pytest.raises(RuntimeError):
            lane.consume(parse_delivery(delivery()))

    stores.now += timedelta(minutes=7)  # 重做发生在另一个时刻
    lane.consume(parse_delivery(delivery()))

    assert sorted(item["behavior"] for item in readable(stores)) == ["洗手", "跑步"]
    assert len(blanks(stores)) == 1
    assert len(stores.receipts.list()) == 3
    assert {item.disposition for item in lane.consume(parse_delivery(delivery()))} == {UnitDisposition.ALREADY_RECORDED}


def test_a_half_written_segment_that_upstream_then_corrects_is_replaced_not_left_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写到一半崩了、之后上游把这一段改了再发：从没写完过，算不上改判。那半截要清掉——留着它，没被回执覆盖的凭据
    会把归约的封口永远卡在它那一刻，这一支之后什么都落不了树。"""

    lane, stores = wired(tmp_path)

    def crash(*_: object) -> None:
        raise RuntimeError("进程崩溃")

    with monkeypatch.context() as patched:
        patched.setattr(stores.receipts, "put", crash)
        with pytest.raises(RuntimeError):
            lane.consume(parse_delivery(delivery()))

    corrected = edited(("segments", 0, "ended_at"), moment("09:13:30"))
    outcomes = lane.consume(parse_delivery(corrected))

    assert outcomes[0].disposition is UnitDisposition.RECORDED
    (wash,) = [item for item in readable(stores) if item["behavior"] == "洗手"]
    assert wash["last_observed_at"].startswith(f"{DAY}T09:13:30")
    # 没有留下没被回执覆盖的凭据
    covered = stores.coverage.covered_observation_ids(stores.now)
    stored = {item.observation_id for envelope in stores.observations.list() for item in envelope.batch.observations}
    assert stored <= covered
