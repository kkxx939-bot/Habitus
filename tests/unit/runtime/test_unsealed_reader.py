"""未封口读口：按归约自己的口径（存储里有、账本里没有）、判断上带着的选词结果，没带的用归约自己的选词器（提前到封口之前、
按链缓存），读成此刻场景的行（词表方案 2026-11-14）。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.reduction.ledger import BehaviorReductionEntry, BehaviorReductionLedger
from habitus.model_client import ModelQuotaError, ModelTransportError
from habitus.runtime.unsealed import UnsealedFromJudgements
from tests.unit.behavior.kinds_fixtures import NameMatchingProvider, ScriptedProvider, kind_components
from tests.unit.behavior.reduction_fixtures import at, judgement_record, record_id

PLAY, WASH = "s-k0001", "s-k0002"


def reader(
    tmp_path: Path, *, names: tuple[str, ...] = ("打球", "洗手"), provider: ScriptedProvider | None = None
) -> tuple[UnsealedFromJudgements, BehaviorJudgementStore, BehaviorReductionLedger, ScriptedProvider]:
    """测试词表是 ``names``（打球 = s-k0001、洗手 = s-k0002）；假模型按名字与词名逐字相同选词，没有就答 ``missing``。"""

    root = tmp_path / "behavior"
    judgements = BehaviorJudgementStore(root)
    ledger = BehaviorReductionLedger(root / "reduction")
    resolved = provider if provider is not None else NameMatchingProvider()
    stamping = kind_components(names=names, provider=resolved, config=BehaviorKindConfig())
    return UnsealedFromJudgements(judgements, ledger, stamping), judgements, ledger, resolved


def prepared(unsealed: UnsealedFromJudgements, until: datetime) -> None:
    asyncio.run(unsealed.prepare(until=until))


def record(seed: str, *, behavior: str | None, start: int, end: int, **overrides):
    return judgement_record(
        seed,
        behavior=behavior,
        started_at=at(start),
        last_observed_at=at(end),
        evidence_ready_at=at(end + 10),
        observation_ids=(f"obs-{seed}",),
        source_refs=(f"src-{seed}",),
        **overrides,
    )


def test_pending_chains_become_rows_and_consumed_ones_do_not(tmp_path) -> None:
    unsealed, judgements, ledger, _provider = reader(tmp_path)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    judgements.put_payload(record("water", behavior="喝口水", start=700, end=720))
    judgements.put_payload(record("done", behavior="收拾球包", start=-900, end=-600))
    ledger.append(
        BehaviorReductionEntry(
            chain_digest="a" * 64,
            kind="occurrence",
            uri="habitus://occurrences/session/2026/08/16/收拾球包--x.md",
            judgement_ids=(record_id("done"),),
            staged_at="2026-08-16T12:00:00Z",
            reduction_version="test",
        )
    )

    prepared(unsealed, at(3600))
    rows = unsealed.rows(since=at(-3600), until=at(3600))

    assert [(row.name, row.kind_token, row.started_at, row.last_observed_at, row.summary) for row in rows] == [
        ("打球", PLAY, at(0), at(600), "打球的摘要"),
        ("喝口水", None, at(700), at(720), "喝口水的摘要"),  # 选词答「表外」：带原名进场景、标未归类，不猜
    ]


def test_unreadable_judgements_are_gap_rows_and_the_window_is_by_overlap(tmp_path) -> None:
    unsealed, judgements, _ledger, _provider = reader(tmp_path)
    judgements.put_payload(record("blur", behavior=None, start=0, end=300))
    judgements.put_payload(record("early", behavior="洗手", start=-7200, end=-7000))
    judgements.put_payload(record("late", behavior="洗手", start=7200, end=7300))
    rows = unsealed.rows(since=at(-100), until=at(100))
    assert [(row.name, row.kind_token, row.started_at, row.last_observed_at) for row in rows] == [
        (None, None, at(0), at(300))
    ]
    assert not rows[0].readable
    # 交集判据：开始在 until 之前、最后看到在 since 之后。
    assert [row.name for row in unsealed.rows(since=at(-7100), until=at(7250))] == ["洗手", None, "洗手"]


def test_a_corrupt_record_is_skipped_rather_than_failing_the_scene(tmp_path) -> None:
    """解析不了的记录归约每轮都会作为隔离报出；这里不报第二遍，也不让此刻场景整个读不出来。"""

    unsealed, judgements, _ledger, _provider = reader(tmp_path)
    judgements.put_payload(record("ok", behavior="洗手", start=0, end=60))
    broken = record("broken", behavior="洗手", start=0, end=60)
    broken["last_observed_at"] = "not-a-time"
    judgements.put_payload(broken)
    assert [row.name for row in unsealed.rows(since=at(-100), until=at(100))] == ["洗手"]


def test_the_reader_insists_on_aware_bounds_and_the_right_stores(tmp_path) -> None:
    unsealed, _judgements, _ledger, _provider = reader(tmp_path)
    with pytest.raises(ValueError, match="since"):
        unsealed.rows(since=datetime(2026, 8, 16, 19), until=at(0))
    with pytest.raises(ValueError, match="until"):
        prepared(unsealed, datetime(2026, 8, 16, 19))


def test_each_judgement_is_classified_once(tmp_path) -> None:
    """按节点身份缓存：同一条下一拍不再问模型；新来一条才问。封口的从缓存里清掉。"""

    unsealed, judgements, _ledger, provider = reader(tmp_path)
    judgements.put_payload(record("first", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    prepared(unsealed, at(800))
    assert provider.calls == 1
    judgements.put_payload(record("second", behavior="打球", start=1200, end=1800))
    prepared(unsealed, at(1900))
    assert provider.calls == 2
    rows = unsealed.rows(since=at(-3600), until=at(3600))
    assert [(row.started_at, row.kind_token) for row in rows] == [(at(0), PLAY), (at(1200), PLAY)]


def test_unclassified_until_prepared_and_carried_picks_are_used_as_they_are(tmp_path) -> None:
    """没带选词结果、还没选过的链标未归类；判断上带着的直接用、不问模型：「非事件」「续话」不进场景（与序列同一口径）。"""

    unsealed, judgements, _ledger, provider = reader(tmp_path)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    judgements.put_payload(record("wash", behavior="洗手", start=610, end=640, kind_token=WASH))
    judgements.put_payload(record("scratch", behavior="挠头", start=650, end=660, kind_token="s-非事件"))
    judgements.put_payload(record("again", behavior="再问", start=670, end=680, kind_token="s-续话"))
    assert [(row.name, row.kind_token) for row in unsealed.rows(since=at(-100), until=at(3600))] == [
        ("打球", None),
        ("洗手", WASH),
    ]
    prepared(unsealed, at(3600))
    assert provider.calls == 1
    assert [(row.name, row.kind_token) for row in unsealed.rows(since=at(-100), until=at(3600))] == [
        ("打球", PLAY),
        ("洗手", WASH),
    ]


def test_a_model_outage_leaves_rows_unclassified_and_retries_next_tick(tmp_path) -> None:
    """模型这一拍不可用：不让此刻场景失败，链留着下一拍再归。"""

    provider = ScriptedProvider([ModelTransportError("down")])
    unsealed, judgements, _ledger, _provider = reader(tmp_path, provider=provider)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    assert [row.kind_token for row in unsealed.rows(since=at(-100), until=at(700))] == [None]
    first = provider.calls
    prepared(unsealed, at(800))
    assert provider.calls > first  # 没缓存失败的结果：下一拍又试


def test_an_exhausted_answer_is_kept_as_pending_like_the_reduction_does(tmp_path) -> None:
    """答复重问用尽：选词器记「待定」（与归约同一口径），按未归类进场景；它就是这一条的结果，下一拍不再问。"""

    provider = ScriptedProvider([{"items": []}])
    unsealed, judgements, _ledger, _provider = reader(tmp_path, provider=provider)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    first = provider.calls
    assert [row.kind_token for row in unsealed.rows(since=at(-100), until=at(700))] == [None]
    prepared(unsealed, at(800))
    assert provider.calls == first


def test_a_quota_error_leaves_the_tick_running(tmp_path) -> None:
    """配额用完不让这一拍的判断丢掉：留着未归类，下一拍再试（E12）。"""

    provider = ScriptedProvider([ModelQuotaError("quota exhausted")])
    unsealed, judgements, _ledger, _provider = reader(tmp_path, provider=provider)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    assert [row.kind_token for row in unsealed.rows(since=at(-100), until=at(700))] == [None]


def test_an_unusable_model_never_raises_for_the_day_expectations_either(tmp_path) -> None:
    """实时化方案第 2 节"模型调不通不停一天"：换代里的日子预期与预测拍读同一座桥、同一口径——模型整个用不了时这一次不归类、
    不抛（原来日子预期那一份 ``in_line`` 原样抛出、整晚停下）。"""

    for error in (ModelQuotaError("quota exhausted"), ModelTransportError("down")):
        unsealed, judgements, _ledger, _provider = reader(tmp_path / type(error).__name__, provider=ScriptedProvider([error]))
        judgements.put_payload(record("play", behavior="打球", start=0, end=600))
        prepared(unsealed, at(700))
        (row,) = unsealed.rows(since=at(0), until=at(700))
        assert row.name == "打球" and row.kind_token is None


def test_an_unobserved_gap_is_not_an_unreadable_row(tmp_path) -> None:
    """「未观测」不是看不到（裁定 68 第 2 条、94 第 3 条）：会话一支写下、还没归约的没在看的时段不进未封口的空白行——不然
    时窗末那一拍会把它当「没读懂」、记「说不准」不开口。"""

    unsealed, judgements, _ledger, _provider = reader(tmp_path)
    unwatched = record("lid", behavior=None, start=0, end=300)
    unwatched["status_basis"] = "unobserved"
    judgements.put_payload(unwatched)
    judgements.put_payload(record("blur", behavior=None, start=400, end=500))
    rows = unsealed.rows(since=at(-100), until=at(600))
    assert [(row.name, row.started_at) for row in rows] == [(None, at(400))]
