"""未封口读口：按归约自己的口径（存储里有、账本里没有）、自己的并链、自己的白天归类（提前到封口之前、按链缓存），读成此刻场景的行。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.changes import AddClass, ChangeReason, VersionRecord
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.reduction.ledger import BehaviorReductionEntry, BehaviorReductionLedger
from habitus.behavior.tree import BehaviorTree
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.model_client import ModelQuotaError, ModelTransportError
from habitus.runtime.unsealed import UnsealedFromJudgements
from tests.unit.behavior.kinds_fixtures import (
    NOW,
    NameMatchingProvider,
    ScriptedProvider,
    kind_components,
    named_classes,
)
from tests.unit.behavior.reduction_fixtures import at, judgement_record, record_id

PLAY, WASH = "s-k0001", "s-k0002"


def reader(
    tmp_path: Path, *, names: tuple[str, ...] = ("打球", "洗手"), provider: ScriptedProvider | None = None
) -> tuple[UnsealedFromJudgements, BehaviorJudgementStore, BehaviorReductionLedger, ScriptedProvider]:
    """词表第 1 版是 ``names``（打球 = s-k0001、洗手 = s-k0002）；假模型按原话与类名逐字相同归类，「挠头」答不是一件事。"""

    root = tmp_path / "behavior"
    judgements = BehaviorJudgementStore(root)
    ledger = BehaviorReductionLedger(root / "reduction")
    resolved = provider if provider is not None else NameMatchingProvider(not_events=frozenset({"挠头"}))
    stamping, _jobs, _store = kind_components(
        BehaviorTree(root / "tree"),
        ProcessLocalLockStore(),
        names=names,
        provider=resolved,
        config=BehaviorKindConfig(transient_retries=0),
    )
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
            uri="behavior://occurrences/2026/08/16/收拾球包--x.md",
            judgement_ids=(record_id("done"),),
            staged_at="2026-08-16T12:00:00Z",
            reduction_version="test",
        )
    )

    prepared(unsealed, at(3600))
    rows = unsealed.rows(since=at(-3600), until=at(3600))

    assert [(row.name, row.kind_token, row.started_at, row.last_observed_at, row.summary) for row in rows] == [
        ("打球", PLAY, at(0), at(600), "打球的摘要"),
        ("喝口水", None, at(700), at(720), "喝口水的摘要"),  # 归类答「都不是」（待定）：带原名进场景、标未归类，不猜
    ]


def test_continues_links_merge_into_one_row_with_the_chain_head_start_and_latest_sight(tmp_path) -> None:
    """并链按归约自己的算法：起点是链头，最后看到是视图里最晚的；不是两行。"""

    unsealed, judgements, _ledger, _provider = reader(tmp_path)
    judgements.put_payload(record("first", behavior="打球", start=0, end=600))
    judgements.put_payload(
        record("second", behavior="打球", start=1200, end=1800, relations=(("continues", record_id("first")),))
    )
    (row,) = unsealed.rows(since=at(-3600), until=at(3600))
    assert (row.name, row.started_at, row.last_observed_at) == ("打球", at(0), at(1800))


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
    with pytest.raises(TypeError):
        UnsealedFromJudgements(object(), BehaviorReductionLedger(tmp_path), unsealed.stamping)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        UnsealedFromJudgements(BehaviorJudgementStore(tmp_path), BehaviorReductionLedger(tmp_path), object())  # type: ignore[arg-type]


def test_each_chain_is_classified_once_and_again_only_when_it_grows(tmp_path) -> None:
    """按链缓存：同一条链下一拍不再问模型；续上了新的一段（链变了）才重归。封口的链从缓存里清掉。"""

    unsealed, judgements, _ledger, provider = reader(tmp_path)
    judgements.put_payload(record("first", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    prepared(unsealed, at(800))
    assert provider.calls == 1
    judgements.put_payload(
        record("second", behavior="打球", start=1200, end=1800, relations=(("continues", record_id("first")),))
    )
    prepared(unsealed, at(1900))
    assert provider.calls == 2
    (row,) = unsealed.rows(since=at(-3600), until=at(3600))
    assert row.kind_token == PLAY


def test_unclassified_until_prepared_and_not_events_stay_out(tmp_path) -> None:
    """还没归过的链标未归类；归为「非事件」的不进场景（与序列同一口径）。"""

    unsealed, judgements, _ledger, _provider = reader(tmp_path)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    judgements.put_payload(record("scratch", behavior="挠头", start=650, end=660))
    assert [(row.name, row.kind_token) for row in unsealed.rows(since=at(-100), until=at(3600))] == [
        ("打球", None),
        ("挠头", None),
    ]
    prepared(unsealed, at(3600))
    assert [(row.name, row.kind_token) for row in unsealed.rows(since=at(-100), until=at(3600))] == [("打球", PLAY)]


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


def test_no_model_call_before_the_vocabulary_exists(tmp_path) -> None:
    """词表还在第 0 版（第一次归约才写入预置清单）：没有类可归，不问模型。"""

    unsealed, judgements, _ledger, provider = reader(tmp_path, names=())
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    assert provider.calls == 0


def test_an_exhausted_answer_is_not_cached_and_is_retried_next_tick(tmp_path) -> None:
    """结构化输出重问用尽给的「待定」不是模型的判断：不缓存，下一拍再试（E12）。"""

    provider = ScriptedProvider([{"items": []}])
    unsealed, judgements, _ledger, _provider = reader(tmp_path, provider=provider)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    first = provider.calls
    assert [row.kind_token for row in unsealed.rows(since=at(-100), until=at(700))] == [None]
    prepared(unsealed, at(800))
    assert provider.calls > first


def test_a_quota_error_leaves_the_tick_running(tmp_path) -> None:
    """配额用完不让这一拍的判断丢掉：留着未归类，下一拍再试（E12）。"""

    provider = ScriptedProvider([ModelQuotaError("quota exhausted")])
    unsealed, judgements, _ledger, _provider = reader(tmp_path, provider=provider)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    assert [row.kind_token for row in unsealed.rows(since=at(-100), until=at(700))] == [None]


def test_a_new_vocabulary_version_reclassifies_the_open_chains(tmp_path) -> None:
    """词表拆改（版本变了）之后，缓存里的编号可能已停用：按新版本重归（E12）。"""

    unsealed, judgements, _ledger, provider = reader(tmp_path)
    judgements.put_payload(record("play", behavior="打球", start=0, end=600))
    prepared(unsealed, at(700))
    assert provider.calls == 1
    store = unsealed.stamping.store
    store.append(
        VersionRecord(2, NOW, ChangeReason.NIGHTLY, (AddClass(named_classes(("打球", "洗手", "跑步"))[2]),)),
        expected_version=1,
    )
    prepared(unsealed, at(800))
    assert provider.calls == 2
