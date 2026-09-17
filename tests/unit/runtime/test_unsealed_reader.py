"""未封口读口：按归约自己的口径（存储里有、账本里没有）、自己的并链、自己的词表，读成此刻场景的行。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.model import BehaviorKindRegistry
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.reduction.ledger import BehaviorReductionEntry, BehaviorReductionLedger
from habitus.runtime.unsealed import UnsealedFromJudgements
from tests.unit.behavior.reduction_fixtures import at, judgement_record, record_id


def reader(tmp_path: Path, *, kinds: dict[str, tuple[str, ...]] | None = None) -> tuple[UnsealedFromJudgements, BehaviorJudgementStore, BehaviorReductionLedger]:
    root = tmp_path / "behavior"
    judgements = BehaviorJudgementStore(root)
    ledger = BehaviorReductionLedger(root / "reduction")
    kind_store = BehaviorKindStore(root / "tree")
    if kinds:
        kind_store.replace(BehaviorKindRegistry(kinds), expected_revision=0, timestamp=datetime(2026, 8, 16, tzinfo=UTC))
    return UnsealedFromJudgements(judgements, ledger, kind_store), judgements, ledger


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
    unsealed, judgements, ledger = reader(tmp_path, kinds={"打球": ()})
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

    rows = unsealed.rows(since=at(-3600), until=at(3600))

    assert [(row.name, row.kind_token, row.started_at, row.last_observed_at, row.summary) for row in rows] == [
        ("打球", "打球", at(0), at(600), "打球的摘要"),
        ("喝口水", None, at(700), at(720), "喝口水的摘要"),  # 词表里没有：带原名进场景，不猜 kind
    ]


def test_continues_links_merge_into_one_row_with_the_chain_head_start_and_latest_sight(tmp_path) -> None:
    """并链按归约自己的算法：起点是链头，最后看到是视图里最晚的；不是两行。"""

    unsealed, judgements, _ledger = reader(tmp_path, kinds={"打球": ()})
    judgements.put_payload(record("first", behavior="打球", start=0, end=600))
    judgements.put_payload(
        record("second", behavior="打球", start=1200, end=1800, relations=(("continues", record_id("first")),))
    )
    (row,) = unsealed.rows(since=at(-3600), until=at(3600))
    assert (row.name, row.started_at, row.last_observed_at) == ("打球", at(0), at(1800))


def test_unreadable_judgements_are_gap_rows_and_the_window_is_by_overlap(tmp_path) -> None:
    unsealed, judgements, _ledger = reader(tmp_path)
    judgements.put_payload(record("blur", behavior=None, start=0, end=300))
    judgements.put_payload(record("early", behavior="洗手", start=-7200, end=-7000))
    judgements.put_payload(record("late", behavior="洗手", start=7200, end=7300))
    rows = unsealed.rows(since=at(-100), until=at(100))
    assert [(row.name, row.kind_token, row.started_at, row.last_observed_at) for row in rows] == [(None, None, at(0), at(300))]
    assert not rows[0].readable
    # 交集判据：开始在 until 之前、最后看到在 since 之后。
    assert [row.name for row in unsealed.rows(since=at(-7100), until=at(7250))] == ["洗手", None, "洗手"]


def test_a_corrupt_record_is_skipped_rather_than_failing_the_scene(tmp_path) -> None:
    """解析不了的记录归约每轮都会作为隔离报出；这里不报第二遍，也不让此刻场景整个读不出来。"""

    unsealed, judgements, _ledger = reader(tmp_path)
    judgements.put_payload(record("ok", behavior="洗手", start=0, end=60))
    broken = record("broken", behavior="洗手", start=0, end=60)
    broken["last_observed_at"] = "not-a-time"
    judgements.put_payload(broken)
    assert [row.name for row in unsealed.rows(since=at(-100), until=at(100))] == ["洗手"]


def test_the_reader_insists_on_aware_bounds_and_the_right_stores(tmp_path) -> None:
    unsealed, _judgements, _ledger = reader(tmp_path)
    with pytest.raises(ValueError, match="since"):
        unsealed.rows(since=datetime(2026, 8, 16, 19), until=at(0))
    with pytest.raises(TypeError):
        UnsealedFromJudgements(object(), BehaviorReductionLedger(tmp_path), BehaviorKindStore(tmp_path))  # type: ignore[arg-type]
