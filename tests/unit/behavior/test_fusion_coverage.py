"""融合覆盖索引：按日分区、窗口读取、整块过期、与回执同步写。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from behavior.fusion.coverage import BehaviorCoverageIndex
from behavior.fusion.errors import BehaviorFusionError
from behavior.fusion.prompt import FUSION_PROMPT_VERSION
from behavior.fusion.receipt import build_fusion_receipt
from behavior.observation import BehaviorObservation, BehaviorObservationConfig

SUBJECT = "家庭成员A"
TZ8 = timezone(timedelta(hours=8))
BASE = datetime(2026, 8, 14, 20, 0, tzinfo=TZ8)
OBSERVATION_CONFIG = BehaviorObservationConfig()


def observation(offset: int, semantics: str) -> BehaviorObservation:
    at = BASE + timedelta(seconds=offset)
    return BehaviorObservation.create(
        observer_id="home-a/hall",
        occurred_at=at,
        available_at=at + timedelta(seconds=1),
        modality="vision",
        semantics=semantics,
        participants=[SUBJECT],
        knowledge_state="observed",
        confidence=0.9,
        evidence_refs=[f"cam:{offset}"],
        config=OBSERVATION_CONFIG,
    )


def receipt(observations: tuple[BehaviorObservation, ...], *, judged_at: datetime):
    return build_fusion_receipt(
        (),
        observations,
        source_refs=("a" * 64,),
        prompt_version=FUSION_PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
        judged_at=judged_at,
    )


def test_recorded_observations_are_covered_inside_the_window(tmp_path: Path) -> None:
    index = BehaviorCoverageIndex(tmp_path, window_days=7)
    first, second = observation(0, "人走到水池边"), observation(4, "人在洗手")
    judged = BASE + timedelta(minutes=5)
    index.record(receipt((first, second), judged_at=judged))
    covered = index.covered_observation_ids(judged + timedelta(days=1))
    assert covered == {first.observation_id, second.observation_id}
    # 记录按 judged_at 的 UTC 日期分目录——过期时整目录删除，不逐条扫描。
    day = judged.astimezone(timezone.utc).date().isoformat()
    assert (tmp_path / "fusion" / "coverage" / day).is_dir()


def test_recording_the_same_receipt_twice_is_idempotent(tmp_path: Path) -> None:
    index = BehaviorCoverageIndex(tmp_path)
    stored = receipt((observation(0, "人走到水池边"),), judged_at=BASE)
    index.record(stored)
    index.record(stored)  # 崩溃重试会再走一次同一步
    assert len(index.covered_observation_ids(BASE)) == 1


def test_records_outside_the_window_are_neither_read_nor_kept(tmp_path: Path) -> None:
    index = BehaviorCoverageIndex(tmp_path, window_days=7)
    old = observation(0, "旧观测")
    fresh = observation(4, "新观测")
    index.record(receipt((old,), judged_at=BASE - timedelta(days=10)))
    index.record(receipt((fresh,), judged_at=BASE))
    assert index.covered_observation_ids(BASE) == {fresh.observation_id}
    assert index.expire(BASE) == 1
    remaining = sorted(path.name for path in (tmp_path / "fusion" / "coverage").iterdir())
    assert remaining == [BASE.astimezone(timezone.utc).date().isoformat()]
    # 窗口内的不动，再过期一次是零。
    assert index.expire(BASE) == 0


def test_an_empty_index_covers_nothing_and_expires_nothing(tmp_path: Path) -> None:
    index = BehaviorCoverageIndex(tmp_path)
    assert index.covered_observation_ids(BASE) == frozenset()
    assert index.expire(BASE) == 0


def test_the_window_must_be_positive_and_now_must_be_aware(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BehaviorCoverageIndex(tmp_path, window_days=0)
    index = BehaviorCoverageIndex(tmp_path)
    with pytest.raises(TypeError):
        index.covered_observation_ids(datetime(2026, 8, 14))
    with pytest.raises(BehaviorFusionError):
        index._path(BASE, "not-a-receipt-id")
