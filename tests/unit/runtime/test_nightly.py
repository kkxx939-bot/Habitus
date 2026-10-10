"""第 N 晚在组合根的串法（第 8 步）：行为树读一次，预测树与语义树用同一份序列。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.prediction.store import PredictionTreeStore
from habitus.runtime.nightly import Nightly
from habitus.runtime.prediction import PredictionRebuilder
from habitus.series import reader
from tests.unit.prediction.prediction_fixtures import config
from tests.unit.runtime.test_scene_night import NOW, TONIGHT, nightly, publish_tonight, site_with_three_bedtimes


def sealed_through(last):  # type: ignore[no-untyped-def]
    """归约说"到 ``last`` 为止都封口了"。"""

    return lambda: tuple(last - timedelta(days=offset) for offset in range(30))


def steady() -> tuple[int, bool]:
    return (1, False)


def assembled(tmp_path, *, clock=lambda: NOW, closed_days=None, vocabulary=steady):  # type: ignore[no-untyped-def]
    site, hits = site_with_three_bedtimes(tmp_path)
    scene, _provider = nightly(tmp_path, hits, site)
    prediction = PredictionRebuilder(
        site.behavior_tree,
        PredictionTreeStore(tmp_path / "prediction", retained_generations=3),
        config=config(),
        zone=ZoneInfo("Asia/Shanghai"),
        clock=clock,
    )
    closed = closed_days if closed_days is not None else sealed_through(TONIGHT)
    return site, Nightly(
        behavior_tree=site.behavior_tree,
        prediction=prediction,
        scene=scene,
        closed_days=closed,
        vocabulary=vocabulary,
        clock=clock,
    )


def test_one_read_feeds_both_trees_for_the_same_night(tmp_path, monkeypatch) -> None:
    site, run = assembled(tmp_path)
    publish_tonight(site)
    reads: list[object] = []
    original = reader.read_series

    def counting(tree, *, cutoff):  # type: ignore[no-untyped-def]
        reads.append(cutoff)
        return original(tree, cutoff=cutoff)

    monkeypatch.setattr("habitus.runtime.nightly.read_series", counting)
    outcome = asyncio.run(run.run_once())
    assert reads == [NOW.astimezone(ZoneInfo("Asia/Shanghai")).date()]  # 行为树只读这一次
    assert outcome.published is not None and outcome.scene is not None
    tree = run.prediction.store.load()
    assert tree is not None and tree.reference_day == outcome.series.last_day
    assert outcome.scene.night == outcome.series.cutoff == outcome.night
    assert TONIGHT in outcome.scene.mapped_days


def test_nothing_runs_on_an_empty_behaviour_tree(tmp_path) -> None:
    _site, run = assembled(tmp_path)
    outcome = asyncio.run(run.run_once())
    assert outcome.published is None and outcome.scene is None


def test_both_runs_must_read_the_same_tree(tmp_path) -> None:
    site, run = assembled(tmp_path)
    _other_site, other = assembled(tmp_path / "other")
    with pytest.raises(ValueError, match="same behaviour tree"):
        Nightly(behavior_tree=site.behavior_tree, prediction=run.prediction, scene=other.scene, closed_days=tuple, vocabulary=steady)


def test_the_night_stops_at_the_last_sealed_day(tmp_path) -> None:
    """昨天的链还没归约完（归约说只封口到前天）：截止日退到昨天，昨天这一晚不算（E1）——不把半天当成完整的一天映射、盖章。"""

    site, run = assembled(tmp_path, closed_days=sealed_through(TONIGHT - timedelta(days=1)))
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.night == TONIGHT and outcome.series.cutoff == TONIGHT
    assert not outcome.series.on(TONIGHT)
    assert outcome.scene is None or TONIGHT not in outcome.scene.mapped_days


def test_nothing_runs_before_any_day_is_sealed(tmp_path) -> None:
    site, run = assembled(tmp_path, closed_days=tuple)
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.published is None and outcome.scene is None


def test_nothing_runs_while_the_vocabulary_is_migrating(tmp_path) -> None:
    """词表迁移做到一半（树上的行改了一部分、新版本还没写）：这一晚不发布、不跑语义树，说出原因（E3）。"""

    site, run = assembled(tmp_path, vocabulary=lambda: (2, True))
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.published is None and outcome.scene is None and outcome.skipped == "词表迁移还没做完"


def test_nothing_runs_when_the_vocabulary_changes_while_reading(tmp_path) -> None:
    states = iter([(1, False), (2, False)])
    site, run = assembled(tmp_path, vocabulary=lambda: next(states))
    publish_tonight(site)
    outcome = asyncio.run(run.run_once())
    assert outcome.published is None and outcome.skipped == "读序列时词表变了"
