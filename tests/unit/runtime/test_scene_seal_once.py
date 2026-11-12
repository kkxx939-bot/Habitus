"""一代只封一次口（裁定 97 第 3 条"按并入之前的概念集封口 → 再并入"；阶段 3 补测：并入之后、完成标记之前被杀，接着做时按
并入之后的概念集又封了一次，新概念提前一代进情境）。

- 封口一做完就落这一代的封口记录，在并入待生效的概念之前；
- 同一代接着做（被杀后、必须成的步骤没成重试）：有记录就不再封，报告用记下的结果；
- 封口没做成也记下：接着做也不再封；
- 记录坏了明确报错、不再封（不当"没有"）；随完成标记一起清。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from habitus.runtime.scene_night_seal import SealReport
from habitus.runtime.scene_seal_files import SealFileError, SealFiles
from habitus.series.reader import read_series
from tests.unit.runtime.test_mapping_by_slot import world
from tests.unit.runtime.test_model_queue import queue_for
from tests.unit.runtime.test_scene_night import NOW, TONIGHT


class Killed(BaseException):
    """模拟进程在并入待生效概念那一刻被杀（不是 ``Exception``：尽力而为的退路接不住它）。"""


def sealed_world(tmp_path):  # type: ignore[no-untyped-def]
    """一份语义树：今天的两条记录都映射好了、明天换代时今天可以盖章；接上封口记录。"""

    site, scene, _provider, _bed, _meal = world(tmp_path)
    tomorrow = TONIGHT + timedelta(days=1)
    asyncio.run(queue_for(tmp_path, scene, site, clock=lambda: NOW, sealed=lambda: tomorrow).drain())
    scene.seal_files = SealFiles(tmp_path / "scene" / "generation" / "seals", artifact_root=tmp_path)
    return site, scene, read_series(site.behavior_tree, cutoff=tomorrow)


def test_the_seal_record_is_on_disk_before_the_pending_concepts_are_merged(tmp_path) -> None:
    """并入那一刻被杀：这一代的封口已经落盘（盖了章、记了结果）。记录写在并入之后，这里会变红。"""

    _site, scene, series = sealed_world(tmp_path)
    seen: list[object] = []

    def killed(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        seen.append(scene.seal_files.read(series.cutoff))
        raise Killed

    scene.merge_pending = killed  # type: ignore[method-assign]
    with pytest.raises(Killed):
        asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1), failures=[]))
    assert len(seen) == 1 and seen[0] is not None
    report, failed = seen[0]  # type: ignore[misc]
    assert not failed and report.sealed == (TONIGHT,) and TONIGHT in scene.hits.days_done()


def test_resuming_the_same_generation_does_not_seal_again_and_reports_the_recorded_seal(tmp_path) -> None:
    """同一代第二次做（上一次并入之后被杀）：不再封，报告里的封口读数是第一次的。去掉"有记录就不再封"，这里会变红。"""

    _site, scene, series = sealed_world(tmp_path)
    first = asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1), failures=[]))
    assert (first.sealed_days, first.seal_rewritten) == (1, 2)

    def no_seal(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("a generation that already sealed must not seal again")

    scene.sealing.seal = no_seal  # type: ignore[method-assign]
    failures: list[tuple[str, BaseException]] = []
    again = asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1, minutes=15), failures=failures))
    assert failures == []
    assert (again.sealed_days, again.seal_rewritten, again.unsealed_days) == (1, 2, 0)
    assert any("已经封过口" in signal for signal in again.signals)


def test_a_failed_seal_is_recorded_and_the_resumed_generation_does_not_seal_either(tmp_path) -> None:
    """封口这一步抛了（尽力而为）：记成没做成；接着做也不封——不中断时这一代就没封口。"""

    _site, scene, series = sealed_world(tmp_path)
    original = scene.sealing.seal

    def broken(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("seal broke")

    scene.sealing.seal = broken  # type: ignore[method-assign]
    failures: list[tuple[str, BaseException]] = []
    first = asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1), failures=failures))
    assert [step for step, _exc in failures] == ["seal"] and first.sealed_days == 0
    recorded = scene.seal_files.read(series.cutoff)
    assert recorded is not None and recorded[1] is True

    scene.sealing.seal = original  # type: ignore[method-assign]
    again = asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1, minutes=15), failures=[]))
    assert again.sealed_days == 0 and TONIGHT not in scene.hits.days_done()
    assert any("已经封过口（没做成）" in signal for signal in again.signals)


def test_a_corrupt_seal_record_is_reported_and_does_not_lead_to_sealing_again(tmp_path) -> None:
    """记录坏了：明确报错（尽力而为记进 failures、严格跑法抛出），不当"没有"再封一次。"""

    _site, scene, series = sealed_world(tmp_path)
    path = scene.seal_files.path(series.cutoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    failures: list[tuple[str, BaseException]] = []
    report = asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1), failures=failures))
    assert [step for step, _exc in failures] == ["seal"] and isinstance(failures[0][1], SealFileError)
    assert report.sealed_days == 0 and TONIGHT not in scene.hits.days_done()
    with pytest.raises(SealFileError):
        asyncio.run(scene.generate(series=series, now=NOW + timedelta(days=1)))


def test_seal_records_round_trip_and_are_pruned_by_day(tmp_path) -> None:
    files = SealFiles(tmp_path / "seals", artifact_root=tmp_path)
    report = SealReport(sealed=(TONIGHT,), rewritten=2, unready_days=1, unready_records=3, readers=4)
    files.write(TONIGHT, report, failed=False)
    files.write(TONIGHT - timedelta(days=10), SealReport(), failed=True)
    assert files.read(TONIGHT) == (report, False)
    assert files.read(TONIGHT - timedelta(days=10)) == (SealReport(), True)
    assert files.read(TONIGHT + timedelta(days=1)) is None
    assert files.prune(TONIGHT, 5) == 1
    assert files.read(TONIGHT - timedelta(days=10)) is None and files.read(TONIGHT) is not None
    files.path(TONIGHT).write_text('{"schema": "generation_seal_v1"}', encoding="utf-8")
    with pytest.raises(SealFileError):
        files.read(TONIGHT)
