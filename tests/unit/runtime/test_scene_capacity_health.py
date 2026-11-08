"""语义树关系表的容量读数交给健康面：稳态留满保留代数不告警，真的多出一晚才报 degraded（v0.12 跑数疑似 bug 清单第 1 条）。

只断言"参与者报的读数交给真实 ``RuntimeHealthService._lifecycle_capacity_check`` 后是否 degraded"，以及告警内容带
``scene.`` 与 lane 名；不假设读数叫什么、count/limit 取什么值。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NoReturn

import pytest

from habitus.config.lifecycle import LifecycleConfig
from habitus.infrastructure.lifecycle import CapacitySnapshot
from habitus.infrastructure.store.contracts import PathLock
from habitus.runtime import RuntimeHealthCheck, RuntimeHealthService, build_runtime
from habitus.runtime.lifecycle_scene import SceneParticipant
from habitus.scene.relations.state import LaneState
from habitus.scene.relations.store import RelationStore
from tests.helpers import BASE_TIME
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies
from tests.lock_store import ProcessLocalLockStore

LANE = "session"
FIRST_NIGHT = date(2026, 4, 15)


class _Context:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def assert_owned(self) -> None:
        return None

    def fence_token(self) -> NoReturn:
        raise AssertionError("not used")


class _Health:
    """一份真实 Runtime 的容量快照存储 + 真实健康面判断。"""

    def __init__(self, tmp_path: Path, share: float) -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path / "runtime"),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        store = runtime.components.workflow.lifecycle_worker.capacity
        assert store is not None
        self.store = store
        self.service = RuntimeHealthService(runtime.components, capacity_warning_share=share)

    def judge(self, participant: SceneParticipant) -> RuntimeHealthCheck:
        report = asyncio.run(participant.maintain(_Context(datetime(2026, 4, 30, tzinfo=UTC))))
        self.store.publish(CapacitySnapshot("scene", BASE_TIME, report.readings))
        return self.service._lifecycle_capacity_check()


def _relations(tmp_path: Path) -> RelationStore:
    return RelationStore(tmp_path / "scene", fallback_generations=LifecycleConfig().fallback_generations)


def _write(store: RelationStore, night: date) -> None:
    store.write(LaneState(lane=LANE, night=night, family=0, relations=()), ())


def _retained() -> int:
    """``RelationStore._prune`` 固定留下的晚数：最晚一晚 + ``fallback_generations`` 晚。"""

    return 1 + LifecycleConfig().fallback_generations


@pytest.mark.parametrize("share", [0.8, 0.99])
def test_steady_state_retention_is_not_a_capacity_warning(tmp_path: Path, share: float) -> None:
    """连写 6 晚：每一晚之后关系表都只留满保留代数（_prune 正常），健康面不该报 degraded。"""

    relations = _relations(tmp_path)
    participant = SceneParticipant(relations)
    health = _Health(tmp_path, share)
    degraded: list[tuple[date, str]] = []
    for offset in range(6):
        night = FIRST_NIGHT + timedelta(days=offset)
        _write(relations, night)
        assert len(relations.nights(LANE)) == min(offset + 1, _retained())  # 前提：稳态，_prune 正常删
        check = health.judge(participant)
        if check.status.value == "degraded":
            degraded.append((night, check.detail))
    assert degraded == []


_Overflow = Callable[[RelationStore, pytest.MonkeyPatch], None]


def _overflow_by_failed_discard(relations: RelationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """这一次 _prune 删不掉：让该实例的 ``_discard`` 成空操作再写一晚。真实运行里删除失败（被占用、权限、
    中途被杀）同样会让状态文件留下，``nights()`` 照实多出一晚。"""

    monkeypatch.setattr(relations, "_discard", lambda path: None)
    _write(relations, FIRST_NIGHT + timedelta(days=_retained()))


def _overflow_by_restored_file(relations: RelationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """把被删掉的最早一晚的原样状态文件复制回目录：文件名与内容都是那一晚 ``write`` 当时真实写出的合法状态
    （``read`` 能解码），等于有人多放回了一晚；``nights()`` 照实多出一晚。"""

    oldest = relations.directory / LANE / f"{FIRST_NIGHT.isoformat()}.json"
    payload = oldest.read_bytes()
    _write(relations, FIRST_NIGHT + timedelta(days=_retained()))
    assert not oldest.exists()  # 正常 _prune 已删
    oldest.write_bytes(payload)
    relations.read(LANE, FIRST_NIGHT)  # 放回来的是一份合法状态


@pytest.mark.parametrize("share", [0.8, 0.01, 0.99])
@pytest.mark.parametrize(
    "overflow", [_overflow_by_failed_discard, _overflow_by_restored_file], ids=["discard", "restore"]
)
def test_retention_overflow_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, share: float, overflow: _Overflow
) -> None:
    """真的超出保留数（_prune 没删掉 / 多放了一晚）时，健康面报 degraded，告警内容带 scene 参与者与 lane 名。"""

    relations = _relations(tmp_path)
    participant = SceneParticipant(relations)
    health = _Health(tmp_path, share)
    for offset in range(_retained()):
        _write(relations, FIRST_NIGHT + timedelta(days=offset))
    overflow(relations, monkeypatch)
    assert len(relations.nights(LANE)) == _retained() + 1  # 前提：盘上真的多出一晚
    check = health.judge(participant)
    assert check.status.value == "degraded"
    assert not check.critical
    assert "scene." in check.detail
    assert LANE in check.detail
