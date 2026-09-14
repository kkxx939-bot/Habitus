"""关联编排的现场：真的行为树与规律级树 + 可拨时钟 + 脚本化关联器。

自成一体，不 import 别的测试文件：归组那份现场在第二刀会跟日情景树一起删。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.prediction import builder, source
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.model import PredictionTree
from habitus.scene.association.model import AssociationAssembly, AssociationDraft, AssociationInput
from habitus.scene.association.refresher import AssociationRefreshConfig, AssociationRefresher
from habitus.scene.backlog import AssociationTask, CauseFacts, backlog, slot_order
from habitus.scene.regularity.store import RegularityTree
from tests.unit.behavior.tree_payloads import gap_payload, occurrence_payload

CST = timezone(timedelta(hours=8))
FRIDAY = date(2026, 9, 4)
NEXT_FRIDAY = date(2026, 9, 11)


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def occurrence_uri(name: str, moment: datetime) -> str:
    return str(
        BehaviorURI.from_address(
            BehaviorAddress(kind=BehaviorKind.OCCURRENCE, occurred_on=moment.date(), name=name, started_at=moment)
        )
    )


def prediction_config(**overrides: object) -> PredictionTreeConfig:
    values: dict[str, object] = dict(
        slot_minutes=15,
        decay_half_life_days=3_650.0,
        recent_half_life_days=14.0,
        recurrence_half_life_days=3_650.0,
        pool_half_width=2,
        shrink_slot_to_pool=0.001,
        shrink_pool_to_weekday=0.001,
        shrink_weekday_to_all_day=0.001,
        laplace_epsilon=0.001,
        transition_window_seconds=7_200.0,
        shrink_edge=0.001,
        recurrence_window_days=90.0,
        rebuild_interval_seconds=86_400.0,
        published_generations=3,
    )
    values.update(overrides)
    return PredictionTreeConfig(**values)  # type: ignore[arg-type]


class ScriptedAssociator:
    """按候选给答复的关联器；记下每次拿到的输入。"""

    version = "scripted_association_v1+schema000000000000"

    def __init__(self, answer: Callable[[AssociationInput], AssociationAssembly] | None = None) -> None:
        self.payloads: list[AssociationInput] = []
        self.explode: Exception | None = None
        self.answer = answer or _plain

    async def associate(self, payload: AssociationInput) -> AssociationAssembly:
        self.payloads.append(payload)
        if self.explode is not None:
            raise self.explode
        return self.answer(payload)


def _plain(payload: AssociationInput) -> AssociationAssembly:
    """默认答复：每个目标一条，引用自己那一行，归进第一种已有情境或开一种新的。

    还会**用掉摆给它的第一条前提、并为下一次铺一条**——不这么做的话，`materialize` 里编号还原成
    URI、两类边的构造与去重那几段在整个测试套件里一次都跑不到。
    """

    drafts = []
    for index, target in enumerate(payload.targets):
        situation_no = 1 if payload.situations else None
        causes = (payload.citable and _first_cause(payload, target)) or ()
        drafts.append(
            AssociationDraft(
                occurrence_no=target,
                context=f"{payload.day.isoformat()} 第 {index + 1} 次{payload.kind_token}",
                cites=tuple(sorted({target, *causes})),
                causes=causes,
                situation_no=situation_no,
                new_situation=None if situation_no else f"{payload.kind_token}的第一种情形",
                consumed=(1,) if payload.pending and index == 0 else (),
                left=((f"为下一次{payload.kind_token}做好了准备", payload.kind_token),) if index == 0 else (),
            )
        )
    return AssociationAssembly(drafts=tuple(drafts))


def _first_cause(payload: AssociationInput, target: int) -> tuple[int, ...]:
    """挑一条真的更早的依据当前因：优先取更早那天的前因候选，其次取当天更早的一行。"""

    if payload.causes:
        return (len(payload.occurrences) + 1,)
    earlier = [row.no for row in payload.occurrences if row.no < target]
    return (earlier[-1],) if earlier else ()


class Ground:
    """两棵真树 + 一个可拨的时钟。"""

    def __init__(self, tmp_path: Path, *, now: datetime | None = None, **config: object) -> None:
        self.root = Path(tmp_path)
        self.behavior_tree = BehaviorTree(self.root / "behavior")
        self.regularity_tree = RegularityTree(self.root / "regularity")
        self.regularity_tree.initialize()
        self.now = now or datetime(2026, 9, 30, tzinfo=UTC)
        self.associator = ScriptedAssociator()
        self.lock_store = ProcessLocalLockStore()
        self.refresher = AssociationRefresher(
            behavior_tree=self.behavior_tree,
            regularity_tree=self.regularity_tree,
            associator=self.associator,
            progress_root=self.root / "progress",
            lock_store=self.lock_store,
            config=AssociationRefreshConfig(**config),  # type: ignore[arg-type]
            clock=lambda: self.now,
        )

    def record(self, day: date, name: str, hour: int, minute: int = 0, *, kind: str | None = None) -> str:
        started = at(day, hour, minute)
        writer = BehaviorDocumentWriter(
            self.behavior_tree, ProcessLocalLockStore(), clock=lambda: started + timedelta(hours=3)
        )
        writer.publish(
            BehaviorKind.OCCURRENCE,
            occurrence_payload(
                occurred_on=day,
                name=name,
                kind_token=kind or name,
                started_at=started,
                last_observed_at=started + timedelta(minutes=20),
                onset_available_at=started + timedelta(seconds=2),
                basis=(),
                goal=None,
                summary=f"{name}的一句话",
            ),
        )
        return occurrence_uri(name, started)

    def gap(self, day: date, start_hour: int, end_hour: int) -> None:
        writer = BehaviorDocumentWriter(self.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(day, 23, 0))
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(occurred_on=day, started_at=at(day, start_hour), ended_at=at(day, end_hour)),
        )

    def tree(self) -> PredictionTree:
        snapshot = source.read(self.behavior_tree)
        latest = snapshot.latest_day
        assert latest is not None
        return builder.build(
            snapshot, config=prediction_config(), reference=latest, built_at=datetime(2026, 12, 31, tzinfo=UTC)
        )

    def tasks(self, *, per_candidate: int = 10, limit: int = 50) -> tuple[AssociationTask, ...]:
        return backlog(
            self.tree(),
            self.refresher.associated_days,
            per_candidate=per_candidate,
            limit=limit,
            blocked=self.refresher.progress,
        )

    def run(self, *, force: bool = False, only: str | None = None):  # type: ignore[no-untyped-def]
        import asyncio

        tasks = self.tasks()
        if only is not None:
            tasks = tuple(task for task in tasks if task.kind_token == only)
        return asyncio.run(self.refresher.refresh(tasks, causes=CauseFacts(self.tree()), force=force))


__all__ = [
    "CST",
    "FRIDAY",
    "NEXT_FRIDAY",
    "Ground",
    "ScriptedAssociator",
    "at",
    "occurrence_uri",
    "prediction_config",
    "slot_order",
]
