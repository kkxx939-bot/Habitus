"""一支的下游管线在组合根的组装（裁定 102，裁定 42 两条独立管线）：``build_runtime`` 对行为侧的每一支调一次
``assemble_pipeline``——这一支的完成标记、trace、预测根租约、预测层、未封口的桥、预测拍，再交 ``runtime.scene.assemble_branch``
装语义树、换代与模型队列。代码一套，按支各装一份、各用各的根；这一支发出的观测事件都带上 lane。"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import date, datetime

from habitus.conditions import DayTypeCalendar
from habitus.config import HabitusConfig
from habitus.foresight import Evidence
from habitus.foundation.observability import Observer
from habitus.foundation.tracing import TraceSink
from habitus.infrastructure.store.contracts import LockStore
from habitus.model_client import StructuredChatClient
from habitus.runtime.behavior import BehaviorRuntimeComponents, LaneObserver, rereader_of
from habitus.runtime.foresight import (
    ForesightRuntimeComponents,
    build_foresight_components,
    build_foresight_days,
    build_foresight_tick,
    build_reminded_lookup,
    foresight_ledger_lease,
    unsealed_bridge,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import ReminderOutlet
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_marks import GenerationMarks
from habitus.runtime.generation_trace import SpanBackend
from habitus.runtime.pipeline import ScenePipeline
from habitus.runtime.scene import assemble_branch, generation_marks, generation_traces, scene_branch_root
from habitus.runtime.unsealed import UnsealedFromJudgements


def assemble_pipeline(
    config: HabitusConfig,
    branch: BehaviorRuntimeComponents,
    *,
    unwatched: Callable[[datetime, datetime], object] | None,
    calendar: DayTypeCalendar | None,
    evidence: Evidence | None,
    ledger: ForesightLedgerStore | None,
    outlet: ReminderOutlet | None,
    structured_chat: StructuredChatClient,
    lock_store: LockStore,
    observer: Observer,
    span_backend: SpanBackend,
) -> ScenePipeline | None:
    """一支的下游管线：这一支的完成标记、trace、预测根租约、预测层（``branch_foresight``），再交 ``assemble_branch`` 装语义树、
    换代与模型队列。这一支发出的观测事件都带上 lane。没开语义树是 None（没有换代就没有这一支的下游）。"""

    if not config.scene.enabled:
        return None
    lane = branch.lane.value
    observer = LaneObserver(observer, branch.lane)
    root = scene_branch_root(config, lane)
    # 换代的完成标记与本地 trace：换代写标记，这一支的结算读它（裁定 94 第 9 条）；trace 换代与预测拍都写
    marks, traces = generation_marks(root), generation_traces(root, span_backend)
    # 这一支的预测根租约：预测 worker 每槽替这一支持一次，在它里面先看一眼换代、再做预测拍（没开预测层时只看换代）
    lease = foresight_ledger_lease(config, lock_store, lane)
    foresight, unsealed, tick = (
        (None, None, None)
        if evidence is None or ledger is None
        else branch_foresight(
            config,
            branch,
            evidence=evidence,
            ledger=ledger,
            outlet=outlet,
            calendar=calendar,
            marks=marks,
            structured_chat=structured_chat,
            observer=observer,
            trace=traces.sink,
            unwatched=unwatched,
            lease=lease,
        )
    )
    return assemble_branch(
        config,
        branch=branch,
        calendar=calendar,
        structured_chat=structured_chat,
        foresight=foresight,
        marks=marks,
        traces=traces,
        ledger_lease=lease,
        lock_store=lock_store,
        observer=observer,
        unsealed=unsealed,
        tick=tick,
    )


def branch_foresight(
    config: HabitusConfig,
    branch: BehaviorRuntimeComponents,
    *,
    evidence: Evidence,
    ledger: ForesightLedgerStore,
    outlet: ReminderOutlet | None,
    calendar: DayTypeCalendar | None,
    marks: GenerationMarks,
    structured_chat: StructuredChatClient,
    observer: Observer,
    trace: Callable[[date], TraceSink],
    unwatched: Callable[[datetime, datetime], object] | None,
    lease: Lease,
) -> tuple[ForesightRuntimeComponents, UnsealedFromJudgements, ForesightTick]:
    """一支的预测层、未封口的桥与预测拍（含当天的日子预期）；这一支的账本接到这一支归约封口时盖 reminded 的口上。"""
    built = build_foresight_components(
        config,
        lane=branch.lane.value,
        ledger=ledger,
        behavior_tree=branch.tree,
        calendar=calendar,
        # "那天定稿了"只有这一支的归约说了算（链都落树、封口视界已过那天的本地结束），结算按它来。
        closed_days=branch.reduction_runner.closed_days,
        evidence=evidence,
        observer=observer,
        kind_store=branch.kind_store,
        outlet=outlet,
        finished=marks.finished,
    )
    assert built is not None  # 有 evidence 就是开了预测层
    # 未封口的判断经那座桥（判断存储 + 消费账本 + 归约的选词器）读：预测拍用它；换代里的日子预期读只读缓存的那一份
    # （换代里没有模型调用；会话一支的判断都带着选词结果，物理一支的由模型队列先选好）
    unsealed = unsealed_bridge(branch)
    tick = build_foresight_tick(
        built,
        config=config,
        unsealed=unsealed,
        evidence=evidence,
        structured_chat=structured_chat,
        vocabulary_version=lambda: branch.kind_store.read().revision,  # 词表版本记在预期上
        observer=observer,
        reread=rereader_of(branch),
        trace=trace,
        unwatched=unwatched,
        lease=lease,
        vocabulary_of=_counted_version(marks),
    )
    # 当天的日子预期：换代必须成的一步，与预测拍读同一座未封口的桥（只读缓存）
    days = build_foresight_days(
        built,
        config=config,
        unsealed=unsealed.cache_only(),
        evidence=evidence,
        vocabulary_version=lambda: branch.kind_store.read().revision,
        observer=observer,
    )
    # 第 0 步：这一支的归约封口时查这一支的预测层账本盖 reminded（归约先装、预测层后装，所以在这里接端口）
    branch.reduction_runner.attach_reminder_lookup(build_reminded_lookup(built))
    return dataclasses.replace(built, days=days), unsealed, tick


def _counted_version(marks: GenerationMarks) -> Callable[[date], int | None]:
    """某一天那一代数词时的词表版本（这一支自己的完成标记；没有标记是 None）：旧表下的日子预期要它与此刻的词表版本相同才建。"""

    def of(night: date) -> int | None:
        mark = marks.read(night)
        return None if mark is None else mark.vocabulary_version

    return of


__all__ = ["assemble_pipeline", "branch_foresight"]
