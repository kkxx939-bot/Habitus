"""白天归类：每条 occurrence 封口就归到同 lane 的一个在用类；归不进打「待定」，不是一件事打「非事件」。

只能归到已有的类，白天不建类（设计 二）。结构不对只重问没答好的那几条，
重问轮数用尽仍答不好的一律当「待定」（宁可待定，不要勉强归类），留信号。模型经注入的 ``KindModelCaller`` 调用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from habitus.behavior.kinds.calls import KindModelCaller, KindModelOutputError
from habitus.behavior.kinds.classify.prompt import (
    CLASSIFY_SYSTEM_PROMPT,
    NOT_EVENT,
    OUTSIDE,
    classify_schema,
    render_user_message,
)
from habitus.behavior.kinds.classify.request import ClassifyRequest
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import Lane, not_event_token, pending_token
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, Vocabulary


class Outcome(str, Enum):
    CLASS = "class"
    PENDING = "pending"
    NOT_EVENT = "not_event"


@dataclass(frozen=True)
class KindVerdict:
    """一条的归类结果：写进 ``kind_token`` 的值、结局、模型的理由、「都不是」时的提议名。"""

    token: str
    outcome: Outcome
    reason: str
    proposed: str | None = None


@dataclass(frozen=True)
class ClassifyResult:
    verdicts: Mapping[str, KindVerdict]
    model_calls: int
    signals: tuple[str, ...]


@dataclass(frozen=True)
class _Batch:
    lane: Lane
    classes: Mapping[str, BehaviorClass]
    items: Mapping[str, ClassifyRequest]

    def narrowed(self, keys: set[str]) -> _Batch:
        return _Batch(self.lane, self.classes, {rid: req for rid, req in self.items.items() if req.key in keys})


class DaytimeClassifier:
    def __init__(self, caller: KindModelCaller) -> None:
        if not isinstance(caller, KindModelCaller):
            raise TypeError("caller must be KindModelCaller")
        self.caller = caller
        self.config: BehaviorKindConfig = caller.config

    async def classify(self, requests: Sequence[ClassifyRequest], vocabulary: Vocabulary) -> ClassifyResult:
        keys = [request.key for request in requests]
        if len(set(keys)) != len(keys):
            raise BehaviorKindError("classify request keys must be unique")
        verdicts: dict[str, KindVerdict] = {}
        signals: list[str] = []
        calls = 0
        for lane in Lane:
            same_lane = [request for request in requests if request.lane is lane]
            if not same_lane:
                continue
            active = vocabulary.active(lane)
            # 这条 lane 还没有类（词表从空开始，裁定 29）也照样问：只剩「都不是」与「不是一件事」两个选项——
            # 判出不是一件事的小动作、单条命令，再给真事起个提议名，待定池里才能按同一件事聚组长出类
            if len(active) > self.config.lane_candidate_cap:
                # TODO(BHV-KINDS-RECALL)：超过上限后应接回向量召回（只召回不判定：按判据文本取最近的若干类
                # 交模型）。这条闸目前只用来报警，不截断清单。
                signals.append(f"kind_lane_over_candidate_cap {lane.value}: {len(active)} active classes")
            classes = {f"C{i}": item for i, item in enumerate(active, start=1)}
            for start in range(0, len(same_lane), self.config.batch_size):
                chunk = same_lane[start : start + self.config.batch_size]
                batch = _Batch(lane, classes, {f"R{i}": r for i, r in enumerate(chunk, start=1)})
                answered, used = await self._judge(batch, signals)
                verdicts.update(answered)
                calls += used
        return ClassifyResult(verdicts, calls, tuple(signals))

    async def _judge(self, batch: _Batch, signals: list[str]) -> tuple[dict[str, KindVerdict], int]:
        verdicts: dict[str, KindVerdict] = {}
        remaining = batch
        calls = 0
        for _ in range(self.config.validation_rounds + 1):
            if not remaining.items:
                break
            calls += 1
            try:
                items = await self._call(remaining)
            except KindModelOutputError as exc:
                signals.append(f"kind_classify_rejected {sorted(r.key for r in remaining.items.values())}: {exc}")
                continue
            accepted = _accepted(items, remaining)
            verdicts.update(accepted)
            missing = {r.key for r in remaining.items.values()} - set(accepted)
            if missing:
                signals.append(f"kind_classify_incomplete {sorted(missing)}")
            remaining = remaining.narrowed(missing)
        for request in remaining.items.values():
            signals.append(f"kind_classify_exhausted {request.key!r} kept pending")
            verdicts[request.key] = _pending(request, EXHAUSTED)
        return verdicts, calls

    async def _call(self, batch: _Batch) -> list[dict[str, Any]]:
        return await self.caller.call(
            system=CLASSIFY_SYSTEM_PROMPT,
            user=render_user_message(batch.lane, batch.classes, batch.items, max_steps=self.config.prompt_max_steps),
            schema=classify_schema(tuple(batch.items), tuple(batch.classes)),
            name="behavior_kind_classify",
            validator=lambda parsed: _validated(parsed, batch),
        )


def _validated(parsed: object, batch: _Batch) -> list[dict[str, Any]]:
    """只拒整体形状错；单条答坏（形状不对、编号不认识、同一条答两遍、选项不在列表、理由为空）只丢那一条，
    交给逐条重问——不让一条坏答案把整批作废。"""

    if not isinstance(parsed, Mapping) or set(parsed) != {"items"} or not isinstance(parsed["items"], list):
        raise BehaviorKindError("classify output must be an object with an `items` list")
    answered: dict[str, dict[str, Any]] = {}
    repeated: set[str] = set()
    for item in parsed["items"]:
        if not isinstance(item, Mapping) or set(item) != {"record", "reason", "choice", "proposed"}:
            continue
        record = item["record"]
        if record not in batch.items:
            continue
        if record in answered:
            repeated.add(record)
            continue
        if item["choice"] not in batch.classes and item["choice"] not in (OUTSIDE, NOT_EVENT):
            continue
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            continue
        answered[record] = dict(item)
    return [item for record, item in answered.items() if record not in repeated]


def _accepted(items: Sequence[Mapping[str, Any]], batch: _Batch) -> dict[str, KindVerdict]:
    accepted: dict[str, KindVerdict] = {}
    for item in items:
        request = batch.items[item["record"]]
        reason = item["reason"].strip()
        choice = item["choice"]
        if choice == NOT_EVENT:
            accepted[request.key] = KindVerdict(not_event_token(batch.lane), Outcome.NOT_EVENT, reason)
        elif choice == OUTSIDE:
            proposed = item["proposed"]
            if isinstance(proposed, str) and proposed.strip():
                accepted[request.key] = KindVerdict(
                    pending_token(batch.lane), Outcome.PENDING, reason, proposed.strip()
                )
        else:
            accepted[request.key] = KindVerdict(str(batch.classes[choice].id), Outcome.CLASS, reason)
    return accepted


#: 结构化输出重问用尽时给的「待定」的原因：这不是模型的判断，调用方据此区分"模型说都不是"与"这次没问成"。
EXHAUSTED = "归类调用重问用尽"


def _pending(request: ClassifyRequest, reason: str) -> KindVerdict:
    return KindVerdict(pending_token(request.lane), Outcome.PENDING, reason, request.content.name)


__all__ = ["EXHAUSTED", "ClassifyResult", "DaytimeClassifier", "KindVerdict", "Outcome"]
