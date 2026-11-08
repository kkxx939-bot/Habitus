"""最终 Context 的召回升温确认：只让最终模型可见的主记忆升温，失败或过期目标从 Context 中剔除。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Protocol

from habitus.memory.document import MemoryDocument
from habitus.memory.recall import MemoryRecallLifecycle, MemoryRecallLifecycleError, MemoryRecallTarget
from habitus.memory.retrieval.assembler import MemoryContextAssembler, MemoryContextAssembly
from habitus.memory.retrieval.model import (
    MemoryMatchedMemory,
    MemorySearchDegradation,
    MemorySearchDegradationStage,
)
from habitus.memory.uri import MemoryURI


class MemoryContextUseReceipt(Protocol):
    @property
    def documents(self) -> tuple[MemoryDocument, ...]: ...

    @property
    def rejected_uris(self) -> tuple[MemoryURI, ...]: ...


class MemoryColdProbeExpander(Protocol):
    """展开活动基线，并在最终 Context 前完成 L2 恢复与使用确认。"""

    def expand_for_probe(self, document: MemoryDocument) -> MemoryDocument: ...

    async def record_context_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime | None = None,
    ) -> MemoryContextUseReceipt: ...


def append_degradation(
    degradations: list[MemorySearchDegradation],
    stage: MemorySearchDegradationStage,
    error: BaseException,
) -> None:
    """每个降级阶段最多记录一次，保留首个错误类型。"""

    if any(item.stage is stage for item in degradations):
        return
    degradations.append(MemorySearchDegradation(stage, type(error).__name__))


class FinalContextAcknowledger(Protocol):
    """对最终 Context 中的直接命中执行使用确认，返回剔除未确认目标后的重新组装结果。"""

    async def acknowledge(
        self,
        assembly: MemoryContextAssembly,
        *,
        targets: dict[MemoryURI, MemoryRecallTarget],
        used_at: datetime,
        degradations: list[MemorySearchDegradation],
    ) -> MemoryContextAssembly: ...


def _retained_targets(
    assembly: MemoryContextAssembly,
    targets: dict[MemoryURI, MemoryRecallTarget],
) -> tuple[MemoryRecallTarget, ...]:
    return tuple(targets[memory.uri] for memory in assembly.memories if memory.uri in targets)


class RecallLifecycleContextAcknowledger:
    """没有冷探针展开器时直接写召回生命周期；无法同步恢复的 COLD_2 从最终 Context 排除。"""

    def __init__(self, *, recall_lifecycle: MemoryRecallLifecycle, assembler: MemoryContextAssembler) -> None:
        self.recall_lifecycle = recall_lifecycle
        self.assembler = assembler

    async def acknowledge(
        self,
        assembly: MemoryContextAssembly,
        *,
        targets: dict[MemoryURI, MemoryRecallTarget],
        used_at: datetime,
        degradations: list[MemorySearchDegradation],
    ) -> MemoryContextAssembly:
        retained_targets = _retained_targets(assembly, targets)
        if not retained_targets:
            return assembly
        cold = {
            memory.uri
            for memory in assembly.memories
            if memory.hit.lifecycle_temperature is not None and memory.hit.lifecycle_temperature.value == "cold_2"
        }
        safe_targets = tuple(target for target in retained_targets if target.uri not in cold)
        try:
            self.recall_lifecycle.record_use(safe_targets, used_at=used_at)
        except MemoryRecallLifecycleError as exc:
            append_degradation(degradations, MemorySearchDegradationStage.RECALL_LIFECYCLE, exc)
            # 普通详细 L2 仍可安全返回；只把无法同步恢复的 COLD_2 排除。
        accepted = {target.uri for target in safe_targets}
        if accepted != {target.uri for target in retained_targets}:
            append_degradation(
                degradations,
                MemorySearchDegradationStage.RECALL_LIFECYCLE,
                MemoryRecallLifecycleError("COLD_2 final Context requires synchronous L2 restore"),
            )
        safe_memories = tuple(memory for memory in assembly.memories if memory.uri in accepted)
        return self.assembler.assemble(
            safe_memories,
            summary_fallbacks=assembly.summary_fallbacks,
        )


class ColdProbeContextAcknowledger:
    """由冷探针展开器完成 L2 恢复与使用确认；被拒绝或未返回的目标从最终 Context 剔除。"""

    def __init__(self, *, expander: MemoryColdProbeExpander, assembler: MemoryContextAssembler) -> None:
        self.expander = expander
        self.assembler = assembler

    async def acknowledge(
        self,
        assembly: MemoryContextAssembly,
        *,
        targets: dict[MemoryURI, MemoryRecallTarget],
        used_at: datetime,
        degradations: list[MemorySearchDegradation],
    ) -> MemoryContextAssembly:
        retained_targets = _retained_targets(assembly, targets)
        if not retained_targets:
            return assembly
        try:
            receipt = await self.expander.record_context_use(retained_targets, used_at=used_at)
            documents = tuple(receipt.documents)
            rejected = set(receipt.rejected_uris)
        except Exception as exc:
            append_degradation(degradations, MemorySearchDegradationStage.RECALL_LIFECYCLE, exc)
            return self.assembler.assemble((), summary_fallbacks=assembly.summary_fallbacks)
        if rejected:
            append_degradation(
                degradations,
                MemorySearchDegradationStage.RECALL_LIFECYCLE,
                MemoryRecallLifecycleError("one or more final Context memories lost their lifecycle fence"),
            )
        by_uri = {MemoryURI.from_address(document.address): document for document in documents}
        retained: list[MemoryMatchedMemory] = []
        for memory in assembly.memories:
            if memory.uri in rejected:
                continue
            document = by_uri.get(memory.uri)
            if document is None:
                continue
            retained.append(replace(memory, document=document))
        return self.assembler.assemble(
            tuple(retained),
            summary_fallbacks=assembly.summary_fallbacks,
        )


def final_context_acknowledger(
    *,
    recall_lifecycle: MemoryRecallLifecycle,
    cold_probe_expander: MemoryColdProbeExpander | None,
    assembler: MemoryContextAssembler,
) -> FinalContextAcknowledger:
    """有冷探针展开器时由它确认，否则直接写召回生命周期。"""

    if cold_probe_expander is None:
        return RecallLifecycleContextAcknowledger(recall_lifecycle=recall_lifecycle, assembler=assembler)
    return ColdProbeContextAcknowledger(expander=cold_probe_expander, assembler=assembler)


__all__ = [
    "ColdProbeContextAcknowledger",
    "FinalContextAcknowledger",
    "MemoryColdProbeExpander",
    "MemoryContextUseReceipt",
    "RecallLifecycleContextAcknowledger",
    "append_degradation",
    "final_context_acknowledger",
]
