"""归约里的词表一步：封口的判断落成 ``kind_token``（词表方案 2026-11-14 第 3 节）。

一条规则：判断上已经带着选词结果的（会话 lane 在 lane 里选好的），直接落；没带的（物理 lane：入口是同步的写盘流程，放不进
模型调用），在这里给它选——一条记录一次调用，看名字、概要、目标、步骤。没问出结论的（这一份被拦、输入超长、模型用不了）
换成「错误」放一边（``SET_ASIDE``）；答复重问之后仍不合格的选词器已经记成「待定」，照常落树。

lane 由组合根给：两条 lane 各是一整套（各自的归约、词表、行为树，见 ``runtime/behavior.py``）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from habitus.behavior.kinds.calls import CallFailure, KindModelOutputError, call_failure
from habitus.behavior.kinds.ids import Lane, not_event_token
from habitus.behavior.kinds.pick.picker import WordPicker
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.behavior.retry import RetryCause

#: 选词没问出结论的那几条换成「错误」时进堆的原因：内容安全拦下的不重试、健康面单列；输入超长是我们自己处理不了的
#: （材料上限与模型窗口没对上，裁定 94 错误分类）；模型整个用不了的放一边、换代前再选一次；别的是 bug，当场报、不白试。
#: 答复不合格（``INVALID``）不会走到这里：选词器把它记成「待定」。
SET_ASIDE: Mapping[CallFailure, RetryCause] = {
    CallFailure.INVALID: RetryCause.INVALID,
    CallFailure.REFUSED: RetryCause.REFUSED,
    CallFailure.TOO_LARGE: RetryCause.DETERMINISTIC,
    CallFailure.UNAVAILABLE: RetryCause.UNAVAILABLE,
    CallFailure.BUG: RetryCause.DETERMINISTIC,
}


@dataclass(frozen=True)
class KindStamps:
    """一批判断的选词结果，按它们落成的节点身份（``chain_digest``）给。"""

    tokens: Mapping[str, str]
    not_events: int
    #: 这一批里在这里选词的条数（判断上没带结果的），一条一次模型调用（答复不合格的重问在结构化客户端里，不另计）。
    model_calls: int
    signals: tuple[str, ...]
    #: 没有选词结果、要换成「错误」放一边的（按节点身份）与为什么（定换代前试不试）。
    set_aside: Mapping[str, CallFailure] = field(default_factory=dict)


class KindStamping:
    """``picker`` 是这条 lane 的选词器；会话 lane 的判断都带着结果，不给选词器（归约对它不调模型）。"""

    def __init__(self, picker: WordPicker | None, *, lane: Lane) -> None:
        self.picker = picker
        self.lane = Lane(lane)

    async def classify(self, records: Sequence[ReducibleJudgement], *, checkpoint: Callable[[], object]) -> KindStamps:
        tokens: dict[str, str] = {}
        set_aside: dict[str, CallFailure] = {}
        signals: list[str] = []
        calls = 0
        for record in records:
            if record.kind_token is not None:
                tokens[record.chain_digest] = record.kind_token
                continue
            material = record_material(record, max_steps=self._max_steps)
            if material is None or self.picker is None:
                set_aside[record.chain_digest] = CallFailure.BUG  # 没有可给选词看的内容，或这条 lane 不该在这里选
                continue
            calls += 1
            try:
                tokens[record.chain_digest] = await self.picker.pick_record(material)
            except KindModelOutputError as exc:
                failure = call_failure(exc)
                set_aside[record.chain_digest] = failure
                signals.append(f"kind_pick_{failure.value} {record.chain_digest[:12]}: {exc}")
            checkpoint()
        not_event = not_event_token(self.lane)
        return KindStamps(
            tokens=tokens,
            not_events=sum(1 for token in tokens.values() if token == not_event),
            model_calls=calls,
            signals=tuple(signals),
            set_aside=set_aside,
        )

    @property
    def _max_steps(self) -> int:
        return 0 if self.picker is None else self.picker.config.prompt_max_steps


def record_material(record: ReducibleJudgement, *, max_steps: int) -> str | None:
    """一条可读判断给选词看的材料：名字 + 概要 + 目标 + 步骤；没有可读名字（绕过融合守卫的坏数据）时给 ``None``。"""

    name = record.behavior
    if not isinstance(name, str) or not name.strip():
        return None
    parts = [name.strip()]
    if record.summary and record.summary.strip():
        parts.append(f"概要：{record.summary.strip()}")
    if record.goal and record.goal.strip():
        parts.append(f"目标：{record.goal.strip()}")
    steps = [semantics.strip() for semantics, _ in record.basis if semantics.strip()]
    if steps:
        shown = steps[:max_steps]
        more = f"（另 {len(steps) - len(shown)} 步）" if len(steps) > len(shown) else ""
        parts.append(f"步骤：{'／'.join(shown)}{more}")
    return "　｜".join(parts)


__all__ = ["SET_ASIDE", "KindStamping", "KindStamps", "record_material"]
