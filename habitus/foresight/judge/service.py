"""调模型：到了该说时的"像不像、先说哪个、一句话"。问两遍（第二遍顺序反过来），逐项对答案；传输层瞬态错有界重试；失败不出本模块。

两遍对不上的：像不像 → 按"像"留着、顺序按算法（时窗先到的在前）；话 → 第一遍的。两遍都说不像的才去掉。
模型不可用 → ``failed``，调用方按算法顺序、模板措辞递。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from habitus.foresight.judge.package import Candidate, Package
from habitus.foresight.judge.prompt import (
    JUDGE_VERSION,
    Answer,
    assemble,
    build_request,
    judge_json_schema,
)
from habitus.model_client import ChatRequest, ModelUnreachable, StructuredChatClient
from habitus.model_client.asking import Failed, ask


@dataclass(frozen=True)
class Verdict:
    """一个候选的最终答案。``like`` 为 None = 两遍不一致（留着）；``evidence`` 是依据那次的记录 URI。"""

    like: bool | None
    text: str | None
    evidence: tuple[str, ...]

    @property
    def kept(self) -> bool:
        return self.like is not False


@dataclass(frozen=True)
class Judgement:
    verdicts: Mapping[str, Verdict]
    #: 说的先后（预期身份）；两遍不一致时是算法顺序（时窗先到的在前）。
    order: tuple[str, ...]
    signals: tuple[str, ...] = ()
    model: str = ""
    failed: bool = False
    version: str = JUDGE_VERSION
    calls: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Judge:
    def __init__(self, client: StructuredChatClient, *, retries: int = 1, delay_seconds: float = 5.0) -> None:
        if retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be a non-negative number")
        self.client = client
        self.retries = retries
        self.delay_seconds = float(delay_seconds)

    async def judge(self, package: Package) -> Judgement:
        forward = package.candidates
        first = await self._ask(package, forward)
        if isinstance(first, str):
            return self._failed(package, first, 1)
        second = await self._ask(package, tuple(reversed(forward)))
        if isinstance(second, str):
            return self._failed(package, second, 2)
        (answer_a, model_a, notes_a), (answer_b, _model_b, notes_b) = first, second
        verdicts: dict[str, Verdict] = {}
        signals = [*notes_a, *notes_b]
        consistent_order = True
        for item in forward:
            like_a, basis_a, text_a = answer_a.verdicts[item.number]
            like_b, basis_b, text_b = answer_b.verdicts[item.number]
            like: bool | None = like_a if like_a == like_b else None
            if like is None:
                signals.append(f"inconsistent: 候选 {item.number} 像不像两遍不一样")
                consistent_order = False
            evidence = tuple(uri for uri in (item.uri_of(basis_a), item.uri_of(basis_b)) if uri)
            verdicts[item.expectation_id] = Verdict(
                like=like,
                text=(text_a or text_b or None) if like is not False else None,
                evidence=tuple(dict.fromkeys(evidence)),
            )
        order_a = tuple(found.expectation_id for n in answer_a.order if (found := package.by_number(n)) is not None)
        order_b = tuple(found.expectation_id for n in answer_b.order if (found := package.by_number(n)) is not None)
        if consistent_order and order_a == order_b:
            order = order_a
        else:
            if order_a != order_b:
                signals.append("inconsistent: 先后两遍不一样，按算法顺序")
            order = tuple(
                item.expectation_id
                for item in sorted(forward, key=lambda candidate: candidate.window[0])
                if verdicts[item.expectation_id].kept
            )
        return Judgement(verdicts=verdicts, order=order, signals=tuple(signals), model=model_a, calls=2)

    async def _ask(self, package: Package, shown: tuple[Candidate, ...]) -> tuple[Answer, str, tuple[str, ...]] | str:
        request = build_request(package, shown)

        def validator(parsed: object) -> Answer:
            return assemble(parsed, shown)

        return await _ask(
            self.client,
            request,
            schema=judge_json_schema([item.number for item in shown]),
            validator=validator,
            retries=self.retries,
            delay_seconds=self.delay_seconds,
        )

    @staticmethod
    def _failed(package: Package, signal: str, calls: int) -> Judgement:
        return Judgement(
            verdicts={},
            order=tuple(item.expectation_id for item in package.candidates),
            signals=(signal,),
            failed=True,
            calls=calls,
        )


async def _ask[T](
    client: StructuredChatClient,
    request: ChatRequest,
    *,
    schema: Mapping[str, Any],
    validator: Callable[[object], T],
    retries: int,
    delay_seconds: float,
    name: str = "foresight_judge",
) -> tuple[T, str, tuple[str, ...]] | str:
    """调模型用各触点共用的那一段（``model_client.asking.ask``）：传输层瞬态错按退避重试，答复不合格由结构层带着错误重问；
    都没成返回信号文本。判决在白天那一拍：模型整个用不了（``ModelUnreachable``）也只是这一次没判成，调用方按算法顺序、模板
    措辞递；模型调不通在哪儿都不停下什么，换代里也只是调模型的那一步走退路（裁定 94 第 8 条）。"""

    try:
        answer = await ask(
            client, request, schema=schema, name=name, validator=validator, retries=retries, delay_seconds=delay_seconds
        )
    except ModelUnreachable as exc:
        return f"model unreachable: {exc}"
    if isinstance(answer, Failed):
        return answer.signal
    return cast(T, answer.value), answer.model, answer.notes


__all__ = ["Judge", "Judgement", "Verdict"]
