"""调模型：到了该说时的"像不像、先说哪个、一句话"。问两遍（第二遍顺序反过来），逐项对答案；传输层瞬态错有界重试；失败不出本模块。

两遍对不上的：像不像 → 按"像"留着、顺序按算法（时窗先到的在前）；话 → 第一遍的。两遍都说不像的才去掉。
模型不可用 → ``failed``，调用方按算法顺序、模板措辞递。
"""

from __future__ import annotations

import asyncio
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
from habitus.foundation.text import clean_line
from habitus.model_client import ChatRequest, ModelClientError, ModelTransportError, StructuredChatClient


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
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, int | float) or delay_seconds < 0:
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
        order_a = tuple(package.by_number(n).expectation_id for n in answer_a.order if package.by_number(n) is not None)  # type: ignore[union-attr]
        order_b = tuple(package.by_number(n).expectation_id for n in answer_b.order if package.by_number(n) is not None)  # type: ignore[union-attr]
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
    """与语义树各触点同一段调法（``scene.llm.ask``；预测层不许 import 那个模块，照样子写一份）：传输层瞬态错按退避重试，
    答复不合格由结构层带着错误重问；都没成返回信号文本。"""

    for attempt in range(retries + 1):
        try:
            response = await client.complete_json_async(request, schema=schema, name=name, validator=validator)
        except ModelTransportError as exc:
            if attempt >= retries:
                return f"model: transport failed after {retries + 1} attempts: {clean_line(str(exc))[:120]}"
            await asyncio.sleep(delay_seconds * (attempt + 1))
            continue
        except ModelClientError as exc:
            cause = exc.__cause__
            detail = f"；{clean_line(str(cause))[:220]}" if cause is not None else ""
            return f"model: {type(exc).__name__}: {clean_line(str(exc))[:120]}{detail}"
        notes: list[str] = []
        if response.validation_attempts > 1:
            notes.append(f"structured: answered on attempt {response.validation_attempts}")
        if response.parse_mode != "strict":
            notes.append(f"structured: json parsed via {response.parse_mode}")
        return cast(T, response.value), response.response.model, tuple(notes)
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["Judge", "Judgement", "Verdict"]
