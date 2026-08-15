"""行为融合评测的用例协议。

## 为什么不复用 ``BenchmarkDataset``

那个协议是"会话 + 问答"的形状（``messages`` / ``questions``），服务的是长期记忆检索。行为融合的
输入是**观测片段序列**、产出是**判断**，两者没有可共享的字段。强行合并只会让两边都变形。

## 只放确定性可判的期望

用例里的 ``expect`` 一律是不需要语义评判就能核对的东西：某条关系出现了没有、对照组有没有瞎标、
多人场景丢没丢人、该判为读不懂的帧判了没有。像"目标判得准不准""分解合不合理"这类需要 Judge 的，
等确定性这一层跑起来、看清哪里真的不稳定之后再加——现在加，连该判什么都说不准。

## 回归用例与探查用例要分开

``probing=True`` 的用例是**用来检验假说的**，不是已经确认该成立的行为。它的失败不代表实现坏了，
所以不计入通过判定，只在报告里单独列出。混在一起会让评测长期红着，而红着的评测没人会认真看。

## 多段用例表达跨窗口

``segments`` 有多段时按顺序融合，后一段拿前一段已落盘的判断作上下文——这正是 ``continues`` /
``supersedes`` / ``results_from`` 唯一能产出的场合（结果往往不是立刻的）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FUSION_CASE_SCHEMA = "behavior_fusion_case_v1"

_RELATION_KINDS = frozenset({"continues", "supersedes", "concurrent_with", "results_from"})


class FusionBenchmarkError(ValueError):
    """评测用例不满足协议。"""


@dataclass(frozen=True)
class FusionFragment:
    """一条观测片段；只带融合层真正会看到的字段。"""

    offset_seconds: int
    semantics: str
    participants: tuple[str, ...]
    knowledge_state: str = "observed"
    modality: str = "vision"

    def __post_init__(self) -> None:
        if not self.semantics.strip():
            raise FusionBenchmarkError("fragment semantics must be non-empty")
        if not self.participants:
            raise FusionBenchmarkError("fragment participants must be non-empty")


@dataclass(frozen=True)
class FusionExpectation:
    """一个用例的确定性期望。字段一律可选——只写这个用例真正想考的那一条。"""

    # 期望出现的跨段关系：(来源段序号, 目标段序号, 关系类型)，段序号从 1 开始。
    relations: tuple[tuple[int, int, str], ...] = ()
    # 不该出现的关系类型；对照组靠它抓"瞎标"。
    forbidden_relations: tuple[str, ...] = ()
    # 期望某一段的 subjects 至少包含这些人。
    subjects_include: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    # 期望某一段产出的判断条数落在这个区间——粗判粒度，不判语义。
    judgement_count: Mapping[int, tuple[int, int]] = field(default_factory=dict)
    # 期望被判为"读不懂"的片段编号（按段内 1..N）。
    unreadable_fragments: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    # 期望**不属于主体**的片段编号。与"读不懂"是两回事：那些帧读得懂，只是做的人不是我们跟踪的。
    # 用条数去测旁人有没有被吸收是测不准的——条数对了不等于那一帧没被塞进主体的判断。
    out_of_scope_fragments: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for _, _, kind in self.relations:
            if kind not in _RELATION_KINDS:
                raise FusionBenchmarkError(f"unknown relation kind: {kind}")
        for kind in self.forbidden_relations:
            if kind not in _RELATION_KINDS:
                raise FusionBenchmarkError(f"unknown relation kind: {kind}")

    @property
    def is_empty(self) -> bool:
        return not (
            self.relations
            or self.forbidden_relations
            or self.subjects_include
            or self.judgement_count
            or self.unreadable_fragments
            or self.out_of_scope_fragments
        )


@dataclass(frozen=True)
class FusionCase:
    """一个用例：若干段观测 + 确定性期望。"""

    case_id: str
    category: str
    intent: str
    segments: tuple[tuple[FusionFragment, ...], ...]
    expect: FusionExpectation
    # 探查用例：期望写的是"假说预测会发生什么"，不是"实现必须做到什么"。
    probing: bool = False
    # 本次跟踪谁；不写就取第一帧的第一个参与者。
    primary_subject: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.category.strip():
            raise FusionBenchmarkError("case_id and category must be non-empty")
        if not self.intent.strip():
            # 每个用例都要说清楚"它在考什么"——否则失败时没人知道该不该在意。
            raise FusionBenchmarkError(f"case {self.case_id} must state its intent")
        if not self.segments or any(not item for item in self.segments):
            raise FusionBenchmarkError(f"case {self.case_id} requires non-empty segments")
        if self.expect.is_empty:
            raise FusionBenchmarkError(f"case {self.case_id} asserts nothing")
        if self.primary_subject is None:
            object.__setattr__(self, "primary_subject", self.segments[0][0].participants[0])


def load_cases(path: str | Path) -> tuple[FusionCase, ...]:
    """从 JSON 读取用例集。"""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != FUSION_CASE_SCHEMA:
        raise FusionBenchmarkError("fusion case file schema is invalid")
    cases = tuple(_case(item) for item in raw["cases"])
    if len({item.case_id for item in cases}) != len(cases):
        raise FusionBenchmarkError("fusion case IDs must be unique")
    return cases


def _case(value: Any) -> FusionCase:
    expect = value.get("expect", {})
    return FusionCase(
        case_id=value["case_id"],
        category=value["category"],
        intent=value["intent"],
        probing=bool(value.get("probing", False)),
        primary_subject=value.get("primary_subject"),
        segments=tuple(
            tuple(_fragment(item) for item in segment) for segment in value["segments"]
        ),
        expect=FusionExpectation(
            relations=tuple(tuple(item) for item in expect.get("relations", ())),
            forbidden_relations=tuple(expect.get("forbidden_relations", ())),
            subjects_include={
                int(key): tuple(item) for key, item in expect.get("subjects_include", {}).items()
            },
            judgement_count={
                int(key): (item[0], item[1])
                for key, item in expect.get("judgement_count", {}).items()
            },
            unreadable_fragments={
                int(key): tuple(item)
                for key, item in expect.get("unreadable_fragments", {}).items()
            },
            out_of_scope_fragments={
                int(key): tuple(item)
                for key, item in expect.get("out_of_scope_fragments", {}).items()
            },
        ),
    )


def _fragment(value: Any) -> FusionFragment:
    return FusionFragment(
        offset_seconds=int(value["at"]),
        semantics=value["says"],
        participants=tuple(value.get("who", ("成年男性A",))),
        knowledge_state=value.get("knowledge_state", "observed"),
        modality=value.get("modality", "vision"),
    )


__all__ = [
    "FUSION_CASE_SCHEMA",
    "FusionBenchmarkError",
    "FusionCase",
    "FusionExpectation",
    "FusionFragment",
    "load_cases",
]
