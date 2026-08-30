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
_STATUS_VALUES = frozenset({"ongoing", "completed", "interrupted", "abandoned"})


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

    # 期望出现的关系：(来源段序号, 目标段序号, 关系类型)，段序号从 1 开始。
    # 段内关系写成 (n, n, kind)——"一边翻炒一边通话"这类并行发生在同一段里。
    relations: tuple[tuple[int, int, str], ...] = ()
    # 不该出现的关系，与 ``relations`` 同形。**必须带段号**：只按类型禁止会让一次段内的合法
    # 延续把跨段对照组染红，红绿与它要抓的滥用就没有对应关系了。
    forbidden_relations: tuple[tuple[int, int, str], ...] = ()
    # 某段里**不该出现**的 status 取值。"人走出画面"这类用例要考的是"不许断言已完成"，
    # 用判断条数考不到——条数对了不等于没瞎标 completed。
    forbidden_status: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    # 某段里**至少有一条判断**要带上的 status 取值。
    status_present: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    # 这些段里任何判断都不该带 goal——目标判不出时硬编一个，产出的是假事实。
    # 只用于目标**确实无从判断**的场景（人站在窗边张望）。若场景本身有一个贴着观测的目标
    # （开冰箱→查看冰箱内物品），要求"必须留空"就是在替现实规定行为该长什么样，不属于本层能立的法。
    goal_absent: tuple[int, ...] = ()
    # 某段里不该出现的目标（子串匹配）。用来抓 intent 点名的那个**具体幻觉**——"开冰箱看一眼"
    # 判成"找食物"是编造，判成"查看冰箱内物品"不是。子串匹配会漏（换个说法就抓不到），但它只会
    # 漏报不会误报，比用判断条数去糊弄要诚实。
    forbidden_goals: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    # **含主体的那些判断**的 subjects 里不该出现的人。旁观者可以有自己的判断（那条会被分流），
    # 但不该被一起写进主体那件事的 subjects——那等于把一件与他无关的事算成他的行为。
    subjects_exclude: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    # 期望某一段的 subjects 至少包含这些人。
    subjects_include: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    # 期望某一段产出的判断条数落在这个区间——粗判粒度，不判语义。
    judgement_count: Mapping[int, tuple[int, int]] = field(default_factory=dict)
    # 期望被判为"读不懂"的片段编号（按段内 1..N）。
    unreadable_fragments: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    # 期望**不属于主体**的片段编号。与"读不懂"是两回事：那些帧读得懂，只是做的人不是我们跟踪的。
    # 用条数去测旁人有没有被吸收是测不准的——条数对了不等于那一帧没被塞进主体的判断。
    out_of_scope_fragments: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    # 期望**无归属**的片段编号：读得懂、但不构成任何事也不是任何事的步骤（无意识小动作、过渡帧）。
    # 与"读不懂""旁人"三者口径各自干净。写成期望是为了两头都抓：该无归属的没无归属（噪声进了事件）、
    # 不该无归属的被扔进去了（压制产出）。
    unowned_fragments: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    # 期望**至少**这些片段无归属（子集匹配）。用在"这段可以判 0 条也可以判 1 条"两种产出都站得住的
    # 真实切片上：扶眼镜、摸脸这些帧无论如何都不该进任何事，但旁边的"点头""看着大家"归给一条
    # "开会"还是也无归属，两种都对——全集精确匹配会把合法的另一种判红。
    unowned_include: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    # 期望这些片段**不在主体的任何判断里**——无归属或旁人的都算。旁人走过那一帧，模型判成
    # "旁人的一条判断"（会被分流）或直接无归属都对，要抓的只是"没被塞进主体那件事"。
    not_owned_by_subject: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    # 期望某段里**至少判出**的行为（子串匹配 behavior）：允许模型不产出之后，"可提醒的单位有没有被
    # 一起扔掉"必须单独考，总条数降了不等于对了。
    behaviors_present: Mapping[int, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for _, _, kind in tuple(self.relations) + tuple(self.forbidden_relations):
            if kind not in _RELATION_KINDS:
                raise FusionBenchmarkError(f"unknown relation kind: {kind}")
        for mapping in (self.forbidden_status, self.status_present):
            for values in mapping.values():
                for value in values:
                    if value not in _STATUS_VALUES:
                        raise FusionBenchmarkError(f"unknown status: {value}")

    @property
    def is_empty(self) -> bool:
        return not (
            self.relations
            or self.forbidden_relations
            or self.subjects_include
            or self.judgement_count
            or self.unreadable_fragments
            or self.out_of_scope_fragments
            or self.unowned_fragments
            or self.unowned_include
            or self.not_owned_by_subject
            or self.behaviors_present
            or self.forbidden_status
            or self.status_present
            or self.goal_absent
            or self.forbidden_goals
            or self.subjects_exclude
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
        segments=tuple(tuple(_fragment(item) for item in segment) for segment in value["segments"]),
        expect=FusionExpectation(
            relations=tuple(tuple(item) for item in expect.get("relations", ())),
            forbidden_relations=tuple(tuple(item) for item in expect.get("forbidden_relations", ())),
            forbidden_status={int(key): tuple(item) for key, item in expect.get("forbidden_status", {}).items()},
            status_present={int(key): tuple(item) for key, item in expect.get("status_present", {}).items()},
            goal_absent=tuple(expect.get("goal_absent", ())),
            forbidden_goals={int(key): tuple(item) for key, item in expect.get("forbidden_goals", {}).items()},
            subjects_exclude={int(key): tuple(item) for key, item in expect.get("subjects_exclude", {}).items()},
            subjects_include={int(key): tuple(item) for key, item in expect.get("subjects_include", {}).items()},
            judgement_count={int(key): (item[0], item[1]) for key, item in expect.get("judgement_count", {}).items()},
            unreadable_fragments={
                int(key): tuple(item) for key, item in expect.get("unreadable_fragments", {}).items()
            },
            out_of_scope_fragments={
                int(key): tuple(item) for key, item in expect.get("out_of_scope_fragments", {}).items()
            },
            unowned_fragments={int(key): tuple(item) for key, item in expect.get("unowned_fragments", {}).items()},
            unowned_include={int(key): tuple(item) for key, item in expect.get("unowned_include", {}).items()},
            not_owned_by_subject={
                int(key): tuple(item) for key, item in expect.get("not_owned_by_subject", {}).items()
            },
            behaviors_present={int(key): tuple(item) for key, item in expect.get("behaviors_present", {}).items()},
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
