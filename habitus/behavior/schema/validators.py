"""两类行为文档的跨字段不变量；只查我们自己产物的自洽，不替现实立法。

会失败的都是"我们自己的产物内部矛盾"：goal 为空却带 basis、步骤引用了本 occurrence 之外的
观测、没读懂的空白不带判断溯源。**不会**因为"步骤时间不连续""步骤时间越出 started_at 范围"
"同一时段有别的行为"而失败——那些是上游数据的事实，不是本层能规定的策略（步骤时间窗规则曾
存在过，因 supersedes 换链头会让真数据被误拒而删除）。
"""

from __future__ import annotations

from typing import Any

from habitus.behavior.model import BehaviorKind, semantic_name
from habitus.behavior.schema.model import BehaviorSchemaError


def validate_payload(kind: BehaviorKind, payload: dict[str, Any]) -> None:
    if kind is BehaviorKind.OCCURRENCE:
        _validate_occurrence(payload)
    else:
        _validate_gap(payload)


def _validate_occurrence(payload: dict[str, Any]) -> None:
    semantic_name(payload["name"], "occurrence name")
    if payload["occurred_on"] != payload["started_at"].date():
        raise BehaviorSchemaError("occurred_on must match the local started_at date")
    if payload["last_observed_at"] < payload["started_at"]:
        raise BehaviorSchemaError("last_observed_at cannot precede started_at")
    if payload["onset_available_at"] < payload["started_at"]:
        raise BehaviorSchemaError("onset_available_at cannot precede started_at")

    if not payload["subjects"]:
        raise BehaviorSchemaError("occurrence must name at least one subject")
    # 消歧记录的自洽性：original_name 非空说明地址名带了序号后缀，两者相等即自相矛盾。
    if payload["original_name"] is not None and payload["original_name"] == payload["name"]:
        raise BehaviorSchemaError("original_name must differ from the disambiguated name")

    # goal 与 basis 的空非空互不约束（2026-08-30 随"一条 occurrence = 可提醒可代劳的单位"改定）：
    # 锁门、喝水说不出目标却有步骤，一次交谈有目标却未必分得出步骤——那是现实的形状。

    # 刻意不校验步骤时间是否落在 [started_at, last_observed_at] 内（用户裁定删除）：
    # supersedes 换链头后 started_at 取新链头，basis 里如实收着的早期步骤可以比它更早——
    # 数据全真却会被这类"现实形状"规则误拒；步骤时间由归约层从观测机械物化，不会说谎。
    # 溯源只剩链身份（原料发布即释放，逐条 id 只会是死引用）；它指向消费账本那一条。
    if not payload["chain_digest"]:
        raise BehaviorSchemaError("occurrence must trace to the chain that produced it")


def _validate_gap(payload: dict[str, Any]) -> None:
    if payload["occurred_on"] != payload["started_at"].date():
        raise BehaviorSchemaError("occurred_on must match the local started_at date")
    # 允许零时长（ended == started）：单条观测构成的"没读懂"段起止同刻，这是真实数据不是矛盾；
    # ended 早于 started 才是我们自己产物的自相矛盾。
    if payload["ended_at"] < payload["started_at"]:
        raise BehaviorSchemaError("a gap must not end before it starts")
    if payload["gap_kind"] == "没读懂":
        # 没读懂来自融合的空判断，溯源必须齐；未观测来自上游覆盖信号（契约未接），
        # 其溯源字段允许为空，接入时随契约补齐。
        if not payload["chain_digest"]:
            raise BehaviorSchemaError("an unreadable gap must trace to the judgements that produced it")


__all__ = ["validate_payload"]
