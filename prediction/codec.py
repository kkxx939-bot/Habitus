"""一代树与其规范 JSON 表示之间的双向翻译。

单独成模块是为了让**序列化的正确性可以脱离磁盘验证**：编解码是纯函数，往返一致性用普通
单测就能穷举，存储层因此只剩"把字节放稳"这一件事。

时间在这里手工转成字符串，不交给 ``canonicalize``——后者会把带偏移的时刻折成 UTC，
而本层的槽位映射依赖本地时分（行为侧同一个坑已经踩过）。树自身的 ``built_at`` 本就是 UTC，
明确写成字符串也顺带把这条纪律固定下来。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any

from prediction.errors import PredictionTreeError
from prediction.model import (
    EdgeStatistics,
    IntervalQuantiles,
    NodeCounts,
    NodeStatistics,
    PredictionTree,
    RecurrenceStatistics,
    SlotExposure,
    SlotKey,
    slot_count,
)

SCHEMA_VERSION = 1


def encode(tree: PredictionTree) -> dict[str, Any]:
    """把一代树摊平成 JSON 安全的对象；键里的元组一律展开成显式字段。"""

    if not isinstance(tree, PredictionTree):
        raise PredictionTreeError("tree must be a PredictionTree")
    return {
        "schema_version": SCHEMA_VERSION,
        "built_at": tree.built_at.isoformat(),
        "reference_day": tree.reference_day.isoformat(),
        "config_digest": tree.config_digest,
        "slot_minutes": tree.slot_minutes,
        "actions": list(tree.actions),
        "observed_days": tree.observed_days,
        "censored_transitions": tree.censored_transitions,
        "exposure": [
            {"slot": _slot(key), "observed_days": value.observed_days, "recent_days": value.recent_days}
            for key, value in sorted(tree.exposure.items(), key=lambda item: _slot(item[0]))
        ],
        "baselines": [
            {"action": action, "rate": rate}
            for action, rate in sorted(tree.baselines.items())
        ],
        "nodes": [
            {"slot": _slot(key[0]), "action": key[1], **_node(value)}
            for key, value in sorted(tree.nodes.items(), key=lambda item: (_slot(item[0][0]), item[0][1]))
        ],
        "edges": _edges(tree.edges),
        "parallels": _edges(tree.parallels),
        "recurrences": [
            {"action": action, "intervals": _intervals(value.intervals)}
            for action, value in sorted(tree.recurrences.items())
        ],
    }


def decode(payload: Mapping[str, Any]) -> PredictionTree:
    """还原一代树；版本不认识就报错，绝不"尽力解析"出一棵半截的树。"""

    if not isinstance(payload, Mapping):
        raise PredictionTreeError("prediction tree payload must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PredictionTreeError("prediction tree payload has an unsupported schema version")
    # 槽宽先取出来：下面每个槽位键都要用它校验上界，否则损坏的 payload 里一个 "0/9999"
    # 能解码成功并混进树，之后每次查询都查不到、也看不出为什么。
    slot_minutes = payload.get("slot_minutes")
    if not isinstance(slot_minutes, int) or isinstance(slot_minutes, bool):
        raise PredictionTreeError("prediction tree payload has an invalid slot width")
    total = slot_count(slot_minutes)
    try:
        return PredictionTree(
            built_at=_moment(payload["built_at"]),
            reference_day=date.fromisoformat(payload["reference_day"]),
            config_digest=payload["config_digest"],
            slot_minutes=slot_minutes,
            nodes={
                (_key(item["slot"], total), item["action"]): _node_statistics(item)
                for item in payload["nodes"]
            },
            edges=_edge_map(payload["edges"], total),
            parallels=_edge_map(payload["parallels"], total),
            recurrences={
                item["action"]: RecurrenceStatistics(intervals=_quantiles(item["intervals"]))
                for item in payload["recurrences"]
            },
            exposure={
                _key(item["slot"], total): SlotExposure(
                    observed_days=item["observed_days"], recent_days=item["recent_days"]
                )
                for item in payload["exposure"]
            },
            baselines={item["action"]: item["rate"] for item in payload["baselines"]},
            actions=tuple(payload["actions"]),
            observed_days=payload["observed_days"],
            censored_transitions=payload["censored_transitions"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionTreeError("prediction tree payload is malformed") from exc


def _edges(edges: Mapping[tuple[str, str], EdgeStatistics]) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "target": target,
            "count": value.count,
            "probability": value.probability,
            "lift": value.lift,
            "n_eff": value.n_eff,
            "intervals": _intervals(value.intervals),
            "slot_histogram": [
                {"slot": _slot(slot), "weight": weight}
                for slot, weight in sorted(value.slot_histogram.items(), key=lambda item: _slot(item[0]))
            ],
        }
        for (source, target), value in sorted(edges.items())
    ]


def _edge_map(items: Any, total: int) -> dict[tuple[str, str], EdgeStatistics]:
    return {
        (item["source"], item["target"]): EdgeStatistics(
            count=item["count"],
            probability=item["probability"],
            lift=item["lift"],
            n_eff=item["n_eff"],
            intervals=_quantiles(item["intervals"]) if item["intervals"] is not None else None,
            slot_histogram={_key(cell["slot"], total): cell["weight"] for cell in item["slot_histogram"]},
        )
        for item in items
    }


def _node(value: NodeStatistics) -> dict[str, Any]:
    return {
        "marginal": value.marginal,
        "hazard": value.hazard,
        "cumulative": value.cumulative,
        "lift_all_day": value.lift_all_day,
        "lift_weekday": value.lift_weekday,
        "n_eff": value.n_eff,
        "trend": value.trend,
        "counts": {
            "occurred_days": value.counts.occurred_days,
            "first_days": value.counts.first_days,
            "earlier_days": value.counts.earlier_days,
            "raw_occurrences": value.counts.raw_occurrences,
            "recent_days": value.counts.recent_days,
        },
    }


def _node_statistics(item: Mapping[str, Any]) -> NodeStatistics:
    return NodeStatistics(
        marginal=item["marginal"],
        hazard=item["hazard"],
        cumulative=item["cumulative"],
        lift_all_day=item["lift_all_day"],
        lift_weekday=item["lift_weekday"],
        n_eff=item["n_eff"],
        trend=item["trend"],
        counts=NodeCounts(**item["counts"]),
    )


def _intervals(value: IntervalQuantiles | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"p10": value.p10, "p50": value.p50, "p90": value.p90, "sample_count": value.sample_count}


def _quantiles(item: Mapping[str, Any]) -> IntervalQuantiles:
    return IntervalQuantiles(
        p10=item["p10"], p50=item["p50"], p90=item["p90"], sample_count=item["sample_count"]
    )


def _slot(key: SlotKey) -> str:
    return f"{key.weekday}/{key.slot}"


def _key(text: Any, total: int) -> SlotKey:
    if not isinstance(text, str):
        raise PredictionTreeError("slot key must be text")
    weekday, _, slot = text.partition("/")
    if not weekday.isdigit() or not slot.isdigit():
        raise PredictionTreeError("slot key must look like '<weekday>/<slot>'")
    if int(slot) >= total:
        raise PredictionTreeError("slot key falls outside this tree's clock face")
    return SlotKey(weekday=int(weekday), slot=int(slot))


def _moment(text: Any) -> datetime:
    if not isinstance(text, str):
        raise PredictionTreeError("built_at must be text")
    return datetime.fromisoformat(text).astimezone(timezone.utc)


__all__ = ["SCHEMA_VERSION", "decode", "encode"]
