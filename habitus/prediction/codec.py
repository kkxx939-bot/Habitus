"""一代树与其规范 JSON 表示之间的双向翻译。

单独成模块是为了让**序列化的正确性可以脱离磁盘验证**：编解码是纯函数，往返一致性用普通
单测就能穷举，存储层因此只剩"把字节放稳"这一件事。

时间在这里手工转成字符串，不交给 ``canonicalize``——后者会把带偏移的时刻折成 UTC，
而本层的槽位映射依赖本地时分（行为侧同一个坑已经踩过）。树自身的 ``built_at`` 本就是 UTC，
明确写成字符串也顺带把这条纪律固定下来。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from habitus.prediction.errors import PredictionTreeError
from habitus.prediction.model import (
    WEEKDAYS,
    DayCurve,
    EdgeStatistics,
    IntervalQuantiles,
    NodeCounts,
    NodeStatistics,
    ParallelStatistics,
    PredictionTree,
    RecurrenceStatistics,
    SlotExposure,
    SlotKey,
    slot_count,
)

# 2：新增密集曲线与并行参与总权重，节点项去掉 trend，并行项换成规范键 + 计数。旧代的字节
# 与这一版形状不兼容——版本号不升，读侧只会报"payload is malformed"，把"上一版规格"说成
# "存储损坏"，而磁盘上最多留着 published_generations 代旧字节。
# 3：节点与边补上**出处日**（这个数字是哪几天攒出来的）。旧代没有这一节，而模型层现在硬性
# 要求每个格子、每条边都说得出自己的日子，照旧读会在构造时抛"published without the days"
# ——那句话指向的是磁盘上的旧字节，不是代码的错，所以这里升版号让它直接报成版本不认识。
SCHEMA_VERSION = 3


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
            {"slot": _slot(key[0]), "action": key[1], **_node(value, tree.reference_day)}
            for key, value in sorted(tree.nodes.items(), key=lambda item: (_slot(item[0][0]), item[0][1]))
        ],
        "curves": [
            {
                "weekday": weekday,
                "action": action,
                "marginal": _runs(value.marginal),
                "hazard": _runs(value.hazard),
                "cumulative": _runs(value.cumulative),
                "trend": value.trend,
                "trend_n_eff": value.trend_n_eff,
            }
            for (weekday, action), value in sorted(tree.curves.items())
        ],
        "weekday_baselines": [
            {"action": action, "rate": _runs(values)}
            for action, values in sorted(tree.weekday_baselines.items())
        ],
        "edges": _edges(tree.edges, tree.reference_day),
        "parallels": [
            {"left": left, "right": right, "count": value.count}
            for (left, right), value in sorted(tree.parallels.items())
        ],
        "parallel_totals": [
            {"action": action, "weight": weight}
            for action, weight in sorted(tree.parallel_totals.items())
        ],
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
        # 出处日按"距基准日多少天"存，所以基准日要先取出来；观测跨度给出它的合法范围。
        reference_day = date.fromisoformat(payload["reference_day"])
        span = payload["observed_days"]
        if isinstance(span, bool) or not isinstance(span, int) or span < 0:
            raise PredictionTreeError("prediction tree payload has an invalid observed span")
        return PredictionTree(
            built_at=_moment(payload["built_at"]),
            reference_day=reference_day,
            config_digest=payload["config_digest"],
            slot_minutes=slot_minutes,
            nodes={
                (_key(item["slot"], total), item["action"]): _node_statistics(item, reference_day, span)
                for item in payload["nodes"]
            },
            curves={
                (_weekday(item["weekday"]), item["action"]): DayCurve(
                    marginal=_expand(item["marginal"], total),
                    hazard=_expand(item["hazard"], total),
                    cumulative=_expand(item["cumulative"], total),
                    trend=item["trend"],
                    trend_n_eff=item["trend_n_eff"],
                )
                for item in payload["curves"]
            },
            weekday_baselines={
                item["action"]: _expand(item["rate"], total)
                for item in payload["weekday_baselines"]
            },
            edges=_edge_map(payload["edges"], total, reference_day, span),
            parallels={
                (item["left"], item["right"]): ParallelStatistics(count=item["count"])
                for item in payload["parallels"]
            },
            parallel_totals=_parallel_totals(payload),
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
    except PredictionTreeError:
        # 已经说清了是哪一处不对（曲线长度、周几范围、并行分母……），别再糊成一句
        # "payload is malformed"——那正是排查时最没用的一句话。
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionTreeError("prediction tree payload is malformed") from exc


def _runs(values: Sequence[float]) -> list[list[float]]:
    """把一条曲线折成 ``[值, 连续长度]`` 的游程。

    曲线是密集的（每条三条数组 × 槽数），而它们高度重复：真实七天树上 590,400 个值只有
    76,912 段游程，``cumulative`` 每条中位只有 2 个不同取值。游程编码把 curves 这一节从
    4.33 MiB 压到 1.17 MiB、整棵树 8.94 → 5.5 MiB 量级，**数值逐位不变**。
    """

    runs: list[list[float]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1.0
        else:
            runs.append([value, 1.0])
    return runs


def _expand(runs: Any, total: int) -> tuple[float, ...]:
    """展开游程；长度必须正好是这棵树的槽数。

    不校验长度的后果实测过：一条长度 2 的曲线能解码成功，之后每次查询抛**裸 IndexError**，
    从一个把所有错误都归一成 ``PredictionTreeError`` 的层里漏出 builtin。
    """

    if not isinstance(runs, list):
        raise PredictionTreeError("a day curve must be encoded as a list of runs")
    values: list[float] = []
    for run in runs:
        if not isinstance(run, list) or len(run) != 2:
            raise PredictionTreeError("a day curve run must be a [value, length] pair")
        length = run[1]
        if isinstance(length, bool) or not isinstance(length, (int, float)) or length < 1:
            raise PredictionTreeError("a day curve run length must be a positive number")
        values.extend([float(run[0])] * int(length))
        if len(values) > total:
            break
    if len(values) != total:
        raise PredictionTreeError("a day curve must cover exactly this tree's clock face")
    return tuple(values)


def _weekday(value: Any) -> int:
    """曲线键的周几：与 ``SlotKey`` 同一个范围，越界的键永远匹配不上、也看不出为什么。"""

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < WEEKDAYS:
        raise PredictionTreeError("a day curve weekday must be an integer below 7")
    return value


def _parallel_totals(payload: Mapping[str, Any]) -> dict[str, float]:
    """并行的参与总权重，并校验它真的盖得住每条并行边。

    分母缺了不会报错，只会让 ``query.parallels`` 静默算出"证据 1.0、概率 0.0"——正是并行边
    不给 lift 想避免的那种"被读成不可能"。
    """

    totals = {item["action"]: item["weight"] for item in payload["parallel_totals"]}
    for item in payload["parallels"]:
        for action in (item["left"], item["right"]):
            if totals.get(action, 0.0) < item["count"]:
                raise PredictionTreeError(
                    "a parallel edge is heavier than its action's participation weight"
                )
    return totals


def _edges(edges: Mapping[tuple[str, str], EdgeStatistics], reference: date) -> list[dict[str, Any]]:
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
            "days": _offsets(value.days, reference),
        }
        for (source, target), value in sorted(edges.items())
    ]


def _edge_map(items: Any, total: int, reference: date, span: int) -> dict[tuple[str, str], EdgeStatistics]:
    return {
        (item["source"], item["target"]): EdgeStatistics(
            count=item["count"],
            probability=item["probability"],
            lift=item["lift"],
            n_eff=item["n_eff"],
            intervals=_quantiles(item["intervals"]) if item["intervals"] is not None else None,
            slot_histogram={_key(cell["slot"], total): cell["weight"] for cell in item["slot_histogram"]},
            days=_dates(item["days"], reference, span),
        )
        for item in items
    }


def _node(value: NodeStatistics, reference: date) -> dict[str, Any]:
    return {
        "n_eff": value.n_eff,
        "days": _offsets(value.days, reference),
        "counts": {
            "occurred_days": value.counts.occurred_days,
            "first_days": value.counts.first_days,
            "earlier_days": value.counts.earlier_days,
            "raw_occurrences": value.counts.raw_occurrences,
            "recent_days": value.counts.recent_days,
        },
    }


def _node_statistics(item: Mapping[str, Any], reference: date, span: int) -> NodeStatistics:
    return NodeStatistics(
        n_eff=item["n_eff"], counts=NodeCounts(**item["counts"]), days=_dates(item["days"], reference, span)
    )


def _offsets(days: Sequence[date], reference: date) -> list[int]:
    """出处日按"距基准日多少天"编码。

    日期一条 ISO 字符串占 12 字节，偏移量占 1–4 字节。出处日的条目数与 occurrence 条数同量级
    （WP4 粒度一年十几万条），按字符串存要给整棵树加一两 MiB，而它现在总共才 4.66 MiB、
    读一次峰值约九倍（见 ``store`` 里 PRED-STORE-002 的实测）。基准日随树一起发布，换算是
    精确的整数运算，不引入任何浮点或时区语义。
    """

    return [(reference - day).days for day in days]


def _dates(raw: Any, reference: date, span: int) -> tuple[date, ...]:
    """还原出处日；顺序原样保留，升序与不重复由 ``PredictionTree`` 自己再查一遍。

    偏移量必须落在 ``[0, span)`` 内，这是**我们自己产出的东西是否自洽**：账本的日集合就是
    ``_calendar_days`` 从最早记录到基准日的那 ``observed_days`` 个连续日历日，所以任何出处日
    都不会晚于基准日、也不会早于跨度的第一天。不查这个范围有两种下场，都实测过：
    ``offset = 999999999`` 让 ``date - timedelta`` 抛 **``OverflowError``**，它继承自
    ``ArithmeticError``，本函数的 ``except (KeyError, TypeError, ValueError)`` 与存储层的
    ``except PredictionTreeError`` 两层都接不住，一个 builtin 直接漏给夜批；``offset = 700000``
    则解出公元 0124 年的日期，升序与上界两道校验都放行，整棵树干干净净地加载，然后读侧把
    公元 0124 年当作"这一格的出处"交给语义层。
    """

    if not isinstance(raw, list):
        raise PredictionTreeError("provenance days must be a list of day offsets")
    days: list[date] = []
    for offset in raw:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise PredictionTreeError("a provenance day offset must be a whole number of days")
        if not 0 <= offset < max(span, 1):
            raise PredictionTreeError("a provenance day offset falls outside this tree's observed span")
        days.append(reference - timedelta(days=offset))
    return tuple(days)


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
    return datetime.fromisoformat(text).astimezone(UTC)


__all__ = ["SCHEMA_VERSION", "decode", "encode"]
