"""把原始记录聚合成判定与指标。

与 ``runner`` 分开，是为了改判定口径时不必重跑模型——同一批 ``records.jsonl`` 可以反复聚合。

## 稳定性是一等指标，不是附注

同一用例多次运行**结果不一致**，与**稳定地失败**是两种完全不同的病，混进同一个通过率就都看不见了：

    稳定失败    实现或提示词有确定的缺口，改了能验证
    抖动        同一输入产出不可预期；就算平均通过率很高也不能依赖

而且抖动会反过来污染用例本身——一个抖动的行为，无论把期望写成"必须 A"还是"必须 B"都会红，
本项目已经在同一个场景上把两个方向各写错过一次。所以报告把两者分开列，并单独给出一致率。

## 指标只覆盖确定性可判的部分

    结构合规    frames 穷尽、编号对齐、装配是否拒绝、重试了几次
    关系命中    期望的关系出现了没有
    关系误标    对照组有没有瞎标（**这一项和命中同等重要**——只看命中会漏掉滥用）
    主体完整    多人场景丢没丢人
    读不懂      该判为读不懂的帧判了没有，以及有没有牵连其它帧
    粒度        判断条数落在期望区间内（粗判，不判语义）

"目标判得准不准""分解合不合理"需要 Judge，不在这里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.fusion.dataset import FusionCase


def evaluate(case: FusionCase, run: Mapping[str, Any]) -> dict[str, Any]:
    """核对一次运行是否满足用例的确定性期望。"""

    checks: list[dict[str, Any]] = []
    segments = {item["index"]: item for item in run["segments"]}
    failure = next((item["failed"] for item in run["segments"] if item["failed"]), None)
    if failure is not None:
        checks.append({"check": "completed", "passed": False, "detail": failure})
        return _summarise(case, run, checks)
    checks.append({"check": "completed", "passed": True, "detail": None})

    observed = [
        (relation["to_segment"], relation["kind"])
        for item in run["segments"]
        for relation in item["relations"]
    ]
    observed_full = [
        (item["index"], relation["to_segment"], relation["kind"])
        for item in run["segments"]
        for relation in item["relations"]
    ]

    for source, target, kind in case.expect.relations:
        hit = (source, target, kind) in observed_full
        checks.append(
            {
                "check": "relation_hit",
                "passed": hit,
                "detail": f"期望 段{source} --{kind}--> 段{target}；实际 {observed_full or '无'}",
            }
        )
    for kind in case.expect.forbidden_relations:
        used = [item for item in observed if item[1] == kind]
        checks.append(
            {
                "check": "relation_not_abused",
                "passed": not used,
                "detail": f"不该出现 {kind}；实际 {used or '无'}",
            }
        )
    for index, expected in case.expect.subjects_include.items():
        actual = {
            name for item in segments.get(index, {}).get("judgements", []) for name in item["subjects"]
        }
        missing = sorted(set(expected) - actual)
        checks.append(
            {
                "check": "subjects_complete",
                "passed": not missing,
                "detail": f"段{index} 缺少主体 {missing}；实际 {sorted(actual)}" if missing else None,
            }
        )
    for index, (low, high) in case.expect.judgement_count.items():
        count = len(segments.get(index, {}).get("judgements", []))
        checks.append(
            {
                "check": "granularity",
                "passed": low <= count <= high,
                "detail": f"段{index} 产出 {count} 条，期望 {low}–{high}",
            }
        )
    for index, expected in case.expect.unreadable_fragments.items():
        actual = tuple(segments.get(index, {}).get("unreadable_fragments", ()))
        checks.append(
            {
                "check": "unreadable",
                "passed": set(expected) == set(actual),
                "detail": f"段{index} 期望读不懂 {list(expected)}，实际 {list(actual)}",
            }
        )
    for index, expected in case.expect.out_of_scope_fragments.items():
        actual = tuple(segments.get(index, {}).get("out_of_scope_fragments", ()))
        checks.append(
            {
                "check": "out_of_scope",
                "passed": set(expected) == set(actual),
                "detail": f"段{index} 期望不属于主体的帧 {list(expected)}，实际 {list(actual)}",
            }
        )
    return _summarise(case, run, checks)


def _summarise(case: FusionCase, run: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for item in checks if item["passed"])
    return {
        "case_id": case.case_id,
        "category": case.category,
        "intent": case.intent,
        "probing": case.probing,
        "attempt": run["attempt"],
        "passed": passed == len(checks),
        "checks": list(checks),
        "model_calls": sum(item["model_calls"] for item in run["segments"]),
        "elapsed_seconds": run["elapsed_seconds"],
    }


def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按用例与类别聚合；同一用例多次运行的稳定性单独报。"""

    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for item in results:
        by_case.setdefault(item["case_id"], []).append(item)

    cases = []
    for case_id, runs in sorted(by_case.items()):
        wins = sum(1 for item in runs if item["passed"])
        total = len(runs)
        cases.append(
            {
                "case_id": case_id,
                "category": runs[0]["category"],
                "probing": runs[0]["probing"],
                "intent": runs[0]["intent"],
                "passed": wins,
                "attempts": total,
                "stable": wins in (0, total),
                # 一致率：最一致的那个结果占多少。1.0 表示每次都一样（无论对错）。
                "agreement": max(wins, total - wins) / total,
                "verdict": (
                    "pass" if wins == total else ("flaky" if wins else "fail")
                ),
                "failures": [
                    check["detail"]
                    for item in runs
                    for check in item["checks"]
                    if not check["passed"]
                ],
            }
        )

    # 探查用例不计入通过判定：它验的是假说，不是实现的义务。
    regression = [item for item in results if not item["probing"]]
    probes = [item for item in results if item["probing"]]

    by_check: dict[str, dict[str, int]] = {}
    for item in regression:
        for check in item["checks"]:
            entry = by_check.setdefault(check["check"], {"passed": 0, "total": 0})
            entry["total"] += 1
            entry["passed"] += int(check["passed"])

    by_category: dict[str, dict[str, int]] = {}
    for item in regression:
        entry = by_category.setdefault(item["category"], {"passed": 0, "total": 0})
        entry["total"] += 1
        entry["passed"] += int(item["passed"])

    return {
        "total_runs": len(regression),
        "passed_runs": sum(1 for item in regression if item["passed"]),
        "probe_runs": len(probes),
        "probe_passed": sum(1 for item in probes if item["passed"]),
        "model_calls": sum(item["model_calls"] for item in results),
        # 同一用例多次运行结果不一致，本身就是一个要盯的信号——比平均通过率更能暴露问题。
        # 稳定失败与抖动分开：前者改了能验证，后者说明产出不可预期，两者的处理方式完全不同。
        "stable_failures": [
            item["case_id"]
            for item in cases
            if item["verdict"] == "fail" and not item["probing"]
        ],
        "flaky_cases": [
            item["case_id"]
            for item in cases
            if item["verdict"] == "flaky" and not item["probing"]
        ],
        "mean_agreement": (
            round(
                sum(item["agreement"] for item in cases if not item["probing"])
                / max(1, sum(1 for item in cases if not item["probing"])),
                3,
            )
        ),
        "by_check": by_check,
        "by_category": by_category,
        "cases": cases,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# 行为融合评测", ""]
    lines.append(
        f"回归用例 {summary['total_runs']} 次，通过 {summary['passed_runs']} 次；"
        f"探查用例 {summary['probe_runs']} 次，其中 {summary['probe_passed']} 次符合假说预测。"
        f"模型调用共 {summary['model_calls']} 次"
    )
    lines.append(f"\n平均一致率 {summary['mean_agreement']}（1.0 = 每次结果都相同，无论对错）")
    if summary["stable_failures"]:
        lines.append(f"\n**稳定失败**（有确定缺口，改了能验证）：{', '.join(summary['stable_failures'])}")
    if summary["flaky_cases"]:
        lines.append(f"\n**抖动**（同一输入产出不可预期，比失败更值得先处理）：{', '.join(summary['flaky_cases'])}")
    lines += ["", "## 按检查项", "", "| 检查项 | 通过 | 总数 |", "| --- | --- | --- |"]
    for name, entry in sorted(summary["by_check"].items()):
        lines.append(f"| {name} | {entry['passed']} | {entry['total']} |")
    lines += ["", "## 按类别", "", "| 类别 | 通过 | 总数 |", "| --- | --- | --- |"]
    for name, entry in sorted(summary["by_category"].items()):
        lines.append(f"| {name} | {entry['passed']} | {entry['total']} |")
    for title, probing in (("## 回归用例", False), ("## 探查用例（验假说，不计入判定）", True)):
        selected = [item for item in summary["cases"] if item["probing"] is probing]
        if not selected:
            continue
        lines += ["", title, ""]
        for item in selected:
            mark = {"pass": "✓", "flaky": "~", "fail": "✗"}[item["verdict"]]
            lines.append(
                f"- {mark} `{item['case_id']}` ({item['passed']}/{item['attempts']}) — {item['intent']}"
            )
            for detail in dict.fromkeys(item["failures"]):
                if detail:
                    lines.append(f"    - {detail}")
    return "\n".join(lines) + "\n"


__all__ = ["aggregate", "evaluate", "render_markdown"]
