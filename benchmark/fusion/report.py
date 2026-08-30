"""把原始记录聚合成判定与指标。

与 ``runner`` 分开，是为了改判定口径时不必重跑模型——同一批 ``records.jsonl`` 可以反复聚合。

## 稳定性是一等指标，不是附注

同一用例多次运行**结果不一致**，与**稳定地失败**是两种完全不同的病，混进同一个通过率就都看不见了：

    稳定失败    实现或提示词有确定的缺口，改了能验证
    抖动        同一输入产出不可预期；就算平均通过率很高也不能依赖

而且抖动会反过来污染用例本身——一个抖动的行为，无论把期望写成"必须 A"还是"必须 B"都会红，
本项目已经在同一个场景上把两个方向各写错过一次。所以报告把两者分开列，并单独给出一致率。

## 指标只覆盖确定性可判的部分

    结构合规    frames 穷尽、编号对齐、装配是否拒绝、结构重试了几次
    关系命中    期望的关系出现了没有（带段号，段内关系写成 (n, n, kind)）
    关系误标    对照组有没有瞎标（**这一项和命中同等重要**——只看命中会漏掉滥用）
    状态        不该断言的状态有没有被断言，该出现的状态出现了没有
    目标        判不出目标时有没有硬编，以及有没有编出 intent 点名的那个幻觉
    主体        旁观者有没有被算进主体那件事的 subjects
    读不懂      该判为读不懂的帧判了没有，以及有没有牵连其它帧
    落在范围外  哪些帧读得懂但做的人不是主体
    粒度        判断条数落在期望区间内（粗判，不判语义）

"目标判得准不准""分解合不合理"需要 Judge，不在这里。

## 期望必须直接测 intent 说的那件事

曾经有七条用例的 intent 写着一件事（不该断言已完成、不该硬编目标、不该把旁观者算成主体），
``expect`` 里却只有判断条数——于是模型犯下 intent 明令禁止的错误，用例照样全绿，而报告还会打印
``status 2/2``、``goal 1/1``，读起来像这几类都判对了。用条数替代语义检查不是"粗判"，是**没判**。

补上状态/目标/主体三类检查之后，第一次跑就炸出两条一直被盖住的真实缺口（模型把"主动放弃"
判成"被打断"、以及一条我自己写错的期望）。所以新增用例时：**先问它的 intent 说的是什么，
再问 expect 里有没有一项在测那个**；只有折叠与粒度这类"条数本身就是被测对象"的才该只写条数。
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
    for source, target, kind in case.expect.forbidden_relations:
        used = [item for item in observed_full if item == (source, target, kind)]
        checks.append(
            {
                "check": "relation_not_abused",
                "passed": not used,
                "detail": f"不该出现 段{source} --{kind}--> 段{target}；实际 {observed_full or '无'}",
            }
        )
    for index, forbidden in case.expect.forbidden_status.items():
        offending = [
            (item["behavior"], item["status"])
            for item in segments.get(index, {}).get("judgements", [])
            if item["status"] in set(forbidden)
        ]
        checks.append(
            {
                "check": "status_not_overclaimed",
                "passed": not offending,
                "detail": f"段{index} 不该出现 status {list(forbidden)}；实际 {offending or '无'}",
            }
        )
    for index, wanted in case.expect.status_present.items():
        seen = {item["status"] for item in segments.get(index, {}).get("judgements", [])}
        status_hit = seen & set(wanted)
        checks.append(
            {
                "check": "status_present",
                "passed": bool(status_hit),
                "detail": f"段{index} 期望出现 status {list(wanted)}；实际 {sorted(x for x in seen if x)}",
            }
        )
    for index in case.expect.goal_absent:
        invented = [
            (item["behavior"], item["goal"])
            for item in segments.get(index, {}).get("judgements", [])
            if item["goal"] is not None
        ]
        checks.append(
            {
                "check": "goal_not_invented",
                "passed": not invented,
                "detail": f"段{index} 判不出目标就该留空；实际硬编了 {invented or '无'}",
            }
        )
    for index, forbidden in case.expect.forbidden_goals.items():
        invented = [
            (item["behavior"], item["goal"])
            for item in segments.get(index, {}).get("judgements", [])
            if item["goal"] and any(word in item["goal"] for word in forbidden)
        ]
        checks.append(
            {
                "check": "goal_not_hallucinated",
                "passed": not invented,
                "detail": f"段{index} 不该编出 {list(forbidden)} 这类目标；实际 {invented or '无'}",
            }
        )
    for index, excluded in case.expect.subjects_exclude.items():
        # 只看含主体的那些判断：旁观者自己那条判断是合法的，不该因为它存在就判红。
        present = {
            name
            for item in segments.get(index, {}).get("judgements", [])
            if case.primary_subject in item["subjects"]
            for name in item["subjects"]
        } & set(excluded)
        checks.append(
            {
                "check": "subjects_not_overreached",
                "passed": not present,
                "detail": (
                    f"段{index} 把 {sorted(present)} 一起写进了主体那件事的 subjects"
                    if present
                    else None
                ),
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
    for index, expected_unreadable in case.expect.unreadable_fragments.items():
        actual_unreadable = tuple(segments.get(index, {}).get("unreadable_fragments", ()))
        checks.append(
            {
                "check": "unreadable",
                "passed": set(expected_unreadable) == set(actual_unreadable),
                "detail": (
                    f"段{index} 期望读不懂 {list(expected_unreadable)}，实际 {list(actual_unreadable)}"
                ),
            }
        )
    for index, expected_out_of_scope in case.expect.out_of_scope_fragments.items():
        actual_out_of_scope = tuple(segments.get(index, {}).get("out_of_scope_fragments", ()))
        checks.append(
            {
                "check": "out_of_scope",
                "passed": set(expected_out_of_scope) == set(actual_out_of_scope),
                "detail": (
                    f"段{index} 期望不属于主体的帧 {list(expected_out_of_scope)}，"
                    f"实际 {list(actual_out_of_scope)}"
                ),
            }
        )
    for index, expected_unowned in case.expect.unowned_fragments.items():
        actual_unowned = tuple(segments.get(index, {}).get("unowned_fragments", ()))
        checks.append(
            {
                "check": "unowned",
                "passed": set(expected_unowned) == set(actual_unowned),
                "detail": (
                    f"段{index} 期望无归属的帧 {list(expected_unowned)}，实际 {list(actual_unowned)}"
                ),
            }
        )
    for index, expected_free in case.expect.subject_free_fragments.items():
        seg = segments.get(index, {})
        # 三类去向都没把这一帧算成主体的行为；只有既不在其中任何一类、又没被判读不懂的帧才是被吸收了。
        free = (
            set(seg.get("out_of_scope_fragments", ()))
            | set(seg.get("unowned_fragments", ()))
            | set(seg.get("unreadable_fragments", ()))
        )
        absorbed = [no for no in expected_free if no not in free]
        checks.append(
            {
                "check": "subject_free",
                "passed": not absorbed,
                "detail": f"段{index} 期望不进主体判断的帧 {list(expected_free)}，被吸收的 {absorbed}",
            }
        )
    for index, expected_behaviors in case.expect.behaviors_present.items():
        actual_behaviors = [
            str(item.get("behavior") or "")
            for item in segments.get(index, {}).get("judgements", ())
        ]
        absent_behaviors = [
            wanted
            for wanted in expected_behaviors
            if not any(wanted in name for name in actual_behaviors)
        ]
        checks.append(
            {
                "check": "behaviors_present",
                "passed": not absent_behaviors,
                "detail": f"段{index} 期望判出 {list(expected_behaviors)}，缺 {absent_behaviors}，实际 {actual_behaviors}",
                # 保留率的分子分母：期望的可提醒单位里有几个真的被判出来了。按单位计数而不是按
                # 用例计数——一段期望三个单位、丢一个，通过率是 0，保留率是 2/3，两者说的不是一件事。
                "expected_units": len(expected_behaviors),
                "retained_units": len(expected_behaviors) - len(absent_behaviors),
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
        # 段数就是"一次成功需要的最少调用次数"；超出的部分全是结构不合法后的重试。
        "structural_retries": sum(
            max(0, item["model_calls"] - 1) for item in run["segments"]
        ),
        "elapsed_seconds": run["elapsed_seconds"],
        "unowned_fragments_total": sum(
            len(item.get("unowned_fragments", ())) for item in run["segments"]
        ),
        "fragment_total": sum(int(item.get("fragment_count", 0) or 0) for item in run["segments"]),
    }


def _remindable_retention(results: Sequence[Mapping[str, Any]]) -> float | None:
    """可提醒单位保留率：期望判出的行为单位里实际判出的比例（全部段合计）。

    WP4 允许模型对无意识小动作不产出之后，这是与无归属占比配对看的数字——占比升、保留率不降
    才是折叠；占比升、保留率跟着掉就是压制过头。没有任何 ``behaviors_present`` 期望时为 None。
    """

    expected = retained = 0
    for item in results:
        for check in item["checks"]:
            if check["check"] == "behaviors_present":
                expected += int(check.get("expected_units", 0))
                retained += int(check.get("retained_units", 0))
    return None if expected == 0 else round(retained / expected, 3)


def _unowned_ratio(results: Sequence[Mapping[str, Any]]) -> float | None:
    unowned = sum(int(item.get("unowned_fragments_total", 0) or 0) for item in results)
    fragments = sum(int(item.get("fragment_total", 0) or 0) for item in results)
    return None if fragments == 0 else round(unowned / fragments, 3)


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
        # 无归属帧占比（全部段合计）："允许不产出"用得多不多，单独看通过率会被"全部无归属"骗过
        "unowned_ratio": _unowned_ratio(results),
        # 与无归属占比配对：允许不产出之后，可提醒的单位有没有被一起扔掉。
        "remindable_retention": _remindable_retention(results),
        "probe_runs": len(probes),
        "probe_passed": sum(1 for item in probes if item["passed"]),
        "model_calls": sum(item["model_calls"] for item in results),
        "structural_retries": sum(item["structural_retries"] for item in results),
        "retried_cases": sorted(
            {item["case_id"] for item in results if item["structural_retries"]}
        ),
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
    unowned = summary.get("unowned_ratio")
    retention = summary.get("remindable_retention")
    lines.append(
        "\n无归属帧占比 "
        + ("—" if unowned is None else f"{unowned:.1%}")
        + "，可提醒单位保留率 "
        + ("—（无 behaviors_present 期望）" if retention is None else f"{retention:.1%}")
        + "（两者配对看：占比升而保留率不降才是折叠，保留率跟着掉是压制过头）"
    )
    if summary["structural_retries"]:
        lines.append(
            f"\n**结构重试 {summary['structural_retries']} 次**"
            f"（模型第一次没吐出合法结构，被 schema 打回重来）："
            f"{', '.join(summary['retried_cases'])}"
        )
    else:
        lines.append("\n结构重试 0 次——每次调用都一次吐出合法结构。")
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
