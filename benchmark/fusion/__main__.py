"""``python -m benchmark.fusion`` —— 跑一批用例并输出报告。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from behavior.fusion import BehaviorFusionConfig
from benchmark.fusion.dataset import load_cases
from benchmark.fusion.report import aggregate, evaluate, render_markdown
from benchmark.fusion.runner import run_case
from Config import HabitusConfig
from ModelClient import StructuredChatClient
from ModelClient.adapters import register_builtin_adapters
from ModelClient.factory import ProviderFactory

_DEFAULT_CASES = Path(__file__).with_name("cases.json")


def _client(config_path: Path, *, retries: int) -> StructuredChatClient:
    """用项目唯一的配置边界构造真实模型客户端——评测不接受 scripted/fake。"""

    config = HabitusConfig.from_file(config_path)
    route = config.models.chat.route
    factory = ProviderFactory()
    register_builtin_adapters(factory)
    # 与 ``Runtime/assembly.py`` 同一条解析路径：凭据只从 Config 这个唯一边界取。
    credentials = config.credentials.resolve(route.credential_ref)
    chat = factory.create_chat_client(config.models.chat, credentials=credentials)
    return StructuredChatClient(chat, validation_retries=retries)


async def _run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    if args.category:
        cases = tuple(item for item in cases if item.category in set(args.category))
    if args.case_id:
        wanted = set(args.case_id)
        cases = tuple(item for item in cases if item.case_id in wanted)
        missing = wanted - {item.case_id for item in cases}
        if missing:
            raise SystemExit(f"没有这些用例：{sorted(missing)}")
    if not cases:
        raise SystemExit("没有匹配的用例")
    client = _client(args.config, retries=args.validation_retries)
    config = BehaviorFusionConfig()

    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "records.jsonl"
    results = []
    with records_path.open("w", encoding="utf-8") as handle:
        for attempt in range(1, args.attempts + 1):
            for case in cases:
                run = await run_case(
                    case, client, attempt=attempt,
                    primary_subject=case.primary_subject or '', config=config
                )
                payload = run.to_dict()
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                result = evaluate(case, payload)
                results.append(result)
                mark = "✓" if result["passed"] else "✗"
                print(f"  {mark} [{attempt}] {case.case_id}")

    summary = aggregate(results)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_markdown(summary)
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0 if summary["passed_runs"] == summary["total_runs"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.fusion",
        description="行为融合的确定性评测：真实模型调用，对照组抓滥用",
    )
    parser.add_argument("--config", type=Path, required=True, help="Habitus 完整运行配置")
    parser.add_argument("--output", type=Path, required=True, help="结果目录")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--attempts", type=int, default=3, help="每个用例跑几次；不稳定本身是信号")
    parser.add_argument("--category", action="append", help="只跑某些类别，可重复")
    parser.add_argument(
        "--case-id", action="append", help="只跑某几条用例，可重复；排查抖动时把次数加大"
    )
    parser.add_argument("--validation-retries", type=int, default=2)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
