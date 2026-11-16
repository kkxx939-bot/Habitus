"""预测不等换代（2026-11-16）：换代余量 3 槽改 2 槽（``预期.md`` D2、C10）。

C10 走组合根装出来的那一支换代（``build_runtime`` 用 ``example.yaml`` 的默认配置）：最后一段跨过零点的「未观测」结束后预测拍水位
过了第 2 槽就换代，第 1 槽时不换（阶段 C 按用户 2026-11-16 第 2 条改读法：今天白天的空白不让等待重来）。
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, time, timedelta
from pathlib import Path

import yaml

from habitus.config.foresight import ForesightConfig
from habitus.runtime.generation_watermark import TickWatermark
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT, _fully_enabled_runtime

#: 改动前 ``ForesightConfig`` 的全部字段（D2：没有新增的键）。
FIELDS = {
    "daily_reminder_cap",
    "enabled",
    "false_alarm_rate",
    "generation_margin_slots",
    "half_life_opportunities",
    "judge_transient_retries",
    "judge_transient_retry_delay_seconds",
    "open_reminders_judged",
    "outlet_claim_limit",
    "outlet_lease_minutes",
    "outlet_liveness_hours",
    "outlet_match_similarity",
    "outlet_max_attempts",
    "shadow_uses_unestablished",
    "store",
    "tick_interval_seconds",
    "unit_prior_share",
    "unseen_day_types",
    "unwatched_tolerance_slots",
    "worker_shutdown_timeout_seconds",
}


def _example_foresight_keys() -> dict[str, str]:
    """``example.yaml`` 里注释掉的 ``foresight:`` 那一组的顶层键 → 那一行的值（含注释）。"""

    lines = (REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8").splitlines()
    start = lines.index("# foresight:")
    found: dict[str, str] = {}
    for line in lines[start + 1 :]:
        if not line.startswith("#   "):
            if line.startswith("#     "):
                continue
            break
        if line.startswith("#     "):
            continue
        key, _, rest = line[4:].partition(":")
        found[key.strip()] = rest.strip()
    return found


def test_d2_the_generation_margin_defaults_to_two_slots_in_code_and_in_the_example() -> None:
    assert ForesightConfig().generation_margin_slots == 2
    value = _example_foresight_keys()["generation_margin_slots"].split("#", 1)[0].strip()
    assert yaml.safe_load(value) == 2


def test_d2_no_foresight_key_was_added() -> None:
    assert {item.name for item in dataclasses.fields(ForesightConfig)} == FIELDS
    assert set(_example_foresight_keys()) == FIELDS


def test_c10_with_the_default_config_the_generation_waits_two_slots_after_the_last_unwatched_stretch(
    tmp_path: Path,
) -> None:
    """C10（按用户 2026-11-16 第 2 条改读法）：开盖以后连着开着。最后一段**跨过昨晚零点**的「未观测」在 09:00 结束、昨天已定稿：
    预测拍水位到 09:15（第 1 槽）不换代，到 09:30（第 2 槽）换代。之后今天白天又合盖写了一段空白（``unwatched_until`` 推到 11:00，
    ``overnight_until`` 仍是 09:00），等待不重来：水位一过就换代。"""

    runtime = _fully_enabled_runtime(tmp_path)
    pipeline = runtime.components.pipeline("session")
    assert pipeline is not None
    generation = pipeline.generation.generation
    zone = runtime.config.locale.zone()
    today = date(2026, 10, 20)
    reopened = datetime.combine(today, time(9, 0), tzinfo=zone)
    generation.clock = lambda: reopened + timedelta(minutes=31)
    generation.closed_days = lambda: (today - timedelta(days=2), today - timedelta(days=1))
    first = TickWatermark(
        ticked_at=reopened + timedelta(minutes=15), unwatched_until=reopened, overnight_until=reopened
    )
    generation.watermark = lambda: first
    early = generation.ready()
    assert early.day is None, f"第 1 槽就换代了：{early}"
    second = TickWatermark(
        ticked_at=reopened + timedelta(minutes=30), unwatched_until=reopened, overnight_until=reopened
    )
    generation.watermark = lambda: second
    ready = generation.ready()
    assert ready.day == today, f"第 2 槽还不换代：{ready.reason}"
    # 今天白天又合盖：10:30 → 11:00 写了一段不跨零点的空白，这一拍水位在 11:00
    daytime = TickWatermark(
        ticked_at=reopened + timedelta(hours=2),
        unwatched_until=reopened + timedelta(hours=2),
        overnight_until=reopened,
    )
    generation.clock = lambda: reopened + timedelta(hours=2, minutes=1)
    generation.watermark = lambda: daytime
    after = generation.ready()
    assert after.day == today, f"今天白天的空白让等待重来了：{after.reason}"
