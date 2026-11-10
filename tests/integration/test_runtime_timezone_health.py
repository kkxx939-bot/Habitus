"""健康面比系统时区与配置时区（第二轮修复 G7）的接线：真 ``build_runtime`` 交给健康服务的是 ``locale.timezone``，不靠换代。

配置一个此刻 UTC 偏移与系统本地时区不同的时区，``runtime.health()`` 报 ``timezone``（non-critical）；配置成与系统此刻偏移相同的
固定偏移时区（``Etc/GMT±N``）时不报。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from habitus.config.locale import LocaleConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.runtime import build_runtime
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies
from tests.lock_store import ProcessLocalLockStore


def _etc_zone(offset: timedelta) -> str | None:
    """整小时偏移对应的 ``Etc/GMT∓N``（符号与习惯相反）；不是整小时没有对应的。"""

    hours, rest = divmod(int(offset.total_seconds()), 3600)
    if rest:
        return None
    return "Etc/GMT" if hours == 0 else f"Etc/GMT{'-' if hours > 0 else '+'}{abs(hours)}"


def _checks(tmp_path: Path, zone: str) -> dict[str, object]:
    config = runtime_config(tmp_path)
    config = replace(config, locale=replace(config.locale, timezone=zone))
    assert isinstance(config.locale, LocaleConfig)
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        config, providers=providers, vector_stores=vectors, path_lock=PathLock(ProcessLocalLockStore())
    )

    async def scenario():  # type: ignore[no-untyped-def]
        try:
            return await runtime.health()
        finally:
            await runtime.close()

    return {check.name: check for check in asyncio.run(scenario()).checks}


def test_the_runtime_health_compares_the_configured_timezone_with_the_system_one(tmp_path: Path) -> None:
    system = datetime.now().astimezone().utcoffset()
    assert system is not None
    other = _etc_zone(system + timedelta(hours=5)) or "Etc/GMT+12"
    if datetime.now(ZoneInfo(other)).utcoffset() == system:  # pragma: no cover - 只在偏移恰好相同的机器上
        other = "Etc/GMT-14"
    found = _checks(tmp_path / "differs", other)
    check = found["timezone"]
    assert check.critical is False and other in check.detail  # type: ignore[attr-defined]
    same = _etc_zone(system)
    if same is not None:
        assert "timezone" not in _checks(tmp_path / "same", same)
