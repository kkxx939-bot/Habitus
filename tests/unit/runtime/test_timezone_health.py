"""健康面比较系统时区与配置时区（06-第二轮修复清单 G7；用户 11-10 换时区，方案 19 第 6.1 节）：此刻两边的 UTC 偏移不一致时
报出两边的名字与偏移（non-critical）；名字不同、偏移相同不报。换代一项（``_generation_checks``）用换代的时区（即
``locale.timezone``）与此刻按系统本地时区的读法调它。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from habitus.runtime.health import RuntimeHealthCheck, RuntimeHealthStatus, timezone_checks

PLUS_EIGHT = timezone(timedelta(hours=8))
NOW = datetime(2026, 4, 28, 2, 0, tzinfo=PLUS_EIGHT)


def test_a_system_offset_other_than_the_configured_one_is_reported() -> None:
    [check] = timezone_checks(ZoneInfo("Europe/London"), NOW)  # 系统 +08，配置 Europe/London（夏令时 +01）
    assert check.name == "timezone" and check.status is RuntimeHealthStatus.DEGRADED and check.critical is False
    assert "+0800" in check.detail and "Europe/London +0100" in check.detail


def test_the_same_offset_under_another_name_is_not_reported() -> None:
    for name in ("Asia/Shanghai", "Asia/Singapore"):
        assert timezone_checks(ZoneInfo(name), NOW) == [], name


def test_the_service_compares_the_configured_zone_even_without_a_generation(monkeypatch) -> None:
    """用户 11-10「都比」：配置时区由 runtime 直接交给健康服务，没接换代时偏移不一致照样报出；没给时区不比。"""

    import asyncio
    import time
    from types import SimpleNamespace

    from habitus.runtime.health import RuntimeHealthService

    async def none_check(*_args):  # type: ignore[no-untyped-def]
        return RuntimeHealthCheck("stub", RuntimeHealthStatus.HEALTHY, "")

    service = RuntimeHealthService(
        SimpleNamespace(
            generation=None,
            memory=SimpleNamespace(vector_index=SimpleNamespace(store=None)),
            conversation=SimpleNamespace(summary_vector_index=SimpleNamespace(store=None)),
        ),
        zone=ZoneInfo("Europe/London"),
    )  # type: ignore[arg-type]
    for name in (
        "_runtime_check",
        "_lifecycle_storage_check",
        "_memory_worker_check",
        "_lifecycle_check",
        "_lifecycle_capacity_check",
        "_observability_check",
    ):
        monkeypatch.setattr(
            service, name, lambda *_args, name=name: RuntimeHealthCheck(name, RuntimeHealthStatus.HEALTHY, "")
        )
    monkeypatch.setattr(service, "_behavior_checks", lambda *_args: [])
    monkeypatch.setattr(service, "_foresight_checks", lambda *_args: [])
    monkeypatch.setattr(service, "_queue_check", lambda: none_check())
    monkeypatch.setattr(service, "_vector_check", lambda *_args: none_check())
    monkeypatch.setenv("TZ", "Asia/Singapore")
    time.tzset()
    try:
        report = asyncio.run(service.report("running"))
        found = {item.name: item for item in report.checks}
        assert found["generation"].detail == "disabled"
        assert found["timezone"].status is RuntimeHealthStatus.DEGRADED and "Europe/London" in found["timezone"].detail
        service.zone = None
        assert "timezone" not in {item.name for item in asyncio.run(service.report("running")).checks}
    finally:
        monkeypatch.undo()
        time.tzset()
