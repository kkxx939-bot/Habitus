"""单用户本地服务的进程、预检和诊断边界。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "DoctorStatus",
    "ServiceInstanceLock",
    "ServiceInstanceLockError",
    "run_doctor",
    "run_doctor_from_env",
    "run_startup_preflight",
]


def __getattr__(name: str) -> Any:
    """插件安装 CLI 不加载配置和服务依赖；实际访问时再解析符号。"""

    if name in {
        "DoctorCheck",
        "DoctorReport",
        "DoctorStatus",
        "run_doctor",
        "run_doctor_from_env",
        "run_startup_preflight",
    }:
        from habitus.integrations.local_service import doctor

        return getattr(doctor, name)
    if name in {"ServiceInstanceLock", "ServiceInstanceLockError"}:
        from habitus.integrations.local_service import instance_lock

        return getattr(instance_lock, name)
    raise AttributeError(name)
