"""基准测量用的进程与机器内存读数（平台相关的部分都收在这里）。

POSIX 用 ``resource`` / ``/proc`` / ``ps`` / ``sysconf``；Windows 没有这些，用标准库 ctypes 调
``GetProcessMemoryInfo`` 读工作集，物理内存总量读不到时明确返回 ``None``。基准是 eval 工具、不进产品运行时。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":

    def _windows_working_set_bytes(*, peak: bool) -> int:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        # 声明原型：不声明时 ctypes 按 32 位 long 传参，64 位伪句柄 (HANDLE)-1 会转换失败。
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not kernel32.K32GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize if peak else counters.WorkingSetSize)


def process_max_rss_bytes() -> int:
    """进程峰值常驻内存（字节）：把 Unix 平台不一致的 ru_maxrss 单位统一成字节；Windows 取峰值工作集。"""

    if sys.platform == "win32":
        return _windows_working_set_bytes(peak=True)
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def process_rss_bytes() -> int:
    """读取当前进程 RSS；不可用时退回平台最大 RSS。Windows 取当前工作集。"""

    if sys.platform == "win32":
        return _windows_working_set_bytes(peak=False)
    if sys.platform.startswith("linux"):
        try:
            pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(page_size, int):
                return pages * page_size
        except (OSError, ValueError, IndexError):
            pass
    try:
        result = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(os.getpid())),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip()) * 1_024
    except (OSError, ValueError, subprocess.SubprocessError):
        return process_max_rss_bytes()


def physical_memory_bytes() -> int | None:
    """机器物理内存总量；Windows 上不读（基准只把它当环境说明），明确返回 ``None``。"""

    if sys.platform == "win32":
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * page_count if isinstance(page_size, int) and isinstance(page_count, int) else None


__all__ = ["physical_memory_bytes", "process_max_rss_bytes", "process_rss_bytes"]
