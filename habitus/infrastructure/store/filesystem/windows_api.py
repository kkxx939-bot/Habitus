"""Windows 文件系统原语：长路径、重解析点感知的打开、写穿式改名与"仅本人"DACL。

只在 Windows 上导入（下面的平台断言让 POSIX 上的类型检查整体跳过本模块）。全部用标准库 ``ctypes``
调 kernel32 / advapi32，不引入 pywin32：气隙或只有内部 PyPI 镜像的机器上也装得上。
"""

from __future__ import annotations

import ctypes
import errno
import msvcrt
import os
import sys
import time
from collections.abc import Callable
from ctypes import wintypes
from functools import cache
from pathlib import Path

assert sys.platform == "win32", "windows_api 只在 Windows 上导入"

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ_WRITE_DELETE = 0x00000001 | 0x00000002 | 0x00000004
_OPEN_EXISTING = 3
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_MOVEFILE_REPLACE_EXISTING = 0x00000001
_MOVEFILE_WRITE_THROUGH = 0x00000008
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_DACL_SECURITY_INFORMATION = 0x00000004
_SDDL_REVISION_1 = 1
_ERROR_INSUFFICIENT_BUFFER = 122
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_CREATE_NEW = 1
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
# 名字代理位：符号链接、目录联接、挂载点带它；云文件占位、去重不带（见 path_safety）。
_REPARSE_TAG_NAME_SURROGATE = 0x20000000
_FILE_ATTRIBUTE_TAG_INFO = 9
_ACL_SIZE_INFORMATION = 2
# DACL 里放权的 ACE：普通、对象、回调、回调对象；拒绝类只收权。认不出的类型一律当"放给了别人"。
_ALLOW_ACE_TYPES = frozenset({0, 5, 9, 11})
_DENY_ACE_TYPES = frozenset({1, 6, 10, 12})
_OBJECT_ACE_TYPES = frozenset({5, 11})

# 杀毒软件、索引器、另一个只读打开的进程会短暂占住目标文件，此时 MoveFileEx/DeleteFile 返回
# 拒绝访问或共享冲突。只对这两类错误做短暂、有界的重试；其余错误立刻上抛。
_TRANSIENT_WINERRORS = frozenset({5, 32, 33})
_TRANSIENT_RETRY_DELAYS_SECONDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)

_CreateFileW = _kernel32.CreateFileW
_CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_CreateFileW.restype = wintypes.HANDLE
_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL
_MoveFileExW = _kernel32.MoveFileExW
_MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
_MoveFileExW.restype = wintypes.BOOL
_GetCurrentProcess = _kernel32.GetCurrentProcess
_GetCurrentProcess.argtypes = []
_GetCurrentProcess.restype = wintypes.HANDLE
_LocalFree = _kernel32.LocalFree
_LocalFree.argtypes = [wintypes.HLOCAL]
_LocalFree.restype = wintypes.HLOCAL

_OpenProcessToken = _advapi32.OpenProcessToken
_OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
_OpenProcessToken.restype = wintypes.BOOL
_GetTokenInformation = _advapi32.GetTokenInformation
_GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_GetTokenInformation.restype = wintypes.BOOL
_ConvertSidToStringSidW = _advapi32.ConvertSidToStringSidW
_ConvertSidToStringSidW.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
_ConvertSidToStringSidW.restype = wintypes.BOOL
_ConvertStringSecurityDescriptorToSecurityDescriptorW = _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
_ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.ULONG),
]
_ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
_ConvertSecurityDescriptorToStringSecurityDescriptorW = _advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW
_ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPWSTR),
    ctypes.POINTER(wintypes.ULONG),
]
_ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = wintypes.BOOL
_SetFileSecurityW = _advapi32.SetFileSecurityW
_SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID]
_SetFileSecurityW.restype = wintypes.BOOL
_GetFileSecurityW = _advapi32.GetFileSecurityW
_GetFileSecurityW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_GetFileSecurityW.restype = wintypes.BOOL
_GetFileInformationByHandleEx = _kernel32.GetFileInformationByHandleEx
_GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
_GetFileInformationByHandleEx.restype = wintypes.BOOL
_GetSecurityDescriptorDacl = _advapi32.GetSecurityDescriptorDacl
_GetSecurityDescriptorDacl.argtypes = [
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.BOOL),
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.BOOL),
]
_GetSecurityDescriptorDacl.restype = wintypes.BOOL
_GetAclInformation = _advapi32.GetAclInformation
_GetAclInformation.argtypes = [wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.c_int]
_GetAclInformation.restype = wintypes.BOOL
_GetAce = _advapi32.GetAce
_GetAce.argtypes = [wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID)]
_GetAce.restype = wintypes.BOOL
_EqualSid = _advapi32.EqualSid
_EqualSid.argtypes = [wintypes.LPVOID, wintypes.LPVOID]
_EqualSid.restype = wintypes.BOOL
_CreateDirectoryW = _kernel32.CreateDirectoryW
_CreateDirectoryW.argtypes = [wintypes.LPCWSTR, wintypes.LPVOID]
_CreateDirectoryW.restype = wintypes.BOOL
_GetFileInformationByHandle = _kernel32.GetFileInformationByHandle
_GetFileInformationByHandle.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
_GetFileInformationByHandle.restype = wintypes.BOOL


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _FileAttributeTagInfo(ctypes.Structure):
    _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]


class _AclSizeInformation(ctypes.Structure):
    _fields_ = [("AceCount", wintypes.DWORD), ("AclBytesInUse", wintypes.DWORD), ("AclBytesFree", wintypes.DWORD)]


class _AceHeader(ctypes.Structure):
    _fields_ = [("AceType", wintypes.BYTE), ("AceFlags", wintypes.BYTE), ("AceSize", wintypes.WORD)]


def _last_error() -> OSError:
    return ctypes.WinError(ctypes.get_last_error())


def extended_path(path: str | os.PathLike[str]) -> str:
    """把绝对路径换成 ``\\\\?\\`` 扩展长度形式，越过 260 字符的 MAX_PATH。

    ``\\\\?\\`` 会关掉 Win32 的路径规范化：调用方必须先给出绝对、无 ``.``/``..``、各段不以空格或点结尾的
    路径（耐久写入层在拼路径前已经逐段拒绝），这里只做形式换算，不做任何清洗。
    """

    text = os.fspath(path)
    if text.startswith("\\\\?\\"):
        return text
    if not Path(text).is_absolute():
        raise ValueError("extended-length path requires an absolute path")
    text = text.replace("/", "\\")
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


def _require_handle(handle: int | None) -> int:
    if handle is None or handle == _INVALID_HANDLE_VALUE:
        raise _last_error()
    return handle


def _descriptor_from_handle(handle: int, flags: int) -> int:
    try:
        return msvcrt.open_osfhandle(handle, flags | os.O_BINARY)
    except BaseException:
        _CloseHandle(handle)
        raise


def _open_read_handle(path: str, *, follow: bool) -> int:
    flags = _FILE_FLAG_BACKUP_SEMANTICS | (0 if follow else _FILE_FLAG_OPEN_REPARSE_POINT)
    return _require_handle(
        _CreateFileW(path, _GENERIC_READ, _FILE_SHARE_READ_WRITE_DELETE, None, _OPEN_EXISTING, flags, None)
    )


def _attribute_tag(handle: int) -> tuple[int, int]:
    """句柄所指对象的属性与重解析标签（``os.fstat`` 在 Windows 上不报标签，这里直接问内核）。"""

    info = _FileAttributeTagInfo()
    if not _GetFileInformationByHandleEx(handle, _FILE_ATTRIBUTE_TAG_INFO, ctypes.byref(info), ctypes.sizeof(info)):
        raise _last_error()
    return int(info.FileAttributes), int(info.ReparseTag)


def open_regular_file_for_read(path: str) -> int:
    """打开一个文件读，返回 CRT 文件描述符；拒绝符号链接与目录联接。

    先以不跟随重解析点的句柄打开，向内核要真实标签：名字代理（符号链接、联接、挂载点）抛 ``ELOOP``；
    云文件占位、去重这类非名字代理重解析点按正常方式重开让过滤器给出真实内容，重开后核对卷序列号与文件 ID 仍是同一个文件。
    共享模式含 ``FILE_SHARE_DELETE``：读者开着句柄时写者仍能删除它；但覆盖式改名仍会被挡住，由写者有界重试。
    """

    handle = _open_read_handle(path, follow=False)
    try:
        attributes, tag = _attribute_tag(handle)
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            followed = _reopen_through_filter(path, handle, tag)
            _CloseHandle(handle)
            handle = followed
    except BaseException:
        _CloseHandle(handle)
        raise
    return _descriptor_from_handle(handle, os.O_RDONLY)


def _reopen_through_filter(path: str, unfollowed: int, tag: int) -> int:
    """非名字代理重解析点（云占位、去重）按正常方式重开，并确认重开落在同一个文件上；名字代理抛 ``ELOOP``。"""

    if tag & _REPARSE_TAG_NAME_SURROGATE:
        raise OSError(errno.ELOOP, "path is a symbolic link, junction or mount point", path)
    followed = _open_read_handle(path, follow=True)
    try:
        same = _file_identity(followed) == _file_identity(unfollowed)
    except BaseException:
        _CloseHandle(followed)
        raise
    if not same:
        _CloseHandle(followed)
        raise OSError(errno.ELOOP, "path changed while it was being opened", path)
    return followed


def _file_identity(handle: int) -> tuple[int, int, int]:
    """(卷序列号, 文件 ID 高, 文件 ID 低)：两次打开是否落在同一个文件上。"""

    info = _ByHandleFileInformation()
    if not _GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise _last_error()
    return int(info.dwVolumeSerialNumber), int(info.nFileIndexHigh), int(info.nFileIndexLow)


def create_owner_only_file(path: str) -> int:
    """排他新建一个文件（已存在抛 ``FileExistsError``），出生时就带仅本人 DACL，返回可写的 CRT 文件描述符。

    安全描述符随 ``CreateFileW`` 一起给出，文件从不以继承来的宽 DACL 存在，别人没有先打开句柄的机会。
    """

    descriptor = _security_descriptor(owner_only_sddl(directory=False))
    try:
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
        handle = _CreateFileW(
            path,
            _GENERIC_WRITE,
            _FILE_SHARE_READ,
            ctypes.addressof(attributes),
            _CREATE_NEW,
            _FILE_ATTRIBUTE_NORMAL,
            None,
        )
        error = ctypes.get_last_error()  # 先存下来：随后的 LocalFree 不保证保留 last-error
    finally:
        _LocalFree(ctypes.cast(descriptor, wintypes.HLOCAL))
    if handle is None or handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(error)
    return _descriptor_from_handle(handle, os.O_WRONLY)


def create_owner_only_directory(path: str) -> None:
    """新建一级目录（已存在抛 ``FileExistsError``），出生时就带仅本人且向下继承的 DACL。"""

    descriptor = _security_descriptor(owner_only_sddl(directory=True))
    try:
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
        created = _CreateDirectoryW(path, ctypes.addressof(attributes))
        error = ctypes.get_last_error()
    finally:
        _LocalFree(ctypes.cast(descriptor, wintypes.HLOCAL))
    if not created:
        raise ctypes.WinError(error)


def _retry_transient(operation: Callable[[], None]) -> None:
    attempt = 0
    while True:
        try:
            operation()
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in _TRANSIENT_WINERRORS:
                raise
            if attempt >= len(_TRANSIENT_RETRY_DELAYS_SECONDS):
                raise
            time.sleep(_TRANSIENT_RETRY_DELAYS_SECONDS[attempt])
            attempt += 1


def move_file_write_through(source: str, destination: str, *, replace: bool) -> None:
    """``MoveFileExW`` 加 ``MOVEFILE_WRITE_THROUGH``：改名落盘后才返回。

    ``replace`` 为假时目标已存在即抛 ``FileExistsError``（不覆盖的原子创建）。目标正被别的句柄打开时
    覆盖式改名会被拒（winerror 5），按瞬态错误有界重试。
    """

    flags = _MOVEFILE_WRITE_THROUGH | (_MOVEFILE_REPLACE_EXISTING if replace else 0)

    def move() -> None:
        if not _MoveFileExW(source, destination, flags):
            raise _last_error()

    _retry_transient(move)


def unlink_with_retry(path: str) -> None:
    _retry_transient(lambda: os.unlink(path))


def rmdir_with_retry(path: str) -> None:
    _retry_transient(lambda: os.rmdir(path))


@cache
def _current_user_token() -> ctypes.Array[ctypes.c_char]:
    """当前进程令牌的 ``TOKEN_USER`` 原始缓冲；缓存住，里面的 SID 指针随它一直有效。"""

    token = wintypes.HANDLE()
    if not _OpenProcessToken(_GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)):
        raise _last_error()
    try:
        needed = wintypes.DWORD()
        _GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not _GetTokenInformation(token, _TOKEN_USER, buffer, needed, ctypes.byref(needed)):
            raise _last_error()
        return buffer
    finally:
        _CloseHandle(token)


def _current_user_sid_pointer() -> int:
    # TOKEN_USER 的第一个字段是 SID_AND_ATTRIBUTES，其第一个字段是 PSID。
    return int(ctypes.cast(_current_user_token(), ctypes.POINTER(wintypes.LPVOID)).contents.value or 0)


@cache
def current_user_sid() -> str:
    """当前进程令牌里用户的 SID 字符串（``S-1-5-21-...``）。"""

    text = wintypes.LPWSTR()
    if not _ConvertSidToStringSidW(_current_user_sid_pointer(), ctypes.byref(text)):
        raise _last_error()
    try:
        value = text.value
    finally:
        _LocalFree(ctypes.cast(text, wintypes.HLOCAL))
    if not value:
        raise OSError("current user SID is empty")
    return value


def owner_only_sddl(*, directory: bool) -> str:
    """只给当前用户完全控制、并关闭继承（``P``）的 DACL；目录的这条 ACE 继承给其下新建的文件与子目录。"""

    inheritance = "OICI" if directory else ""
    return f"D:P(A;{inheritance};FA;;;{current_user_sid()})"


def _security_descriptor(sddl: str) -> int:
    """SDDL 换成自相关安全描述符（``LocalAlloc`` 分配，调用方 ``LocalFree``）。"""

    descriptor = wintypes.LPVOID()
    if not _ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, _SDDL_REVISION_1, ctypes.byref(descriptor), None
    ):
        raise _last_error()
    return int(descriptor.value or 0)


def set_dacl_from_sddl(path: str, sddl: str) -> None:
    descriptor = _security_descriptor(sddl)
    try:
        if not _SetFileSecurityW(path, _DACL_SECURITY_INFORMATION, descriptor):
            raise _last_error()
    finally:
        _LocalFree(ctypes.cast(descriptor, wintypes.HLOCAL))


def _file_dacl_descriptor(path: str) -> ctypes.Array[ctypes.c_char]:
    needed = wintypes.DWORD()
    _GetFileSecurityW(path, _DACL_SECURITY_INFORMATION, None, 0, ctypes.byref(needed))
    error = ctypes.get_last_error()
    if needed.value == 0:
        raise ctypes.WinError(error)
    buffer = ctypes.create_string_buffer(needed.value)
    if not _GetFileSecurityW(path, _DACL_SECURITY_INFORMATION, buffer, needed, ctypes.byref(needed)):
        raise _last_error()
    return buffer


def _ace_sid_address(ace: int, ace_type: int) -> int:
    """ACE 里 SID 的地址：普通/回调 ACE 紧跟在头（4 字节）和掩码（4 字节）后；对象 ACE 还有标志和可选的两个 GUID。"""

    if ace_type not in _OBJECT_ACE_TYPES:
        return ace + 8
    flags = wintypes.DWORD.from_address(ace + 8).value
    return ace + 12 + (16 if flags & 0x1 else 0) + (16 if flags & 0x2 else 0)


def dacl_grants_only_current_user(path: str) -> bool:
    """DACL 是否只把访问权限放给当前用户（逐条读二进制 ACE、按 ``EqualSid`` 比）。

    没有 DACL 或空指针 DACL 等于对所有人敞开，判假；认不出的 ACE 类型也判假。拒绝类 ACE 不放权，跳过。
    """

    descriptor = _file_dacl_descriptor(path)
    present, defaulted, acl = wintypes.BOOL(), wintypes.BOOL(), wintypes.LPVOID()
    if not _GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(acl), ctypes.byref(defaulted)):
        raise _last_error()
    if not present.value or not acl.value:
        return False
    size = _AclSizeInformation()
    if not _GetAclInformation(acl, ctypes.byref(size), ctypes.sizeof(size), _ACL_SIZE_INFORMATION):
        raise _last_error()
    user = _current_user_sid_pointer()
    for index in range(size.AceCount):
        ace = wintypes.LPVOID()
        if not _GetAce(acl, index, ctypes.byref(ace)) or not ace.value:
            raise _last_error()
        ace_type = int(_AceHeader.from_address(ace.value).AceType)
        if ace_type in _DENY_ACE_TYPES:
            continue
        if ace_type not in _ALLOW_ACE_TYPES or not _EqualSid(_ace_sid_address(ace.value, ace_type), user):
            return False
    return True


def dacl_sddl(path: str) -> str:
    """读回一个文件或目录的 DACL 文本，形如 ``D:P(A;;FA;;;S-1-5-21-...)``（只给人看与诊断用，不拿来判定）。"""

    buffer = _file_dacl_descriptor(path)
    text = wintypes.LPWSTR()
    if not _ConvertSecurityDescriptorToStringSecurityDescriptorW(
        buffer,
        _SDDL_REVISION_1,
        _DACL_SECURITY_INFORMATION,
        ctypes.byref(text),
        None,
    ):
        raise _last_error()
    try:
        return text.value or ""
    finally:
        _LocalFree(ctypes.cast(text, wintypes.HLOCAL))


__all__ = [
    "create_owner_only_directory",
    "create_owner_only_file",
    "current_user_sid",
    "dacl_grants_only_current_user",
    "dacl_sddl",
    "extended_path",
    "move_file_write_through",
    "open_regular_file_for_read",
    "owner_only_sddl",
    "rmdir_with_retry",
    "set_dacl_from_sddl",
    "unlink_with_retry",
]
