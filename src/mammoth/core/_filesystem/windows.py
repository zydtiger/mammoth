"""Windows handle primitives for local artifacts and direct execution.

Directory handles deny delete sharing while paths are used by serializers.
File readers use binary CRT descriptors over no-follow Win32 handles. This
module is importable elsewhere, but its operations require Windows.
"""

from __future__ import annotations

import ctypes
import os
import stat
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_winapi: Any = ctypes
_kernel: Any = None
_crt: Any = None

if os.name == "nt":
    import msvcrt

    _crt = msvcrt
    _kernel = _winapi.WinDLL("kernel32", use_last_error=True)
    _kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _kernel.CreateFileW.restype = wintypes.HANDLE
    _kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel.CloseHandle.restype = wintypes.BOOL
    _kernel.GetFileInformationByHandle.argtypes = (wintypes.HANDLE, wintypes.LPVOID)
    _kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    _kernel.GetDriveTypeW.argtypes = (wintypes.LPCWSTR,)
    _kernel.GetDriveTypeW.restype = wintypes.UINT
    _kernel.SetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _kernel.SetFileInformationByHandle.restype = wintypes.BOOL


class _FileInformation(ctypes.Structure):
    _fields_ = [
        ("attributes", wintypes.DWORD),
        ("creation", wintypes.FILETIME),
        ("access", wintypes.FILETIME),
        ("write", wintypes.FILETIME),
        ("volume", wintypes.DWORD),
        ("size_high", wintypes.DWORD),
        ("size_low", wintypes.DWORD),
        ("links", wintypes.DWORD),
        ("index_high", wintypes.DWORD),
        ("index_low", wintypes.DWORD),
    ]


def _open_handle(path: Path, *, access: int, share: int, creation: int) -> int:
    if _kernel is None:
        raise NotImplementedError("Windows filesystem handles require Windows")
    handle = _kernel.CreateFileW(
        str(path),
        access,
        share,
        None,
        creation,
        0x00200000 | 0x02000000,
        None,  # OPEN_REPARSE_POINT | BACKUP_SEMANTICS
    )
    if handle == ctypes.c_void_p(-1).value:
        raise _winapi.WinError(_winapi.get_last_error())
    try:
        info = _FileInformation()
        if not _kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise _winapi.WinError(_winapi.get_last_error())
        if info.attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise ValueError(f"Windows reparse points are not supported: {path}")
    except BaseException:
        _kernel.CloseHandle(handle)
        raise
    return int(handle)


def open_file(
    path: Path,
    *,
    write: bool = False,
    create: bool = False,
    append: bool = False,
    share: int = 7,
) -> int:
    """Open a regular no-follow file as a non-inheritable binary descriptor."""
    handle = _open_handle(
        Path(path),
        access=0x80000000 | (0x40000000 if write else 0),
        share=share,
        creation=4 if create else 3,  # OPEN_ALWAYS / OPEN_EXISTING
    )
    try:
        flags = getattr(os, "O_BINARY", 0) | (os.O_RDWR if write else os.O_RDONLY)
        if append:
            flags |= os.O_APPEND
        descriptor = int(_crt.open_osfhandle(handle, flags))
    except BaseException:
        _kernel.CloseHandle(handle)
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact path must be a regular file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@dataclass
class PinnedDirectory:
    """Keep every ancestor non-renamable while a path-based writer uses it."""

    path: Path
    _handles: list[int] = field(default_factory=list, repr=False)

    def close(self) -> None:
        """Release pins in reverse traversal order, including on repeated close."""
        while self._handles:
            _kernel.CloseHandle(self._handles.pop())

    def __enter__(self) -> PinnedDirectory:
        """Return the pinned directory."""
        return self

    def __exit__(self, *args: object) -> None:
        """Release every handle."""
        self.close()


def pin_directory(path: Path, *, create: bool = False) -> PinnedDirectory:
    """Walk a local absolute path without reparse points and pin its ancestors."""
    absolute = Path(os.path.abspath(path))
    if _kernel is None:
        raise NotImplementedError("Windows filesystem handles require Windows")
    if absolute.drive.startswith("\\\\") or _kernel.GetDriveTypeW(absolute.anchor) == 4:
        raise ValueError("Windows training requires a local filesystem")
    pinned = PinnedDirectory(absolute)
    current = Path(absolute.anchor)
    try:
        for component in ("", *absolute.parts[1:]):
            current /= component
            if create and component:
                current.mkdir(exist_ok=True)
            handle = _open_handle(
                current,
                access=0x80,
                share=3,
                creation=3,  # READ_ATTRIBUTES, no DELETE sharing
            )
            pinned._handles.append(handle)
            if not current.is_dir():
                raise NotADirectoryError(str(current))
        return pinned
    except BaseException:
        pinned.close()
        raise


class _RenameInformation(ctypes.Structure):
    _fields_ = [
        ("flags", wintypes.DWORD),
        ("root", wintypes.HANDLE),
        ("length", wintypes.DWORD),
        ("name", wintypes.WCHAR * 1),
    ]


def replace_file(source: Path, destination: Path) -> None:
    """Atomically replace a target, including a retained read-only checkpoint.

    FileRenameInfoEx can ignore the target's read-only attribute without
    changing its permissions before commit. ACL checks and delete-sharing
    restrictions still apply; unsupported filesystems propagate their error.
    """
    try:
        os.replace(source, destination)
        return
    except PermissionError as error:
        try:
            info = destination.lstat()
        except FileNotFoundError:
            raise error from None
        if not stat.S_ISREG(info.st_mode) or not getattr(info, "st_file_attributes", 0) & 1:
            raise
    name = str(Path(os.path.abspath(destination))).encode("utf-16-le")
    size = max(ctypes.sizeof(_RenameInformation), _RenameInformation.name.offset + len(name) + 2)
    buffer = ctypes.create_string_buffer(size)
    rename = _RenameInformation.from_buffer(buffer)
    rename.flags = 0x1 | 0x2 | 0x40  # REPLACE_IF_EXISTS | POSIX_SEMANTICS | IGNORE_READONLY
    rename.root = None
    rename.length = len(name)
    ctypes.memmove(ctypes.addressof(buffer) + _RenameInformation.name.offset, name, len(name))
    handle = _open_handle(source, access=0x10000 | 0x80, share=7, creation=3)
    try:
        if not _kernel.SetFileInformationByHandle(handle, 22, buffer, size):
            raise _winapi.WinError(_winapi.get_last_error())
    finally:
        _kernel.CloseHandle(handle)
