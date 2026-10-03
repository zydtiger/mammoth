"""Seekable Zstandard storage for single-producer JSONL event streams.

Frames retain compression history across records and flush every published
record. A disposable sidecar indexes closed frames while the producer is live;
closed streams carry the standard Zstandard seek table in the file itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, Union

import zstandard as zstd

from mammoth.core._filesystem.io import open_event_file, read_at
from mammoth.core.artifacts import atomic_write_json

FRAME_BYTES = 256 * 1024
_SEEK_MAGIC = 0x8F92EAB1
_TABLE_MAGIC = 0x184D2A5E
_ENTRY = struct.Struct("<II")
_FOOTER = struct.Struct("<IBI")
_HEADER = struct.Struct("<II")
_GUARD_BYTES = 4096


def resolve_event_path(path: Path) -> Path:
    """Resolve one producer's old or compressed stream, rejecting ambiguous copies."""
    path = Path(path)
    other = Path(str(path)[:-4]) if path.name.endswith(".jsonl.zst") else Path(str(path) + ".zst")
    if path.exists() and other.exists():
        raise OSError(f"Both event stream formats exist: {path} and {other}")
    return other if not path.exists() and other.exists() else path


def _table(entries: list[tuple[int, int]]) -> bytes:
    body = b"".join(_ENTRY.pack(*entry) for entry in entries)
    body += _FOOTER.pack(len(entries), 0, _SEEK_MAGIC)
    return _HEADER.pack(_TABLE_MAGIC, len(body)) + body


def _read_table(descriptor: int, size: int) -> Union[list[tuple[int, int]], None]:
    if size < _HEADER.size + _FOOTER.size:
        return None
    footer = read_at(descriptor, _FOOTER.size, size - _FOOTER.size)
    if len(footer) != _FOOTER.size:
        raise OSError("event stream changed while reading seek table")
    count, flags, magic = _FOOTER.unpack(footer)
    if magic != _SEEK_MAGIC:
        return None
    if flags != 0:
        raise OSError("unsupported event seek table flags")
    length = _HEADER.size + count * _ENTRY.size + _FOOTER.size
    if length > size:
        raise OSError("invalid event seek table size")
    raw = read_at(descriptor, length, size - length)
    if len(raw) != length or _HEADER.unpack(raw[:8]) != (_TABLE_MAGIC, length - 8):
        raise OSError("invalid event seek table header")
    entries = list(_ENTRY.iter_unpack(raw[8:-9]))
    if any(csize == 0 for csize, _ in entries) or sum(c for c, _ in entries) != size - length:
        raise OSError("event seek table does not cover the stream")
    return entries


def _index_checksum(identity: list[int], entries: list[tuple[int, int]]) -> str:
    """Check cached metadata without replaying every indexed compression frame."""
    raw = json.dumps([identity, entries], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _live_entries(path: Path, file_stat: os.stat_result) -> Union[list[tuple[int, int]], None]:
    """Treat absent, stale, or invalid index caches as a sequential-read fallback."""
    try:
        descriptor = open_event_file(Path(str(path) + ".idx"))
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            payload = json.load(handle)
        if payload["identity"] != [file_stat.st_dev, file_stat.st_ino]:
            return None
        entries = payload["frames"]
        if not isinstance(entries, list):
            return None
        if any(
            not isinstance(entry, list)
            or len(entry) != 2
            or any(type(value) is not int for value in entry)
            or not 0 < entry[0] <= 0xFFFFFFFF
            or not 0 <= entry[1] <= 0xFFFFFFFF
            for entry in entries
        ):
            return None
        if sum(entry[0] for entry in entries) > file_stat.st_size:
            return None
        frames = [(entry[0], entry[1]) for entry in entries]
        if payload["checksum"] != _index_checksum(payload["identity"], frames):
            return None
        return frames
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return None


class ZstdEventWriter:
    """Append flushed records and a standard seek table without rewriting history."""

    def __init__(self, stream: BinaryIO, path: Path) -> None:
        self.stream = stream
        self.path = path
        size = os.fstat(stream.fileno()).st_size
        entries = _read_table(stream.fileno(), size)
        if size and entries is None:
            raise OSError("cannot append to an unsealed Zstandard event stream")
        self._entries = entries or []
        if size:
            # Old tables are legal skippable frames. Keeping them preserves the
            # byte offsets and append guards of already attached readers.
            self._entries.append((size - sum(c for c, _ in self._entries), 0))
        self._compressor: Union[zstd.ZstdCompressionObj, None] = None
        self._compressed = 0
        self._uncompressed = 0
        self._closed = False
        self._failed = False
        self._publish_index()

    def _publish_index(self) -> None:
        file_stat = os.fstat(self.stream.fileno())
        identity = [file_stat.st_dev, file_stat.st_ino]
        # This is only an accelerator: loss must never disable event logging.
        with suppress(OSError):
            atomic_write_json(
                Path(str(self.path) + ".idx"),
                {
                    "identity": identity,
                    "frames": self._entries,
                    "checksum": _index_checksum(identity, self._entries),
                },
            )

    def _append(self, data: bytes) -> None:
        try:
            if self.stream.write(data) != len(data):
                raise OSError("short compressed event append")
            self.stream.flush()
        except BaseException:
            self._failed = True
            raise

    def write(self, record: bytes) -> int:
        """Publish one complete record without ending its compression frame."""
        if self._closed or self._failed:
            raise OSError("compressed event stream is closed or failed")
        try:
            if self._compressor is None:
                self._compressor = zstd.ZstdCompressor(level=1, write_checksum=True).compressobj()
            data = self._compressor.compress(record)
            data += self._compressor.flush(zstd.COMPRESSOBJ_FLUSH_BLOCK)
            self._append(data)
            self._compressed += len(data)
            self._uncompressed += len(record)
            if self._uncompressed >= FRAME_BYTES:
                self._finish_frame()
            return len(record)
        except BaseException:
            self._failed = True
            raise

    def _finish_frame(self) -> None:
        if self._compressor is None:
            return
        data = self._compressor.flush(zstd.COMPRESSOBJ_FLUSH_FINISH)
        self._append(data)
        self._entries.append((self._compressed + len(data), self._uncompressed))
        self._compressor = None
        self._compressed = self._uncompressed = 0
        self._publish_index()

    def flush(self) -> None:
        """Flush the underlying descriptor; write already flushes compressed blocks."""
        self.stream.flush()

    def close(self) -> None:
        """Seal the stream, preserving readable data if finalization fails."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._failed:
                self._finish_frame()
                self._append(_table(self._entries))
        finally:
            self.stream.close()


class CompressedReadError(OSError):
    """Retain decoded records preceding damaged compressed bytes."""

    def __init__(self, detail: str, valid_bytes: bytes = b"") -> None:
        super().__init__(detail)
        self.valid_bytes = valid_bytes


class ZstdEventReader:
    """Incrementally decode one append-only stream, optionally starting near its tail."""

    def __init__(self, path: Path, *, tail_window_bytes: Union[int, None] = None) -> None:
        self.path = path
        self.tail_window_bytes = tail_window_bytes
        self.offset = 0
        self.skipped = False
        self._identity: Union[tuple[int, int], None] = None
        self._guard = b""
        self._decoder: Union[zstd.ZstdDecompressionObj, None] = None

    @property
    def identity(self) -> Union[tuple[int, int], None]:
        """Return the identity of the file already opened by this reader."""
        return self._identity

    def _start(self, descriptor: int, file_stat: os.stat_result) -> None:
        if self.tail_window_bytes is None:
            return
        try:
            entries = _read_table(descriptor, file_stat.st_size)
        except OSError:
            # Seek metadata is optional for readers; frame decoding remains
            # authoritative. Writer reopen validation stays strict.
            entries = None
        if entries is None:
            entries = _live_entries(self.path, file_stat)
        if not entries:
            return
        target = max(0, sum(u for _, u in entries) - self.tail_window_bytes)
        uncompressed = 0
        for compressed, length in entries:
            if length and uncompressed + length > target:
                break
            self.offset += compressed
            uncompressed += length
        self.skipped = uncompressed > 0

    def poll(self) -> bytes:
        """Decode only appended bytes, retaining state for an unfinished live frame."""
        descriptor = open_event_file(self.path)
        chunks: list[bytes] = []
        first = self._identity is None
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError(f"Event stream must be a regular file: {self.path}")
            identity = (file_stat.st_dev, file_stat.st_ino)
            if self._identity is not None and identity != self._identity:
                raise OSError("active stream file identity changed")
            if file_stat.st_size < self.offset:
                raise OSError(f"active stream was truncated from offset {self.offset}")
            if (
                self._guard
                and read_at(descriptor, len(self._guard), self.offset - len(self._guard))
                != self._guard
            ):
                raise OSError("consumed stream prefix changed")
            if first:
                self._start(descriptor, file_stat)
            self._identity = identity
            while self.offset < file_stat.st_size:
                data = read_at(
                    descriptor, min(64 * 1024, file_stat.st_size - self.offset), self.offset
                )
                if not data:
                    raise OSError("active stream changed while polling")
                self.offset += len(data)
                self._guard = (self._guard + data)[-_GUARD_BYTES:]
                while data:
                    if self._decoder is None:
                        self._decoder = zstd.ZstdDecompressor().decompressobj()
                    chunks.append(self._decoder.decompress(data))
                    if not self._decoder.eof:
                        break
                    data = self._decoder.unused_data
                    self._decoder = None
        except (OSError, zstd.ZstdError) as error:
            decoded = b"".join(chunks)
            if first:
                decoded = self._first_poll_window(decoded)
            raise CompressedReadError(str(error), decoded) from error
        finally:
            os.close(descriptor)
        decoded = b"".join(chunks)
        return self._first_poll_window(decoded) if first else decoded

    def _first_poll_window(self, decoded: bytes) -> bytes:
        """Apply identical tail semantics to clean data and a damaged stream's prefix."""
        if self.tail_window_bytes is not None and len(decoded) > self.tail_window_bytes:
            start = len(decoded) - self.tail_window_bytes
            if decoded[start - 1 : start] != b"\n":
                start = decoded.find(b"\n", start) + 1
            if start and b"\n" in decoded[start:]:
                self.skipped = True
                decoded = decoded[start:]
            else:
                # Preserve the plain reader's oversized-record fallback.
                self.offset = 0
                self.skipped = False
                self._identity = None
                self._guard = b""
                self._decoder = None
                self.tail_window_bytes = None
                return self.poll()
        return decoded
