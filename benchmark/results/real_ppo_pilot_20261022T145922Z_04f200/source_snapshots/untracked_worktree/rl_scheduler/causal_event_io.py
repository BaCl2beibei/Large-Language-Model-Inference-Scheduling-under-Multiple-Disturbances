"""Helpers for reading and losslessly archiving causal JSONL event streams."""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path
from typing import IO, Any


def event_stream_paths(event_dir: Path) -> list[Path]:
    """Return event streams, preferring a verified archive over its plain copy."""
    paths = []
    for path in sorted(event_dir.glob("events*.jsonl*")):
        if path.name.endswith(".tmp"):
            continue
        if path.name.endswith(".jsonl") and path.with_suffix(path.suffix + ".gz").exists():
            continue
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            paths.append(path)
    return paths


def open_event_stream(path: Path) -> IO[str]:
    """Open a plain or gzip-compressed causal JSONL event stream."""
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compress_event_stream(path: Path) -> dict[str, Any]:
    """Gzip a JSONL stream, verify its full decoded hash, then remove the source."""
    if path.name.endswith(".gz"):
        raise ValueError(f"event stream is already compressed: {path}")
    archive = path.with_suffix(path.suffix + ".gz")
    temporary = archive.with_name(archive.name + ".tmp")
    uncompressed_bytes = path.stat().st_size
    uncompressed_sha256 = _sha256(path)
    try:
        with path.open("rb") as source, temporary.open("wb") as target:
            with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=1, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
        digest = hashlib.sha256()
        decoded_bytes = 0
        with gzip.open(temporary, "rb") as restored:
            for block in iter(lambda: restored.read(1024 * 1024), b""):
                digest.update(block)
                decoded_bytes += len(block)
        if decoded_bytes != uncompressed_bytes or digest.hexdigest() != uncompressed_sha256:
            raise IOError(f"gzip verification failed for {path}")
        temporary.replace(archive)
        path.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "encoding": "gzip",
        "uncompressed_bytes": uncompressed_bytes,
        "uncompressed_sha256": uncompressed_sha256,
    }
