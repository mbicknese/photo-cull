"""XMP metadata handling via ExifTool (spec section 3, 27).

Design decisions:
  * We never modify the RAF's own image data or embedded metadata; the
    RAF file itself is opened read-only (or not at all).
  * Ratings are written to a sidecar `.xmp` file next to the RAF (or next
    to the JPEG if there is no RAF).
  * `-XMP:Rating=<n>` only touches the Rating tag; ExifTool merges this
    into the existing sidecar rather than replacing it, so any existing
    editing metadata (crops, develop settings, keywords, etc.) survives.
  * `-overwrite_original_in_place` is used so ExifTool performs an
    in-place, backup-free (and effectively atomic, write-temp-then-rename)
    update instead of leaving a `<file>_original` copy behind.
  * ExifTool can create a brand-new XMP file from nothing (XMP is one of
    the standalone metadata formats it supports natively), so the same
    code path handles "create sidecar" and "update sidecar".
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


class ExifToolNotFoundError(RuntimeError):
    """Raised when the `exiftool` executable cannot be located."""


class MetadataWriteError(RuntimeError):
    """Raised when ExifTool fails to write metadata."""


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def _require_exiftool() -> str:
    path = shutil.which("exiftool")
    if not path:
        raise ExifToolNotFoundError(
            "exiftool was not found on PATH. Install it (e.g. `brew install "
            "exiftool`) to read or write XMP ratings."
        )
    return path


def xmp_sidecar_path(target: Path) -> Path:
    """The `.xmp` sidecar path for a RAF (or JPEG-only) target file."""
    return target.with_suffix(".xmp")


def read_existing_rating(metadata_path: Path) -> Optional[int]:
    """Read `XMP:Rating` from a sidecar/image file, if present.

    Returns None if the file doesn't exist, has no rating, or ExifTool
    fails to read it (treated as "no rating" so analysis proceeds).
    """
    if not metadata_path.exists():
        return None

    exiftool = _require_exiftool()
    try:
        proc = subprocess.run(
            [exiftool, "-j", "-XMP:Rating", str(metadata_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ExifToolNotFoundError(str(exc)) from exc

    if proc.returncode != 0 or not proc.stdout.strip():
        return None

    import json

    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

    if not parsed:
        return None

    rating = parsed[0].get("Rating")
    if rating is None:
        return None
    try:
        return int(rating)
    except (TypeError, ValueError):
        return None


def write_rating(sidecar_path: Path, rating: int) -> None:
    """Write (create or update) `XMP:Rating` on a sidecar file.

    Only the Rating tag is touched; other existing tags in the sidecar
    are preserved. Never touches the RAF/JPEG image files themselves.
    """
    if not (1 <= rating <= 5):
        raise ValueError(f"rating must be 1-5, got {rating}")

    exiftool = _require_exiftool()
    try:
        proc = subprocess.run(
            [
                exiftool,
                "-overwrite_original_in_place",
                f"-XMP:Rating={rating}",
                str(sidecar_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ExifToolNotFoundError(str(exc)) from exc

    if proc.returncode != 0:
        raise MetadataWriteError(
            f"exiftool failed to write rating to {sidecar_path}: {proc.stderr.strip()}"
        )
