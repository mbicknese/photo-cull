"""File discovery: locate JPEG/RAF photo pairs in a directory.

Design notes (spec section 2):
  * Non-recursive by default.
  * JPEG extensions matched case-insensitively: .jpg / .jpeg
  * RAW extension matched case-insensitively: .raf
  * Pairing is by exact filename stem match only -- never by capture time.
  * A JPEG without a RAF is still usable (included, analysable).
  * A RAF without a JPEG cannot be visually scored; it is reported as a
    warning and skipped for scoring, but does not abort the run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import PhotoPair

JPEG_EXTENSIONS = {".jpg", ".jpeg"}
RAF_EXTENSIONS = {".raf"}


@dataclass
class DiscoveryResult:
    pairs: list[PhotoPair]
    warnings: list[str]


def discover_photo_pairs(directory: Path, recursive: bool = False) -> DiscoveryResult:
    """Scan `directory` and return matched JPEG/RAF photo pairs.

    Returns pairs sorted by stem for deterministic ordering. Also returns
    a list of human-readable warnings (e.g. RAF with no matching JPEG).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    iterator = directory.rglob("*") if recursive else directory.iterdir()

    jpeg_by_stem: dict[str, Path] = {}
    raf_by_stem: dict[str, Path] = {}

    for entry in iterator:
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix in JPEG_EXTENSIONS:
            jpeg_by_stem.setdefault(entry.stem, entry)
        elif suffix in RAF_EXTENSIONS:
            raf_by_stem.setdefault(entry.stem, entry)
        # Unrelated files are silently ignored.

    all_stems = sorted(set(jpeg_by_stem) | set(raf_by_stem))

    pairs: list[PhotoPair] = []
    warnings: list[str] = []

    for stem in all_stems:
        jpeg_path = jpeg_by_stem.get(stem)
        raf_path = raf_by_stem.get(stem)

        if raf_path is not None and jpeg_path is None:
            warnings.append(
                f"RAF without matching JPEG, skipping visual scoring: {raf_path.name}"
            )

        pairs.append(PhotoPair(stem=stem, jpeg_path=jpeg_path, raf_path=raf_path))

    return DiscoveryResult(pairs=pairs, warnings=warnings)
