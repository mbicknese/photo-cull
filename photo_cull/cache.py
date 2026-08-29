"""On-disk cache for expensive per-image computations (spec section 18).

Keyed by (file hash, config key) so that:
  * if the JPEG bytes change, the cache entry is invalidated;
  * if scoring-relevant configuration changes (model name, prompt
    version, embedding backend), the cache entry is invalidated too,
    without needing to touch the file itself.

Stored as a single JSON file for simplicity -- this tool processes at
most a few thousand images per shoot, so a flat JSON document is more
than fast enough and is trivial to inspect/debug by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

DEFAULT_CACHE_FILENAME = ".photo-cull-cache.json"
CACHE_VERSION = 1


class AnalysisCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {"version": CACHE_VERSION, "entries": {}}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and loaded.get("version") == CACHE_VERSION:
                self._data = loaded
        except (json.JSONDecodeError, OSError):
            # Corrupt/unreadable cache: start fresh rather than failing the run.
            pass

    @staticmethod
    def make_key(file_hash: str, config_key: str) -> str:
        return f"{file_hash}:{config_key}"

    def get(self, file_hash: str, config_key: str) -> Optional[dict]:
        return self._data["entries"].get(self.make_key(file_hash, config_key))

    def set(self, file_hash: str, config_key: str, value: dict) -> None:
        self._data["entries"][self.make_key(file_hash, config_key)] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        tmp_path.replace(self.path)
        self._dirty = False
