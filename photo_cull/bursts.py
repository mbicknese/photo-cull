"""Burst / near-duplicate sequence detection (spec sections 9-10).

Two signals are required for two frames to be considered part of the
same burst:
  1. Temporal proximity  -- capture timestamps within `max_gap_seconds`.
  2. Visual similarity    -- cosine similarity of image embeddings above
                              `similarity_threshold`.

Grouping strategy
------------------
We deliberately do NOT build a similarity graph and take its connected
components. Doing so risks exactly the failure mode called out in the
spec: "image A resembles B and B resembles C but A and C clearly
represent different moments" producing one giant transitive cluster.

Instead, frames are sorted by capture time and scanned in order. A new
frame extends the current burst only if it is both temporally close
AND visually similar to the *immediately preceding* frame in time. This
naturally supports gradual sequences (e.g. "child turns head -> smiles
-> looks away -> smiles at camera", where each consecutive pair is
similar even though the first and last frame may differ substantially)
while still preventing unrelated look-alike photos taken minutes apart,
or photos taken quickly but of different subjects, from being merged.

A group of size 1 is not a burst and is reported as standalone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from .embeddings import cosine_similarity
from .models import BurstGroup

DEFAULT_MAX_GAP_SECONDS = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.9


@dataclass
class BurstCandidate:
    """Per-photo inputs needed for burst detection."""

    stem: str
    capture_time: Optional[datetime]
    embedding: Optional[np.ndarray]


def detect_bursts(
    candidates: list[BurstCandidate],
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[BurstGroup]:
    """Group candidates into bursts of size >= 2.

    Candidates missing a capture time or embedding cannot be linked to
    neighbours and are always returned as their own standalone group of
    size 1 (filtered out of the returned list, same as any other
    singleton chain).
    """
    usable = [c for c in candidates if c.capture_time is not None and c.embedding is not None]
    usable.sort(key=lambda c: c.capture_time)

    groups: list[list[str]] = []
    current: list[BurstCandidate] = []

    def flush() -> None:
        if len(current) >= 1:
            groups.append([c.stem for c in current])
        current.clear()

    previous: Optional[BurstCandidate] = None
    for cand in usable:
        if previous is None:
            current.append(cand)
        else:
            gap = (cand.capture_time - previous.capture_time).total_seconds()
            sim = cosine_similarity(previous.embedding, cand.embedding)
            if gap <= max_gap_seconds and sim >= similarity_threshold:
                current.append(cand)
            else:
                flush()
                current.append(cand)
        previous = cand
    flush()

    burst_groups: list[BurstGroup] = []
    counter = 0
    for stems in groups:
        if len(stems) < 2:
            continue  # a single image is not a burst
        counter += 1
        burst_groups.append(BurstGroup(id=f"burst-{counter:03d}", stems=stems))

    return burst_groups


def shortlist_for_comparison(
    ordered_stems: list[str], potentials: dict[str, int], max_count: int
) -> tuple[list[str], list[str]]:
    """Pick which burst members to send to the VLM for relative comparison.

    For bursts larger than `max_count`, the top `max_count` frames by
    individual `potential` are shortlisted (spec 11: "intelligently split
    or shortlist"). The shortlist preserves the original temporal order
    so the model can still reason about timing/sequence. Remaining
    frames are returned separately; the pipeline assigns them a neutral
    "normal" tier since they were not competitive enough to be worth the
    model's direct comparison.
    """
    if len(ordered_stems) <= max_count:
        return list(ordered_stems), []

    ranked = sorted(ordered_stems, key=lambda s: potentials.get(s, 0), reverse=True)
    shortlist_set = set(ranked[:max_count])
    shortlisted_ordered = [s for s in ordered_stems if s in shortlist_set]
    remainder_ordered = [s for s in ordered_stems if s not in shortlist_set]
    return shortlisted_ordered, remainder_ordered
