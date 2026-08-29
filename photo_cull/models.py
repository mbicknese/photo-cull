"""Shared data models used across the photo_cull pipeline.

Keeping these in one module avoids circular imports between files.py,
vision.py, bursts.py, scoring.py, and cache.py, all of which need to
refer to the same small set of value objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


@dataclass
class PhotoPair:
    """A JPEG/RAF pairing identified by a shared filename stem."""

    stem: str
    jpeg_path: Optional[Path] = None
    raf_path: Optional[Path] = None

    @property
    def has_jpeg(self) -> bool:
        return self.jpeg_path is not None

    @property
    def has_raf(self) -> bool:
        return self.raf_path is not None

    @property
    def sidecar_target(self) -> Optional[Path]:
        """The file the XMP sidecar should be written next to.

        Prefer sitting the sidecar next to the RAF (the canonical
        photograph). If there is no RAF, fall back to the JPEG so the
        image can still receive a rating.
        """
        if self.raf_path is not None:
            return self.raf_path
        return self.jpeg_path


# ---------------------------------------------------------------------------
# Vision model outputs
# ---------------------------------------------------------------------------


@dataclass
class IndividualAnalysis:
    """Structured result of judging a single photograph in isolation."""

    composition: int
    exposure: int
    sharpness: int
    moment: int
    potential: int
    confidence: int
    primary_strength: str = ""
    primary_problem: str = ""
    fixable_issues: list[str] = field(default_factory=list)
    nonfixable_issues: list[str] = field(default_factory=list)
    explanation: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "composition": self.composition,
            "exposure": self.exposure,
            "sharpness": self.sharpness,
            "moment": self.moment,
            "potential": self.potential,
            "confidence": self.confidence,
            "primary_strength": self.primary_strength,
            "primary_problem": self.primary_problem,
            "fixable_issues": self.fixable_issues,
            "nonfixable_issues": self.nonfixable_issues,
            "explanation": self.explanation,
        }


# Burst comparison tiers, ordered strongest -> weakest. Values are the
# score adjustments applied on top of the individual `potential` score.
#
# These are deliberately large relative to the ~12-18 point width of each
# star band (see scoring.py): a "clear_winner" should usually pull a frame
# up a full star versus its burst-mates, and a "redundant"/near-duplicate
# frame should usually drop a full star or more, so that a burst rarely
# ends up with every frame at the same star rating.
BURST_TIER_ADJUSTMENTS: dict[str, int] = {
    "clear_winner": 10,
    "close_second": 3,
    "normal": 0,
    "weaker": -8,
    "redundant": -18,
}


@dataclass
class BurstComparisonEntry:
    """One photograph's outcome from a relative burst comparison."""

    stem: str
    rank: int
    tier: str  # one of BURST_TIER_ADJUSTMENTS keys
    notes: str = ""

    @property
    def adjustment(self) -> int:
        return BURST_TIER_ADJUSTMENTS.get(self.tier, 0)


@dataclass
class BurstGroup:
    """A temporally + visually linked sequence of near-duplicate frames."""

    id: str
    stems: list[str]
    comparison: Optional[list[BurstComparisonEntry]] = None

    @property
    def size(self) -> int:
        return len(self.stems)


# ---------------------------------------------------------------------------
# Per-photo result assembled by the pipeline
# ---------------------------------------------------------------------------


@dataclass
class PhotoResult:
    stem: str
    jpeg_path: Optional[Path]
    raf_path: Optional[Path]
    status: str = "ok"  # ok | skipped | failed
    error: Optional[str] = None
    capture_time: Optional[datetime] = None
    individual: Optional[IndividualAnalysis] = None
    burst_id: Optional[str] = None
    burst_rank: Optional[int] = None
    burst_size: Optional[int] = None
    burst_tier: Optional[str] = None
    burst_adjustment: int = 0
    final_score: Optional[int] = None
    rating: Optional[int] = None
    existing_rating: Optional[int] = None
    note: str = ""
    # Runtime-only field (not persisted in the JSON report): the image
    # embedding vector, used solely for in-memory burst similarity detection.
    embedding: Optional[list[float]] = None

    def to_json_dict(self) -> dict:
        data: dict = {
            "jpeg": self.jpeg_path.name if self.jpeg_path else None,
            "raw": self.raf_path.name if self.raf_path else None,
            "status": self.status,
        }
        if self.error:
            data["error"] = self.error
        if self.individual is not None:
            data["individual"] = self.individual.to_dict()
        if self.burst_id is not None:
            data["burst"] = {
                "id": self.burst_id,
                "rank": self.burst_rank,
                "size": self.burst_size,
                "tier": self.burst_tier,
                "adjustment": self.burst_adjustment,
            }
        if self.final_score is not None:
            data["final_score"] = self.final_score
        if self.rating is not None:
            data["rating"] = self.rating
        if self.existing_rating is not None:
            data["existing_rating"] = self.existing_rating
        if self.individual is not None:
            data["confidence"] = self.individual.confidence
        if self.note:
            data["note"] = self.note
        return data
