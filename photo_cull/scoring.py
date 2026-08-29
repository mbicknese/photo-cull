"""Score-to-star conversion and burst-adjustment application (spec 12-13, 24)."""
from __future__ import annotations

from .models import BURST_TIER_ADJUSTMENTS, IndividualAnalysis

# Star boundaries (inclusive lower bound), spec section 13 & 26.
#   0-44  -> 1
#   45-61 -> 2
#   62-77 -> 3
#   78-89 -> 4
#   90-100 -> 5
_STAR_BOUNDARIES: list[tuple[int, int]] = [
    (90, 5),
    (78, 4),
    (62, 3),
    (45, 2),
    (0, 1),
]


def score_to_stars(score: int) -> int:
    """Convert a 0-100 final potential score into a 1-5 star rating."""
    for threshold, stars in _STAR_BOUNDARIES:
        if score >= threshold:
            return stars
    return 1


def clamp_score(score: int) -> int:
    return max(0, min(100, score))


def burst_adjustment_for_tier(tier: str) -> int:
    return BURST_TIER_ADJUSTMENTS.get(tier, 0)


def apply_burst_adjustment(potential: int, tier: str) -> int:
    """Apply a burst-comparison tier adjustment, clamped to 0-100."""
    return clamp_score(potential + burst_adjustment_for_tier(tier))


def deterministic_potential(individual: IndividualAnalysis) -> int:
    """Fallback potential score if the model's own `potential` is unusable.

    Not used when the model provides a valid `potential` value -- the VLM
    is asked for `potential` explicitly and that value is authoritative.
    This exists only as a documented, deliberately moment-weighted safety
    net (spec section 24).
    """
    score = (
        individual.composition * 0.30
        + individual.sharpness * 0.25
        + individual.moment * 0.30
        + individual.exposure * 0.15
    )
    return clamp_score(round(score))
