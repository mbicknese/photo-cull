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



# Baseline weights for the balanced average, spec section 24: moment and
# composition matter most, sharpness next, exposure least (it's the most
# forgiving of RAW recovery). This average alone is *not* the final
# answer -- see the ceiling functions below.
_BASE_WEIGHTS: dict[str, float] = {
    "composition": 0.30,
    "sharpness": 0.25,
    "moment": 0.30,
    "exposure": 0.15,
}

# sharpness/moment are nonfixable: real focus/motion blur and a missed
# decisive moment can't be recovered afterwards. Below this floor, a
# nonfixable dimension's ceiling tracks its own raw score 1:1, so a badly
# out-of-focus or moment-less shot is dragged down near that low value
# regardless of how good everything else is. Above the floor the ceiling
# opens up quickly, since *mild* softness shouldn't alone keep an
# otherwise-strong photo out of the top bands.
_NONFIXABLE_FLOOR = 50
_NONFIXABLE_SLOPE = 1.5

# composition/exposure are fixable: a crop/perspective pass or RAW
# exposure recovery can usually rescue them, so even a middling score
# here still leaves real headroom for a high final score.
_FIXABLE_BASE = 60
_FIXABLE_SLOPE = 0.4


def _nonfixable_ceiling(score: int) -> float:
    """Ceiling implied by a nonfixable (sharpness/moment) dimension."""
    if score < _NONFIXABLE_FLOOR:
        return score
    return min(100.0, _NONFIXABLE_FLOOR + (score - _NONFIXABLE_FLOOR) * _NONFIXABLE_SLOPE)


def _fixable_ceiling(score: int) -> float:
    """Ceiling implied by a fixable (composition/exposure) dimension."""
    return min(100.0, _FIXABLE_BASE + score * _FIXABLE_SLOPE)


def dimension_ceiling(individual: IndividualAnalysis) -> float:
    """The most `potential` can be, given the four dimension scores alone.

    Categories do not simply average out -- each one independently caps
    ("vetoes") how high `potential` can go, and the *lowest* cap wins:

    - sharpness and moment are nonfixable (blur and a missed moment
      can't be fixed in post), so a severe failure in either caps the
      whole photo near its own low value no matter how well everything
      else scored.
    - composition and exposure are fixable (crop, perspective, RAW
      exposure recovery), so even a weak score there still leaves
      headroom for a high final score -- a composition of 50 with
      everything else strong should still allow a 4-star result, not
      get dragged down to match its weakest number.

    Used both by `deterministic_potential` (the fallback formula) and by
    `reconcile_potential` (sanity-checking the model's own `potential`).
    """
    return min(
        _nonfixable_ceiling(individual.sharpness),
        _nonfixable_ceiling(individual.moment),
        _fixable_ceiling(individual.composition),
        _fixable_ceiling(individual.exposure),
    )


def deterministic_potential(individual: IndividualAnalysis) -> int:
    """Fallback potential score if the model's own `potential` is unusable.

    Not used when the model provides a valid `potential` value -- the VLM
    is asked for `potential` explicitly and `reconcile_potential` is used
    to sanity-check it instead. This exists only as a documented safety
    net (spec section 24).
    """
    weighted_avg = (
        individual.composition * _BASE_WEIGHTS["composition"]
        + individual.sharpness * _BASE_WEIGHTS["sharpness"]
        + individual.moment * _BASE_WEIGHTS["moment"]
        + individual.exposure * _BASE_WEIGHTS["exposure"]
    )
    score = min(weighted_avg, dimension_ceiling(individual))
    return clamp_score(round(score))


def reconcile_potential(individual: IndividualAnalysis) -> int:
    """Sanity-check the model's self-reported `potential` against its own
    dimension scores, and cap it if it's inconsistent.

    VLMs are asked to weigh nonfixable defects (sharpness/moment) far
    more harshly than fixable ones (composition/exposure), but numeric
    self-scoring is unreliable even with explicit prompt guidance: a
    model can describe a photo as "slightly out of focus" and still
    report a high `potential` that doesn't reflect that. This applies
    the same nonfixable/fixable ceiling used by `deterministic_potential`
    as a hard backstop, so a stated nonfixable defect always constrains
    the final score regardless of what `potential` the model returned.
    """
    return clamp_score(min(individual.potential, round(dimension_ceiling(individual))))
