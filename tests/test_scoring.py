import pytest

from photo_cull.models import IndividualAnalysis
from photo_cull.scoring import (
    apply_burst_adjustment,
    clamp_score,
    deterministic_potential,
    reconcile_potential,
    score_to_stars,
)


def _individual(
    composition: int = 90,
    exposure: int = 90,
    sharpness: int = 90,
    moment: int = 90,
) -> IndividualAnalysis:
    return IndividualAnalysis(
        composition=composition,
        exposure=exposure,
        sharpness=sharpness,
        moment=moment,
        potential=0,
        confidence=100,
    )


@pytest.mark.parametrize(
    "score,expected_stars",
    [
        (0, 1),
        (44, 1),
        (45, 2),
        (61, 2),
        (62, 3),
        (77, 3),
        (78, 4),
        (89, 4),
        (90, 5),
        (100, 5),
    ],
)
def test_star_boundaries(score: int, expected_stars: int) -> None:
    assert score_to_stars(score) == expected_stars


def test_burst_adjustment_clear_winner_crosses_into_five_stars() -> None:
    adjusted = apply_burst_adjustment(89, "clear_winner")
    assert adjusted == 99
    assert score_to_stars(adjusted) == 5


def test_burst_adjustment_does_not_inappropriately_promote_weak_photo() -> None:
    adjusted = apply_burst_adjustment(43, "clear_winner")
    assert adjusted == 53
    assert score_to_stars(adjusted) == 2


def test_burst_adjustment_clamped_to_100() -> None:
    assert apply_burst_adjustment(99, "clear_winner") == 100


def test_burst_adjustment_clamped_to_0() -> None:
    assert apply_burst_adjustment(1, "redundant") == 0


def test_burst_adjustment_normal_is_zero() -> None:
    assert apply_burst_adjustment(70, "normal") == 70


def test_burst_adjustment_values() -> None:
    assert apply_burst_adjustment(80, "clear_winner") == 90
    assert apply_burst_adjustment(80, "close_second") == 83
    assert apply_burst_adjustment(80, "weaker") == 72
    assert apply_burst_adjustment(80, "redundant") == 62


def test_clamp_score() -> None:
    assert clamp_score(-5) == 0
    assert clamp_score(150) == 100
    assert clamp_score(50) == 50


def test_deterministic_potential_out_of_focus_caps_score_low_despite_other_strengths() -> None:
    # Nonfixable defect (sharpness) should veto an otherwise-excellent photo.
    individual = _individual(composition=95, exposure=95, sharpness=10, moment=95)
    score = deterministic_potential(individual)
    assert score <= 20
    assert score_to_stars(score) == 1


def test_deterministic_potential_weak_composition_still_reaches_four_stars() -> None:
    # Fixable defect (composition) shouldn't drag down an otherwise-strong photo.
    individual = _individual(composition=50, exposure=90, sharpness=90, moment=90)
    score = deterministic_potential(individual)
    assert score_to_stars(score) == 4


def test_deterministic_potential_weak_moment_caps_score_low_despite_other_strengths() -> None:
    individual = _individual(composition=95, exposure=95, sharpness=95, moment=10)
    score = deterministic_potential(individual)
    assert score <= 20
    assert score_to_stars(score) == 1


def test_deterministic_potential_weak_exposure_still_reaches_solid_tier() -> None:
    # Fixable defect (exposure) shouldn't be as punishing as a nonfixable one.
    individual = _individual(composition=95, exposure=10, sharpness=95, moment=95)
    score = deterministic_potential(individual)
    assert score_to_stars(score) >= 3


def test_deterministic_potential_uniform_mediocre_scores_matches_plain_average() -> None:
    individual = _individual(composition=70, exposure=70, sharpness=70, moment=70)
    assert deterministic_potential(individual) == 70


def test_deterministic_potential_all_excellent_scores_near_top() -> None:
    individual = _individual(composition=95, exposure=95, sharpness=95, moment=95)
    score = deterministic_potential(individual)
    assert score_to_stars(score) == 5


def test_reconcile_potential_caps_overconfident_model_score_on_soft_focus() -> None:
    # Regression test: the model reported potential=78 (4-star) with
    # sharpness=55 ("slightly out of focus") and listed nonfixable issues --
    # that mismatch should be caught and capped, not trusted verbatim.
    individual = IndividualAnalysis(
        composition=85,
        exposure=85,
        sharpness=55,
        moment=85,
        potential=78,
        confidence=90,
        nonfixable_issues=["slightly out of focus"],
    )
    reconciled = reconcile_potential(individual)
    assert reconciled < 78
    assert score_to_stars(reconciled) <= 3


def test_reconcile_potential_leaves_consistent_score_untouched() -> None:
    individual = _individual(composition=80, exposure=80, sharpness=80, moment=80)
    individual.potential = 80
    assert reconcile_potential(individual) == 80


def test_reconcile_potential_never_raises_a_conservative_model_score() -> None:
    # If the model itself is more conservative than the dimension ceiling
    # allows, that lower number should still win -- this only ever caps.
    individual = _individual(composition=95, exposure=95, sharpness=95, moment=95)
    individual.potential = 40
    assert reconcile_potential(individual) == 40
