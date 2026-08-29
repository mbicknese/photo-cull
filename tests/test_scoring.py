import pytest

from photo_cull.scoring import apply_burst_adjustment, clamp_score, score_to_stars


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
