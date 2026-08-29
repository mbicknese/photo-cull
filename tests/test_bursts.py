from datetime import datetime, timedelta

import numpy as np

from photo_cull.bursts import BurstCandidate, detect_bursts, shortlist_for_comparison


def _t(seconds_offset: int) -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0) + timedelta(seconds=seconds_offset)


def test_no_bursts_when_temporally_far_apart() -> None:
    similar = np.array([1.0, 0.0, 0.0])
    candidates = [
        BurstCandidate("A", _t(0), similar),
        BurstCandidate("B", _t(120), similar),
    ]
    groups = detect_bursts(candidates, max_gap_seconds=5, similarity_threshold=0.9)
    assert groups == []


def test_no_bursts_when_visually_dissimilar() -> None:
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    candidates = [
        BurstCandidate("A", _t(0), v1),
        BurstCandidate("B", _t(2), v2),
    ]
    groups = detect_bursts(candidates, max_gap_seconds=5, similarity_threshold=0.9)
    assert groups == []


def test_burst_formed_when_close_and_similar() -> None:
    v = np.array([1.0, 0.0, 0.0])
    candidates = [
        BurstCandidate("A", _t(0), v),
        BurstCandidate("B", _t(2), v),
        BurstCandidate("C", _t(4), v),
    ]
    groups = detect_bursts(candidates, max_gap_seconds=5, similarity_threshold=0.9)
    assert len(groups) == 1
    assert groups[0].stems == ["A", "B", "C"]


def test_single_image_is_not_a_burst() -> None:
    v = np.array([1.0, 0.0, 0.0])
    candidates = [BurstCandidate("A", _t(0), v)]
    groups = detect_bursts(candidates)
    assert groups == []


def test_gradual_sequence_chains_into_one_burst() -> None:
    # Each consecutive pair is similar even though first/last differ a lot,
    # e.g. "child turns head -> smiles -> looks away -> smiles at camera".
    vectors = [
        np.array([1.0, 0.0]),
        np.array([0.95, 0.05]),
        np.array([0.85, 0.15]),  # still >=0.9 sim to previous but drifting
    ]

    def sim_ok(a, b):
        na, nb = a / np.linalg.norm(a), b / np.linalg.norm(b)
        return np.dot(na, nb)

    candidates = [
        BurstCandidate("A", _t(0), vectors[0]),
        BurstCandidate("B", _t(2), vectors[1]),
        BurstCandidate("C", _t(4), vectors[2]),
    ]
    groups = detect_bursts(candidates, max_gap_seconds=5, similarity_threshold=0.9)
    assert len(groups) == 1
    assert groups[0].stems == ["A", "B", "C"]


def test_transitive_dissimilar_frames_do_not_merge_into_one_giant_cluster() -> None:
    # A~B similar, B~C similar, but A and C are wildly different. Because
    # grouping is chained by consecutive frames only, A/B/C still end up
    # in the same burst (a smooth continuous drift is intentional), but a
    # break in either time or similarity anywhere in the chain must split
    # the group -- verified here by forcing a large gap before D.
    v = np.array([1.0, 0.0])
    candidates = [
        BurstCandidate("A", _t(0), v),
        BurstCandidate("B", _t(2), v),
        BurstCandidate("D", _t(200), v),  # big time gap breaks the chain
    ]
    groups = detect_bursts(candidates, max_gap_seconds=5, similarity_threshold=0.9)
    assert len(groups) == 1
    assert groups[0].stems == ["A", "B"]


def test_missing_capture_time_or_embedding_excluded() -> None:
    v = np.array([1.0, 0.0])
    candidates = [
        BurstCandidate("A", _t(0), v),
        BurstCandidate("B", None, v),
        BurstCandidate("C", _t(2), None),
    ]
    groups = detect_bursts(candidates)
    assert groups == []


def test_configurable_max_gap() -> None:
    v = np.array([1.0, 0.0])
    candidates = [
        BurstCandidate("A", _t(0), v),
        BurstCandidate("B", _t(8), v),
    ]
    assert detect_bursts(candidates, max_gap_seconds=5) == []
    groups = detect_bursts(candidates, max_gap_seconds=10)
    assert len(groups) == 1


def test_shortlist_for_comparison_small_group_returns_all() -> None:
    stems = ["A", "B", "C"]
    potentials = {"A": 80, "B": 70, "C": 90}
    shortlist, remainder = shortlist_for_comparison(stems, potentials, max_count=6)
    assert shortlist == stems
    assert remainder == []


def test_shortlist_for_comparison_large_group_picks_top_by_potential() -> None:
    stems = ["A", "B", "C", "D"]
    potentials = {"A": 50, "B": 90, "C": 60, "D": 95}
    shortlist, remainder = shortlist_for_comparison(stems, potentials, max_count=2)
    assert set(shortlist) == {"B", "D"}
    assert set(remainder) == {"A", "C"}
    # Order preserved from the original temporal order.
    assert shortlist == ["B", "D"]
