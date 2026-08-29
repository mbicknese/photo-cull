import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
from PIL import Image

from photo_cull.embeddings import PerceptualHashEmbeddingModel
from photo_cull.models import BurstComparisonEntry, IndividualAnalysis
from photo_cull.pipeline import RunOptions, run
from photo_cull.vision import VisionModel


def _make_gradient_jpeg(path: Path, orientation: str, size: int = 64, quality: int = 95) -> None:
    arr = np.zeros((size, size), dtype=np.uint8)
    if orientation == "horizontal":
        for x in range(size):
            arr[:, x] = int(x / size * 255)
    else:
        for y in range(size):
            arr[y, :] = int(y / size * 255)
    # `quality` only affects JPEG byte-level encoding (and therefore the
    # content hash used for caching), not the underlying pixel pattern,
    # so two "near-duplicate" test images can have distinct file hashes
    # while remaining visually/embedding-similar.
    Image.fromarray(arr).convert("RGB").save(path, "JPEG", quality=quality)


def _individual(potential: int) -> IndividualAnalysis:
    return IndividualAnalysis(
        composition=potential,
        exposure=potential,
        sharpness=potential,
        moment=potential,
        potential=potential,
        confidence=90,
        explanation="test",
    )


class FakeVisionModel(VisionModel):
    """Deterministic stand-in for the real MLX vision model in tests."""

    def __init__(self, individual_by_call: list[IndividualAnalysis], burst_tiers: Optional[dict] = None):
        self._queue = list(individual_by_call)
        self._burst_tiers = burst_tiers or {}
        self.individual_calls = 0
        self.burst_calls = 0

    def analyze_individual(self, images):
        self.individual_calls += 1
        return self._queue.pop(0)

    def compare_burst(self, members):
        self.burst_calls += 1
        entries = []
        for i, (stem, _img) in enumerate(members):
            tier = self._burst_tiers.get(stem, "normal")
            entries.append(BurstComparisonEntry(stem=stem, rank=i + 1, tier=tier))
        return entries


def _patch_capture_times(monkeypatch, times: dict):
    def fake(jpeg_path: Path):
        return times.get(jpeg_path.stem)

    monkeypatch.setattr("photo_cull.pipeline.get_capture_time", fake)


def test_burst_pair_gets_modest_adjustment(tmp_path: Path, monkeypatch) -> None:
    _make_gradient_jpeg(tmp_path / "DSCF1001.JPG", "horizontal", quality=95)
    _make_gradient_jpeg(tmp_path / "DSCF1002.JPG", "horizontal", quality=90)
    (tmp_path / "DSCF1001.RAF").write_bytes(b"raw")
    (tmp_path / "DSCF1002.RAF").write_bytes(b"raw")

    base = datetime(2024, 1, 1, 12, 0, 0)
    _patch_capture_times(monkeypatch, {"DSCF1001": base, "DSCF1002": base + timedelta(seconds=2)})

    vision = FakeVisionModel(
        [_individual(84), _individual(83)],
        burst_tiers={"DSCF1001": "clear_winner", "DSCF1002": "weaker"},
    )

    options = RunOptions(path=tmp_path, dry_run=True, no_burst_analysis=False)

    with patch("photo_cull.metadata.exiftool_available", return_value=True):
        exit_code = run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    assert exit_code == 0
    doc = json.loads((tmp_path / ".photo-cull.json").read_text())
    images = doc["images"]
    assert images["DSCF1001"]["burst"]["id"] == images["DSCF1002"]["burst"]["id"]
    assert images["DSCF1001"]["final_score"] == 94  # 84 + 10 (clear_winner)
    assert images["DSCF1001"]["rating"] == 5
    assert images["DSCF1002"]["final_score"] == 75  # 83 - 8 (weaker)
    assert images["DSCF1002"]["rating"] == 3
    assert vision.burst_calls == 1


def test_dissimilar_but_close_in_time_does_not_form_burst(tmp_path: Path, monkeypatch) -> None:
    _make_gradient_jpeg(tmp_path / "DSCF2001.JPG", "horizontal")
    _make_gradient_jpeg(tmp_path / "DSCF2002.JPG", "vertical")
    (tmp_path / "DSCF2001.RAF").write_bytes(b"raw")
    (tmp_path / "DSCF2002.RAF").write_bytes(b"raw")

    base = datetime(2024, 1, 1, 12, 0, 0)
    _patch_capture_times(monkeypatch, {"DSCF2001": base, "DSCF2002": base + timedelta(seconds=1)})

    vision = FakeVisionModel([_individual(70), _individual(70)])
    options = RunOptions(path=tmp_path, dry_run=True)

    with patch("photo_cull.metadata.exiftool_available", return_value=True):
        run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    doc = json.loads((tmp_path / ".photo-cull.json").read_text())
    assert "burst" not in doc["images"]["DSCF2001"]
    assert "burst" not in doc["images"]["DSCF2002"]
    assert doc["images"]["DSCF2001"]["final_score"] == 70
    assert vision.burst_calls == 0


def test_existing_rating_is_preserved_by_default(tmp_path: Path, monkeypatch) -> None:
    _make_gradient_jpeg(tmp_path / "DSCF3001.JPG", "horizontal")
    (tmp_path / "DSCF3001.RAF").write_bytes(b"raw")

    def raise_if_called(*args, **kwargs):
        raise AssertionError("vision model should not be called for already-rated photos")

    vision = FakeVisionModel([])
    vision.analyze_individual = raise_if_called  # type: ignore[assignment]

    options = RunOptions(path=tmp_path, dry_run=True)

    with patch("photo_cull.metadata.exiftool_available", return_value=True), patch(
        "photo_cull.metadata.read_existing_rating", return_value=4
    ):
        exit_code = run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    assert exit_code == 0
    doc = json.loads((tmp_path / ".photo-cull.json").read_text())
    assert doc["images"]["DSCF3001"]["status"] == "skipped"
    assert doc["images"]["DSCF3001"]["existing_rating"] == 4
    assert doc["images"]["DSCF3001"]["rating"] == 4


def test_overwrite_ratings_forces_recompute(tmp_path: Path, monkeypatch) -> None:
    _make_gradient_jpeg(tmp_path / "DSCF3002.JPG", "horizontal")
    (tmp_path / "DSCF3002.RAF").write_bytes(b"raw")

    vision = FakeVisionModel([_individual(95)])
    options = RunOptions(path=tmp_path, dry_run=True, overwrite_ratings=True)

    with patch("photo_cull.metadata.exiftool_available", return_value=True), patch(
        "photo_cull.metadata.read_existing_rating", return_value=2
    ):
        run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    assert vision.individual_calls == 1
    doc = json.loads((tmp_path / ".photo-cull.json").read_text())
    assert doc["images"]["DSCF3002"]["rating"] == 5


def test_dry_run_does_not_write_metadata(tmp_path: Path, monkeypatch) -> None:
    _make_gradient_jpeg(tmp_path / "DSCF4001.JPG", "horizontal")
    (tmp_path / "DSCF4001.RAF").write_bytes(b"raw")

    vision = FakeVisionModel([_individual(80)])
    options = RunOptions(path=tmp_path, dry_run=True)

    with patch("photo_cull.metadata.exiftool_available", return_value=True), patch(
        "photo_cull.metadata.read_existing_rating", return_value=None
    ), patch("photo_cull.metadata.write_rating") as mock_write:
        run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    mock_write.assert_not_called()


def test_writes_metadata_when_not_dry_run(tmp_path: Path, monkeypatch) -> None:
    _make_gradient_jpeg(tmp_path / "DSCF4002.JPG", "horizontal")
    (tmp_path / "DSCF4002.RAF").write_bytes(b"raw")

    vision = FakeVisionModel([_individual(80)])
    options = RunOptions(path=tmp_path, dry_run=False)

    with patch("photo_cull.metadata.exiftool_available", return_value=True), patch(
        "photo_cull.metadata.read_existing_rating", return_value=None
    ), patch("photo_cull.metadata.write_rating") as mock_write:
        run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    mock_write.assert_called_once()
    args, kwargs = mock_write.call_args
    assert args[0] == tmp_path / "DSCF4002.xmp"
    assert args[1] == 4  # score 80 -> 4 stars


def test_raf_without_jpeg_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "DSCF5001.RAF").write_bytes(b"raw")

    vision = FakeVisionModel([])
    options = RunOptions(path=tmp_path, dry_run=True)

    with patch("photo_cull.metadata.exiftool_available", return_value=True):
        exit_code = run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    assert exit_code == 0
    doc = json.loads((tmp_path / ".photo-cull.json").read_text())
    assert doc["images"]["DSCF5001"]["status"] == "skipped"


def test_vision_failure_does_not_abort_run_and_sets_exit_code(tmp_path: Path) -> None:
    _make_gradient_jpeg(tmp_path / "DSCF6001.JPG", "horizontal")
    _make_gradient_jpeg(tmp_path / "DSCF6002.JPG", "horizontal")
    (tmp_path / "DSCF6001.RAF").write_bytes(b"raw")
    (tmp_path / "DSCF6002.RAF").write_bytes(b"raw")

    from photo_cull.vision import VisionAnalysisError

    class FlakyVisionModel(VisionModel):
        def __init__(self):
            self.calls = 0

        def analyze_individual(self, images):
            self.calls += 1
            if self.calls == 1:
                raise VisionAnalysisError("model returned garbage")
            return _individual(80)

        def compare_burst(self, members):
            return []

    vision = FlakyVisionModel()
    options = RunOptions(path=tmp_path, dry_run=True)

    with patch("photo_cull.metadata.exiftool_available", return_value=True), patch(
        "photo_cull.metadata.read_existing_rating", return_value=None
    ):
        exit_code = run(options, vision_model=vision, embedding_model=PerceptualHashEmbeddingModel())

    assert exit_code == 1
    doc = json.loads((tmp_path / ".photo-cull.json").read_text())
    statuses = {stem: img["status"] for stem, img in doc["images"].items()}
    assert "failed" in statuses.values()
    assert "ok" in statuses.values()
