import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from photo_cull import metadata


def _fake_which(name: str):
    return "/usr/bin/exiftool" if name == "exiftool" else None


def test_exiftool_not_found_raises(tmp_path: Path) -> None:
    with patch("shutil.which", return_value=None):
        with pytest.raises(metadata.ExifToolNotFoundError):
            metadata.write_rating(tmp_path / "DSCF1234.xmp", 5)


def test_write_rating_invalid_value_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        metadata.write_rating(tmp_path / "DSCF1234.xmp", 0)
    with pytest.raises(ValueError):
        metadata.write_rating(tmp_path / "DSCF1234.xmp", 6)


def test_write_rating_new_sidecar(tmp_path: Path) -> None:
    xmp = tmp_path / "DSCF1234.xmp"
    with patch("shutil.which", side_effect=_fake_which), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        metadata.write_rating(xmp, 5)

    args = mock_run.call_args.args[0]
    assert args[0] == "/usr/bin/exiftool"
    assert "-overwrite_original_in_place" in args
    assert "-XMP:Rating=5" in args
    assert str(xmp) in args


def test_write_rating_failure_raises(tmp_path: Path) -> None:
    xmp = tmp_path / "DSCF1234.xmp"
    with patch("shutil.which", side_effect=_fake_which), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="some exiftool error"
        )
        with pytest.raises(metadata.MetadataWriteError):
            metadata.write_rating(xmp, 3)


def test_read_existing_rating_none_when_missing_file(tmp_path: Path) -> None:
    xmp = tmp_path / "DSCF9999.xmp"
    assert metadata.read_existing_rating(xmp) is None


def test_read_existing_rating_parses_value(tmp_path: Path) -> None:
    xmp = tmp_path / "DSCF1234.xmp"
    xmp.write_text("<xmp/>")  # existence is all that matters; exiftool call is mocked
    stdout = json.dumps([{"SourceFile": str(xmp), "Rating": 4}])
    with patch("shutil.which", side_effect=_fake_which), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        rating = metadata.read_existing_rating(xmp)

    assert rating == 4


def test_read_existing_rating_no_rating_field(tmp_path: Path) -> None:
    xmp = tmp_path / "DSCF1234.xmp"
    xmp.write_text("<xmp/>")
    stdout = json.dumps([{"SourceFile": str(xmp)}])
    with patch("shutil.which", side_effect=_fake_which), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        rating = metadata.read_existing_rating(xmp)

    assert rating is None


def test_xmp_sidecar_path() -> None:
    assert metadata.xmp_sidecar_path(Path("/photos/DSCF1234.RAF")) == Path("/photos/DSCF1234.xmp")
