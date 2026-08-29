from pathlib import Path

from photo_cull.files import discover_photo_pairs


def _touch(path: Path) -> None:
    path.write_bytes(b"x")


def test_basic_jpeg_raf_pair(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF1234.JPG")
    _touch(tmp_path / "DSCF1234.RAF")

    result = discover_photo_pairs(tmp_path)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.stem == "DSCF1234"
    assert pair.has_jpeg
    assert pair.has_raf
    assert result.warnings == []


def test_lowercase_extensions(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF0001.jpeg")
    _touch(tmp_path / "DSCF0001.raf")

    result = discover_photo_pairs(tmp_path)

    assert len(result.pairs) == 1
    assert result.pairs[0].has_jpeg
    assert result.pairs[0].has_raf


def test_mixed_case_extensions(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF0002.Jpg")
    _touch(tmp_path / "DSCF0002.Raf")

    result = discover_photo_pairs(tmp_path)

    assert len(result.pairs) == 1
    assert result.pairs[0].has_jpeg
    assert result.pairs[0].has_raf


def test_missing_raf(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF0003.JPG")

    result = discover_photo_pairs(tmp_path)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.has_jpeg
    assert not pair.has_raf
    assert result.warnings == []  # JPEG-only pairs don't generate a warning


def test_missing_jpeg(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF0004.RAF")

    result = discover_photo_pairs(tmp_path)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert not pair.has_jpeg
    assert pair.has_raf
    assert len(result.warnings) == 1
    assert "DSCF0004.RAF" in result.warnings[0]


def test_unrelated_files_ignored(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF0005.JPG")
    _touch(tmp_path / "DSCF0005.RAF")
    _touch(tmp_path / "readme.txt")
    _touch(tmp_path / ".DS_Store")

    result = discover_photo_pairs(tmp_path)

    assert len(result.pairs) == 1


def test_non_recursive_by_default(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF0006.JPG")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    _touch(subdir / "DSCF0007.JPG")

    result = discover_photo_pairs(tmp_path)

    assert len(result.pairs) == 1
    assert result.pairs[0].stem == "DSCF0006"


def test_recursive_scan(tmp_path: Path) -> None:
    _touch(tmp_path / "DSCF0006.JPG")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    _touch(subdir / "DSCF0007.JPG")

    result = discover_photo_pairs(tmp_path, recursive=True)

    stems = {p.stem for p in result.pairs}
    assert stems == {"DSCF0006", "DSCF0007"}
