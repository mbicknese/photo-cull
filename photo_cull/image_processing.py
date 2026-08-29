"""JPEG preparation utilities.

Only the rendered JPEG is ever used for visual analysis (spec section 5).
RAF data is never decoded here.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import ExifTags, Image, ImageOps

# Whole-image representation: large enough to preserve photographic detail
# (framing, expression, moment) without sending an unnecessarily huge image
# to the vision model.
DEFAULT_MAX_DIMENSION = 1280

# Centre crop used as a supplementary, higher-relative-resolution view for
# judging focus/sharpness on the main subject. This deliberately avoids
# aggressively shrinking the image before asking about sharpness.
DEFAULT_CROP_FRACTION = 0.55
DEFAULT_CROP_MAX_DIMENSION = 1024


def load_image(jpeg_path: Path) -> Image.Image:
    """Open a JPEG and normalise orientation.

    Uses `ImageOps.exif_transpose` so downstream consumers never need to
    reason about EXIF orientation tags. No aesthetic enhancement is
    applied -- this is a faithful decode of the camera/photographer's
    rendering.
    """
    with Image.open(jpeg_path) as img:
        img.load()
        transposed = ImageOps.exif_transpose(img)
        return transposed.convert("RGB")


def prepare_whole_image(img: Image.Image, max_dimension: int = DEFAULT_MAX_DIMENSION) -> Image.Image:
    """Return a resized copy suitable for composition/exposure/moment judging."""
    resized = img.copy()
    resized.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    return resized


def prepare_center_crop(
    img: Image.Image,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
    max_dimension: int = DEFAULT_CROP_MAX_DIMENSION,
) -> Image.Image:
    """Return a centre crop at a higher relative resolution for sharpness judging."""
    width, height = img.size
    crop_w = int(width * crop_fraction)
    crop_h = int(height * crop_fraction)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    cropped.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    return cropped


def prepare_representations(
    jpeg_path: Path,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
    crop_max_dimension: int = DEFAULT_CROP_MAX_DIMENSION,
) -> list[Image.Image]:
    """Build the set of image representations sent to the vision model."""
    img = load_image(jpeg_path)
    whole = prepare_whole_image(img, max_dimension=max_dimension)
    crop = prepare_center_crop(img, crop_fraction=crop_fraction, max_dimension=crop_max_dimension)
    return [whole, crop]


_DATE_TAG_NAMES = ("DateTimeOriginal", "DateTimeDigitized", "DateTime")
_EXIF_TAG_IDS = {v: k for k, v in ExifTags.TAGS.items()}


def get_capture_time(jpeg_path: Path) -> Optional[datetime]:
    """Best-effort read of the JPEG's EXIF capture timestamp."""
    try:
        with Image.open(jpeg_path) as img:
            exif = img.getexif()
            if not exif:
                return None
            for tag_name in _DATE_TAG_NAMES:
                tag_id = _EXIF_TAG_IDS.get(tag_name)
                if tag_id is None:
                    continue
                value = exif.get(tag_id)
                if not value:
                    continue
                try:
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    continue
    except Exception:
        return None
    return None


def compute_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 hash of a file's contents, used for cache invalidation."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
