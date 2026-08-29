# photo-cull

A local, MLX-powered pre-edit culling tool for macOS/Apple Silicon. Points at a
directory of Fujifilm `JPEG` + `RAF` pairs, judges each photo's _editing
potential_ with a local vision-language model, resolves near-duplicate bursts,
and writes a 1-5 star `XMP:Rating` to a `.xmp` sidecar next to each `RAF`.

This tool is for **culling before editing** -- it judges whether a capture is
worth your time, not whether the out-of-camera JPEG looks polished.

## Requirements

- macOS on Apple Silicon
- Python 3.10+
- [ExifTool](https://exiftool.org/) on `PATH` (`brew install exiftool`) --
  required to read/write `XMP:Rating`
- `mlx` + `mlx-vlm` (Apple Silicon only) for real model inference:
  `pip install photo-cull[mlx]`
- Optional: [`mlx-embeddings`](https://pypi.org/project/mlx-embeddings/) for
  real SigLIP burst-similarity embeddings: `pip install photo-cull[embeddings]`.
  Without it, a much weaker perceptual-hash fallback is used (with a logged
  warning) so the tool still runs end-to-end.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mlx,embeddings,dev]"
```

## Usage

```bash
python photo_cull.py /path/to/photos
python photo_cull.py /path/to/photos --dry-run --verbose
python photo_cull.py /path/to/photos --model mlx-community/Qwen3-VL-8B-Instruct-4bit
```

Key options:

| Flag                                 | Meaning                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| `--model`                            | MLX vision-language model id (default `mlx-community/Qwen3-VL-8B-Instruct-4bit`) |
| `--dry-run`                          | Analyse and print results; write no metadata                                     |
| `--force`                            | Ignore cache and existing ratings; recompute everything                          |
| `--overwrite-ratings`                | Recompute ratings for photos that already have one                               |
| `--verbose`                          | Print component scores and model explanations                                    |
| `--output-json <path>`               | Analysis JSON path (default `<path>/.photo-cull.json`)                           |
| `--burst-max-gap <seconds>`          | Max capture-time gap for burst candidates (default 5)                            |
| `--burst-similarity-threshold <0-1>` | Min embedding cosine similarity for burst candidates (default 0.9)               |
| `--no-burst-analysis`                | Disable burst detection/comparison entirely                                      |
| `--recursive`                        | Scan subdirectories (default: off)                                               |

Exit code is non-zero if any photo failed analysis or metadata writing.

## How it works

```
File discovery -> JPEG/RAF pairing -> metadata inspection -> JPEG preparation
  -> individual VLM analysis -> image embeddings -> temporal+visual burst
  detection -> relative burst comparison -> small burst adjustment
  -> 0-100 final potential -> 1-5 star conversion -> XMP Rating
```

- Only the rendered JPEG is ever decoded/analysed; RAF bytes are never opened.
- Bursts are grouped by scanning frames in capture-time order and only chaining
  a frame onto the current burst if it is _both_ within `--burst-max-gap`
  seconds _and_ above the similarity threshold **relative to the immediately
  preceding frame** (not an all-pairs similarity graph). This supports gradual
  sequences (e.g. a subject slowly turning/smiling) while avoiding the "A
  resembles B, B resembles C, but A and C are unrelated" transitive-clustering
  trap.
- Burst comparison nudges scores by a small, bounded amount (`clear_winner +3`,
  `close_second +1`, `normal 0`, `weaker -2`, `redundant -4`), so the best frame
  in a mediocre burst still isn't promoted into a different quality tier -- it
  just becomes easier to pick out.
- Already-rated photos are skipped by default (unless `--force` or
  `--overwrite-ratings`) and are excluded from burst grouping in this first
  version -- keeping burst logic simple, at the cost of not being able to
  compare a newly-shot frame against an already-rated sibling from the same
  burst. This is a deliberate v1 simplification.
- Individual analysis and image embeddings are cached in
  `.photo-cull-cache.json` (directory-local), keyed by JPEG content hash +
  model/prompt version, so re-runs only pay for inference on new/changed photos.
- Metadata writes go through ExifTool with `-overwrite_original_in_place`,
  touching only `XMP:Rating` so any existing sidecar metadata (crops, develop
  settings, keywords, etc.) is preserved. Sidecars are created fresh if missing
  (ExifTool supports this natively for XMP).

## Architecture

```
photo_cull/
    cli.py              argument parsing, thin wrapper around pipeline.run()
    pipeline.py         end-to-end orchestration
    files.py            JPEG/RAF discovery + pairing
    metadata.py          XMP read/write via ExifTool
    image_processing.py  JPEG loading, EXIF orientation/time, resizing/cropping
    embeddings.py        image-embedding abstraction (SigLIP via mlx-embeddings + fallback)
    bursts.py            temporal+visual burst grouping, shortlist selection
    vision.py            VLM abstraction, prompts, JSON parsing/validation
    scoring.py           star boundaries, burst-tier adjustments
    cache.py             on-disk cache keyed by file hash + config
    models.py            shared dataclasses
```

## Tests

```bash
pytest
```

Covers pairing edge cases, star-rating boundaries, burst-adjustment math,
metadata read/write (ExifTool calls mocked), vision-model JSON
parsing/validation/error-handling, burst grouping/shortlisting, and end-to-end
pipeline behaviour (existing-rating skip, `--force`/ `--overwrite-ratings`,
`--dry-run`, and non-fatal per-photo failures) using an injectable fake
`VisionModel`.
