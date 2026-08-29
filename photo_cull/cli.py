"""Command-line interface for photo_cull (spec section 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bursts import DEFAULT_MAX_GAP_SECONDS, DEFAULT_SIMILARITY_THRESHOLD
from .pipeline import RunOptions, run

DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo_cull.py",
        description=(
            "Assign 1-5 star XMP ratings to Fujifilm JPEG+RAF photo pairs "
            "based on pre-edit photographic potential, using a local MLX "
            "vision-language model."
        ),
    )
    parser.add_argument("path", type=Path, help="Directory containing JPEG/RAF photo pairs")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MLX vision-language model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively (default: off)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and print results but do not write any metadata",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute everything, ignoring the cache and existing ratings",
    )
    parser.add_argument(
        "--overwrite-ratings",
        action="store_true",
        help="Recompute ratings for photographs that already have an XMP rating",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print component scores and model explanations",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write the machine-readable analysis JSON (default: <path>/.photo-cull.json)",
    )
    parser.add_argument(
        "--burst-max-gap",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
        help=f"Maximum seconds between frames to be considered same-burst candidates (default: {DEFAULT_MAX_GAP_SECONDS})",
    )
    parser.add_argument(
        "--burst-similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Minimum cosine similarity for same-burst candidates (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--no-burst-analysis",
        action="store_true",
        help="Disable burst detection and relative burst comparison entirely",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    options = RunOptions(
        path=args.path,
        model=args.model,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
        output_json=args.output_json,
        burst_max_gap=args.burst_max_gap,
        burst_similarity_threshold=args.burst_similarity_threshold,
        no_burst_analysis=args.no_burst_analysis,
        overwrite_ratings=args.overwrite_ratings,
        recursive=args.recursive,
    )

    if not options.path.is_dir():
        print(f"Error: not a directory: {options.path}", file=sys.stderr)
        return 2

    return run(options)


if __name__ == "__main__":
    sys.exit(main())
