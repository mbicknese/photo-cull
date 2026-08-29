"""Pipeline orchestration: wires together discovery, metadata, vision,
embeddings, bursts, and scoring into the end-to-end culling workflow
described in spec section 23.

`cli.py` stays a thin argument-parsing wrapper around `run(args)`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import metadata
from .bursts import (
    BurstCandidate,
    detect_bursts,
    shortlist_for_comparison,
)
from .cache import AnalysisCache, DEFAULT_CACHE_FILENAME
from .embeddings import EmbeddingModel, get_default_embedding_model
from .files import discover_photo_pairs
from .image_processing import (
    compute_file_hash,
    get_capture_time,
    prepare_representations,
)
from .models import BurstComparisonEntry, IndividualAnalysis, PhotoResult
from .scoring import apply_burst_adjustment, clamp_score, score_to_stars
from .vision import (
    MLXVisionModel,
    PROMPT_VERSION,
    VisionAnalysisError,
    VisionModel,
)

logger = logging.getLogger("photo_cull")

MAX_BURST_COMPARE = 6
DEFAULT_OUTPUT_JSON_NAME = ".photo-cull.json"
ANALYSIS_VERSION = 1

STAR_GLYPHS = {5: "★★★★★", 4: "★★★★", 3: "★★★", 2: "★★", 1: "★"}


@dataclass
class RunOptions:
    path: Path
    model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    dry_run: bool = False
    force: bool = False
    verbose: bool = False
    output_json: Optional[Path] = None
    burst_max_gap: float = 5.0
    burst_similarity_threshold: float = 0.9
    no_burst_analysis: bool = False
    overwrite_ratings: bool = False
    recursive: bool = False


@dataclass
class RunStats:
    analyzed: int = 0
    skipped_existing: int = 0
    failures: int = 0
    burst_groups: int = 0
    standalone: int = 0


def _config_key(model_name: str) -> str:
    return f"individual:{model_name}:v{PROMPT_VERSION}"


def _embedding_config_key(embedding_model: EmbeddingModel) -> str:
    return f"embedding:{embedding_model.name}"


def _individual_from_cache(cached: dict) -> IndividualAnalysis:
    return IndividualAnalysis(
        composition=cached["composition"],
        exposure=cached["exposure"],
        sharpness=cached["sharpness"],
        moment=cached["moment"],
        potential=cached["potential"],
        confidence=cached["confidence"],
        primary_strength=cached.get("primary_strength", ""),
        primary_problem=cached.get("primary_problem", ""),
        fixable_issues=cached.get("fixable_issues", []),
        nonfixable_issues=cached.get("nonfixable_issues", []),
        explanation=cached.get("explanation", ""),
        raw=cached,
    )


def run(
    options: RunOptions,
    vision_model: Optional[VisionModel] = None,
    embedding_model: Optional[EmbeddingModel] = None,
) -> int:
    """Execute the full culling pipeline. Returns a process exit code.

    `vision_model`/`embedding_model` can be injected (e.g. in tests) to
    avoid loading real MLX models; production callers normally leave
    these as None so the configured defaults are constructed.
    """
    logging.basicConfig(
        level=logging.DEBUG if options.verbose else logging.INFO,
        format="%(message)s",
    )

    directory = Path(options.path)
    discovery = discover_photo_pairs(directory, recursive=options.recursive)
    for warning in discovery.warnings:
        logger.warning("Warning: %s", warning)

    if not metadata.exiftool_available():
        logger.warning(
            "Warning: exiftool not found on PATH. Existing ratings cannot be "
            "read and no ratings will be written. Install it with "
            "`brew install exiftool`."
        )

    cache = AnalysisCache(directory / DEFAULT_CACHE_FILENAME)
    output_json_path = Path(options.output_json) if options.output_json else directory / DEFAULT_OUTPUT_JSON_NAME

    if vision_model is None:
        vision_model = MLXVisionModel(options.model)
    if embedding_model is None and not options.no_burst_analysis:
        try:
            embedding_model = get_default_embedding_model()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Burst similarity detection disabled: %s", exc)

    results: dict[str, PhotoResult] = {}
    stats = RunStats()

    exiftool_ok = metadata.exiftool_available()

    total_pairs = len(discovery.pairs)
    for idx, pair in enumerate(discovery.pairs, start=1):
        progress = f"[{idx}/{total_pairs}] {pair.stem}"
        result = PhotoResult(stem=pair.stem, jpeg_path=pair.jpeg_path, raf_path=pair.raf_path)
        results[pair.stem] = result

        if not pair.has_jpeg:
            result.status = "skipped"
            result.note = "no JPEG available for visual scoring"
            print(f"{progress}: skipped (no JPEG)", flush=True)
            continue

        sidecar_target = pair.sidecar_target
        existing_rating = None
        if exiftool_ok and sidecar_target is not None:
            xmp_path = metadata.xmp_sidecar_path(sidecar_target)
            try:
                existing_rating = metadata.read_existing_rating(xmp_path)
            except metadata.ExifToolNotFoundError as exc:
                logger.warning("Could not read existing rating for %s: %s", pair.stem, exc)

        result.existing_rating = existing_rating

        should_recompute = options.force or options.overwrite_ratings
        if existing_rating is not None and not should_recompute:
            result.status = "skipped"
            result.rating = existing_rating
            result.note = "existing rating preserved"
            print(f"{progress}: skipped (existing rating {existing_rating})", flush=True)
            continue

        # --- Individual analysis (cached by JPEG content hash) ---
        try:
            file_hash = compute_file_hash(pair.jpeg_path)
        except OSError as exc:
            result.status = "failed"
            result.error = f"could not read JPEG: {exc}"
            logger.warning("Failed %s: %s", pair.stem, result.error)
            print(f"{progress}: failed ({result.error})", flush=True)
            continue

        result.capture_time = get_capture_time(pair.jpeg_path)

        cache_key = _config_key(options.model)
        cached_entry = None if options.force else cache.get(file_hash, cache_key)

        if cached_entry is not None:
            result.individual = _individual_from_cache(cached_entry)
            cache_note = " (cached)"
        else:
            print(f"{progress}: analyzing...", flush=True)
            try:
                images = prepare_representations(pair.jpeg_path)
                individual = vision_model.analyze_individual(images)
            except (VisionAnalysisError, ImportError, OSError) as exc:
                result.status = "failed"
                result.error = f"vision analysis failed: {exc}"
                logger.warning("Failed %s: %s", pair.stem, result.error)
                print(f"{progress}: failed ({result.error})", flush=True)
                continue
            result.individual = individual
            cache.set(file_hash, cache_key, individual.to_dict())
            cache_note = ""

        # --- Embedding (cached separately; independent of prompt/model config) ---
        if embedding_model is not None:
            embed_key = _embedding_config_key(embedding_model)
            cached_embedding = cache.get(file_hash, embed_key)
            if cached_embedding is not None:
                result_embedding = cached_embedding.get("vector")
            else:
                try:
                    whole_image = prepare_representations(pair.jpeg_path)[0]
                    vector = embedding_model.embed(whole_image)
                    result_embedding = vector.tolist()
                    cache.set(file_hash, embed_key, {"vector": result_embedding})
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Embedding failed for %s: %s", pair.stem, exc)
                    result_embedding = None
            result.embedding = result_embedding

        result.status = "ok"
        print(f"{progress}: potential={result.individual.potential}{cache_note}", flush=True)

    cache.save()

    # --- Burst detection ---
    burst_eligible = [r for r in results.values() if r.status == "ok" and r.individual is not None]
    candidates = [
        BurstCandidate(
            stem=r.stem,
            capture_time=r.capture_time,
            embedding=(np.array(r.embedding) if r.embedding is not None else None),
        )
        for r in burst_eligible
    ]

    burst_groups = []
    if not options.no_burst_analysis and candidates:
        burst_groups = detect_bursts(
            candidates,
            max_gap_seconds=options.burst_max_gap,
            similarity_threshold=options.burst_similarity_threshold,
        )

    stats.burst_groups = len(burst_groups)
    grouped_stems: set[str] = set()

    for group in burst_groups:
        grouped_stems.update(group.stems)
        potentials = {s: results[s].individual.potential for s in group.stems}
        shortlisted, remainder = shortlist_for_comparison(group.stems, potentials, MAX_BURST_COMPARE)

        try:
            members = [
                (stem, prepare_representations(results[stem].jpeg_path)[0])
                for stem in shortlisted
            ]
            entries = vision_model.compare_burst(members)
        except (VisionAnalysisError, ImportError, OSError) as exc:
            logger.warning("Burst comparison failed for %s: %s", group.id, exc)
            entries = [
                BurstComparisonEntry(stem=s, rank=i + 1, tier="normal")
                for i, s in enumerate(shortlisted)
            ]

        entry_by_stem = {e.stem: e for e in entries}
        next_rank = max((e.rank for e in entries), default=0) + 1

        for stem in group.stems:
            r = results[stem]
            r.burst_id = group.id
            r.burst_size = group.size
            entry = entry_by_stem.get(stem)
            if entry is not None:
                r.burst_rank = entry.rank
                r.burst_tier = entry.tier
            else:
                r.burst_rank = next_rank
                r.burst_tier = "normal"
                next_rank += 1
            r.burst_adjustment = apply_burst_adjustment(r.individual.potential, r.burst_tier) - r.individual.potential
            r.final_score = clamp_score(r.individual.potential + r.burst_adjustment)
            r.rating = score_to_stars(r.final_score)

    for r in burst_eligible:
        if r.stem in grouped_stems:
            continue
        stats.standalone += 1
        r.final_score = clamp_score(r.individual.potential)
        r.rating = score_to_stars(r.final_score)

    # --- Metadata writes ---
    if not options.dry_run:
        for pair in discovery.pairs:
            r = results[pair.stem]
            if r.status != "ok" or r.rating is None:
                continue
            sidecar_target = pair.sidecar_target
            if sidecar_target is None:
                continue
            if not exiftool_ok:
                r.status = "failed"
                r.error = "exiftool not available; rating not written"
                logger.warning("Failed %s: %s", pair.stem, r.error)
                continue
            xmp_path = metadata.xmp_sidecar_path(sidecar_target)
            try:
                metadata.write_rating(xmp_path, r.rating)
            except (metadata.ExifToolNotFoundError, metadata.MetadataWriteError) as exc:
                r.status = "failed"
                r.error = f"metadata write failed: {exc}"
                logger.warning("Failed %s: %s", pair.stem, r.error)

    stats.failures = sum(1 for r in results.values() if r.status == "failed")
    stats.skipped_existing = sum(
        1 for r in results.values() if r.status == "skipped" and r.existing_rating is not None
    )
    stats.analyzed = sum(1 for r in results.values() if r.status == "ok")

    # --- Persist analysis JSON ---
    output_doc = {
        "version": ANALYSIS_VERSION,
        "model": options.model,
        "images": {stem: r.to_json_dict() for stem, r in results.items()},
    }
    tmp_path = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2, default=str)
    tmp_path.replace(output_json_path)

    _print_report(results, burst_groups, stats, options)

    return 1 if stats.failures > 0 else 0


def _print_report(results, burst_groups, stats: RunStats, options: RunOptions) -> None:
    if options.dry_run or options.verbose:
        grouped_stems = {s for g in burst_groups for s in g.stems}
        for group in burst_groups:
            print(f"\nBurst {group.id}:")
            ordered = sorted(
                group.stems, key=lambda s: (results[s].burst_rank if results[s].burst_rank is not None else 0)
            )
            best_rank = min(
                (results[s].burst_rank for s in group.stems if results[s].burst_rank is not None),
                default=None,
            )
            for stem in ordered:
                r = results[stem]
                marker = "  BEST" if r.burst_rank == best_rank else ""
                stars = STAR_GLYPHS.get(r.rating or 1, "")
                print(f"{stars:6s} {stem}  {r.final_score}{marker}")
                if options.verbose and r.individual is not None:
                    print(f"       {r.individual.explanation}")

        standalone = [
            r
            for r in results.values()
            if r.status == "ok" and r.stem not in grouped_stems
        ]
        if standalone:
            print("\nStandalone:")
            for r in standalone:
                stars = STAR_GLYPHS.get(r.rating or 1, "")
                note = f"  {r.note}" if r.note else ""
                print(f"{stars:6s} {r.stem}  {r.final_score}{note}")
                if options.verbose and r.individual is not None:
                    print(f"       {r.individual.explanation}")

    total_rated = stats.analyzed
    star_counts = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for r in results.values():
        if r.status == "ok" and r.rating is not None:
            star_counts[r.rating] += 1

    print(f"\nAnalysed: {total_rated} photographs\n")
    for stars in (5, 4, 3, 2, 1):
        print(f"{STAR_GLYPHS[stars]:6s} {star_counts[stars]}")

    print(f"\nBurst groups: {stats.burst_groups}")
    print(f"Standalone photos: {stats.standalone}")
    print(f"Skipped existing ratings: {stats.skipped_existing}")
    print(f"Failures: {stats.failures}")

    failed = [r for r in results.values() if r.status == "failed"]
    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  {r.stem}: {r.error}")

    if options.dry_run:
        print("\nDry run: no metadata was written.")
    else:
        print("\nRatings written to XMP sidecars.")
