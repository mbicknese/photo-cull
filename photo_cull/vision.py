"""Vision-language model abstraction (spec sections 4, 7-8, 11, 19).

`VisionModel` is the seam that keeps the culling pipeline independent of
any specific MLX model. The default implementation drives a local
`mlx-vlm` model (e.g. `mlx-community/Qwen3-VL-8B-Instruct-4bit`), but a
different MLX vision model can be substituted by implementing this
interface.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from .models import BurstComparisonEntry, IndividualAnalysis

PROMPT_VERSION = 2

INDIVIDUAL_ANALYSIS_PROMPT = """You are a professional, hard-to-please photographic culling assistant.

Evaluate this photograph as an unedited capture.

Your task is not to judge whether the current JPEG is a polished finished photograph.
Your task is to decide how much potential the underlying photograph has after normal RAW editing.

Judge whether this shot deserves editing time.

Prioritise:
- strength of the captured moment
- subject expression and pose
- composition
- framing
- visual balance
- focus accuracy on important subjects
- useful sharpness
- motion blur and camera shake
- subject separation
- exposure recoverability
- highlight preservation
- distracting or accidental elements

Do not strongly penalise normal post-processing issues such as:
- imperfect white balance
- low contrast
- muted colours
- modest exposure correction
- crop refinement
- shadow/highlight adjustment
- flat camera JPEG rendering

Strongly penalise problems that cannot realistically be fixed in post.

Do not reward an image merely because its subject is beautiful, dramatic or interesting.
Judge the photographic capture.

A technically imperfect photograph may still have high potential if it captures an exceptional moment.

Be a harsh, discerning critic. Most casual/travel/snapshot photographs are ordinary and
should score in the middle or lower part of the range -- do not default to a high score
out of politeness. Use the entire 0-100 range; do not cluster your answers in the
70s-80s. In a typical batch of everyday photographs, expect the majority to score below
60, a smaller portion in the 60s-70s, only a handful in the 80s, and 90+ reserved for a
truly exceptional, portfolio-worthy capture.

Calibration anchors for the `potential` score:
- 0-29:  no editing value. Badly blurred/out of focus, eyes closed, subject cut off or
  turned away, chaotic or accidental composition, nothing noteworthy captured.
- 30-44: weak. Technically usable but dull -- flat moment, awkward pose, cluttered or
  boring composition, unremarkable framing.
- 45-61: average snapshot. Nothing wrong enough to discard, but nothing special either --
  ordinary composition, mild distractions, so-so timing.
- 62-77: solid. Good composition and timing with only minor, easily-fixed flaws.
- 78-89: strong. Clearly a keeper: good moment, clean composition, accurate focus, only
  cosmetic/editable issues remain.
- 90-100: exceptional. A rare, decisive-moment capture with excellent composition,
  expression and technical execution -- the kind of shot you'd feature. Only a small
  fraction of any shoot should reach this band.

Return JSON only, matching exactly this schema:
{
  "composition": <0-100 int>,
  "exposure": <0-100 int>,
  "sharpness": <0-100 int>,
  "moment": <0-100 int>,
  "potential": <0-100 int>,
  "confidence": <0-100 int>,
  "primary_strength": <string>,
  "primary_problem": <string>,
  "fixable_issues": [<string>, ...],
  "nonfixable_issues": [<string>, ...],
  "explanation": <string>
}
"""

BURST_COMPARISON_PROMPT = """These images are near-duplicate photographs from the same shooting sequence.

Compare them against each other.

The goal is to identify which frame or frames are most worth keeping.

Differences between these frames matter more than their absolute photographic quality.

Pay particular attention to:
- expression
- eye position
- blink/closed eyes
- gesture
- body position
- subject pose
- interaction between people
- timing of movement
- focus accuracy
- motion blur
- accidental occlusion
- distracting elements
- composition
- subject separation

For wildlife also consider:
- head angle
- eye visibility
- wing/body position
- action
- gesture
- separation from background
- whether the animal is partly obstructed

Do not automatically choose the technically sharpest image if another frame captures a substantially better moment.

Rank every image from strongest to weakest.

Be decisive: in most bursts there is a meaningfully best frame. Only use "close_second"
for a frame that is genuinely indistinguishable in quality from the best one -- do not
use it just to avoid making a call. Reserve "normal" for frames that are clearly behind
the leader but still fine on their own, "weaker" for frames with a real flaw versus the
leader (imperfect timing, softer focus, less flattering expression), and "redundant" for
frames that add nothing beyond what a stronger frame already captures.

Identify whether the best image is clearly superior, slightly superior, or effectively tied with another frame.

The images are labelled in order: {labels}.

Return JSON only, matching exactly this schema:
{{
  "ranking": [
    {{"label": <string, one of the labels above>, "rank": <int, 1 = best>, "tier": <one of "clear_winner", "close_second", "normal", "weaker", "redundant">, "notes": <string>}},
    ...
  ]
}}
Exactly one entry should use tier "clear_winner" or the top-ranked entries may share "close_second" if effectively tied for best. Every label must appear exactly once.
"""

REQUIRED_INDIVIDUAL_FIELDS = (
    "composition",
    "exposure",
    "sharpness",
    "moment",
    "potential",
    "confidence",
)
_VALID_TIERS = {"clear_winner", "close_second", "normal", "weaker", "redundant"}


class VisionAnalysisError(RuntimeError):
    """Raised when the vision model produces no usable structured result."""


def _extract_json_object(text: str) -> dict:
    """Pull the first top-level JSON object out of `text` and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def parse_individual_analysis(text: str) -> IndividualAnalysis:
    """Parse and validate the model's individual-analysis JSON.

    Raises ValueError on any structural or range problem. Malformed
    responses are rejected rather than silently clamped/guessed (spec 19,
    26): a value outside 0-100 or a missing required field is an error.
    """
    data = _extract_json_object(text)

    for field_name in REQUIRED_INDIVIDUAL_FIELDS:
        if field_name not in data:
            raise ValueError(f"missing required field: {field_name}")
        value = data[field_name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"field {field_name} is not numeric: {value!r}")
        if not (0 <= value <= 100):
            raise ValueError(f"field {field_name} out of range 0-100: {value!r}")

    return IndividualAnalysis(
        composition=int(data["composition"]),
        exposure=int(data["exposure"]),
        sharpness=int(data["sharpness"]),
        moment=int(data["moment"]),
        potential=int(data["potential"]),
        confidence=int(data["confidence"]),
        primary_strength=str(data.get("primary_strength", "")),
        primary_problem=str(data.get("primary_problem", "")),
        fixable_issues=list(data.get("fixable_issues", []) or []),
        nonfixable_issues=list(data.get("nonfixable_issues", []) or []),
        explanation=str(data.get("explanation", "")),
        raw=data,
    )


def parse_burst_comparison(text: str, expected_labels: list[str]) -> list[dict]:
    """Parse and validate the model's burst-comparison JSON.

    Returns a list of dicts with keys label/rank/tier/notes. Raises
    ValueError if the ranking doesn't cover every expected label exactly
    once or uses an unrecognised tier.
    """
    data = _extract_json_object(text)
    ranking = data.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("missing or empty 'ranking' list")

    seen_labels: set[str] = set()
    entries: list[dict] = []
    for item in ranking:
        label = item.get("label")
        tier = item.get("tier")
        rank = item.get("rank")
        if label not in expected_labels:
            raise ValueError(f"unexpected label in ranking: {label!r}")
        if tier not in _VALID_TIERS:
            raise ValueError(f"unexpected tier in ranking: {tier!r}")
        if not isinstance(rank, int):
            raise ValueError(f"rank is not an int: {rank!r}")
        seen_labels.add(label)
        entries.append(
            {
                "label": label,
                "rank": rank,
                "tier": tier,
                "notes": str(item.get("notes", "")),
            }
        )

    if seen_labels != set(expected_labels):
        missing = set(expected_labels) - seen_labels
        raise ValueError(f"ranking is missing labels: {missing}")

    return entries


class VisionModel(ABC):
    """Abstraction over a local MLX vision-language model."""

    name: str = "vision-model"

    @abstractmethod
    def analyze_individual(self, images: list[Image.Image]) -> IndividualAnalysis:
        """Judge a single photograph's editing potential."""
        raise NotImplementedError

    @abstractmethod
    def compare_burst(self, members: list[tuple[str, Image.Image]]) -> list[BurstComparisonEntry]:
        """Rank a burst's frames against each other.

        `members` is a list of (stem, whole-image representation) pairs.
        Returns one BurstComparisonEntry per member.
        """
        raise NotImplementedError


class MLXVisionModel(VisionModel):
    """Default VisionModel backed by `mlx-vlm`."""

    MAX_BURST_COMPARE = 6

    def __init__(self, model_id: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"):
        self.name = model_id
        self._model = None
        self._processor = None
        self._config = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_vlm import load  # type: ignore
            from mlx_vlm.utils import load_config  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "mlx-vlm is not installed. Install the 'mlx' extra "
                "(`pip install photo-cull[mlx]`) on Apple Silicon to run "
                "local vision-model inference."
            ) from exc
        self._model, self._processor = load(self.name)
        self._config = load_config(self.name)

    def _generate(self, images: list[Image.Image], prompt: str) -> str:
        self._ensure_loaded()
        from mlx_vlm import generate  # type: ignore
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore

        formatted_prompt = apply_chat_template(
            self._processor, self._config, prompt, num_images=len(images)
        )
        result = generate(
            self._model,
            self._processor,
            formatted_prompt,
            images,
            temperature=0.0,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)

    def _generate_with_retry(self, images: list[Image.Image], prompt: str, parse_fn):
        text = self._generate(images, prompt)
        try:
            return parse_fn(text)
        except (ValueError, json.JSONDecodeError) as first_error:
            repair_prompt = (
                prompt
                + "\n\nYour previous response could not be parsed as valid JSON "
                "matching the schema above. Return ONLY a single valid JSON "
                "object matching the schema, with no surrounding text."
            )
            text = self._generate(images, repair_prompt)
            try:
                return parse_fn(text)
            except (ValueError, json.JSONDecodeError) as second_error:
                raise VisionAnalysisError(
                    f"model did not return valid JSON after retry: {second_error}"
                ) from second_error

    def analyze_individual(self, images: list[Image.Image]) -> IndividualAnalysis:
        return self._generate_with_retry(
            images, INDIVIDUAL_ANALYSIS_PROMPT, parse_individual_analysis
        )

    def compare_burst(self, members: list[tuple[str, Image.Image]]) -> list[BurstComparisonEntry]:
        labels = [stem for stem, _ in members]
        prompt = BURST_COMPARISON_PROMPT.format(labels=", ".join(labels))
        images = [img for _, img in members]

        def parse(text: str) -> list[dict]:
            return parse_burst_comparison(text, labels)

        parsed = self._generate_with_retry(images, prompt, parse)
        return [
            BurstComparisonEntry(
                stem=entry["label"],
                rank=entry["rank"],
                tier=entry["tier"],
                notes=entry["notes"],
            )
            for entry in parsed
        ]
