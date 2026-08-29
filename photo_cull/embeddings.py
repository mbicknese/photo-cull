"""Image embedding abstraction used for visual-similarity burst detection.

The pipeline only needs a vector per image and a cosine similarity
function; the concrete embedding backend is swappable. The preferred
backend is a local MLX SigLIP model via the `mlx-embeddings` package
(https://pypi.org/project/mlx-embeddings/). When that optional
dependency isn't installed, we fall back to a cheap deterministic
perceptual-hash-style embedding so the rest of the pipeline (burst
grouping, tests) keeps working -- with a logged warning since it is a
much weaker similarity signal than a real SigLIP embedding.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstraction over a local image-embedding model."""

    name: str = "embedding-model"

    @abstractmethod
    def embed(self, image: Image.Image) -> np.ndarray:
        """Return a 1-D, L2-normalised embedding vector for `image`."""
        raise NotImplementedError


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class MLXSiglipEmbeddingModel(EmbeddingModel):
    """SigLIP image embeddings computed locally via `mlx-embeddings`."""

    name = "mlx-embeddings-siglip"

    def __init__(self, model_id: str = "mlx-community/siglip-so400m-patch14-384"):
        try:
            from mlx_embeddings.utils import load  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without dep
            raise ImportError(
                "mlx-embeddings is not installed. Install the 'embeddings' "
                "extra (`pip install photo-cull[embeddings]`) to enable real "
                "SigLIP embeddings for burst similarity detection."
            ) from exc
        self._model, self._processor = load(model_id)

    def embed(self, image: Image.Image) -> np.ndarray:
        import mlx.core as mx  # type: ignore

        # The processor requires a text argument even though we only need
        # the image tower; the accompanying text is a throwaway.
        inputs = self._processor(
            text=[""], images=image, padding="max_length", return_tensors="np"
        )
        pixel_values = mx.array(inputs.pixel_values).transpose(0, 2, 3, 1).astype(mx.float32)
        input_ids = mx.array(inputs.input_ids)
        outputs = self._model(pixel_values=pixel_values, input_ids=input_ids)
        vec = np.asarray(outputs.image_embeds[0], dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class PerceptualHashEmbeddingModel(EmbeddingModel):
    """Deterministic, dependency-free fallback embedding.

    Downscales the image to a small grayscale grid and flattens it into a
    normalised vector. This is a much cruder similarity signal than a real
    SigLIP embedding and should only be used when `mlx-embeddings` is
    unavailable.
    """

    name = "perceptual-hash-fallback"

    def __init__(self, size: int = 32):
        self.size = size

    def embed(self, image: Image.Image) -> np.ndarray:
        small = image.convert("L").resize((self.size, self.size), Image.LANCZOS)
        vec = np.asarray(small, dtype=np.float64).reshape(-1)
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def get_default_embedding_model() -> EmbeddingModel:
    """Prefer a real SigLIP embedding; fall back with a warning."""
    try:
        return MLXSiglipEmbeddingModel()
    except ImportError as exc:
        logger.warning(
            "Falling back to a low-quality perceptual-hash embedding for "
            "burst similarity detection: %s",
            exc,
        )
        return PerceptualHashEmbeddingModel()
