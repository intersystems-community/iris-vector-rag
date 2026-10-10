"""The test embedder: a deterministic stand-in for a real embedding model.

It is reachable only by name (`embeddings: {provider: stub}`) or by a test handing an
instance in. Nothing substitutes it when a real model fails to load. Its vectors carry the
embedding identity `stub`, so they are never mixed with a real model's.
"""

import hashlib
import logging
import random
import threading
from typing import Any, List, Sequence, Union

logger = logging.getLogger(__name__)

MODEL_KEY = "stub"

_warn_lock = threading.Lock()
_warned = False


def _warn_once() -> None:
    global _warned
    with _warn_lock:
        if _warned:
            return
        _warned = True
    logger.warning(
        "The test embedder is in use: its vectors are deterministic noise, not a real "
        "embedding model. Similarity between them means nothing. Never use it outside "
        "tests."
    )


class TestEmbedder:
    """Deterministic, unit-length vectors of a caller-chosen dimension."""

    __test__ = False  # not a pytest test class

    model_key = MODEL_KEY

    def __init__(self, dimension: int):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError(
                f"The test embedder needs a positive integer dimension, got {dimension!r}"
            )
        self.dimension = dimension
        _warn_once()

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def get_sentence_embedding_dimension(self) -> int:
        # Kept for callers still on the pre-6.x name; the package probes the new one.
        return self.get_embedding_dimension()

    def _vector(self, text: str) -> List[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        raw = [rng.random() - 0.5 for _ in range(self.dimension)]
        norm = sum(x * x for x in raw) ** 0.5
        if norm == 0.0:  # a single-element draw of exactly zero
            raw[0] = 1.0
            norm = 1.0
        return [x / norm for x in raw]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vector(str(text)) for text in texts]

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        **_: Any,
    ) -> List[List[float]]:
        """The slice of the SentenceTransformer API the code under test calls."""
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_texts(texts)
