"""The real LLM and the real embedder, for integration and e2e tests.

The LLM is OpenAI's, reached through the project's own `get_llm_func` with the
`OPENAI_API_KEY` of the environment. The embedder is the real sentence-transformer the
project ships as its default. Nothing here fakes either; fakes belong in tests/unit.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List

import numpy as np

from iris_vector_rag.common.utils import get_embedding_func, get_llm_func

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_WIDTH = 384  # the width of EMBEDDING_MODEL's vectors


def real_llm() -> Callable[[str], str]:
    """The real OpenAI model as a `prompt -> text` function."""
    return get_llm_func(provider="openai", model_name=LLM_MODEL)


def real_embedder() -> Callable:
    """The real sentence-transformer: a string gives a vector, a list gives a list."""
    return get_embedding_func(model_name_override=EMBEDDING_MODEL)


COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLBERT_TOKEN_WIDTH = 128


class RealColBERTEncoder:
    """ColBERTv2 token embeddings from the real model: BERT plus its 128-wide projection.

    Same contract as the pylate model the ColBERT pipelines load:
    `encode(texts, is_query=False)` gives one `(tokens, 128)` float32 array per text.
    This environment cannot run the project's default GTE-ModernColBERT (its tokenizer
    file does not parse with the installed `tokenizers`, and `pylate` is not installed),
    so the test encoder is ColBERTv2 itself, without the [Q]/[D] marker tokens.
    """

    def __init__(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, BertModel

        path = snapshot_download(COLBERT_MODEL)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._bert = BertModel.from_pretrained(path).eval()
        weights = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
        self._projection = weights["linear.weight"]  # (128, 768)

    def encode(self, texts: List[str], is_query: bool = False) -> List[np.ndarray]:
        torch = self._torch
        out = []
        for text in texts:
            batch = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64
            )
            with torch.no_grad():
                hidden = self._bert(**batch).last_hidden_state[0]  # (tokens, 768)
                vectors = torch.nn.functional.normalize(
                    hidden @ self._projection.T, dim=-1
                )
            out.append(vectors.numpy().astype(np.float32))
        return out


@lru_cache(maxsize=1)
def real_colbert_encoder() -> RealColBERTEncoder:
    """The real ColBERT token encoder, loaded once per process."""
    return RealColBERTEncoder()
