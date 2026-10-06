"""Fused documents carry rrf_score, rrf_rank (1-based, sequential) and score (the
fused score, the key every pipeline's documents share)."""

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines import multi_query_rrf as module
from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline


def _docs(*ids):
    return [Document(id=i, page_content=f"text {i}", metadata={}) for i in ids]


@pytest.mark.parametrize("use_ivg", [True, False])
def test_fused_documents_carry_score_and_rank(monkeypatch, use_ivg):
    monkeypatch.setattr(module, "_HAS_IVG_RRF", use_ivg and module._HAS_IVG_RRF)
    pipeline = MultiQueryRRFPipeline.__new__(MultiQueryRRFPipeline)
    pipeline.rrf_k = 60
    fused = pipeline._reciprocal_rank_fusion(
        [_docs("a", "b"), _docs("b", "c")], top_k=3
    )
    assert [d.id for d in fused] == ["b", "a", "c"]
    assert [d.metadata["rrf_rank"] for d in fused] == [1, 2, 3]
    assert fused[0].metadata["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    scores = [d.metadata["rrf_score"] for d in fused]
    assert scores == sorted(scores, reverse=True)
    assert [d.metadata["score"] for d in fused] == scores
