"""Fusion keeps real content and sane scores (CR-01, CR-02, CR-03, CR-09 filter half).

Every expected number below is worked out by hand in the comment next to it.

Rank contribution (weighted fusion): a hit at 1-based rank r in a list of n hits earns
``weight * (n - r + 1) / n``. The top hit earns the full weight, the last earns
``weight / n``, and a one-hit list earns its whole weight. The old formula,
``weight * (1 - r / n)``, gave the last hit 0, so a sole hit scored nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.core.query_options import QueryOptions
from iris_vector_rag.retrieval import engine as engine_module
from iris_vector_rag.retrieval.engine import (
    RetrievalEngine,
    _reciprocal_rank_fusion,
    _weighted_score_fusion,
)


def doc(doc_id: str, content: str = "", **metadata: Any) -> Document:
    return Document(id=doc_id, page_content=content, metadata=dict(metadata))


# ---------------------------------------------------------------------------
# CR-01: one document per id, and the one with content wins
# ---------------------------------------------------------------------------


def test_rrf_overlap_keeps_the_vector_content_when_the_text_hit_is_blank():
    vector = [doc("a", "alpha text", source="bio")]
    text = [doc("a", "", text_hint=4.0)]

    fused = _reciprocal_rank_fusion([vector, text], top_k=5)

    assert [d.id for d in fused] == ["a"]
    assert fused[0].page_content == "alpha text"
    # Metadata of both legs survives the merge.
    assert fused[0].metadata["source"] == "bio"
    assert fused[0].metadata["text_hint"] == 4.0


def test_rrf_overlap_keeps_the_content_when_the_blank_hit_comes_first():
    vector = [doc("a", "", vector_only=True)]
    text = [doc("a", "alpha text")]

    fused = _reciprocal_rank_fusion([vector, text], top_k=5)

    assert fused[0].page_content == "alpha text"
    assert fused[0].metadata["vector_only"] is True


def test_rrf_scores_of_an_overlap_are_the_sum_of_both_legs():
    # a: rank 1 in vector (1/61) and rank 2 in text (1/62); b: rank 2 in vector only
    # (1/62); c: rank 1 in text only (1/61).
    vector = [doc("a", "A"), doc("b", "B")]
    text = [doc("c", "C"), doc("a", "A")]

    fused = _reciprocal_rank_fusion([vector, text], top_k=5)

    by_id = {d.id: d for d in fused}
    assert by_id["a"].metadata["fusion_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["b"].metadata["fusion_score"] == pytest.approx(1 / 62)
    assert by_id["c"].metadata["fusion_score"] == pytest.approx(1 / 61)
    assert by_id["a"].metadata["vector_score"] == pytest.approx(1 / 61)
    assert by_id["a"].metadata["text_score"] == pytest.approx(1 / 62)
    assert [d.id for d in fused] == ["a", "c", "b"]


def test_weighted_overlap_keeps_the_content_and_one_entry_per_id():
    vector = [doc("a", "alpha text")]
    text = [doc("a", "")]

    fused = _weighted_score_fusion([vector, text], [0.7, 0.3], top_k=5)

    assert [d.id for d in fused] == ["a"]
    assert fused[0].page_content == "alpha text"


# ---------------------------------------------------------------------------
# CR-03: rank contribution never gives 0 to a sole or last hit
# ---------------------------------------------------------------------------


def test_weighted_sole_hit_scores_its_whole_weight():
    fused = _weighted_score_fusion([[doc("a", "A")], []], [0.7, 0.3], top_k=5)

    assert fused[0].metadata["fusion_score"] == pytest.approx(0.7)
    assert fused[0].metadata["vector_score"] == pytest.approx(0.7)


def test_weighted_scores_by_hand():
    # vector (w 0.7), n=3: a -> 0.7*3/3 = 0.7; b -> 0.7*2/3; c -> 0.7*1/3
    # text (w 0.3), n=1: b -> 0.3*1/1 = 0.3
    # totals: a 0.7; b 0.7*2/3 + 0.3 = 0.76667; c 0.23333
    vector = [doc("a", "A"), doc("b", "B"), doc("c", "C")]
    text = [doc("b", "")]

    fused = _weighted_score_fusion([vector, text], [0.7, 0.3], top_k=5)

    assert [d.id for d in fused] == ["b", "a", "c"]
    scores = {d.id: d.metadata["fusion_score"] for d in fused}
    assert scores["a"] == pytest.approx(0.7)
    assert scores["b"] == pytest.approx(0.7 * 2 / 3 + 0.3)
    assert scores["c"] == pytest.approx(0.7 / 3)
    assert all(s > 0 for s in scores.values())


# ---------------------------------------------------------------------------
# CR-02: three lists need three names
# ---------------------------------------------------------------------------


def test_weighted_three_lists_keep_all_three_and_label_them():
    high = [doc("h", "H")]
    low = [doc("l", "L")]
    naive = [doc("n", "N")]

    fused = _weighted_score_fusion(
        [high, low, naive],
        [0.5, 0.3, 0.2],
        top_k=5,
        source_names=["high_level", "low_level", "naive"],
    )

    scores = {d.id: d.metadata["fusion_score"] for d in fused}
    # Each list has one hit, so each earns its whole weight.
    assert scores == {
        "h": pytest.approx(0.5),
        "l": pytest.approx(0.3),
        "n": pytest.approx(0.2),
    }
    by_id = {d.id: d for d in fused}
    assert by_id["n"].metadata["naive_score"] == pytest.approx(0.2)
    assert "vector_score" not in by_id["n"].metadata


def test_fusion_with_mismatched_names_raises_instead_of_truncating():
    with pytest.raises(ValueError, match="source_names"):
        _weighted_score_fusion([[doc("a", "A")]] * 3, [0.5, 0.3, 0.2], top_k=5)
    with pytest.raises(ValueError, match="weights"):
        _weighted_score_fusion(
            [[doc("a", "A")]] * 2, [0.5], top_k=5, source_names=["x", "y"]
        )
    with pytest.raises(ValueError, match="source_names"):
        _reciprocal_rank_fusion([[doc("a", "A")]] * 3, top_k=5)


# ---------------------------------------------------------------------------
# The engine: hydrate text-only hits by id, honour filters
# ---------------------------------------------------------------------------


CORPUS = {
    "a": ("alpha insulin", {"source": "bio"}),
    "b": ("beta glucose", {"source": "bio"}),
    "c": ("gamma network", {"source": "cs"}),
}


class FakeStore:
    """A vector store with a fixed ranking; records how it was called."""

    def __init__(
        self,
        ranking: List[str],
        scores: Optional[Dict[str, float]] = None,
        text: Optional[List[tuple]] = None,
    ):
        self.ranking = ranking
        self.scores = scores or {}
        self.text = text or []
        self.calls: List[tuple] = []
        self.fetched: List[List[str]] = []

    def _doc(self, doc_id: str) -> Document:
        content, meta = CORPUS[doc_id]
        return Document(id=doc_id, page_content=content, metadata=dict(meta))

    def search_by_text(self, query, top_k=5, metadata_filter=None):
        self.calls.append(("search_by_text", query, top_k, metadata_filter))
        docs = [self._doc(i) for i in self.ranking]
        if metadata_filter:
            docs = [
                d
                for d in docs
                if all(d.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
        return docs[:top_k]

    def similarity_search_with_score(self, query, k=4, filter=None):
        self.calls.append(("similarity_search_with_score", query, k, filter))
        docs = [self._doc(i) for i in self.ranking]
        if filter:
            docs = [
                d
                for d in docs
                if all(d.metadata.get(key) == v for key, v in filter.items())
            ]
        return [(d, self.scores[d.id]) for d in docs][:k]

    def text_search(self, query, top_k=5, metadata_filter=None):
        """The store's full-text search: complete documents, scored, filter applied."""
        self.calls.append(("text_search", query, top_k, metadata_filter))
        docs = []
        for doc_id, score in self.text:
            doc = self._doc(doc_id)
            doc.metadata["text_score"] = score
            if metadata_filter and not all(
                doc.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            docs.append(doc)
        return docs[:top_k]


def run(store, retrieval, **kw):
    opts = QueryOptions(query="insulin", retrieval=retrieval, top_k=3, **kw)
    return RetrievalEngine(vector_store=store, connection=object()).retrieve(opts)


def test_text_mode_returns_documents_with_content_and_scores():
    store = FakeStore(ranking=[], text=[("b", 3.5), ("a", 1.0)])

    docs = run(store, "text")

    assert [d.id for d in docs] == ["b", "a"]
    assert [d.page_content for d in docs] == ["beta glucose", "alpha insulin"]
    assert docs[0].metadata["text_score"] == 3.5
    assert docs[0].metadata["source"] == "bio"


def test_rrf_mode_text_only_hits_come_back_with_content():
    store = FakeStore(ranking=["a", "b"], text=[("c", 2.0)])

    docs = run(store, "rrf")

    assert all(d.page_content for d in docs), [d.id for d in docs if not d.page_content]
    # a: vector rank 1 -> 1/61; b: vector rank 2 -> 1/62; c: text rank 1 -> 1/61.
    assert {d.id for d in docs} == {"a", "b", "c"}
    assert [d.id for d in docs][-1] == "b"


def test_rrf_mode_ranks_the_overlap_first_with_content():
    store = FakeStore(ranking=["a", "b"], text=[("b", 2.0)])

    docs = run(store, "rrf")

    # b: 1/62 (vector rank 2) + 1/61 (text rank 1) beats a: 1/61.
    assert [d.id for d in docs] == ["b", "a"]
    assert docs[0].page_content == "beta glucose"


def test_hybrid_mode_top_vector_hit_is_not_zero():
    store = FakeStore(ranking=["a"], text=[])

    docs = run(store, "hybrid")

    # One vector hit, default weight 0.7: 0.7 * 1/1.
    assert docs[0].metadata["fusion_score"] == pytest.approx(0.7)


# --- CR-09 filter half ------------------------------------------------------


def test_vector_leg_receives_the_metadata_filter():
    store = FakeStore(ranking=["a", "c"], text=[])

    docs = run(store, "rrf", metadata_filter={"source": "cs"})

    assert [d.id for d in docs] == ["c"]
    assert store.calls[0][3] == {"source": "cs"}


def test_text_mode_passes_the_filter_to_the_store():
    store = FakeStore(ranking=[], text=[("a", 3.0), ("c", 2.0), ("b", 1.0)])

    docs = run(store, "text", metadata_filter={"source": "cs"})

    assert [d.id for d in docs] == ["c"]


@pytest.mark.parametrize("mode", ["hybrid", "rrf"])
def test_fusion_modes_filter_both_legs(mode):
    store = FakeStore(ranking=["b", "c"], text=[("a", 3.0), ("c", 2.0)])

    docs = run(store, mode, metadata_filter={"source": "cs"})

    assert [d.id for d in docs] == ["c"]


def test_similarity_threshold_applies_to_the_vector_leg_before_fusion():
    store = FakeStore(
        ranking=["a", "b"], text=[("c", 2.0)], scores={"a": 0.9, "b": 0.4}
    )

    docs = run(store, "rrf", similarity_threshold=0.5)

    assert store.calls[0][0] == "similarity_search_with_score"
    # a passes; b is below; c is a text hit and carries no similarity.
    assert {d.id for d in docs} == {"a", "c"}
    assert all(d.page_content for d in docs)


def test_threshold_on_text_mode_raises():
    with pytest.raises(ValueError, match="similarity_threshold"):
        run(FakeStore(ranking=[], text=[("a", 1.0)]), "text", similarity_threshold=0.5)


def test_vector_mode_through_the_engine_honours_filter_and_threshold():
    store = FakeStore(ranking=["a", "b", "c"], scores={"a": 0.9, "b": 0.3, "c": 0.8})

    docs = run(
        store, "vector", metadata_filter={"source": "bio"}, similarity_threshold=0.5
    )

    assert [d.id for d in docs] == ["a"]
    assert docs[0].metadata["score"] == 0.9


# --- mix mode (CR-02 at the engine) ------------------------------------------


class FakeRelationStore:
    def __init__(self, hits):
        self.hits = hits

    def count_embedded(self):
        return len(self.hits)

    def search(self, emb, top_k=5):
        return self.hits


def mix_engine(store):
    eng = RetrievalEngine(vector_store=store, connection=object())
    eng.keyword_extractor = type(
        "K", (), {"extract": lambda s, q: (["theme"], ["entity"])}
    )()
    eng._embed_text_for_search = lambda text: [0.0]
    return eng


def relation(rid):
    return {
        "relationship_id": rid,
        "source_entity_id": "s",
        "target_entity_id": "t",
        "relationship_type": "R",
        "score": 0.9,
    }


def test_mix_with_weights_keeps_the_naive_list():
    store = FakeStore(ranking=["a"])
    eng = mix_engine(store)
    opts = QueryOptions(
        query="q", retrieval="mix", top_k=5, weights={"relation": 0.5, "vector": 0.2}
    )
    with patch.object(
        engine_module,
        "RelationEmbeddingStore",
        return_value=FakeRelationStore([relation("r1")]),
    ):
        result = eng._retrieve_mix(opts)

    ids = [d.id for d in result["retrieved_documents"]]
    assert "a" in ids and "r1" in ids
    by_id = {d.id: d for d in result["retrieved_documents"]}
    # "a" is in the naive leg and in the low-level leg (same fake store), each with
    # one hit: naive 0.2 (vector weight) + low_level 0.3 (default) = 0.5.
    assert by_id["a"].metadata["naive_score"] == pytest.approx(0.2)
    assert by_id["a"].metadata["low_level_score"] == pytest.approx(0.3)
    assert by_id["a"].metadata["fusion_score"] == pytest.approx(0.5)
    assert by_id["r1"].metadata["high_level_score"] == pytest.approx(0.5)


def test_mix_with_a_filter_scopes_document_legs_and_says_relations_were_skipped():
    store = FakeStore(ranking=["a", "c"])
    eng = mix_engine(store)
    opts = QueryOptions(
        query="q", retrieval="mix", top_k=5, metadata_filter={"source": "cs"}
    )
    with patch.object(
        engine_module,
        "RelationEmbeddingStore",
        return_value=FakeRelationStore([relation("r1")]),
    ):
        result = eng._retrieve_mix(opts)

    assert {d.id for d in result["retrieved_documents"]} == {"c"}
    meta = result["metadata"]
    assert meta["degraded"] is True
    assert "metadata_filter" in meta["degradation_reason"]
    assert all(c[3] == {"source": "cs"} for c in store.calls)


def test_global_with_a_filter_raises():
    eng = mix_engine(FakeStore(ranking=[]))
    opts = QueryOptions(
        query="q", retrieval="global", top_k=5, metadata_filter={"source": "cs"}
    )
    with patch.object(
        engine_module,
        "RelationEmbeddingStore",
        return_value=FakeRelationStore([relation("r1")]),
    ):
        with pytest.raises(ValueError, match="metadata_filter"):
            eng._retrieve_global(opts)


@pytest.mark.parametrize("mode", ["text", "hybrid", "rrf"])
def test_a_missing_text_index_is_raised_not_swallowed(mode):
    from iris_vector_rag.exceptions import TextIndexMissingError

    class NoIndexStore(FakeStore):
        def text_search(self, query, top_k=5, metadata_filter=None):
            raise TextIndexMissingError("no index; run ensure_text_index")

    with pytest.raises(TextIndexMissingError, match="ensure_text_index"):
        run(NoIndexStore(ranking=["a"]), mode)


def test_a_store_without_text_search_says_so():
    class VectorOnly:
        def search_by_text(self, *args, **kwargs):
            return []

    with pytest.raises(NotImplementedError, match="text_search"):
        run(VectorOnly(), "text")
