"""
E2E: a metadata filter must scope recall to the filtered subset, in SQL.

Reproduces the scenario from opsreview's upstream report
(docs/upstream/ivr-metadata-filter-pushdown.md): a shared RAG.SourceDocuments
table where one noisy tenant owns most documents and a small tenant owns two.
Ranking globally and filtering afterwards loses the small tenant's weaker
document; scoping in the WHERE clause returns both.

NO MOCKS - real IRIS, synthetic 384-d embeddings so the geometry is exact.
"""

from __future__ import annotations

import logging
import math
import random
from typing import List

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

logger = logging.getLogger(__name__)

DIM = 384
NOISY_DOCS = 60  # enough that even a 5x over-fetch of top_k=5 (25) misses the weak doc


def _unit(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _near(query: List[float], cosine: float, rng: random.Random) -> List[float]:
    """A unit vector at (approximately) the requested cosine to `query`."""
    noise = _unit([rng.gauss(0, 1) for _ in range(DIM)])
    # remove the component along query so the mix has a predictable angle
    dot = sum(a * b for a, b in zip(noise, query))
    ortho = _unit([n - dot * q for n, q in zip(noise, query)])
    sin = math.sqrt(max(0.0, 1 - cosine * cosine))
    return _unit([cosine * q + sin * o for q, o in zip(query, ortho)])


@pytest.mark.true_e2e
class TestMetadataFilterPushdownE2E:
    def test_small_tenant_recall_is_independent_of_noisy_neighbours(
        self,
        fresh_iris_vector_store: IRISVectorStore,
        e2e_database_cleanup,
    ):
        rng = random.Random(62)
        query = _unit([1.0] + [0.0] * (DIM - 1))

        docs: List[Document] = []
        vecs: List[List[float]] = []
        # Noisy tenant: many documents all closer to the query than the small
        # tenant's weak document.
        for i in range(NOISY_DOCS):
            docs.append(
                Document(
                    id=f"e2e_test_noisy_{i}",
                    page_content=f"noisy tenant document {i}",
                    metadata={"source": "tenant-noisy"},
                )
            )
            vecs.append(_near(query, rng.uniform(0.60, 0.95), rng))
        # Small tenant: one strong match, one weak match (below every noisy doc).
        docs.append(
            Document(
                id="e2e_test_small_strong",
                page_content="small tenant strong",
                metadata={"source": "tenant-small"},
            )
        )
        vecs.append(_near(query, 0.58, rng))
        docs.append(
            Document(
                id="e2e_test_small_weak",
                page_content="small tenant weak",
                metadata={"source": "tenant-small"},
            )
        )
        vecs.append(_near(query, 0.23, rng))

        stored = fresh_iris_vector_store.add_documents(docs, embeddings=vecs)
        assert len(stored) == len(docs), "setup: every document must be stored"

        # Sanity: without a filter the small tenant is invisible in a top-5 page.
        unfiltered = fresh_iris_vector_store.similarity_search_by_embedding(
            query, top_k=5
        )
        assert all(
            d.metadata.get("source") == "tenant-noisy" for d, _ in unfiltered
        ), "setup: the noisy tenant must dominate the unfiltered page"

        # The property under test: the filter scopes recall to the tenant.
        filtered = fresh_iris_vector_store.similarity_search_by_embedding(
            query, top_k=5, filter={"source": "tenant-small"}
        )
        ids = [d.id for d, _ in filtered]
        assert ids == ["e2e_test_small_strong", "e2e_test_small_weak"], (
            "both of the small tenant's documents must be returned, strongest first; got "
            f"{[(i, round(s, 2)) for i, (_, s) in zip(ids, filtered)]}"
        )
        assert all(d.metadata.get("source") == "tenant-small" for d, _ in filtered)
        scores = [s for _, s in filtered]
        assert scores == sorted(scores, reverse=True)
        logger.info(
            "filtered page: %s", [(i, round(s, 3)) for i, s in zip(ids, scores)]
        )

    def test_filter_with_no_matching_documents_returns_empty(
        self,
        fresh_iris_vector_store: IRISVectorStore,
        e2e_database_cleanup,
    ):
        rng = random.Random(7)
        query = _unit([0.0, 1.0] + [0.0] * (DIM - 2))
        docs = [
            Document(
                id=f"e2e_test_other_{i}",
                page_content=f"other {i}",
                metadata={"source": "tenant-other"},
            )
            for i in range(3)
        ]
        fresh_iris_vector_store.add_documents(
            docs, embeddings=[_near(query, 0.9, rng) for _ in docs]
        )

        results = fresh_iris_vector_store.similarity_search_by_embedding(
            query, top_k=5, filter={"source": "tenant-nobody"}
        )
        assert results == []
