"""IVGTextSearchBackend — pluggable BM25 text search using IVG Graph.KG.BM25Index.

Implements the same call contract as IRISGraphEngine.kg_TXT so it can be injected
into HybridRetrievalMethods.text_engine at construction time (Outcome B, spec 021
Phase 8c T086).

Usage:
    from iris_vector_rag.pipelines.ivg_bm25_backend import IVGTextSearchBackend

    backend = IVGTextSearchBackend(iris_engine=engine, index_name="isc-obs-logs")
    retrieval_methods.text_engine = backend

    # In retrieve_via_enhanced_text, replace:
    #   self.iris_engine.kg_TXT(query_text, k, min_confidence)
    # with:
    #   self.text_engine.search(query_text, k, min_confidence)
"""

from __future__ import annotations

from typing import List, Tuple


class IVGTextSearchBackend:
    """Wraps IRISGraphEngine.bm25_search to match the kg_TXT call contract.

    IRISGraphEngine.kg_TXT calls the IRIS stored procedure iris_vector_graph.kg_TXT,
    which runs iFind on Graph_KG.rdf_props.val — tightly coupled to the IVG schema.
    IVG BM25 (Graph.KG.BM25Index) indexes are built at ingest time on node text
    fields directly relevant to the analysis domain (alerts, logs, spans).

    Both methods return List[Tuple[str, float]] so the downstream
    convert_text_results_to_documents() helper works unchanged.
    """

    def __init__(self, iris_engine, index_name: str) -> None:
        self.iris_engine = iris_engine
        self.index_name = index_name

    def search(
        self,
        query_text: str,
        k: int,
        min_confidence: int = 0,  # iFind artefact — ignored by IVG BM25
    ) -> List[Tuple[str, float]]:
        """Return List[Tuple[entity_id, relevance_score]] matching kg_TXT shape."""
        return self.iris_engine.bm25_search(self.index_name, query_text, k)
