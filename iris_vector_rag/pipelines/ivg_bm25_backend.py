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

With iris-vector-graph 4.2.0 two things matter to a caller:

* ``max_df_fraction`` (constructor argument, or ``retrieval.bm25.max_df_fraction`` through
  :meth:`from_config`) makes ivg skip query terms found in more than that fraction of the
  documents. Off (0) by default, so results are unchanged unless it is asked for.
* An index built before 4.2.0 is schema 1: it can be searched, but ``Insert`` and
  ``Delete`` refuse it until ``engine.bm25_upgrade(name)`` has run
  (``SchemaManager.upgrade_bm25_indexes``). Searching one logs a warning, once.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIG_KEY = "retrieval.bm25.max_df_fraction"


def _checked_fraction(value: Any, source: str) -> float:
    """A number from 0 to 1, or a ValueError that names where it came from."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or math.isnan(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(
            f"{source} (max_df_fraction) must be a number from 0 to 1, got {value!r}"
        )
    return float(value)


class IVGTextSearchBackend:
    """Wraps IRISGraphEngine.bm25_search to match the kg_TXT call contract.

    IRISGraphEngine.kg_TXT calls the IRIS stored procedure iris_vector_graph.kg_TXT,
    which runs iFind on Graph_KG.rdf_props.val — tightly coupled to the IVG schema.
    IVG BM25 (Graph.KG.BM25Index) indexes are built at ingest time on node text
    fields directly relevant to the analysis domain (alerts, logs, spans).

    Both methods return List[Tuple[str, float]] so the downstream
    convert_text_results_to_documents() helper works unchanged.
    """

    def __init__(
        self,
        iris_engine,
        index_name: str,
        max_df_fraction: Optional[float] = None,
    ) -> None:
        self.iris_engine = iris_engine
        self.index_name = index_name
        self.max_df_fraction = (
            0.0
            if max_df_fraction is None
            else _checked_fraction(max_df_fraction, "max_df_fraction")
        )
        self._schema_checked = False

    @classmethod
    def from_config(
        cls,
        iris_engine,
        index_name: str,
        config_manager,
        max_df_fraction: Optional[float] = None,
    ) -> "IVGTextSearchBackend":
        """Build from ``retrieval.bm25.max_df_fraction``; an argument beats the file."""
        if max_df_fraction is None:
            configured = config_manager.get("retrieval:bm25:max_df_fraction", 0.0)
            max_df_fraction = _checked_fraction(configured, CONFIG_KEY)
        return cls(iris_engine, index_name, max_df_fraction=max_df_fraction)

    def _warn_if_schema_1(self) -> None:
        """Once per backend: say that writers will be refused. Never raises."""
        if self._schema_checked:
            return
        self._schema_checked = True
        info_of = getattr(self.iris_engine, "bm25_info", None)
        if not callable(info_of):
            return
        try:
            info = info_of(self.index_name) or {}
            schema = int(info.get("schema", 1)) if info else None
        except Exception as exc:  # a search must not fail because Info did
            logger.debug("BM25 index %r: schema not read: %s", self.index_name, exc)
            return
        if schema is not None and schema < 2:
            logger.warning(
                "BM25 index %r is schema 1: Search works, but Insert and Delete are "
                "refused until engine.bm25_upgrade(%r) has run "
                "(SchemaManager.upgrade_bm25_indexes). Scores change once it has.",
                self.index_name,
                self.index_name,
            )

    def search(
        self,
        query_text: str,
        k: int,
        min_confidence: int = 0,  # iFind artefact — ignored by IVG BM25
    ) -> List[Tuple[str, float]]:
        """Return List[Tuple[entity_id, relevance_score]] matching kg_TXT shape."""
        self._warn_if_schema_1()
        if self.max_df_fraction:
            return self.iris_engine.bm25_search(
                self.index_name,
                query_text,
                k,
                max_df_fraction=self.max_df_fraction,
            )
        return self.iris_engine.bm25_search(self.index_name, query_text, k)
