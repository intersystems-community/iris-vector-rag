"""
GraphRAGToolSet — iris_llm.ToolSet subclass exposing GraphRAG pipeline operations
as agent-callable tools.

Requires iris_llm to be installed. Raises ImportError on module load if absent.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from iris_llm import ToolSet, tool  # type: ignore[import]
except ImportError as _iris_llm_err:
    raise ImportError(
        "iris_vector_rag.tools.graphrag requires the iris_llm wheel. "
        "Install with: pip install iris_llm-*.whl"
    ) from _iris_llm_err

if TYPE_CHECKING:
    from iris_vector_rag.executor import SqlExecutor


class GraphRAGToolSet(ToolSet):
    """
    Agent-facing toolset wrapping
    :class:`~iris_vector_rag.pipelines.hybrid_graphrag.HybridGraphRAGPipeline`.

    The toolset owns the pipeline instance. The caller supplies a
    :class:`~iris_vector_rag.executor.SqlExecutor` and gets fully wired GraphRAG
    tools back — no GraphRAG code needed in the consumer.

    Construction::

        from iris_vector_rag.tools import GraphRAGToolSet
        toolset = GraphRAGToolSet(executor=MyExecutor(connection))

    RBAC checks are the caller's responsibility; this class performs none.
    """

    def __init__(self, executor: "SqlExecutor") -> None:
        super().__init__()
        from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        self._executor = executor
        self._pipeline = HybridGraphRAGPipeline(executor=executor)

    @tool
    def search_entities(self, query: str, limit: int = 5) -> str:
        """
        Search the medical knowledge graph for entities matching query terms.

        Returns JSON: ``{"query": str, "entities_found": int, "entities": [...]}``.
        """
        rows = self._executor.execute(
            "SELECT entity_id, entity_name, entity_type "
            "FROM RAG.Entities "
            "WHERE entity_name LIKE ? "
            "ORDER BY entity_name",
            [f"%{query}%"],
        )
        entities = [
            {
                "entity_id": r.get("entity_id") or r.get("ENTITY_ID", ""),
                "entity_name": r.get("entity_name") or r.get("ENTITY_NAME", ""),
                "entity_type": r.get("entity_type") or r.get("ENTITY_TYPE", ""),
            }
            for r in rows[:limit]
        ]
        return json.dumps(
            {"query": query, "entities_found": len(entities), "entities": entities},
            default=str,
        )

    @tool
    def traverse_relationships(self, entity_text: str, max_depth: int = 2) -> str:
        """
        Traverse knowledge graph relationships from a seed entity (1–3 hops).

        *max_depth* is clamped to [1, 3].

        Returns JSON: ``{"seed_entity": str, "entities_found": int,
        "relationships_found": int, "graph": {...}}``.
        """
        max_depth = max(1, min(3, max_depth))

        seed_rows = self._executor.execute(
            "SELECT entity_id FROM RAG.Entities WHERE entity_name LIKE ?",
            [f"%{entity_text}%"],
        )
        seed_ids = {
            r.get("entity_id") or r.get("ENTITY_ID", "")
            for r in seed_rows
            if r.get("entity_id") or r.get("ENTITY_ID")
        }

        visited: set[str] = set(seed_ids)
        frontier: set[str] = set(seed_ids)
        all_relationships: list[dict] = []

        for _ in range(max_depth):
            if not frontier:
                break
            placeholders = ",".join(["?" for _ in frontier])
            rel_rows = self._executor.execute(
                f"SELECT source_entity_id, target_entity_id, relationship_type "
                f"FROM RAG.EntityRelationships "
                f"WHERE source_entity_id IN ({placeholders}) "
                f"OR target_entity_id IN ({placeholders})",
                list(frontier) + list(frontier),
            )
            new_frontier: set[str] = set()
            for r in rel_rows:
                src = r.get("source_entity_id") or r.get("SOURCE_ENTITY_ID", "")
                tgt = r.get("target_entity_id") or r.get("TARGET_ENTITY_ID", "")
                rel_type = r.get("relationship_type") or r.get("RELATIONSHIP_TYPE", "")
                all_relationships.append({"source": src, "target": tgt, "type": rel_type})
                for node in (src, tgt):
                    if node and node not in visited:
                        new_frontier.add(node)
            visited.update(new_frontier)
            frontier = new_frontier

        return json.dumps(
            {
                "seed_entity": entity_text,
                "entities_found": len(visited),
                "relationships_found": len(all_relationships),
                "graph": {"nodes": list(visited), "edges": all_relationships},
            },
            default=str,
        )

    @tool
    def hybrid_search(self, query: str, top_k: int = 5) -> str:
        """
        Combined vector + graph search using Reciprocal Rank Fusion.

        Returns JSON: ``{"query": str, "fused_results": int, "top_documents": [...]}``.
        """
        result = self._pipeline.query(query, top_k=top_k, generate_answer=False)
        docs = result.get("retrieved_documents") or result.get("contexts") or []
        top_documents = [
            {
                "id": getattr(d, "id", ""),
                "content": getattr(d, "page_content", "")[:500],
                "score": float(getattr(d, "metadata", {}).get("similarity_score", 0.0)),
            }
            for d in docs[:top_k]
        ]
        return json.dumps(
            {
                "query": query,
                "fused_results": len(top_documents),
                "top_documents": top_documents,
            },
            default=str,
        )
