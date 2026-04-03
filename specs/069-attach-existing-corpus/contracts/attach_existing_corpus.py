"""Contract: HybridGraphRAGPipeline.attach_existing_corpus()

This is the API contract, not implementation. Tests verify this signature and return type.
"""

from typing import Optional, TypedDict


class AttachResult(TypedDict):
    table: str
    label: str
    id_col: str
    embedding_col: str
    dimension: int
    row_count: int
    has_hnsw_index: Optional[bool]  # None = detection not supported on this IRIS build


class DimensionMismatchError(ValueError):
    """Raised when query vector dimension != attached table embedding dimension."""

    pass


# Method signature contract:
# HybridGraphRAGPipeline.attach_existing_corpus(
#     source_table: str,       # e.g., "RAG.SourceDocuments"
#     id_col: str,             # e.g., "doc_id"
#     text_col: str,           # e.g., "text_content"
#     embedding_col: str,      # e.g., "embedding"
#     graph_label: str,        # e.g., "Document"
# ) -> AttachResult
#
# Private helpers (not public API):
# _detect_hnsw_index(source_table, embedding_col) -> Optional[bool]
# _validate_query_dimension(graph_label, query_vec) -> None  (raises DimensionMismatchError)
