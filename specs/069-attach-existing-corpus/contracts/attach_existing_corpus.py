"""Contract: HybridGraphRAGPipeline.attach_existing_corpus()

This is the API contract, not implementation. Tests verify this signature and return type.
"""

from typing import TypedDict


class AttachResult(TypedDict):
    table: str
    label: str
    id_col: str
    embedding_col: str
    dimension: int
    row_count: int
    has_hnsw_index: bool


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
