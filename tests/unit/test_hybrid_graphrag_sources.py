"""HybridGraphRAGPipeline._extract_sources called `super()._extract_sources`, which
no parent defines: every query that reached the response step raised AttributeError
(seen through the MCP `rag_graphrag` tool)."""

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline


def test_sources_have_the_shape_the_other_pipelines_use():
    pipeline = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    docs = [
        Document(id="a", page_content="x", metadata={"source": "s.pdf"}),
        Document(id="b", page_content="y", metadata={"chunk_index": 2}),
    ]
    assert pipeline._extract_sources(docs) == [
        {"document_id": "a", "source": "s.pdf", "filename": "Unknown"},
        {
            "document_id": "b",
            "source": "Unknown",
            "filename": "Unknown",
            "chunk_index": 2,
        },
    ]
