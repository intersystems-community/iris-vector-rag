"""include_sources=True on HybridGraphRAG lists the sources, as GraphRAG does.

`_extract_sources` delegated to a parent method that GraphRAGPipeline never had, so
every hybrid query with include_sources=True (the MCP default) raised AttributeError.
"""

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline


def test_sources_are_titles_falling_back_to_ids():
    pipeline = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    documents = [
        Document(id="d1", page_content="a", metadata={"title": "Diabetes"}),
        Document(id="d2", page_content="b", metadata={}),
    ]

    assert pipeline._extract_sources(documents) == ["Diabetes", "d2"]
