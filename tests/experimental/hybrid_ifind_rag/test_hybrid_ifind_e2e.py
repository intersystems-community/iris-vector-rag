import pytest
from unittest.mock import MagicMock, patch
import os # Added for sys.path
import sys # Added for sys.path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline
from common.utils import Document

@pytest.fixture
def mock_iris_connector():
    return MagicMock()

@pytest.fixture
def mock_embedding_func():
    return MagicMock(return_value=[0.1] * 768) # Example embedding

@pytest.fixture
def mock_llm_func():
    return MagicMock(return_value="Generated answer based on hybrid context.")

@pytest.fixture
def hybrid_pipeline(mock_iris_connector, mock_embedding_func, mock_llm_func):
    return HybridiFindRAGPipeline(mock_iris_connector, mock_embedding_func, mock_llm_func)

def test_hybrid_ifind_rag_e2e_combined_retrieval(hybrid_pipeline, mock_llm_func):
    query = "What are the treatments for neurodegenerative diseases and their link to protein aggregation?"

    # Mock documents from BasicRAG
    doc_basic1 = Document(id="basic_doc_1", content="Content from BasicRAG about treatments.", score=0.8)
    doc_basic1._metadata = {"title": "Basic Doc 1"}
    
    # Mock documents, entities, and relationships from GraphRAG
    doc_graph1 = Document(id="graph_doc_1", content="Content from GraphRAG about protein aggregation.", score=0.9)
    doc_graph1._metadata = {"title": "Graph Doc 1"}
    doc_shared = Document(id="shared_doc_1", content="Shared content relevant to treatments and proteins.", score=0.85)
    doc_shared._metadata = {"title": "Shared Doc 1"} # Also found by GraphRAG
    entities_graph = [
        {"entity_id": "E1", "entity_name": "Neurodegenerative Diseases", "entity_type": "Condition"},
        {"entity_id": "E2", "entity_name": "Protein Aggregation", "entity_type": "Process"}
    ]
    relationships_graph = [{"source_id": "E1", "target_id": "E2", "type": "LINKED_TO"}]

    # Mock documents and hypothetical document from HyDE
    doc_hyde1 = Document(id="hyde_doc_1", content="Content from HyDE based on hypothetical answer.", score=0.75)
    doc_hyde1._metadata = {"title": "HyDE Doc 1"}
    hypothetical_doc_hyde = "Hypothetical answer discussing treatments and protein aggregation link."

    # Patch the individual retrieval methods on the pipeline instance
    with patch.object(hybrid_pipeline, 'retrieve_with_basic_rag', return_value=[doc_basic1, doc_shared]) as mock_retrieve_basic, \
         patch.object(hybrid_pipeline, 'retrieve_with_graphrag', return_value=([doc_graph1, doc_shared], entities_graph, relationships_graph)) as mock_retrieve_graph, \
         patch.object(hybrid_pipeline, 'retrieve_with_hyde', return_value=([doc_hyde1], hypothetical_doc_hyde)) as mock_retrieve_hyde:

        result = hybrid_pipeline.run(query, top_k=3)

        # Assertions for retrieval methods being called
        mock_retrieve_basic.assert_called_once_with(query, top_k=3)
        mock_retrieve_graph.assert_called_once_with(query, top_k=3)
        mock_retrieve_hyde.assert_called_once_with(query, top_k=3)

        # Assertions for the final result
        assert result["query"] == query
        assert result["answer"] == "Generated answer based on hybrid context."
        
        retrieved_docs = result["retrieved_documents"]
        assert len(retrieved_docs) == 3 # top_k=3

        doc_ids_retrieved = [doc['id'] for doc in retrieved_docs]
        
        # Check for presence of documents from different sources (or the shared one)
        # The exact order depends on merged scores, so check for IDs and their metadata
        assert "shared_doc_1" in doc_ids_retrieved
        assert "basic_doc_1" in doc_ids_retrieved or "graph_doc_1" in doc_ids_retrieved or "hyde_doc_1" in doc_ids_retrieved

        for doc_info in retrieved_docs:
            assert "hybrid_metadata" in doc_info
            hybrid_meta = doc_info["hybrid_metadata"]
            assert "sources" in hybrid_meta
            assert "individual_scores" in hybrid_meta
            assert "combined_score" in hybrid_meta

            if doc_info["id"] == "shared_doc_1":
                assert "BasicRAG" in hybrid_meta["sources"] # Updated from BasicRAG_V2
                assert "GraphRAG" in hybrid_meta["sources"] # Updated from GraphRAG_V2
                assert "basic" in hybrid_meta["individual_scores"]
                assert hybrid_meta["individual_scores"]["basic"] == 0.85 # score from doc_shared when returned by basic
                assert "graph" in hybrid_meta["individual_scores"] 
                assert hybrid_meta["individual_scores"]["graph"] == 0.85 # score from doc_shared when returned by graph
                # Expected combined score: (0.85 * 0.3) + (0.85 * 0.4) = 0.255 + 0.34 = 0.595
                assert hybrid_meta["combined_score"] == pytest.approx(0.595) 
            
            elif doc_info["id"] == "basic_doc_1":
                assert hybrid_meta["sources"] == ["BasicRAG"] # Updated from BasicRAG_V2
                assert hybrid_meta["individual_scores"]["basic"] == 0.8
                # Expected combined score: 0.8 * 0.3 = 0.24
                assert hybrid_meta["combined_score"] == pytest.approx(0.24)

            elif doc_info["id"] == "graph_doc_1":
                assert hybrid_meta["sources"] == ["GraphRAG"] # Updated from GraphRAG_V2
                assert hybrid_meta["individual_scores"]["graph"] == 0.9
                # Expected combined score: 0.9 * 0.4 = 0.36
                assert hybrid_meta["combined_score"] == pytest.approx(0.36)

            elif doc_info["id"] == "hyde_doc_1":
                assert hybrid_meta["sources"] == ["HyDE"] # Updated from HyDE_V2
                assert hybrid_meta["individual_scores"]["hyde"] == 0.75
                # Expected combined score: 0.75 * 0.3 = 0.225
                assert hybrid_meta["combined_score"] == pytest.approx(0.225)

        # Check metadata in the result
        assert result["metadata"]["pipeline"] == "HybridiFindRAG" # Updated from HybridiFindRAG_V2
        assert result["metadata"]["top_k"] == 3
        assert len(result["entities"]) == 2
        assert result["hypothetical_document"].startswith(hypothetical_doc_hyde[:200])

        # Check that LLM was called with context containing elements from all sources
        llm_call_args = mock_llm_func.call_args[0][0] # Get the prompt string
        assert "Hybrid Context:" in llm_call_args
        assert "Hypothetical Answer:" in llm_call_args
        assert hypothetical_doc_hyde[:50] in llm_call_args # Check part of hypothetical doc
        assert "Key Entities:" in llm_call_args
        assert "Neurodegenerative Diseases" in llm_call_args # Check entity name
        assert "Content from BasicRAG" in llm_call_args or \
               "Content from GraphRAG" in llm_call_args or \
               "Shared content" in llm_call_args # Check document content based on top merged docs

        # Verify that the context for LLM includes source and score information for documents
        assert "Sources: BasicRAG, GraphRAG" in llm_call_args or \
               "Sources: BasicRAG" in llm_call_args or \
               "Sources: GraphRAG" in llm_call_args or \
               "Sources: HyDE" in llm_call_args
        
        assert "Scores: basic=" in llm_call_args or \
               "Scores: graph=" in llm_call_args or \
               "Scores: hyde=" in llm_call_args