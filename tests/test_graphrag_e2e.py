import pytest
import sys # Added for path manipulation
import os # Added for path manipulation

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from iris_rag.pipelines.graphrag import GraphRAGPipeline

# According to .clinerules, tests use real data and pytest fixtures.
# We assume the database is populated by fixtures in a main conftest.py
# (e.g., @pytest.mark.usefixtures("loaded_db_with_graph_data"))
# This ensures that entities like 'BRCA1' and their relationships exist.

def test_graphrag_e2e_protein_interaction_and_pathways():
    """
    Tests GraphRAG's ability to answer complex queries requiring graph traversal
    for entity relationships, such as protein interactions within specific pathways.
    This test relies on the database having relevant entities (e.g., BRCA1,
    interacting proteins, cancer-related pathways) and their relationships populated.
    The entity types in the database must align with what GraphRAG expects or
    be broad enough to capture these biological entities.
    """
    pipeline = GraphRAGPipeline() 

    # Query designed to test graph traversal for relationships and context
    query = "What proteins interact with BRCA1 in cancer pathways?"
    
    result = pipeline.run(query_text=query, top_k=5) # top_k for documents

    # Basic assertions for pipeline execution
    assert result is not None, "Pipeline should return a result."
    assert "answer" in result, "Result should contain an answer."
    assert "retrieved_documents" in result, "Result should contain retrieved documents."
    assert "method" in result, "Result should specify the retrieval method."
    assert "query" in result and result["query"] == query, "Result should echo the query."

    answer = result["answer"]
    answer_lower = answer.lower() # For case-insensitive checks
    retrieved_docs_count = result["document_count"]
    method = result["method"]

    # Print details for debugging and manual verification
    print(f"\n--- GraphRAG E2E Test ---")
    print(f"Query: {query}")
    print(f"Method Used: {method}")
    print(f"Answer: {answer}")
    print(f"Retrieved Documents Count: {retrieved_docs_count}")
    
    for i, doc_data in enumerate(result['retrieved_documents']):
        doc_id = doc_data.get('id', 'N/A')
        doc_score = doc_data.get('score')
        score_str = f"{doc_score:.3f}" if isinstance(doc_score, float) else str(doc_score)
        # Ensure content is a string before slicing
        content_snippet = str(doc_data.get('content', ''))[:150] 
        print(f"  Doc {i+1}: ID={doc_id}, Score={score_str}, Content='{content_snippet}...'")
    print(f"--- End of Test Details ---")

    # Assert that GraphRAG's specific method was used
    # OriginalGraphRAGPipeline (now always used) reports "knowledge_graph_traversal"
    expected_method = "knowledge_graph_traversal"
    assert method == expected_method, \
        f"GraphRAG should use '{expected_method}', but used '{method}'. This might indicate a fallback or wrong pipeline implementation."

    # Assert that relevant information was found
    assert retrieved_docs_count > 0, \
        "GraphRAG should retrieve at least one document for this type of query."
    
    # Check for generic failure messages in the answer
    failure_phrases = [
        "could not find relevant information", 
        "cannot answer based on the provided information",
        "i'm sorry",
        "i do not have enough information"
    ]
    assert not any(phrase in answer_lower for phrase in failure_phrases), \
        f"Answer appears to be a generic failure message: '{answer}'"
    
    assert len(answer) > 20, \
        f"Answer is too short, potentially indicating a problem: '{answer}'" # Min length for a meaningful answer

    # Assertions related to the query's specific entities and concepts
    # These depend on the LLM's generation and the underlying graph data.
    
    # BRCA1 should be central
    assert "brca1" in answer_lower or \
           any("brca1" in str(doc_data.get('content', '')).lower() for doc_data in result['retrieved_documents']), \
           "The entity 'BRCA1' from the query should be present in the answer or retrieved documents."
    
    # Keywords indicating interactions or pathways should be present
    # This demonstrates that the graph traversal likely found related entities/concepts.
    relationship_keywords = ["interact", "interaction", "binds", "binding", "complex", "associate", "pathway", "regulation", "role in cancer"]
    assert any(keyword in answer_lower for keyword in relationship_keywords) or \
           any(keyword in str(doc_data.get('content', '')).lower() for doc_data in result['retrieved_documents'] for keyword in relationship_keywords), \
           "The answer or retrieved documents should contain keywords related to protein interactions or pathways."

    # Optional: If specific interacting proteins or pathway names are expected from test data,
    # they could be asserted here. e.g., if "TP53" is a known interactor of BRCA1 in the test graph.
    # Example:
    # expected_partner = "tp53"
    # assert expected_partner in answer_lower or \
    #        any(expected_partner in str(doc_data.get('content', '')).lower() for doc_data in result['retrieved_documents']), \
    #        f"Expected interacting protein '{expected_partner}' not found in results for BRCA1 query."
    # This is commented out as it requires specific test data knowledge.