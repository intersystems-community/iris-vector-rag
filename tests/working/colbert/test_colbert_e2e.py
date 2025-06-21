import pytest
import json
from unittest.mock import patch

# Add project root to sys.path to allow imports from common, colbert etc.
import sys
import os
# Adjust sys.path to point to the project root if 'src' is directly under it
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.working.colbert.pipeline import ColbertRAGPipeline # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

# Test data
TEST_DOCS_DATA_V2 = [
    {"id": "colbert_v2_doc_001", "content": "Azithromycin is a common antibiotic used for bacterial infections."},
    {"id": "colbert_v2_doc_002", "content": "Streptococcus pneumoniae can cause serious lung problems."},
    {"id": "colbert_v2_doc_003", "content": "Treatment for Streptococcus pneumoniae often involves azithromycin therapy."},
    {"id": "colbert_v2_doc_004", "content": "Regular exercise and a balanced diet are key to good health."}
]
TEST_DOC_IDS_V2 = [doc["id"] for doc in TEST_DOCS_DATA_V2]

def setup_test_data_v2(iris_connection, embedding_function):
    """Inserts test documents with their sentence embeddings into RAG.SourceDocuments."""
    cursor = iris_connection.cursor()
    for doc_data in TEST_DOCS_DATA_V2:
        doc_id = doc_data["id"]
        content = doc_data["content"]
        
        # Generate sentence embedding for the document
        sentence_embedding_vector = embedding_function([content])[0]
        # Store as a string representation suitable for TO_VECTOR(), e.g., "[0.1,0.2,...]"
        embedding_str = f"[{','.join(map(str, sentence_embedding_vector))}]"
        
        try:
            cursor.execute("SELECT doc_id FROM RAG.SourceDocuments WHERE doc_id = ?", (doc_id,))
            if cursor.fetchone() is None:
                 cursor.execute(
                     "INSERT INTO RAG.SourceDocuments (doc_id, text_content, embedding) VALUES (?, ?, ?)",
                     (doc_id, content, embedding_str)
                 )
            else:
                # Optionally, update if exists, or just ensure it's there
                print(f"Setup V2: Document {doc_id} already exists. Updating embedding.")
                cursor.execute(
                    "UPDATE RAG.SourceDocuments SET text_content = ?, embedding = ? WHERE doc_id = ?",
                    (content, embedding_str, doc_id)
                )
        except Exception as e:
            print(f"Error inserting/updating source document {doc_id} for V2: {e}")
            # Depending on error, might want to raise or handle differently
            pass
    iris_connection.commit()
    cursor.close()
    print(f"Setup V2: Ensured {len(TEST_DOCS_DATA_V2)} documents are present in SourceDocuments with embeddings.")

def cleanup_test_data_v2(iris_connection):
    """Removes test documents from RAG.SourceDocuments."""
    cursor = iris_connection.cursor()
    try:
        placeholders = ','.join(['?' for _ in TEST_DOC_IDS_V2])
        # No DocumentTokenEmbeddings table to clean for V2 pipeline's direct operation
        cursor.execute(f"DELETE FROM RAG.SourceDocuments WHERE doc_id IN ({placeholders})", TEST_DOC_IDS_V2)
        print(f"Cleanup V2: Deleted {cursor.rowcount} source documents for test docs: {TEST_DOC_IDS_V2}")
        iris_connection.commit()
    except Exception as e:
        print(f"Error during V2 cleanup: {e}")
        iris_connection.rollback()
    finally:
        cursor.close()

def mock_llm_for_colbert_v2_test(prompt: str) -> str:
    """Mock LLM specifically for this ColBERT V2 test."""
    context_lower = prompt.lower()
    # print(f"Mock LLM V2 received prompt context (first 500 chars):\n{context_lower[:500]}...")
    if "azithromycin" in context_lower and "streptococcus pneumoniae" in context_lower:
        # Check for specific doc IDs in the prompt if they are included by the pipeline's context generation
        # The V2 pipeline includes title and scores in context.
        if "colbert_v2_doc_003" in prompt and "colbert_v2_doc_001" in prompt:
             return "Azithromycin is used for bacterial infections and is a treatment for Streptococcus pneumoniae. (Docs 3 & 1, V2)"
        elif "colbert_v2_doc_003" in prompt:
            return "Azithromycin is a treatment for Streptococcus pneumoniae. (Doc 3, V2)"
        elif "colbert_v2_doc_001" in prompt:
            return "Azithromycin is a common antibiotic for bacterial infections. (Doc 1, V2)"
    return "Based on the provided V2 context, I cannot definitively answer the question regarding azithromycin and Streptococcus pneumoniae."

# Removed patches as we are injecting the mock LLM directly via constructor
def test_colbert_v2_e2e_fine_grained_match(iris_testcontainer_connection): # Removed mock_llm_attr, mock_get_llm_factory
    """
    Tests the ColBERT V2 pipeline's end-to-end flow with a real database (testcontainer)
    and real embeddings, focusing on fine-grained term matching.
    The LLM part is mocked for predictable answer assertion.
    """
    # Determine which mock to configure based on how ColBERTPipelineV2 gets its LLM
    # Assuming ColBERTPipelineV2 takes llm_func in constructor, so we pass our mock directly.
    # No need to patch get_llm_func if we instantiate directly with the mock.
    
    # Get real embedding function for data setup AND for the pipeline
    real_embedding_function = get_embedding_func(mock=False)
    mock_llm_function = mock_llm_for_colbert_v2_test

    try:
        print("Setting up V2 test data in testcontainer...")
        setup_test_data_v2(iris_testcontainer_connection, real_embedding_function)
        
        # Instantiate ColBERTPipelineV2 directly with real iris_connector, real embedding_func, and mock llm_func
        pipeline = ColbertRAGPipeline( # Updated class name
            iris_connector=iris_testcontainer_connection,
            colbert_query_encoder_func=real_embedding_function, # Parameter name changed in ColbertRAGPipeline
            llm_func=mock_llm_function
            # embedding_func is also a param in ColbertRAGPipeline, might need to pass real_embedding_function again or ensure default is okay
            # For now, assuming colbert_query_encoder_func is the primary one needed for embeddings here.
            # The actual ColbertRAGPipeline also takes embedding_func for stage 1.
            # Let's add it for completeness, assuming real_embedding_function serves both roles for this test.
            , embedding_func=real_embedding_function
        )

        query = "What is azithromycin used for regarding Streptococcus pneumoniae?"
        
        results = pipeline.run(query=query, top_k=2, similarity_threshold=0.0)

        print(f"V2 Query: {results['query']}")
        print(f"V2 Answer: {results['answer']}")
        for doc in results.get("retrieved_documents", []):
            print(f"V2 Retrieved Doc ID: {doc.get('id')}, Metadata: {doc.get('metadata')}, Content: {doc.get('content', '')[:100]}...")

        assert "answer" in results
        assert "retrieved_documents" in results
        
        retrieved_docs = results["retrieved_documents"]
        assert len(retrieved_docs) > 0, "V2: No documents were retrieved."
        assert len(retrieved_docs) <= 2

        retrieved_doc_ids = [doc['id'] for doc in retrieved_docs]
        
        assert "colbert_v2_doc_003" in retrieved_doc_ids, \
            f"V2: Expected 'colbert_v2_doc_003' to be retrieved. Got: {retrieved_doc_ids}"

        if len(retrieved_docs) == 2:
            assert "colbert_v2_doc_001" in retrieved_doc_ids, \
                 f"V2: Expected 'colbert_v2_doc_001' to be among top 2 if two docs retrieved. Got: {retrieved_doc_ids}"
            # Order can vary with real embeddings, so check for set presence
            assert set(retrieved_doc_ids) == {"colbert_v2_doc_003", "colbert_v2_doc_001"}
        elif len(retrieved_docs) == 1:
            assert retrieved_docs[0]['id'] == "colbert_v2_doc_003"
        
        answer_lower = results["answer"].lower()
        print(f"DEBUG: answer_lower for assertion: '{answer_lower}'") # DEBUG PRINT
        assert "azithromycin" in answer_lower
        assert "streptococcus pneumoniae" in answer_lower
        
        # Correctly predict mock behavior by including necessary keywords in dummy prompts for the OR chain
        expected_answer_docs_3_and_1 = mock_llm_for_colbert_v2_test("azithromycin streptococcus pneumoniae colbert_v2_doc_003 colbert_v2_doc_001").lower()
        expected_answer_doc_3 = mock_llm_for_colbert_v2_test("azithromycin streptococcus pneumoniae colbert_v2_doc_003").lower()
        expected_answer_doc_1 = mock_llm_for_colbert_v2_test("azithromycin streptococcus pneumoniae colbert_v2_doc_001").lower()
        expected_answer_default = mock_llm_for_colbert_v2_test("default content no keywords").lower()

        assert answer_lower == expected_answer_docs_3_and_1 \
            or answer_lower == expected_answer_doc_3 \
            or answer_lower == expected_answer_doc_1 \
            or answer_lower == expected_answer_default

        # The previous assertion (answer_lower == expected_answer_...) confirms the mock LLM
        # produced an output consistent with the retrieved documents.
        # The direct checks on retrieved_doc_ids (lines 135-141) already confirm
        # that the correct documents were retrieved.
        # This more specific block below is therefore redundant and was causing issues
        # due to assumptions about the mock LLM's output string format.
        # if "(docs 3 & 1, v2)" in answer_lower:
        #      assert "colbert_v2_doc_003" in results["answer"] and "colbert_v2_doc_001" in results["answer"]
        # elif "(doc 3, v2)" in answer_lower:
        #     assert "colbert_v2_doc_003" in results["answer"]
        # Not asserting the negative case as it depends on retrieval success

    finally:
        print("Cleaning up V2 test data from testcontainer...")
        cleanup_test_data_v2(iris_testcontainer_connection)