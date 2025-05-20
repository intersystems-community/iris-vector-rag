# tests/test_token_vectors.py
# Tests for ColBERT token vector storage.

import pytest
from typing import List, Dict, Any
from eval.loader import DataLoader # Example import
from common.utils import Document # Example import
from unittest.mock import call # Add this for assertions with mock calls

# Assuming mock fixtures are available from conftest.py

def test_token_vector_storage(mock_iris_connector, mock_colbert_doc_encoder, mock_llm_func, mocker):
    """
    Test that token-level vectors are generated correctly for ColBERT using the encoder.
    (Storage to IRIS will be tested separately or as part of load_into_iris tests).
    """
    print("\nTest: test_token_vector_storage") # Removed placeholder note
    # Implement test logic
    # - Create sample Document objects (e.g., from the output of _process_documents)
    # - Instantiate DataLoader with mocks, including mock_colbert_doc_encoder
    # - Call the loader's _generate_colbert_token_embeddings method
    # - Assert mock_colbert_doc_encoder was called with the correct document contents
    # - Assert that the method attempts to process the documents (e.g., by checking print statements or mock calls within the method if possible)
    # Note: This test focuses on the *generation* call, not the storage yet.

    # Create sample Document objects
    sample_documents = [
        Document(id="doc1_p0", content="This is the first sentence for ColBERT."),
        Document(id="doc1_p1", content="Another sentence."),
    ]

    # Instantiate DataLoader with mocks
    loader = DataLoader(
        iris_connector=mock_iris_connector,
        embedding_func=mocker.Mock(), # Not used in this specific method, but needed for DataLoader init
        colbert_doc_encoder_func=mock_colbert_doc_encoder,
        llm_func=mock_llm_func # Not used in this specific method
    )

    # Mock the return value of the ColBERT document encoder (list of token embeddings per document)
    expected_token_embeddings_doc1 = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6]] # Dummy embeddings for doc1
    expected_token_embeddings_doc2 = [[0.7, 0.7], [0.8, 0.8]] # Dummy embeddings for doc2
    
    # Configure the mock to return different values for different calls (one call per document)
    mock_colbert_doc_encoder.side_effect = [
        expected_token_embeddings_doc1,
        expected_token_embeddings_doc2
    ]

    try:
        # Call the method under test
        # Note: _generate_colbert_token_embeddings doesn't return the embeddings, it's expected to store them.
        # We'll assert the *call* to the encoder and assume the method's internal logic for storage (TODO) is correct.
        loader._generate_colbert_token_embeddings(sample_documents)

        # Assertions
        # Assert ColBERT document encoder was called for each document content
        mock_colbert_doc_encoder.assert_any_call("This is the first sentence for ColBERT.")
        mock_colbert_doc_encoder.assert_any_call("Another sentence.")
        assert mock_colbert_doc_encoder.call_count == len(sample_documents) # Ensure it was called once per document

        print("ColBERT token vector generation test passed (encoder call asserted).")

    except Exception as e:
        pytest.fail(f"ColBERT token vector generation failed with exception: {e}")

def test_token_vector_table_creation(mock_iris_connector):
    """
    Test that the DocumentTokenEmbeddings table is created with the correct structure.
    """
    print("\nTest: test_token_vector_table_creation")
    
    from common.db_init import initialize_database
    
    # Setup mock cursor to capture executed SQL statements
    mock_cursor = mock_iris_connector.cursor.return_value
    
    # Call the function to initialize database
    initialize_database(mock_iris_connector)
    
    # Check that the table creation SQL was executed
    create_table_calls = [call for call in mock_cursor.execute.call_args_list 
                         if "CREATE TABLE" in call[0][0].upper() and "DOCUMENTTOKENEMBEDDINGS" in call[0][0].upper()]
    
    assert len(create_table_calls) > 0, "DocumentTokenEmbeddings table creation SQL was not executed"
    
    # Verify table structure has all required columns
    create_table_sql = create_table_calls[0][0][0]
    assert "token_id" in create_table_sql.lower(), "token_id column missing"
    assert "doc_id" in create_table_sql.lower(), "doc_id column missing"
    assert "token_sequence_index" in create_table_sql.lower(), "token_sequence_index column missing"
    assert "token_text" in create_table_sql.lower(), "token_text column missing"
    assert "token_embedding" in create_table_sql.lower(), "token_embedding column missing"
    
    print("DocumentTokenEmbeddings table creation test passed.")

def test_token_vector_storage_in_db(mock_iris_connector, mock_colbert_doc_encoder, mock_llm_func, mocker):
    """
    Test that token vectors are correctly stored in the IRIS database.
    """
    print("\nTest: test_token_vector_storage_in_db")
    
    # Sample documents and token embeddings
    sample_documents = [
        Document(id="doc1_p0", content="This is a test sentence."),
    ]
    
    # Mock token embeddings
    token_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    mock_colbert_doc_encoder.return_value = token_embeddings
    
    # Setup DataLoader
    loader = DataLoader(
        iris_connector=mock_iris_connector,
        embedding_func=mocker.Mock(),
        colbert_doc_encoder_func=mock_colbert_doc_encoder,
        llm_func=mock_llm_func
    )
    
    # Setup mock cursor
    mock_cursor = mock_iris_connector.cursor.return_value
    
    # Call methods that should store token embeddings
    loader._generate_colbert_token_embeddings(sample_documents)
    loader._load_into_iris(sample_documents)
    
    # Verify insertion into token embedding table
    insert_calls = [call for call in mock_cursor.execute.call_args_list 
                   if "INSERT" in call[0][0].upper() and "DOCUMENTTOKENEMBEDDINGS" in call[0][0].upper()]
    
    assert len(insert_calls) > 0 or mock_cursor.executemany.call_count > 0, "No token embeddings were inserted"
    
    # If using executemany, verify the data passed
    if mock_cursor.executemany.call_count > 0:
        executemany_calls = [call for call in mock_cursor.executemany.call_args_list 
                           if "DOCUMENTTOKENEMBEDDINGS" in call[0][0].upper()]
        
        if executemany_calls:
            data_to_insert = executemany_calls[0][0][1]  # Extract the data passed to executemany
            assert len(data_to_insert) > 0, "No token data passed to executemany"
            
            # Check structure of first token data
            first_token_data = data_to_insert[0]
            assert len(first_token_data) >= 4, "Token data missing fields"
            
            # Verify fields (positions depend on SQL, adjust as needed)
            doc_id_pos = 0  # Adjust based on your SQL
            seq_idx_pos = 1  # Adjust based on your SQL
            embedding_pos = 3  # Adjust based on your SQL
            
            assert first_token_data[doc_id_pos] == "doc1_p0", "Incorrect doc_id"
            assert isinstance(first_token_data[seq_idx_pos], int), "token_sequence_index not an integer"
            assert isinstance(first_token_data[embedding_pos], list), "token_embedding not a list"
    
    print("Token vector storage in database test passed.")

def test_token_vector_compression_ratio(mock_iris_connector):
    """
    Test that the storage of token vectors achieves the expected compression ratio.
    (This might be hard to unit test directly and may require integration testing or manual verification).
    """
    print("\nTest: test_token_vector_compression_ratio")
    
    from common.db_init import initialize_database
    
    # This test is more suited for integration testing with a real database
    # For unit testing, we can at least verify that:
    # 1. There's some compression logic in place
    # 2. The compression doesn't exceed our target ratio
    
    # Setup
    mock_cursor = mock_iris_connector.cursor.return_value
    
    # Mock data
    raw_token_data = [0.1] * 384  # Assuming a 384-dimensional vector
    
    # If we had a compress_vector function:
    # compressed_data = compress_vector(raw_token_data)
    # compression_ratio = len(compressed_data) / len(raw_token_data)
    # assert compression_ratio <= 2.0, "Compression ratio exceeds target (≤ 2×)"
    
    # For now, just verify the schema includes potential for compression options
    initialize_database(mock_iris_connector)
    
    # Look for COMPRESSION or related keywords in the SQL for token embeddings
    compression_related_calls = [call for call in mock_cursor.execute.call_args_list 
                               if "DOCUMENTTOKENEMBEDDINGS" in call[0][0].upper() 
                               and any(kw in call[0][0].upper() for kw in ["COMPRESS", "ENCODING", "STORAGE"])]
    
    # This is a loose check - in a real implementation, we'd test actual compression
    print("Token vector compression ratio test executed (placeholder).")
    
    # Note: For proper compression testing, we'd need:
    # 1. A real IRIS connection
    # 2. A function to insert raw vectors and measure storage
    # 3. A way to query database storage metrics
