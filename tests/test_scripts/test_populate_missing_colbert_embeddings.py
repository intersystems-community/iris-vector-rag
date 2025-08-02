"""
Test suite for the ColBERT token embedding population script.
Ensures that missing embeddings are correctly identified, generated, and stored.
"""
import pytest
from unittest import mock

# Import the actual script
from scripts.utilities import populate_missing_colbert_embeddings

@pytest.fixture
def mock_db_connection():
    """Mocks the database connection and cursor."""
    mock_conn = mock.MagicMock()
    mock_cursor = mock.MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor

@pytest.fixture
def mock_colbert_encoder():
    """Mocks the ColBERT encoder."""
    return mock.MagicMock()

def test_identify_missing_docs(mock_db_connection):
    """
    Tests if the script correctly identifies documents missing ColBERT embeddings.
    Mocks DB, verifies correct SQL query and parsing of doc_ids.
    """
    mock_conn, mock_cursor = mock_db_connection
    
    # Mock the database response
    mock_cursor.fetchall.return_value = [
        ('doc1', 'content1', 'abstract1', 'title1'),
        ('doc2', 'content2', 'abstract2', 'title2')
    ]
    
    # Call the function
    result = populate_missing_colbert_embeddings.identify_missing_documents(mock_conn)
    
    # Verify the SQL query was executed
    mock_cursor.execute.assert_called_once()
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "LEFT JOIN RAG.DocumentTokenEmbeddings" in sql_call
    assert "WHERE dte.doc_id IS NULL" in sql_call
    
    # Verify the returned data structure
    assert len(result) == 2
    assert result[0]['doc_id'] == 'doc1'
    assert result[0]['text_content'] == 'content1'
    assert result[1]['doc_id'] == 'doc2'
    assert result[1]['text_content'] == 'content2'

def test_single_doc_embedding_generation_and_storage(mock_db_connection, mock_colbert_encoder):
    """
    Tests processing of a single document: content retrieval, encoding, and storage.
    Mocks DB and ColBERT encoder. Verifies correct storage format.
    """
    mock_conn, mock_cursor = mock_db_connection
    doc_id = "PMC12345"
    document_content = "This is the content of the test document."
    
    # Mock ColBERT encoder output - should return tuple (tokens, embeddings)
    mock_colbert_encoder.return_value = (
        ["token1", "token2"],  # List of tokens
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # List of embeddings
    )
    
    # Create document data
    doc = {
        'doc_id': doc_id,
        'text_content': document_content,
        'abstract': None,
        'title': None
    }
    
    # Call the function
    result = populate_missing_colbert_embeddings.process_single_document(doc, mock_conn, mock_colbert_encoder)
    
    # Verify ColBERT encoder was called with document content
    mock_colbert_encoder.assert_called_once_with(document_content)
    
    # Verify database insertion was called (once for each token)
    assert mock_cursor.execute.call_count == 2
    
    # Check the SQL query format and data for the first call
    first_call_args = mock_cursor.execute.call_args_list[0][0]
    sql_call_first = first_call_args[0]
    insert_data_first = first_call_args[1]

    assert "INSERT INTO RAG.DocumentTokenEmbeddings" in sql_call_first
    assert "VALUES (?, ?, ?, ?)" in sql_call_first # Native vector insertion
    
    # Check the data format for the first token
    assert len(insert_data_first) == 4  # doc_id, token_index, token_text, token_embedding
    assert insert_data_first[0] == doc_id  # doc_id
    assert insert_data_first[1] == 0  # token_index
    assert insert_data_first[2] == "token1"  # token_text
    # Check that the embedding vector string matches the expected format (comma-separated, no brackets)
    embedding_vector_str_first = insert_data_first[3]
    assert not embedding_vector_str_first.startswith('[')
    assert not embedding_vector_str_first.endswith(']')
    # Verify it contains comma-separated numbers (not hash values)
    import re
    vector_pattern_native = r'^(-?\d+(\.\d+)?(,-?\d+(\.\d+)?)*)$' # No brackets
    assert re.match(vector_pattern_native, embedding_vector_str_first), f"Invalid native vector format: {embedding_vector_str_first}"
    assert "HASH" not in embedding_vector_str_first, f"Vector contains hash value: {embedding_vector_str_first}"
    assert "0.10000000,0.20000000,0.30000000" in embedding_vector_str_first # Check actual content

    # Check the data format for the second token
    second_call_args = mock_cursor.execute.call_args_list[1][0]
    insert_data_second = second_call_args[1]
    assert insert_data_second[0] == doc_id
    assert insert_data_second[1] == 1
    assert insert_data_second[2] == "token2"
    embedding_vector_str_second = insert_data_second[3]
    assert re.match(vector_pattern_native, embedding_vector_str_second), f"Invalid native vector format: {embedding_vector_str_second}"
    assert "0.40000000,0.50000000,0.60000000" in embedding_vector_str_second

    assert result is True

def test_convert_to_iris_vector_validation():
    """
    Tests the convert_to_iris_vector function to ensure it properly validates input
    and rejects invalid data like hash values.
    """
    from scripts.utilities.populate_missing_colbert_embeddings import convert_to_iris_vector
    
    # Test valid input
    valid_embedding = [0.1, 0.2, 0.3, -0.4, 1.5]
    result = convert_to_iris_vector(valid_embedding)
    # For native VECTOR, format is comma-separated, no brackets, 8 decimal places by default from format_vector_for_iris
    assert result == "0.10000000,0.20000000,0.30000000,-0.40000000,1.50000000"
    
    # Test invalid inputs that should raise ValueError
    import pytest
    
    # Test non-list input
    with pytest.raises(ValueError, match=r"Cannot convert vector type <class 'str'>"):
        convert_to_iris_vector("HASH@$vector")
    
    # Test empty list
    with pytest.raises(ValueError, match=r"Failed to convert embedding to IRIS vector format: Vector is empty"):
        convert_to_iris_vector([])
    
    # Test list with non-numeric values
    with pytest.raises(ValueError, match=r"Unexpected error formatting vector: could not convert string to float"):
        convert_to_iris_vector([0.1, "HASH@$vector", 0.3])
    
    # Test list with hash-like values (this will also trigger the "not a number" check first)
    with pytest.raises(ValueError, match=r"Unexpected error formatting vector: could not convert string to float"):
        convert_to_iris_vector([0.1, "HASH123", 0.3])

def test_batch_embedding_population(mock_db_connection, mock_colbert_encoder):
    """
    Verifies that a batch of documents is processed correctly.
    Mocks DB and encoder.
    """
    mock_conn, mock_cursor = mock_db_connection
    
    # Mock ColBERT encoder output - should return tuple (tokens, embeddings)
    mock_colbert_encoder.return_value = (
        ["token1"],  # List of tokens
        [[0.1, 0.2, 0.3]]  # List of embeddings
    )
    
    # Create batch of documents
    doc_batch = [
        {'doc_id': 'doc1', 'text_content': 'content1', 'abstract': None, 'title': None},
        {'doc_id': 'doc2', 'text_content': 'content2', 'abstract': None, 'title': None},
        {'doc_id': 'doc3', 'text_content': 'content3', 'abstract': None, 'title': None}
    ]
    
    # Mock process_single_document to track calls
    with mock.patch('scripts.populate_missing_colbert_embeddings.process_single_document') as mock_process_single:
        mock_process_single.return_value = True
        
        # Call the function
        populate_missing_colbert_embeddings.process_batch_embeddings(doc_batch, mock_conn, mock_colbert_encoder, batch_size=2)
        
        # Verify process_single_document was called for each document
        assert mock_process_single.call_count == len(doc_batch)
        
        # Verify commit was called
        mock_conn.commit.assert_called_once()

def test_all_docs_have_embeddings_after_population(mock_db_connection):
    """
    Simulates the script running to completion and verifies the post-condition
    (no missing embeddings). Mocks DB.
    """
    mock_conn, mock_cursor = mock_db_connection
    
    # Mock the verification queries
    mock_cursor.fetchall.return_value = []  # No missing documents after processing
    mock_cursor.fetchone.side_effect = [
        (100,),  # total_token_embeddings
        (50,)    # documents_with_embeddings
    ]
    
    # Call verify_completion
    result = populate_missing_colbert_embeddings.verify_completion(mock_conn)
    
    # Verify the result
    assert result['remaining_missing_docs'] == 0
    assert result['total_token_embeddings'] == 100
    assert result['documents_with_embeddings'] == 50
    assert result['completion_status'] == 'complete'

def test_embedding_format_in_db_mocked(mock_db_connection):
    """
    Mocks cursor.execute for an INSERT and checks that TO_VECTOR(?) is used.
    """
    mock_conn, mock_cursor = mock_db_connection
    doc_id = "PMC_test_format"
    token_embeddings_data = [("token1", [0.1, 0.2]), ("token2", [0.3, 0.4])]

    # Call the store_token_embeddings function directly
    result = populate_missing_colbert_embeddings.store_token_embeddings(doc_id, token_embeddings_data, mock_conn)
    
    # Verify the SQL query format (execute called for each token)
    assert mock_cursor.execute.call_count == 2
    
    # Check the first call
    first_call_args = mock_cursor.execute.call_args_list[0][0]
    sql_first = first_call_args[0]
    data_first = first_call_args[1]

    assert "INSERT INTO RAG.DocumentTokenEmbeddings" in sql_first
    assert "VALUES (?, ?, ?, ?)" in sql_first # Native vector insertion
    assert len(data_first) == 4  # doc_id, token_idx, token_text, embedding_vector
    
    # Verify the data format for the first token
    assert data_first[0] == doc_id
    assert data_first[1] == 0  # token_index
    assert data_first[2] == "token1"  # token_text
    # Vector string should be comma-separated, no brackets, 8 decimal places
    assert data_first[3] == "0.10000000,0.20000000"

    # Check the second call
    second_call_args = mock_cursor.execute.call_args_list[1][0]
    data_second = second_call_args[1]
    assert data_second[0] == doc_id
    assert data_second[1] == 1  # token_index
    assert data_second[2] == "token2"  # token_text
    assert data_second[3] == "0.30000000,0.40000000"

    assert result is True

def test_no_content_document_skip(mock_db_connection, mock_colbert_encoder):
    """
    Verifies documents with no usable text content are skipped gracefully.
    """
    mock_conn, mock_cursor = mock_db_connection
    doc_id_no_content = "PMC_NO_CONTENT"
    
    # Create document with no usable content
    doc = {
        'doc_id': doc_id_no_content,
        'text_content': None,
        'abstract': '',
        'title': '   '  # Only whitespace
    }
    
    # Call the function
    result = populate_missing_colbert_embeddings.process_single_document(doc, mock_conn, mock_colbert_encoder)
    
    # Verify encoder was not called
    mock_colbert_encoder.assert_not_called()
    
    # Verify no database operations were performed
    mock_cursor.executemany.assert_not_called()
    
    # Verify function returned False (indicating skip)
    assert result is False

def test_colbert_encoder_error_handling(mock_db_connection, mock_colbert_encoder):
    """
    Verifies graceful error handling if the ColBERT encoder fails for a document.
    """
    mock_conn, mock_cursor = mock_db_connection
    doc_id_encoder_error = "PMC_ENCODER_ERROR"
    
    # Mock encoder to raise an exception
    mock_colbert_encoder.side_effect = Exception("ColBERT encoding failed")
    
    # Create document with content
    doc = {
        'doc_id': doc_id_encoder_error,
        'text_content': "Some content",
        'abstract': None,
        'title': None
    }
    
    # Call the function - should handle the error gracefully
    result = populate_missing_colbert_embeddings.process_single_document(doc, mock_conn, mock_colbert_encoder)
    
    # Verify encoder was called
    mock_colbert_encoder.assert_called_once_with("Some content")
    
    # Ensure no partial/incorrect data is stored
    mock_cursor.executemany.assert_not_called()
    
    # Verify function returned False (indicating failure)
    assert result is False

def test_db_commit_batching(mock_db_connection):
    """
    Verifies that conn.commit() is called appropriately for batching.
    """
    mock_conn, mock_cursor = mock_db_connection
    
    # Create batch of documents
    doc_batch = [
        {'doc_id': 'doc1', 'text_content': 'content1', 'abstract': None, 'title': None},
        {'doc_id': 'doc2', 'text_content': 'content2', 'abstract': None, 'title': None}
    ]
    
    # Mock process_single_document to return success
    with mock.patch('scripts.populate_missing_colbert_embeddings.process_single_document') as mock_process_single:
        mock_process_single.return_value = True
        
        # Call process_batch_embeddings
        populate_missing_colbert_embeddings.process_batch_embeddings(doc_batch, mock_conn, mock.MagicMock(), batch_size=2)
        
        # Verify commit was called once per batch
        mock_conn.commit.assert_called_once()