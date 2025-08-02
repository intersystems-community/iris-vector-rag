"""
Test suite for the comprehensive RAGAS evaluation script.
Ensures that the RAGAS evaluation framework is correctly executed,
including API key checks, dataset validation, pipeline evaluation,
metric calculation, and report generation.
"""
import pytest
import os
import json
from unittest import mock

# Import the actual script now that it's created
from scripts.utilities.evaluation import execute_comprehensive_ragas_evaluation

@pytest.fixture
def mock_db_connection():
    """Mocks the database connection and cursor."""
    mock_conn = mock.MagicMock()
    mock_cursor = mock.MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor

@pytest.fixture
def mock_config_manager():
    """Mocks the ConfigurationManager."""
    mock_cfg = mock.MagicMock()
    # Example: mock_cfg.get_setting.return_value = "some_value"
    return mock_cfg

@pytest.fixture
def sample_evaluation_queries():
    """Provides sample evaluation queries similar to 'eval/sample_queries.json'."""
    return [
        {"query_id": "q1", "query_text": "What is Colbert?"},
        {"query_id": "q2", "query_text": "How does RAGAS work?"}
    ]

def test_openai_api_key_check_present(monkeypatch):
    """
    Mocks os.environ to provide the API key, ensures no error.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_present")
    # Should not raise an exception
    execute_comprehensive_ragas_evaluation.validate_openai_api_key()

def test_openai_api_key_check_missing(monkeypatch):
    """
    Mocks os.environ to remove/empty the API key, asserts SystemExit is raised.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "") # Also test empty string
    with pytest.raises(SystemExit):
        execute_comprehensive_ragas_evaluation.validate_openai_api_key()

def test_dataset_completeness_check_sufficient_data(mock_db_connection):
    """
    Mocks DB to return counts indicating 1000 docs, all with embeddings.
    Ensures no error.
    """
    mock_conn, mock_cursor = mock_db_connection
    # Mock return values for document count and embeddings count
    mock_cursor.fetchone.side_effect = [(1000,), (1000,)]  # Total docs, docs with embeddings
    
    # Should not raise an exception
    execute_comprehensive_ragas_evaluation.validate_dataset_completeness(mock_conn)
    
    # Verify the correct queries were executed
    assert mock_cursor.execute.call_count == 2

def test_dataset_completeness_check_insufficient_docs(mock_db_connection):
    """
    Mocks DB for < 1000 docs. Asserts SystemExit.
    """
    mock_conn, mock_cursor = mock_db_connection
    # Mock return values for insufficient documents
    mock_cursor.fetchone.side_effect = [(999,), (999,)]  # Total docs, docs with embeddings
    
    with pytest.raises(SystemExit):
        execute_comprehensive_ragas_evaluation.validate_dataset_completeness(mock_conn)

def test_dataset_completeness_check_missing_embeddings(mock_db_connection):
    """
    Mocks DB for 1000 docs but some missing embeddings. Asserts SystemExit.
    """
    mock_conn, mock_cursor = mock_db_connection
    # Mock return values for missing embeddings
    mock_cursor.fetchone.side_effect = [(1000,), (995,)]  # Total docs, docs with embeddings
    
    with pytest.raises(SystemExit):
        execute_comprehensive_ragas_evaluation.validate_dataset_completeness(mock_conn)

@mock.patch('eval.execute_comprehensive_ragas_evaluation.ChatOpenAI')
@mock.patch('eval.execute_comprehensive_ragas_evaluation.OpenAIEmbeddings')
def test_ragas_framework_initialization(MockOpenAIEmbeddings, MockChatOpenAI):
    """
    Verifies ChatOpenAI and OpenAIEmbeddings are initialized.
    Can mock the classes themselves to check instantiation parameters.
    """
    llm, embeddings, metrics = execute_comprehensive_ragas_evaluation.initialize_ragas_framework()
    
    # Verify the classes were instantiated
    MockChatOpenAI.assert_called_once_with(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1000
    )
    MockOpenAIEmbeddings.assert_called_once_with(
        model="text-embedding-3-small"
    )
    
    # Verify metrics list is returned
    assert len(metrics) == 6

def test_load_evaluation_queries(tmp_path, sample_evaluation_queries):
    """
    Verifies 'eval/sample_queries.json' is loaded and parsed correctly.
    """
    queries_file = tmp_path / "sample_queries.json"
    with open(queries_file, 'w') as f:
        json.dump(sample_evaluation_queries, f)

    with mock.patch('eval.execute_comprehensive_ragas_evaluation.QUERIES_FILE_PATH', str(queries_file)):
        loaded_queries = execute_comprehensive_ragas_evaluation.load_evaluation_queries()
        assert loaded_queries == sample_evaluation_queries

def test_evaluate_single_pipeline_mocked(sample_evaluation_queries, mock_config_manager, mock_db_connection):
    """
    Mocks a single RAG pipeline's 'query' method.
    Verifies the evaluation loop correctly calls it for all test queries
    and formats the pipeline_responses.
    """
    mock_pipeline_instance = mock.MagicMock()
    mock_pipeline_instance.query.side_effect = lambda q_text, **kwargs: {
        "query": q_text,
        "answer": f"Answer for {q_text}",
        "retrieved_documents": [{"doc_id": "d1", "text": "context1"}]
    }

    responses = execute_comprehensive_ragas_evaluation.evaluate_single_pipeline(
        mock_pipeline_instance, sample_evaluation_queries
    )
    
    assert mock_pipeline_instance.query.call_count == len(sample_evaluation_queries)
    assert len(responses) == len(sample_evaluation_queries)
    for i, response in enumerate(responses):
        assert response["question"] == sample_evaluation_queries[i]["query_text"]
        assert "answer" in response
        assert "contexts" in response
        assert response["ground_truth"] == execute_comprehensive_ragas_evaluation.NOT_APPLICABLE_GROUND_TRUTH


def test_all_pipelines_are_instantiated_and_called(sample_evaluation_queries, mock_config_manager, mock_db_connection):
    """
    Mocks all RAG pipeline classes. Verifies that the main evaluation loop
    attempts to instantiate and call evaluate_single_pipeline for each.
    """
    with mock.patch('eval.execute_comprehensive_ragas_evaluation.evaluate_single_pipeline') as mock_eval_single:
        # Mock the pipeline classes
        with mock.patch('eval.execute_comprehensive_ragas_evaluation.BasicRAGPipeline') as MockBasic:
            with mock.patch('eval.execute_comprehensive_ragas_evaluation.HyDERAGPipeline') as MockHyDE:
                mock_eval_single.return_value = []  # Return empty results
                
                results = execute_comprehensive_ragas_evaluation.execute_pipeline_evaluations(
                    sample_evaluation_queries, mock_config_manager, mock_db_connection[0]
                )
                
                # Should have attempted to instantiate pipelines
                MockBasic.assert_called_once_with(mock_config_manager, mock_db_connection[0])
                MockHyDE.assert_called_once_with(mock_config_manager, mock_db_connection[0])
                
                # Should have results for all pipelines (even if empty due to mocking)
                assert isinstance(results, dict)

@mock.patch('eval.execute_comprehensive_ragas_evaluation.evaluate') # RAGAS evaluate
@mock.patch('eval.execute_comprehensive_ragas_evaluation.Dataset')
def test_ragas_metrics_calculation_mocked(MockDataset, mock_ragas_evaluate):
    """
    Mocks the RAGAS 'evaluate' function. Provides sample pipeline_responses
    and asserts that 'evaluate' is called with the correctly formatted Dataset
    and that its (mocked) output is processed.
    """
    pipeline_results = {
        "TestPipeline": [
            {"question": "q1", "answer": "a1", "contexts": ["c1"], "ground_truth": "gt1", "success": True},
            {"question": "q2", "answer": "a2", "contexts": ["c2"], "ground_truth": "gt2", "success": True}
        ]
    }
    
    # Mock RAGAS metrics and LLM/embeddings
    mock_metrics = [mock.MagicMock(name="faithfulness"), mock.MagicMock(name="answer_relevancy")]
    mock_llm = mock.MagicMock()
    mock_embeddings = mock.MagicMock()
    mock_ragas_evaluate.return_value = {"faithfulness": 0.9, "answer_relevancy": 0.8}

    results = execute_comprehensive_ragas_evaluation.calculate_ragas_metrics(
        pipeline_results, mock_llm, mock_embeddings, mock_metrics
    )
    
    MockDataset.from_dict.assert_called_once_with({
        'question': ['q1', 'q2'],
        'answer': ['a1', 'a2'],
        'contexts': [['c1'], ['c2']],
        'ground_truth': ['gt1', 'gt2']
    })
    mock_ragas_evaluate.assert_called_once()
    args, kwargs = mock_ragas_evaluate.call_args
    assert kwargs['dataset'] == MockDataset.from_dict.return_value
    assert kwargs['metrics'] == mock_metrics
    assert kwargs['llm'] == mock_llm
    assert kwargs['embeddings'] == mock_embeddings
    assert results["TestPipeline"]["faithfulness"] == 0.9
    assert results["TestPipeline"]["answer_relevancy"] == 0.8

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('json.dump')
@mock.patch('os.makedirs')
@mock.patch('os.path.exists', return_value=False) # Assume dir doesn't exist initially
def test_report_generation_mocked(mock_path_exists, mock_makedirs, mock_json_dump, mock_file_open):
    """
    Mocks file system operations. Verifies report generation functions
    are called with correct filenames and data.
    """
    pipeline_results = {
        "PipelineA": [{"success": True, "execution_time": 1.0}],
        "PipelineB": [{"success": True, "execution_time": 2.0}]
    }
    ragas_results = {
        "PipelineA": {"faithfulness": 0.9},
        "PipelineB": {"answer_relevancy": 0.8}
    }
    
    report_dir = execute_comprehensive_ragas_evaluation.generate_evaluation_report(
        pipeline_results, ragas_results, 10.0
    )

    # Verify directory creation
    mock_makedirs.assert_called_once()
    
    # Verify files were opened for writing
    assert mock_file_open.call_count >= 2  # At least raw results and summary
    
    # Verify JSON dump was called
    assert mock_json_dump.call_count >= 1
    
    # Verify report directory name format
    assert "comprehensive_ragas_results_" in report_dir