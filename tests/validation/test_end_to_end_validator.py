import pytest
from unittest import mock
from iris_rag.validation.end_to_end_validator import EndToEndValidator

@pytest.fixture
def db_connection_mock():
    """Fixture for a mock database connection."""
    return mock.Mock()

@pytest.fixture
def config_mock():
    """Fixture for a mock configuration."""
    # Add specific config needed by EndToEndValidator if any
    return mock.Mock()

@pytest.fixture
def e2e_validator(config_mock, db_connection_mock):
    """Fixture for EndToEndValidator."""
    validator = EndToEndValidator(config=config_mock, db_connection=db_connection_mock)
    # For these tests, we'll mock the pipeline instances within the validator
    validator.pipelines_to_test = {
        "basic_rag": mock.Mock(name="BasicRAGPipeline"),
        "colbert_rag": mock.Mock(name="ColBERTRAGPipeline"),
        "hyde_rag": mock.Mock(name="HyDERAGPipeline"),
        "crag": mock.Mock(name="CRAGPipeline"),
        "hybrid_ifind_rag": mock.Mock(name="HybridIFindRAGPipeline"),
        "graph_rag": mock.Mock(name="GraphRAGPipeline"),
        "node_rag": mock.Mock(name="NodeRAGPipeline"),
    }
    return validator

@pytest.fixture
def sample_queries():
    return ["What is COVID-19?", "Tell me about gene therapy."]

def test_all_pipelines_all_pass(e2e_validator, sample_queries):
    """Test test_all_pipelines when all individual pipeline tests pass."""
    with mock.patch.object(e2e_validator, '_test_single_pipeline', return_value=True) as mock_test_single:
        assert e2e_validator.test_all_pipelines(sample_queries) == True
        results = e2e_validator.get_results()
        assert results['overall_e2e_status'] == 'pass'
        assert mock_test_single.call_count == len(e2e_validator.pipelines_to_test)
        for pipeline_name, pipeline_instance in e2e_validator.pipelines_to_test.items():
            mock_test_single.assert_any_call(pipeline_name, pipeline_instance, sample_queries)
            # Assuming _test_single_pipeline would populate detailed results if it were fully implemented
            # For now, the overall status is the main check.

def test_all_pipelines_one_fails(e2e_validator, sample_queries):
    """Test test_all_pipelines when one individual pipeline test fails."""
    pipeline_names = list(e2e_validator.pipelines_to_test.keys())
    
    def side_effect_test_single(name, instance, queries):
        if name == pipeline_names[0]: # Fail the first pipeline
            # Simulate _test_single_pipeline updating its specific results
            e2e_validator.results[f'{name}_execution'] = {'status': 'failed', 'details': 'Simulated failure'}
            return False
        e2e_validator.results[f'{name}_execution'] = {'status': 'passed', 'details': 'Simulated success'}
        return True

    with mock.patch.object(e2e_validator, '_test_single_pipeline', side_effect=side_effect_test_single) as mock_test_single:
        assert e2e_validator.test_all_pipelines(sample_queries) == False
        results = e2e_validator.get_results()
        assert results['overall_e2e_status'] == 'fail'
        assert mock_test_single.call_count == len(e2e_validator.pipelines_to_test)
        assert results[f'{pipeline_names[0]}_execution']['status'] == 'failed'
        if len(pipeline_names) > 1:
            assert results[f'{pipeline_names[1]}_execution']['status'] == 'passed'


def test_all_pipelines_all_fail(e2e_validator, sample_queries):
    """Test test_all_pipelines when all individual pipeline tests fail."""
    def side_effect_all_fail(name, instance, queries):
        e2e_validator.results[f'{name}_execution'] = {'status': 'failed', 'details': 'Simulated failure'}
        return False

    with mock.patch.object(e2e_validator, '_test_single_pipeline', side_effect=side_effect_all_fail) as mock_test_single:
        assert e2e_validator.test_all_pipelines(sample_queries) == False
        results = e2e_validator.get_results()
        assert results['overall_e2e_status'] == 'fail'
        assert mock_test_single.call_count == len(e2e_validator.pipelines_to_test)
        for pipeline_name in e2e_validator.pipelines_to_test.keys():
            assert results[f'{pipeline_name}_execution']['status'] == 'failed'

def test_all_pipelines_some_not_instantiated(e2e_validator, sample_queries):
    """Test test_all_pipelines when some pipelines are not instantiated (None)."""
    # Set one pipeline to None
    pipeline_names = list(e2e_validator.pipelines_to_test.keys())
    uninstantiated_pipeline_name = pipeline_names[1]
    e2e_validator.pipelines_to_test[uninstantiated_pipeline_name] = None
    
    # Mock _test_single_pipeline to always return True for those that are called
    with mock.patch.object(e2e_validator, '_test_single_pipeline', return_value=True) as mock_test_single:
        # The overall result might depend on how 'skipped' pipelines are treated.
        # Current implementation of test_all_pipelines does not set all_pipelines_passed to False for skipped.
        # Let's assume for now that skipping doesn't automatically fail the overall run if others pass.
        # If strictness is required, the main method should handle this.
        # For this test, we'll check that the skipped one is marked and others are attempted.
        
        e2e_validator.test_all_pipelines(sample_queries) # Call the method
        results = e2e_validator.get_results()
        
        assert results[f'{uninstantiated_pipeline_name}_execution']['status'] == 'skipped'
        # Ensure _test_single_pipeline was called for non-None pipelines
        expected_calls = len([p for p in e2e_validator.pipelines_to_test.values() if p is not None])
        assert mock_test_single.call_count == expected_calls


# Placeholder for other EndToEndValidator tests

def test_validate_response_quality_placeholder(e2e_validator):
    """Test the placeholder validate_response_quality method."""
    pipeline_name = "test_pipeline"
    query = "test_query"
    # Simulate a response structure that the method might expect
    mock_response = {"answer": "Test answer", "retrieved_documents": []}
    
    # Current placeholder returns False and sets status to 'pending'
    assert not e2e_validator.validate_response_quality(pipeline_name, query, mock_response)
    results = e2e_validator.get_results()
    assert results[f'{pipeline_name}_quality'][query]['status'] == 'pending'
    assert "Not yet implemented" in results[f'{pipeline_name}_quality'][query]['details']

def test_monitor_performance_placeholder(e2e_validator):
    """Test the placeholder monitor_performance method."""
    pipeline_name = "test_pipeline"
    query = "test_query"
    # Current placeholder returns True and sets status to 'pending'
    assert e2e_validator.monitor_performance(pipeline_name, query, execution_time=0.1, resource_usage={})
    results = e2e_validator.get_results()
    assert results[f'{pipeline_name}_performance'][query]['status'] == 'pending'
    assert "Not yet implemented" in results[f'{pipeline_name}_performance'][query]['details']