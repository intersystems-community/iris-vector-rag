import pytest
from unittest import mock
from iris_rag.validation.comprehensive_validation_runner import ComprehensiveValidationRunner
from iris_rag.validation.environment_validator import EnvironmentValidator
from iris_rag.validation.data_population_orchestrator import DataPopulationOrchestrator
from iris_rag.validation.end_to_end_validator import EndToEndValidator

@pytest.fixture
def mock_config():
    return mock.Mock()

@pytest.fixture
def mock_db_connection():
    return mock.Mock()

@pytest.fixture
def runner(mock_config, mock_db_connection):
    # Patch the constructors of the sub-validators to use mocks
    with mock.patch('iris_rag.validation.comprehensive_validation_runner.EnvironmentValidator') as MockEnvValidator, \
         mock.patch('iris_rag.validation.comprehensive_validation_runner.DataPopulationOrchestrator') as MockDataOrchestrator, \
         mock.patch('iris_rag.validation.comprehensive_validation_runner.EndToEndValidator') as MockE2EValidator:
        
        # Configure the mock instances that will be created
        mock_env_validator_instance = MockEnvValidator.return_value
        mock_env_validator_instance.run_all_checks.return_value = True # Assume pass for now
        mock_env_validator_instance.get_results.return_value = {'overall_status': 'pass', 'details': 'Env OK'}

        mock_data_orchestrator_instance = MockDataOrchestrator.return_value
        mock_data_orchestrator_instance.populate_all_tables.return_value = True # Assume pass
        mock_data_orchestrator_instance.get_results.return_value = {'overall_population_status': 'pass', 'details': 'Data OK'}
        
        mock_e2e_validator_instance = MockE2EValidator.return_value
        mock_e2e_validator_instance.test_all_pipelines.return_value = True # Assume pass
        mock_e2e_validator_instance.get_results.return_value = {'overall_e2e_status': 'pass', 'details': 'E2E OK'}

        runner_instance = ComprehensiveValidationRunner(config=mock_config, db_connection=mock_db_connection)
        
        # Replace the validator instances on the runner with our configured mocks
        runner_instance.environment_validator = mock_env_validator_instance
        runner_instance.data_population_orchestrator = mock_data_orchestrator_instance
        runner_instance.end_to_end_validator = mock_e2e_validator_instance
        
        return runner_instance

@pytest.fixture
def sample_queries():
    return ["What is test query 1?", "What is test query 2?"]

def test_run_complete_validation_all_pass(runner, sample_queries):
    """Test run_complete_validation when all sub-validators report success."""
    runner.environment_validator.run_all_checks.return_value = True
    runner.data_population_orchestrator.populate_all_tables.return_value = True
    runner.end_to_end_validator.test_all_pipelines.return_value = True
    
    # Set reliability score threshold to a value that will pass
    runner.reliability_score_threshold = 0.9
    
    assert runner.run_complete_validation(sample_queries=sample_queries) == True
    results = runner.get_results()
    
    assert results['environment_validation']['overall_status'] == 'pass'
    assert results['data_population']['overall_population_status'] == 'pass'
    assert results['end_to_end_validation']['overall_e2e_status'] == 'pass'
    assert results['overall_reliability_score'] == 1.0 # Based on current _calculate_reliability_score
    assert results['production_ready'] == True
    
    runner.environment_validator.run_all_checks.assert_called_once()
    runner.data_population_orchestrator.populate_all_tables.assert_called_once()
    runner.end_to_end_validator.test_all_pipelines.assert_called_once_with(sample_queries)

def test_run_complete_validation_env_fails(runner, sample_queries):
    """Test run_complete_validation when environment validation fails."""
    runner.environment_validator.run_all_checks.return_value = False
    runner.environment_validator.get_results.return_value = {'overall_status': 'fail'}
    runner.data_population_orchestrator.populate_all_tables.return_value = True
    runner.end_to_end_validator.test_all_pipelines.return_value = True
    runner.reliability_score_threshold = 0.95

    assert runner.run_complete_validation(sample_queries=sample_queries) == False # Production ready should be False
    results = runner.get_results()
    assert results['environment_validation']['overall_status'] == 'fail'
    assert results['overall_reliability_score'] < 0.95 # Check against threshold
    assert results['production_ready'] == False

def test_run_complete_validation_data_fails(runner, sample_queries):
    """Test run_complete_validation when data population fails."""
    runner.environment_validator.run_all_checks.return_value = True
    runner.data_population_orchestrator.populate_all_tables.return_value = False
    runner.data_population_orchestrator.get_results.return_value = {'overall_population_status': 'fail'}
    runner.end_to_end_validator.test_all_pipelines.return_value = True
    runner.reliability_score_threshold = 0.95

    assert runner.run_complete_validation(sample_queries=sample_queries) == False
    results = runner.get_results()
    assert results['data_population']['overall_population_status'] == 'fail'
    assert results['overall_reliability_score'] < 0.95
    assert results['production_ready'] == False

def test_run_complete_validation_e2e_fails(runner, sample_queries):
    """Test run_complete_validation when end-to-end validation fails."""
    runner.environment_validator.run_all_checks.return_value = True
    runner.data_population_orchestrator.populate_all_tables.return_value = True
    runner.end_to_end_validator.test_all_pipelines.return_value = False
    runner.end_to_end_validator.get_results.return_value = {'overall_e2e_status': 'fail'}
    runner.reliability_score_threshold = 0.95

    assert runner.run_complete_validation(sample_queries=sample_queries) == False
    results = runner.get_results()
    assert results['end_to_end_validation']['overall_e2e_status'] == 'fail'
    assert results['overall_reliability_score'] < 0.95
    assert results['production_ready'] == False

def test_calculate_reliability_score(runner):
    """Test the _calculate_reliability_score method directly."""
    # All pass
    assert runner._calculate_reliability_score(True, True, True) == pytest.approx(1.0)
    # Env fails
    assert runner._calculate_reliability_score(False, True, True) == pytest.approx(0.8) # 0.4 (data) + 0.4 (e2e)
    # Data fails
    assert runner._calculate_reliability_score(True, False, True) == pytest.approx(0.6) # 0.2 (env) + 0.4 (e2e)
    # E2E fails
    assert runner._calculate_reliability_score(True, True, False) == pytest.approx(0.6) # 0.2 (env) + 0.4 (data)
    # All fail
    assert runner._calculate_reliability_score(False, False, False) == pytest.approx(0.0)

# Placeholder for other ComprehensiveValidationRunner tests
# def test_placeholder_comprehensive_validation_runner():
#     """Placeholder test for ComprehensiveValidationRunner."""
#     assert True