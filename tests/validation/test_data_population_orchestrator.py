import pytest
from unittest import mock
from rag_templates.validation.data_population_orchestrator import DataPopulationOrchestrator

@pytest.fixture
def db_connection_mock():
    """Fixture for a mock database connection."""
    return mock.Mock()

@pytest.fixture
def config_mock():
    """Fixture for a mock configuration."""
    return mock.Mock()

@pytest.fixture
def orchestrator(config_mock, db_connection_mock):
    """Fixture for DataPopulationOrchestrator."""
    return DataPopulationOrchestrator(config=config_mock, db_connection=db_connection_mock)

def test_populate_all_tables_success(orchestrator):
    """Test populate_all_tables when all individual table populations succeed."""
    def side_effect_populate_table_success(table_name):
        # Simulate _populate_table updating its result for success
        orchestrator.results[f'{table_name}_population'] = {'status': 'success', 'details': 'Simulated success'}
        return True

    with mock.patch.object(orchestrator, '_populate_table', side_effect=side_effect_populate_table_success) as mock_populate_single:
        assert orchestrator.populate_all_tables() == True
        results = orchestrator.get_results()
        assert results['overall_population_status'] == 'pass'
        
        # Check if _populate_table was called for each table in TABLE_ORDER
        assert mock_populate_single.call_count == len(orchestrator.TABLE_ORDER)
        for table_name in orchestrator.TABLE_ORDER:
            mock_populate_single.assert_any_call(table_name)
            assert results[f'{table_name}_population']['status'] == 'success' # Assuming _populate_table updates this

def test_populate_all_tables_one_fails(orchestrator):
    """Test populate_all_tables when one individual table population fails."""
    # Make the first table population fail, others succeed
    def side_effect_populate_table(table_name):
        if table_name == orchestrator.TABLE_ORDER[0]:
            # Simulate failure for the first table by updating its result and returning False
            orchestrator.results[f'{table_name}_population'] = {'status': 'failed', 'details': 'Simulated failure for first table'}
            return False
        orchestrator.results[f'{table_name}_population'] = {'status': 'success', 'details': 'Simulated success'}
        return True

    with mock.patch.object(orchestrator, '_populate_table', side_effect=side_effect_populate_table) as mock_populate_single:
        assert orchestrator.populate_all_tables() == False
        results = orchestrator.get_results()
        assert results['overall_population_status'] == 'fail'
        
        # Ensure it was called for all tables despite the failure (or up to the point of failure, depending on design)
        # Current design calls all, so check all.
        assert mock_populate_single.call_count == len(orchestrator.TABLE_ORDER)
        
        # Check status of the first table (failed) and a subsequent one (should be success if called)
        first_table_name = orchestrator.TABLE_ORDER[0]
        assert results[f'{first_table_name}_population']['status'] == 'failed'
        
        if len(orchestrator.TABLE_ORDER) > 1:
            second_table_name = orchestrator.TABLE_ORDER[1]
            # This assertion depends on whether _populate_table is called for subsequent tables after a failure.
            # The current orchestrator.populate_all_tables continues, so this should be 'success'.
            assert results[f'{second_table_name}_population']['status'] == 'success'


def test_populate_all_tables_all_fail(orchestrator):
    """Test populate_all_tables when all individual table populations fail."""
    with mock.patch.object(orchestrator, '_populate_table', return_value=False) as mock_populate_single:
        # Simulate _populate_table updating results to 'failed'
        def update_results_on_fail(table_name):
            orchestrator.results[f'{table_name}_population'] = {'status': 'failed', 'details': 'Simulated failure'}
            return False
        mock_populate_single.side_effect = update_results_on_fail

        assert orchestrator.populate_all_tables() == False
        results = orchestrator.get_results()
        assert results['overall_population_status'] == 'fail'
        assert mock_populate_single.call_count == len(orchestrator.TABLE_ORDER)
        for table_name in orchestrator.TABLE_ORDER:
            assert results[f'{table_name}_population']['status'] == 'failed'

# Placeholder for other DataPopulationOrchestrator tests

def test_run_self_healing_placeholder(orchestrator):
    """Test the placeholder run_self_healing method."""
    # Current placeholder returns False and sets status to 'pending'
    assert not orchestrator.run_self_healing()
    results = orchestrator.get_results()
    assert results['self_healing_status']['status'] == 'pending'
    assert "Not yet implemented" in results['self_healing_status']['details']

def test_verify_data_dependencies_placeholder(orchestrator):
    """Test the placeholder verify_data_dependencies method."""
    # Current placeholder returns False and sets status to 'pending'
    assert not orchestrator.verify_data_dependencies()
    results = orchestrator.get_results()
    assert results['data_dependency_status']['status'] == 'pending'
    assert "Not yet implemented" in results['data_dependency_status']['details']