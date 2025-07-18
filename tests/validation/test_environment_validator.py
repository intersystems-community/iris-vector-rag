import pytest
import os
from unittest import mock
from rag_templates.validation.environment_validator import EnvironmentValidator

@pytest.fixture
def validator():
    """Fixture for EnvironmentValidator."""
    return EnvironmentValidator()

def test_conda_activation_not_active(validator, monkeypatch):
    """Test validate_conda_activation when CONDA_DEFAULT_ENV is not set (inactive)."""
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert not validator.validate_conda_activation()
    assert validator.results['conda_activation']['status'] == 'fail'
    assert "CONDA_DEFAULT_ENV not set" in validator.results['conda_activation']['details']

def test_conda_activation_active_correct_env_name(validator, monkeypatch):
    """Test validate_conda_activation when CONDA_DEFAULT_ENV is set to the expected name."""
    # Assuming the expected env name can be configured or is a known constant
    expected_env_name = "rag_dev_env" # Example, should come from config or be a constant
    validator.expected_conda_env_name = expected_env_name # Simulate setting expected name
    monkeypatch.setenv("CONDA_DEFAULT_ENV", expected_env_name)
    monkeypatch.setenv("CONDA_PREFIX", f"/some/path/to/envs/{expected_env_name}")
    assert validator.validate_conda_activation()
    assert validator.results['conda_activation']['status'] == 'pass'
    assert validator.results['conda_activation']['details'] == f"Conda environment '{expected_env_name}' is active."

def test_conda_activation_active_incorrect_env_name(validator, monkeypatch):
    """Test validate_conda_activation when a conda env is active, but it's not the expected one."""
    expected_env_name = "rag_dev_env"
    actual_active_env = "base"
    validator.expected_conda_env_name = expected_env_name
    monkeypatch.setenv("CONDA_DEFAULT_ENV", actual_active_env)
    monkeypatch.setenv("CONDA_PREFIX", f"/some/path/to/envs/{actual_active_env}")
    assert not validator.validate_conda_activation()
    assert validator.results['conda_activation']['status'] == 'fail'
    assert f"Expected conda environment '{expected_env_name}' but found '{actual_active_env}'" in validator.results['conda_activation']['details']

def test_conda_prefix_used_if_name_matches_config(validator, monkeypatch):
    """Test that CONDA_PREFIX is checked if CONDA_DEFAULT_ENV matches expected name from config."""
    expected_env_name = "rag_dev_env"
    # Simulate config where expected_conda_env_name is set
    validator.config = {'expected_conda_env_name': expected_env_name}
    monkeypatch.setenv("CONDA_DEFAULT_ENV", expected_env_name)
    monkeypatch.setenv("CONDA_PREFIX", f"/path/to/envs/{expected_env_name}")
    
    # We need to define expected_conda_env_name on the validator instance for the current implementation
    # or refactor validate_conda_activation to use self.config
    validator.expected_conda_env_name = expected_env_name

    assert validator.validate_conda_activation()
    assert validator.results['conda_activation']['status'] == 'pass'

def test_conda_prefix_mismatch_env_name(validator, monkeypatch):
    """Test validate_conda_activation when CONDA_PREFIX does not match CONDA_DEFAULT_ENV name."""
    expected_env_name = "rag_dev_env"
    validator.expected_conda_env_name = expected_env_name
    monkeypatch.setenv("CONDA_DEFAULT_ENV", expected_env_name)
    monkeypatch.setenv("CONDA_PREFIX", "/path/to/envs/other_env") # Mismatch
    assert not validator.validate_conda_activation()
    assert validator.results['conda_activation']['status'] == 'fail'
    assert "does not appear to match the environment name" in validator.results['conda_activation']['details']

# Tests for verify_package_dependencies
def test_verify_package_dependencies_all_present_correct_version(validator, monkeypatch):
    """Test verify_package_dependencies when all packages are present with correct versions."""
    
    # Simulate config for expected packages
    expected_packages = {
        "requests": "2.31.0",
        "numpy": "1.24.0"
    }
    validator.config = {'expected_packages': expected_packages}

    def mock_version(package_name):
        if package_name == "requests":
            return "2.31.0"
        if package_name == "numpy":
            return "1.24.0"
        raise importlib.metadata.PackageNotFoundError
    
    monkeypatch.setattr("importlib.metadata.version", mock_version)
    
    # Import importlib.metadata here as it's used in the test
    import importlib.metadata
    
    assert validator.verify_package_dependencies()
    results = validator.results['package_dependencies']
    assert results['status'] == 'pass'
    assert "All required packages are installed with compatible versions." in results['details']
    assert results['packages']['requests']['status'] == 'pass'
    assert results['packages']['numpy']['status'] == 'pass'

def test_verify_package_dependencies_package_missing(validator, monkeypatch):
    """Test verify_package_dependencies when a required package is missing."""
    expected_packages = {"requests": "2.31.0", "nonexistent_package": "1.0.0"}
    validator.config = {'expected_packages': expected_packages}

    def mock_version(package_name):
        if package_name == "requests":
            return "2.31.0"
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", mock_version)
    import importlib.metadata # Ensure importlib.metadata is available

    assert not validator.verify_package_dependencies()
    results = validator.results['package_dependencies']
    assert results['status'] == 'fail'
    assert "nonexistent_package is not installed." in results['packages']['nonexistent_package']['details']
    assert results['packages']['nonexistent_package']['status'] == 'fail'
    assert results['packages']['requests']['status'] == 'pass'

def test_verify_package_dependencies_incorrect_version(validator, monkeypatch):
    """Test verify_package_dependencies when a package has an incompatible version."""
    expected_packages = {"requests": "2.31.0", "numpy": "1.24.0"}
    validator.config = {'expected_packages': expected_packages}

    def mock_version(package_name):
        if package_name == "requests":
            return "2.30.0"  # Incorrect version
        if package_name == "numpy":
            return "1.24.0"
        raise importlib.metadata.PackageNotFoundError
        
    monkeypatch.setattr("importlib.metadata.version", mock_version)
    import importlib.metadata # Ensure importlib.metadata is available

    assert not validator.verify_package_dependencies()
    results = validator.results['package_dependencies']
    assert results['status'] == 'fail'
    assert "Installed version 2.30.0 does not meet requirement 2.31.0 (parsed as ==2.31.0)" in results['packages']['requests']['details']
    assert results['packages']['requests']['status'] == 'fail'
    assert results['packages']['numpy']['status'] == 'pass'

def test_verify_package_dependencies_no_expected_packages_in_config(validator):
    """Test behavior when no expected_packages are defined in config."""
    validator.config = {} # No expected_packages
    assert validator.verify_package_dependencies() # Should pass if nothing is expected
    results = validator.results['package_dependencies']
    assert results['status'] == 'pass'
    assert "No packages specified in configuration to verify." in results['details']

def test_verify_package_dependencies_version_specifier_greater_equal(validator, monkeypatch):
    """Test verify_package_dependencies with '>=' version specifier."""
    expected_packages = {"numpy": ">=1.23.0"}
    validator.config = {'expected_packages': expected_packages}

    def mock_version(package_name):
        if package_name == "numpy":
            return "1.24.0" # Meets >=1.23.0
        raise importlib.metadata.PackageNotFoundError
        
    monkeypatch.setattr("importlib.metadata.version", mock_version)
    import importlib.metadata

    assert validator.verify_package_dependencies()
    results = validator.results['package_dependencies']
    assert results['status'] == 'pass'
    assert results['packages']['numpy']['status'] == 'pass'
    assert results['packages']['numpy']['found'] == '1.24.0'

def test_verify_package_dependencies_version_specifier_less_than(validator, monkeypatch):
    """Test verify_package_dependencies with '<' version specifier."""
    expected_packages = {"numpy": "<1.25.0"}
    validator.config = {'expected_packages': expected_packages}

    def mock_version(package_name):
        if package_name == "numpy":
            return "1.24.0" # Meets <1.25.0
        raise importlib.metadata.PackageNotFoundError
        
    monkeypatch.setattr("importlib.metadata.version", mock_version)
    import importlib.metadata

    assert validator.verify_package_dependencies()
    results = validator.results['package_dependencies']
    assert results['status'] == 'pass'
    assert results['packages']['numpy']['status'] == 'pass'

def test_verify_package_dependencies_version_specifier_tilde_equal(validator, monkeypatch):
    """Test verify_package_dependencies with '~=' (compatible release) version specifier."""
    expected_packages = {"numpy": "~=1.24.0"} # Means >=1.24.0, <1.25.0
    validator.config = {'expected_packages': expected_packages}

    def mock_version_pass(package_name):
        if package_name == "numpy":
            return "1.24.3"
        raise importlib.metadata.PackageNotFoundError
    
    def mock_version_fail_minor(package_name):
        if package_name == "numpy":
            return "1.25.0" # Fails ~=1.24.0
        raise importlib.metadata.PackageNotFoundError

    import importlib.metadata
    monkeypatch.setattr("importlib.metadata.version", mock_version_pass)
    assert validator.verify_package_dependencies()
    assert validator.results['package_dependencies']['packages']['numpy']['status'] == 'pass'
    
    # Reset validator for next check
    validator = EnvironmentValidator(config={'expected_packages': expected_packages})
    monkeypatch.setattr("importlib.metadata.version", mock_version_fail_minor)
    assert not validator.verify_package_dependencies()
    assert validator.results['package_dependencies']['packages']['numpy']['status'] == 'fail'
# Tests for test_ml_ai_function_availability
@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_embedding_model')
@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_llm')
def test_ml_ai_functions_all_available(mock_get_llm, mock_get_embedding_model, validator):
    """Test test_ml_ai_function_availability when all functions are available and working."""
    # Mock embedding model
    mock_embedding_model_instance = mock.Mock()
    mock_embedding_model_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_get_embedding_model.return_value = mock_embedding_model_instance

    # Mock LLM
    mock_llm_instance = mock.Mock()
    mock_llm_instance.invoke.return_value = "LLM response"
    mock_get_llm.return_value = mock_llm_instance
    
    validator.config = {
        'test_embedding_text': 'sample text for embedding',
        'test_llm_prompt': 'sample prompt for llm'
    }

    assert validator.test_ml_ai_function_availability()
    results = validator.results['ml_ai_functions']
    assert results['status'] == 'pass'
    assert results['embedding_model_status']['status'] == 'pass'
    assert results['llm_status']['status'] == 'pass'
    mock_embedding_model_instance.embed_query.assert_called_once_with('sample text for embedding')
    mock_llm_instance.invoke.assert_called_once_with('sample prompt for llm')

@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_embedding_model')
@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_llm')
def test_ml_ai_functions_embedding_fails(mock_get_llm, mock_get_embedding_model, validator):
    """Test test_ml_ai_function_availability when embedding model fails."""
    mock_get_embedding_model.return_value.embed_query.side_effect = Exception("Embedding error")
    
    mock_llm_instance = mock.Mock()
    mock_llm_instance.invoke.return_value = "LLM response"
    mock_get_llm.return_value = mock_llm_instance

    validator.config = {
        'test_embedding_text': 'sample text',
        'test_llm_prompt': 'sample prompt'
    }

    assert not validator.test_ml_ai_function_availability()
    results = validator.results['ml_ai_functions']
    assert results['status'] == 'fail'
    assert results['embedding_model_status']['status'] == 'fail'
    assert "Embedding error" in results['embedding_model_status']['details']
    assert results['llm_status']['status'] == 'pass' # LLM should still be checked if embedding fails first

@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_embedding_model')
@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_llm')
def test_ml_ai_functions_llm_fails(mock_get_llm, mock_get_embedding_model, validator):
    """Test test_ml_ai_function_availability when LLM fails."""
    mock_embedding_model_instance = mock.Mock()
    mock_embedding_model_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_get_embedding_model.return_value = mock_embedding_model_instance

    mock_get_llm.return_value.invoke.side_effect = Exception("LLM error")

    validator.config = {
        'test_embedding_text': 'sample text',
        'test_llm_prompt': 'sample prompt'
    }

    assert not validator.test_ml_ai_function_availability()
    results = validator.results['ml_ai_functions']
    assert results['status'] == 'fail'
    assert results['embedding_model_status']['status'] == 'pass'
    assert results['llm_status']['status'] == 'fail'
    assert "LLM error" in results['llm_status']['details']

@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_embedding_model', return_value=None)
@mock.patch('rag_templates.validation.environment_validator.EnvironmentValidator._get_llm', return_value=None)
def test_ml_ai_functions_models_not_configured(mock_get_llm, mock_get_embedding_model, validator):
    """Test test_ml_ai_function_availability when models are not configured (return None)."""
    validator.config = {
        'test_embedding_text': 'sample text',
        'test_llm_prompt': 'sample prompt'
    }
    assert not validator.test_ml_ai_function_availability()
    results = validator.results['ml_ai_functions']
    assert results['status'] == 'fail'
    assert results['embedding_model_status']['status'] == 'fail'
    assert "Embedding model not configured or failed to load" in results['embedding_model_status']['details']
    assert results['llm_status']['status'] == 'fail'
    assert "LLM not configured or failed to load" in results['llm_status']['details']