# Quick Start CLI Wizard Test Guide

This guide provides comprehensive documentation for testing the Quick Start CLI wizard system, including test execution, requirements, and implementation guidelines.

## Overview

The CLI wizard test suite follows Test-Driven Development (TDD) principles to ensure comprehensive coverage of the Quick Start CLI wizard functionality. The tests are designed to fail initially (red phase) and guide the implementation of the CLI wizard components.

## Test Structure

### Main Test Files

- **`test_cli_wizard.py`** - Main comprehensive test suite covering all CLI wizard functionality
- **`test_cli_wizard_fixtures.py`** - Test fixtures, utilities, and mock objects
- **`test_data/cli_wizard_test_configs.yaml`** - Test configuration files for various scenarios

### Test Categories

1. **Profile Selection Tests**
   - Interactive profile selection (minimal, standard, extended, custom)
   - Non-interactive profile selection via CLI arguments
   - Profile validation and error handling
   - Profile characteristics display

2. **Environment Configuration Tests**
   - Database connection configuration
   - LLM provider configuration
   - Embedding model selection
   - Environment variable generation and validation

3. **Template Generation Tests**
   - Configuration file generation
   - Environment file (.env) creation
   - Docker Compose file generation
   - Sample data script generation

4. **Validation and Testing Integration Tests**
   - Database connectivity validation
   - LLM provider credential validation
   - Embedding model availability checks
   - System health check integration

5. **CLI Interface Tests**
   - Command-line argument parsing
   - Interactive prompt handling
   - Output formatting and display
   - Progress indicators and status updates

6. **Integration Tests**
   - Integration with TemplateEngine
   - Integration with SchemaValidator
   - Integration with IntegrationFactory
   - Integration with SampleDataManager

7. **Error Handling and Edge Case Tests**
   - Invalid profile handling
   - Network connectivity issues
   - File permission errors
   - Concurrent wizard instances

8. **End-to-End Workflow Tests**
   - Complete profile setup workflows
   - Non-interactive automation
   - Environment-specific configurations
   - Multi-tenant setups

## CLI Wizard Requirements

### Expected CLI Interface

The CLI wizard should support both interactive and non-interactive modes:

#### Interactive Mode
```bash
python -m quick_start.cli.wizard
```

#### Non-Interactive Mode
```bash
python -m quick_start.cli.wizard --profile standard --database-host localhost --llm-provider openai
```

#### Help and Options
```bash
python -m quick_start.cli.wizard --help
python -m quick_start.cli.wizard --list-profiles
python -m quick_start.cli.wizard --validate-only
```

### Required CLI Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--profile` | Profile to use (minimal/standard/extended) | `--profile standard` |
| `--database-host` | IRIS database host | `--database-host localhost` |
| `--database-port` | IRIS database port | `--database-port 1972` |
| `--database-namespace` | IRIS namespace | `--database-namespace USER` |
| `--database-username` | Database username | `--database-username demo` |
| `--database-password` | Database password | `--database-password demo` |
| `--llm-provider` | LLM provider (openai/anthropic) | `--llm-provider openai` |
| `--llm-model` | LLM model name | `--llm-model gpt-4` |
| `--llm-api-key` | LLM API key | `--llm-api-key sk-...` |
| `--embedding-provider` | Embedding provider | `--embedding-provider openai` |
| `--embedding-model` | Embedding model | `--embedding-model text-embedding-ada-002` |
| `--document-count` | Number of sample documents | `--document-count 100` |
| `--output-dir` | Output directory for files | `--output-dir ./config` |
| `--generate-docker-compose` | Generate docker-compose.yml | `--generate-docker-compose` |
| `--generate-sample-script` | Generate sample data script | `--generate-sample-script` |
| `--non-interactive` | Run without prompts | `--non-interactive` |
| `--validate-only` | Only validate configuration | `--validate-only` |
| `--list-profiles` | List available profiles | `--list-profiles` |
| `--help` | Show help message | `--help` |

### Expected CLI Wizard Architecture

```
quick_start/cli/
├── __init__.py
├── wizard.py              # Main CLI wizard implementation
├── prompts.py             # Interactive prompt utilities
├── validators.py          # CLI-specific validation functions
└── formatters.py          # Output formatting and display utilities
```

### Required Classes and Methods

#### QuickStartCLIWizard Class

```python
class QuickStartCLIWizard:
    def __init__(self):
        """Initialize CLI wizard with required components."""
    
    # Profile Selection
    def select_profile_interactive(self) -> CLIWizardResult:
        """Interactive profile selection menu."""
    
    def select_profile_from_args(self, profile: str = None) -> CLIWizardResult:
        """Non-interactive profile selection from CLI args."""
    
    def get_profile_characteristics(self, profile: str) -> Dict[str, Any]:
        """Get profile characteristics and resource requirements."""
    
    # Environment Configuration
    def configure_database_interactive(self) -> Dict[str, Any]:
        """Interactive database configuration prompts."""
    
    def configure_llm_provider_interactive(self) -> Dict[str, Any]:
        """Interactive LLM provider configuration."""
    
    def configure_embeddings_interactive(self) -> Dict[str, Any]:
        """Interactive embedding model selection."""
    
    def generate_env_file(self, config: Dict[str, Any], path: Path) -> Path:
        """Generate environment variable file."""
    
    def validate_environment_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate environment configuration."""
    
    # Template Generation
    def generate_configuration_file(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate configuration file from profile."""
    
    def create_env_file(self, env_vars: Dict[str, str], path: Path) -> Path:
        """Create environment file."""
    
    def generate_docker_compose(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate docker-compose file."""
    
    def generate_sample_data_script(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate sample data setup script."""
    
    # Validation and Testing
    def test_database_connection(self, config: Dict[str, Any]) -> MockConnectionResult:
        """Test database connectivity."""
    
    def test_llm_credentials(self, config: Dict[str, Any]) -> MockConnectionResult:
        """Test LLM provider credentials."""
    
    def test_embedding_model(self, config: Dict[str, Any]) -> MockConnectionResult:
        """Test embedding model availability."""
    
    def run_system_health_check(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive system health check."""
    
    # CLI Interface
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments."""
    
    def prompt_for_input(self, prompt: str, input_type: type) -> Any:
        """Handle interactive prompts with validation."""
    
    def display_message(self, message: str, level: str) -> None:
        """Display formatted messages to user."""
    
    def show_progress(self, message: str, current: int, total: int) -> None:
        """Show progress indicators."""
    
    # Integration
    def get_available_profiles(self) -> List[str]:
        """Get available profiles from TemplateEngine."""
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration using SchemaValidator."""
    
    def integrate_with_existing_systems(self, config: Dict[str, Any]) -> Any:
        """Integrate with existing systems using IntegrationFactory."""
    
    def get_available_data_sources(self) -> List[Dict[str, Any]]:
        """Get available data sources from SampleDataManager."""
    
    # Workflows
    def run_interactive_setup(self, output_dir: Path) -> CLIWizardResult:
        """Run complete interactive setup workflow."""
    
    def run_non_interactive_setup(self) -> CLIWizardResult:
        """Run complete non-interactive setup workflow."""
    
    def run_complete_setup(self, profile: str, output_dir: Path, non_interactive: bool = False) -> CLIWizardResult:
        """Run complete setup workflow."""
```

## Test Execution

### Running All CLI Wizard Tests

```bash
# Run all CLI wizard tests
pytest tests/quick_start/test_cli_wizard.py -v

# Run with coverage
pytest tests/quick_start/test_cli_wizard.py --cov=quick_start.cli --cov-report=html

# Run specific test categories
pytest tests/quick_start/test_cli_wizard.py::TestQuickStartCLIWizard::test_profile_selection_interactive_minimal -v
```

### Running Tests in TDD Mode

Since the CLI wizard is not yet implemented, all tests will initially fail (red phase). This is expected and follows TDD principles:

```bash
# Expected to fail initially - this is the RED phase
pytest tests/quick_start/test_cli_wizard.py -v --tb=short

# After implementing CLI wizard components, tests should pass - GREEN phase
pytest tests/quick_start/test_cli_wizard.py -v

# Refactor and ensure tests still pass - REFACTOR phase
pytest tests/quick_start/test_cli_wizard.py -v
```

### Test Fixtures and Utilities

The test suite includes comprehensive fixtures:

```python
# Use fixtures in tests
def test_example(sample_profiles, mock_template_engine, temp_dir):
    # Test implementation using fixtures
    pass
```

Available fixtures:
- `sample_profiles` - Sample profile configurations
- `sample_environment_variables` - Sample environment variables
- `mock_user_inputs` - Mock user inputs for interactive testing
- `mock_cli_arguments` - Mock CLI arguments for non-interactive testing
- `mock_template_engine` - Mock TemplateEngine
- `mock_schema_validator` - Mock SchemaValidator
- `mock_integration_factory` - Mock IntegrationFactory
- `mock_sample_manager` - Mock SampleDataManager

## Implementation Guidelines

### TDD Workflow

1. **Red Phase**: Run tests to see them fail
   ```bash
   pytest tests/quick_start/test_cli_wizard.py::TestQuickStartCLIWizard::test_profile_selection_interactive_minimal -v
   ```

2. **Green Phase**: Implement minimal code to make test pass
   ```python
   # quick_start/cli/wizard.py
   class QuickStartCLIWizard:
       def select_profile_interactive(self):
           # Minimal implementation to pass test
           return CLIWizardResult(success=True, profile="quick_start_minimal", ...)
   ```

3. **Refactor Phase**: Improve code while keeping tests passing
   ```python
   # Refactor for better design, maintainability
   # Ensure all tests still pass
   ```

### Implementation Order

Recommended implementation order based on test dependencies:

1. **Basic CLI Structure**
   - `QuickStartCLIWizard` class
   - Basic argument parsing
   - Error handling framework

2. **Profile Selection**
   - Interactive profile menu
   - Profile validation
   - Profile characteristics display

3. **Environment Configuration**
   - Database configuration prompts
   - LLM provider configuration
   - Environment variable handling

4. **Template Generation**
   - Configuration file generation
   - Environment file creation
   - Docker Compose generation

5. **Validation Integration**
   - Connection testing
   - System health checks
   - Error reporting

6. **End-to-End Workflows**
   - Complete setup workflows
   - Non-interactive automation
   - Integration with existing components

### Error Handling Requirements

The CLI wizard must handle various error conditions gracefully:

- **Network Errors**: Database connection failures, API timeouts
- **Validation Errors**: Invalid configurations, missing required fields
- **File System Errors**: Permission denied, disk space issues
- **User Input Errors**: Invalid selections, malformed input
- **Integration Errors**: Component failures, version mismatches

### Performance Requirements

- **Startup Time**: < 2 seconds for wizard initialization
- **Response Time**: < 1 second for user input processing
- **File Generation**: < 5 seconds for all configuration files
- **Validation**: < 10 seconds for complete system validation

### Security Requirements

- **Credential Handling**: Secure storage of API keys and passwords
- **File Permissions**: Proper permissions for generated files
- **Input Validation**: Sanitize all user inputs
- **Environment Variables**: Secure handling of sensitive environment variables

## Test Data and Configurations

### Test Configuration Files

The test suite includes various configuration scenarios:

- **Valid Configurations**: `minimal_profile_valid`, `standard_profile_valid`, `extended_profile_valid`
- **Invalid Configurations**: `invalid_missing_metadata`, `invalid_minimal_too_many_docs`
- **Environment Variables**: `config_with_env_vars`
- **Production Scenarios**: `production_config`, `multi_tenant_config`
- **Development Scenarios**: `development_config`
- **Migration Scenarios**: `migration_test_config`

### Mock Data

The test suite provides comprehensive mock data for:

- User inputs for interactive scenarios
- CLI arguments for non-interactive scenarios
- Environment variables for various environments
- Profile configurations for all supported profiles
- Error scenarios and edge cases

## Continuous Integration

### Test Requirements for CI

```yaml
# .github/workflows/cli-wizard-tests.yml
name: CLI Wizard Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run CLI wizard tests
        run: |
          pytest tests/quick_start/test_cli_wizard.py -v --cov=quick_start.cli
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Quality Gates

- **Test Coverage**: Minimum 90% code coverage for CLI wizard components
- **Test Success**: All tests must pass before merge
- **Performance**: CLI wizard startup time < 2 seconds
- **Documentation**: All public methods must have docstrings

## Troubleshooting

### Common Test Issues

1. **Import Errors**: Ensure `quick_start.cli.wizard` module exists
2. **Mock Failures**: Check mock object configurations
3. **File Permission Errors**: Ensure test directories are writable
4. **Environment Variable Issues**: Check test environment setup

### Debug Mode

Run tests with debug output:

```bash
pytest tests/quick_start/test_cli_wizard.py -v -s --log-cli-level=DEBUG
```

### Test Isolation

Each test is designed to be independent:
- Uses temporary directories for file operations
- Mocks external dependencies
- Cleans up resources after execution

## Contributing

When adding new CLI wizard functionality:

1. **Write Tests First**: Follow TDD principles
2. **Update Fixtures**: Add new mock data as needed
3. **Document Changes**: Update this guide and docstrings
4. **Test Coverage**: Ensure new code is fully tested
5. **Integration**: Test with existing Quick Start components

## References

- [Quick Start Configuration Templates](../../quick_start/config/template_engine.py)
- [Schema Validation](../../quick_start/config/schema_validator.py)
- [Integration Factory](../../quick_start/config/integration_factory.py)
- [Sample Data Manager](../../quick_start/data/sample_manager.py)
- [TDD Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)