# Docker-compose Integration Test Guide

This guide explains the comprehensive test suite for Docker-compose integration in the Quick Start system.

## Overview

The Docker-compose integration tests ensure that the Quick Start system can generate, configure, and manage containerized RAG environments seamlessly. These tests follow TDD principles and provide comprehensive coverage of all Docker-related functionality.

## Test Structure

### Test Categories

1. **Docker-compose File Generation Tests**
   - Test generation of docker-compose.yml for each profile (minimal, standard, extended, custom)
   - Validate service configuration and dependencies
   - Test volume and network configuration
   - Test environment variable injection

2. **Container Configuration Tests**
   - Test IRIS database container configuration
   - Test RAG application container configuration
   - Test MCP server container configuration
   - Test monitoring services configuration

3. **Profile-Specific Tests**
   - Test minimal profile (50 docs, basic services)
   - Test standard profile (500 docs, includes MCP server)
   - Test extended profile (5000 docs, full monitoring stack)
   - Test custom profile (user-defined configurations)

4. **Integration Tests**
   - Test integration with CLI wizard
   - Test integration with setup pipeline
   - Test integration with sample data manager
   - Test integration with template engine

5. **Docker Operations Tests**
   - Test docker-compose up/down operations
   - Test service health checks and readiness
   - Test volume persistence and data integrity
   - Test network connectivity between services

6. **Development Workflow Tests**
   - Test development mode configuration
   - Test hot reloading functionality
   - Test debug port configuration
   - Test log aggregation and monitoring

## Test Files

### Main Test File
- `test_docker_compose_integration.py` - Comprehensive test suite with all test categories

### Test Data Files
- `test_data/docker_compose_test_configs.yaml` - Test configurations for all profiles
- `test_data/docker_compose_templates.yaml` - Sample Docker compose templates
- `test_data/cli_wizard_test_configs.yaml` - CLI wizard test configurations (existing)

### Test Fixtures
- Defined in `conftest.py` and within test files
- Temporary directories for isolated testing
- Mock Docker commands and responses
- Sample configurations for each profile

## Running the Tests

### Prerequisites
- Docker and docker-compose installed
- Python 3.11+ with pytest
- All Quick Start dependencies installed

### Test Execution

```bash
# Run all Docker-compose integration tests
pytest tests/quick_start/test_docker_compose_integration.py -v

# Run specific test categories
pytest tests/quick_start/test_docker_compose_integration.py::TestDockerComposeIntegration::test_docker_compose_file_generation_minimal -v

# Run with coverage
pytest tests/quick_start/test_docker_compose_integration.py --cov=quick_start.docker --cov-report=html

# Run in parallel (if pytest-xdist is installed)
pytest tests/quick_start/test_docker_compose_integration.py -n auto
```

### Test Profiles

The tests cover the following Docker-compose profiles:

#### Minimal Profile
- **Services**: IRIS database, RAG application
- **Document Count**: 50
- **Use Case**: Development and testing
- **Resources**: Low resource requirements

#### Standard Profile
- **Services**: IRIS database, RAG application, MCP server
- **Document Count**: 500
- **Use Case**: Standard development and demo
- **Resources**: Moderate resource requirements

#### Extended Profile
- **Services**: IRIS database, RAG application, MCP server, Nginx, Prometheus, Grafana
- **Document Count**: 5000
- **Use Case**: Production-like environment
- **Resources**: High resource requirements with monitoring

#### Development Profile
- **Services**: IRIS database, RAG application, MCP server (with debug ports)
- **Features**: Hot reloading, debug ports, development tools
- **Use Case**: Active development

#### Production Profile
- **Services**: Full stack with SSL, monitoring, backup
- **Features**: SSL termination, automated backups, resource limits
- **Use Case**: Production deployment

#### Testing Profile
- **Services**: Test-specific services with isolated data
- **Features**: Test database, mock services, test data volumes
- **Use Case**: Automated testing

## Test Implementation Strategy

### TDD Approach

1. **Red Phase**: Write failing tests first
   - Tests expect Docker integration modules that don't exist yet
   - Tests define the expected API and behavior
   - All tests initially fail with ImportError or NotImplementedError

2. **Green Phase**: Implement minimal code to pass tests
   - Create Docker integration modules
   - Implement basic functionality to satisfy test requirements
   - Focus on making tests pass, not on optimization

3. **Refactor Phase**: Improve code while keeping tests passing
   - Optimize Docker-compose generation
   - Improve error handling and validation
   - Add performance optimizations

### Test Isolation

- Each test uses temporary directories
- Docker commands are mocked to avoid actual container operations
- Tests don't depend on external Docker services
- Cleanup is performed after each test

### Mock Strategy

- Mock `subprocess.run` for Docker commands
- Mock file system operations where appropriate
- Mock external service dependencies
- Provide realistic mock responses for Docker operations

## Expected Implementation Modules

The tests expect the following modules to be implemented:

### `quick_start.docker.compose_generator`
- `DockerComposeGenerator` class
- Methods for generating docker-compose.yml files
- Profile-specific generation logic
- Template integration

### `quick_start.docker.container_config`
- `ContainerConfigManager` class
- Service configuration generation
- Environment variable management
- Resource limit configuration

### `quick_start.docker.service_manager`
- `DockerServiceManager` class
- Docker-compose operations (up, down, logs)
- Health check management
- Service connectivity testing

### `quick_start.docker.volume_manager`
- `VolumeManager` class
- Volume creation and management
- Backup and restore operations
- Data persistence handling

### `quick_start.docker.templates`
- `DockerTemplateEngine` class
- Template loading and processing
- Variable substitution
- Template validation

## Test Data Structure

### Configuration Files
```yaml
# docker_compose_test_configs.yaml
minimal_profile:
  profile: minimal
  document_count: 50
  # ... configuration details

standard_profile:
  profile: standard
  document_count: 500
  # ... configuration details
```

### Template Files
```yaml
# docker_compose_templates.yaml
minimal_template: |
  version: '3.8'
  services:
    iris:
      # ... service configuration
```

## Validation Criteria

### Docker-compose File Validation
- Valid YAML syntax
- Required services present for each profile
- Proper service dependencies
- Correct port mappings
- Valid environment variables
- Appropriate volume configurations
- Network configuration

### Service Configuration Validation
- Container images specified
- Health checks configured
- Resource limits set (for production profiles)
- Environment variables properly injected
- Volumes mounted correctly

### Integration Validation
- CLI wizard integration works
- Setup pipeline integration works
- Template engine integration works
- Sample data manager integration works

## Error Handling Tests

### Invalid Configuration Handling
- Invalid profile names
- Missing required configuration
- Invalid Docker service definitions
- Port conflicts
- Invalid volume mounts

### Docker Environment Issues
- Docker not installed
- Docker-compose not available
- Permission issues
- Network conflicts

### Resource Constraint Handling
- Insufficient memory
- CPU limits exceeded
- Disk space issues
- Network port conflicts

## Performance Considerations

### Test Execution Speed
- Mock Docker operations to avoid slow container operations
- Use temporary directories for fast file I/O
- Parallel test execution where possible
- Efficient fixture setup and teardown

### Resource Usage
- Minimal memory footprint for test execution
- Clean up temporary files and directories
- Avoid actual Docker container creation during tests
- Mock external service calls

## Continuous Integration

### CI/CD Integration
- Tests run in GitHub Actions/GitLab CI
- Docker-in-Docker configuration for CI environments
- Test result reporting and coverage metrics
- Automated test execution on pull requests

### Test Environment Setup
- Consistent test environment across CI/CD platforms
- Docker and docker-compose installation in CI
- Test data and fixture management
- Artifact collection for failed tests

## Troubleshooting

### Common Test Failures
- Import errors for non-existent modules (expected in TDD)
- Mock configuration issues
- Temporary directory cleanup problems
- YAML parsing errors in test data

### Debugging Tips
- Use `pytest -v -s` for verbose output
- Check temporary directory contents for generated files
- Verify mock configurations match expected calls
- Use `pytest --pdb` for interactive debugging

## Future Enhancements

### Additional Test Coverage
- Multi-platform Docker testing (Linux, macOS, Windows)
- Docker Swarm mode testing
- Kubernetes deployment testing
- Performance benchmarking of Docker operations

### Advanced Features
- Docker image building tests
- Registry integration tests
- Secret management tests
- Multi-environment deployment tests

## Contributing

### Adding New Tests
1. Follow TDD principles - write failing tests first
2. Use existing fixtures and patterns
3. Ensure test isolation and cleanup
4. Add appropriate documentation
5. Update this guide with new test categories

### Test Naming Conventions
- Use descriptive test names that explain what is being tested
- Group related tests in the same test class
- Use consistent naming patterns across test files
- Include profile names in test names where relevant

### Code Quality
- Follow existing code style and patterns
- Add type hints where appropriate
- Include docstrings for test methods
- Ensure comprehensive test coverage