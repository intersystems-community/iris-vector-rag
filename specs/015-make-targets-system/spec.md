# Make Targets System Specification

## Overview
The RAG Templates framework provides a comprehensive Makefile-based automation system that simplifies Docker container management, testing workflows, and development operations. The make targets serve as the primary interface for developers, DevOps engineers, and CI/CD systems to interact with the framework's infrastructure.

## Primary User Story
Developers and operators need a unified, discoverable, and reliable command interface to manage the complex multi-service RAG framework deployment without needing to memorize lengthy Docker Compose commands or configuration details. The make targets must provide consistent behavior, comprehensive error handling, and clear feedback for all common development and operational workflows.

## Acceptance Scenarios

### AC-001: Environment Setup and Validation
**GIVEN** a fresh clone of the RAG Templates repository
**WHEN** a developer runs `make setup`
**THEN** the system creates necessary directories, copies environment files, and makes scripts executable
**AND** running `make env-check` validates the configuration without errors

### AC-002: Docker Service Management
**GIVEN** a properly configured environment
**WHEN** a developer runs `make docker-up`
**THEN** core services (IRIS, Redis, API, Streamlit) start successfully
**AND** service URLs are displayed upon completion
**AND** health checks pass within the timeout period

### AC-003: Development Environment Orchestration
**GIVEN** a development workstation
**WHEN** a developer runs `make docker-dev`
**THEN** the development environment starts with Jupyter notebooks
**AND** sample data is automatically loaded
**AND** all services are healthy before completion

### AC-004: Testing Integration
**GIVEN** a running development environment
**WHEN** an operator runs `make test-ragas-sample`
**THEN** RAGAS evaluation runs on sample documents
**AND** reports are generated in the outputs directory
**AND** the process completes without manual intervention

### AC-005: Production Deployment
**GIVEN** a production server environment
**WHEN** an operator runs `make docker-prod`
**THEN** production services start with monitoring
**AND** all production health checks pass
**AND** appropriate security configurations are applied

## Functional Requirements

### Core Infrastructure Management
- **FR-001**: System MUST provide environment setup targets (`setup`, `env-check`) that validate prerequisites including Docker and Docker Compose dependencies, failing immediately with installation instructions when missing or incompatible, validate configuration files and environment variables failing with specific missing configuration guidance, and prepare the workspace
- **FR-002**: System MUST provide Docker lifecycle management targets (`docker-up`, `docker-down`, `docker-restart`) with profile support
- **FR-003**: System MUST provide environment-specific deployment targets (`docker-dev`, `docker-prod`, `docker-quick`) with appropriate service configurations
- **FR-004**: System MUST provide build and image management targets (`docker-build`, `docker-pull`, `docker-update`) with parallel execution support

### Monitoring and Debugging
- **FR-005**: System MUST provide monitoring targets (`docker-health`, `docker-stats`, `docker-logs`) with JSON output options for automation
- **FR-006**: System MUST provide shell access targets (`docker-shell`, `docker-iris-shell`, `docker-redis-shell`) for debugging and maintenance
- **FR-007**: System MUST provide service URL discovery (`docker-urls`) that dynamically detects running services

### Testing and Validation
- **FR-008**: System MUST provide enterprise-scale testing targets (`test-enterprise-10k`, `test-graphrag-scale`) with mock and real database options
- **FR-009**: System MUST provide RAGAS evaluation targets (`test-ragas-sample`, `test-ragas-1000`) with configurable pipeline selection
- **FR-010**: System MUST provide functional testing targets (`docker-test`, `docker-test-query`) for continuous integration workflows

### Data Management
- **FR-011**: System MUST provide data initialization targets (`docker-init-data`) with force reload capabilities
- **FR-012**: System MUST provide backup and recovery targets (`docker-backup`) with timestamped archives
- **FR-013**: System MUST provide cleanup targets (`docker-clean`, `docker-clean-all`) with interactive Y/N confirmation prompts defaulting to abort for destructive operations

### Developer Experience
- **FR-014**: System MUST provide comprehensive help (`help`, `info`) with categorized target listings and example usage
- **FR-015**: System MUST provide colored output and progress indicators for improved user experience
- **FR-016**: System MUST provide wait mechanisms (`docker-wait`) with timeout handling and retry logic using exponential backoff up to 3 maximum attempts for reliable automation

## Non-Functional Requirements

### Reliability
- **NFR-001**: All make targets MUST be idempotent and safe to run multiple times
- **NFR-002**: All make targets MUST provide appropriate error handling and exit codes for automation
- **NFR-003**: All make targets MUST validate prerequisites before executing destructive operations and require interactive Y/N confirmation prompts defaulting to abort

### Performance
- **NFR-004**: Docker operations MUST support parallel execution where applicable (builds, pulls)
- **NFR-005**: Health check operations MUST complete within configurable timeout periods (300s default) with retry logic using exponential backoff up to 3 maximum attempts when failures occur
- **NFR-006**: Large-scale testing targets MUST provide progress indicators and intermediate status updates

### Usability
- **NFR-007**: All make targets MUST include descriptive help text accessible via `make help`
- **NFR-008**: All make targets MUST provide consistent colored output and clear success/failure indicators
- **NFR-009**: All make targets MUST follow naming conventions for discoverability (`docker-*`, `test-*`)

## Key Entities

### Target Categories
- **Environment**: Setup, configuration validation, and workspace preparation
- **Docker**: Container lifecycle, image management, and service orchestration
- **Testing**: Unit tests, integration tests, enterprise scale validation, RAGAS evaluation
- **Monitoring**: Health checks, logging, statistics, debugging access
- **Data**: Initialization, backup, recovery, cleanup operations
- **Maintenance**: Updates, resets, optimization operations

### Configuration Dependencies
- **Environment File**: `.env` with API keys and database connections
- **Compose Files**: `docker-compose.full.yml` with service definitions
- **Scripts**: `scripts/docker/*.sh` for health checks and data initialization
- **Profiles**: Docker Compose profiles for different deployment scenarios (`core`, `dev`, `prod`)

## Implementation Guidelines

### Target Structure
```makefile
.PHONY: target-name
target-name: dependencies ## Help text for the target
	$(call print_message,$(COLOR),Status message)
	@command execution
	$(call print_message,$(GREEN),Completion message)
```

### Error Handling
- All targets must check prerequisites before execution
- Destructive operations must require confirmation
- Failed operations must provide actionable error messages
- Timeout mechanisms must be implemented for long-running operations

### Documentation Integration
- All targets must include help text in `## Comment` format
- Complex targets must display example usage
- Service URLs must be automatically discovered and displayed
- Configuration status must be validated and reported

### Testing Requirements
- All make targets must be testable in CI/CD environments
- Mock data options must be available for testing targets
- Health checks must be integrated into deployment workflows
- Performance benchmarks must be established for large-scale operations

## Dependencies

### External Dependencies
- Docker Engine (20.10+)
- Docker Compose (1.28+)
- GNU Make (3.81+)
- Bash shell environment

### Internal Dependencies
- IRIS database container health
- Environment configuration validation
- Docker network connectivity
- Volume mount permissions

### Service Integration
- All targets must integrate with the existing Docker Compose service definitions
- Health check scripts must be maintained and executable
- Configuration files must be validated before service startup
- Inter-service dependencies must be respected in startup sequences

## Clarifications

### Session 2025-01-28
- Q: What should happen when Docker or Docker Compose dependencies are missing or incompatible? → A: Fail immediately with installation instructions
- Q: How should the system handle timeout failures during health checks and service startup? → A: Retry with exponential backoff up to maximum attempts
- Q: What confirmation mechanism should be used for destructive operations like cleanup targets? → A: Interactive Y/N prompt with default to abort
- Q: What should be the default behavior when environment variables or configuration files are missing? → A: Fail with specific missing configuration guidance
- Q: What should be the maximum retry attempts for health checks and startup operations? → A: 3

## Success Metrics

### Developer Productivity
- Reduce setup time from manual Docker commands to single `make docker-dev` execution
- Provide zero-configuration testing workflows for common scenarios
- Enable reliable automation through consistent exit codes and error handling

### Operational Reliability
- Achieve 99%+ success rate for automated deployments using make targets
- Reduce deployment-related incidents through comprehensive validation
- Enable rapid troubleshooting through integrated monitoring targets

### Framework Adoption
- Provide comprehensive documentation through discoverable help system
- Enable self-service operations for common development tasks
- Support both development and production deployment scenarios