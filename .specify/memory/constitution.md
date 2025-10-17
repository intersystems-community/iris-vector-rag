<!--
Sync Impact Report:
- Version change: 1.5.0 → 1.6.0
- List of modified principles:
  * Development Standards (enhanced with uv package management requirement)
- Amendment details:
  * Added mandatory uv usage for Python package management
  * Deprecates traditional pip/virtualenv workflows
  * Emphasizes uv's superior performance and reliability
  * Ensures framework-wide consistency in dependency management
- Added sections: Package Management requirement within existing Development Standards
- Removed sections: N/A
- Templates requiring updates:
  ⏳ Makefile (update to use uv commands)
  ⏳ README.md (update setup instructions to use uv)
  ⏳ requirements files (consider pyproject.toml migration)
- Follow-up TODOs: Migrate existing projects to use uv for package management
-->

# RAG-Templates Constitution

## Core Principles

### I. Framework-First Architecture

Every component MUST be designed as a reusable framework element. Pipeline implementations MUST extend the abstract RAGPipeline base class. Components MUST be independently testable and documented with clear interfaces. No application-specific logic in framework components. All components MUST expose functionality via CLI interface (Make targets or direct CLI commands).

**Rationale**: Ensures the framework can be consumed by multiple applications while maintaining clean separation of concerns, reusability, and operational accessibility.

### II. Pipeline Validation & Requirements

All pipelines MUST implement automated requirement validation and setup orchestration. Pipeline creation MUST fail fast with clear error messages when prerequisites are missing. Setup procedures MUST be idempotent and recoverable.

**Rationale**: Enterprise environments require reliable, self-validating components that can detect and resolve configuration issues automatically.

### III. Test-Driven Development with Live Database Validation (NON-NEGOTIABLE)

Tests MUST be written before implementation. All contract tests MUST fail initially, then pass after implementation. Unit, integration, and end-to-end tests are mandatory for all pipeline implementations.

**IRIS Database Requirement**: All RAG framework tests that involve data storage, vector operations, schema management, or pipeline operations MUST execute against a running IRIS database instance. Tests MUST use either:
1. **Framework-managed Docker IRIS** (preferred): Use available licensed IRIS images with dynamic port allocation
2. **External IRIS instance**: When configured via environment variables

**IRIS Docker Management Procedures**:
- **Required IRIS Image**: ALL Docker Compose files MUST use `docker.iscinternal.com/intersystems/iris:2025.3.0EHAT.127.0-linux-arm64v8` (note: NOT iris-lockeddown)
- **Licensed IRIS Priority**: Use `docker.iscinternal.com/intersystems/iris:2025.3.0EHAT.127.0-linux-arm64v8` or similar licensed images when available
- **Standardized Port Mapping**: ALL Docker Compose files MUST follow the framework's port mapping strategy:
  * **Container ports**: Always use IRIS standard ports (1972 SuperServer, 52773 Management Portal)
  * **Host port ranges**:
    - Default IRIS: `11972:1972` and `152773:52773` (docker-compose.yml)
    - Licensed IRIS: `21972:1972` and `252773:52773` (docker-compose.licensed.yml)
    - Reserved: `1972:1972` and `52773:52773` (for existing IRIS installations)
  * **Rationale**: Predictable ports avoid conflicts, enable easy configuration, support multiple IRIS instances
- **Port Discovery**: Use the framework's `common/iris_port_discovery.py` utility to locate running IRIS instances across mapped ports
- **Connection Management**: Use `common/iris_connection_manager.py` for standardized IRIS connections with automatic fallback patterns
- **Health Validation**: Always run `python evaluation_framework/test_iris_connectivity.py` before database-dependent testing

**Test Categories with IRIS Requirements**:
- `@pytest.mark.requires_database`: MUST connect to live IRIS
- `@pytest.mark.integration`: MUST use IRIS for data operations
- `@pytest.mark.e2e`: MUST use complete IRIS + vector workflow
- `@pytest.mark.clean_iris`: MUST start from fresh/empty IRIS instance to validate complete setup workflow
- Unit tests MAY mock IRIS for isolated component testing

**Clean IRIS Testing Requirement (NON-NEGOTIABLE)**: All RAG pipeline implementations MUST provide test variants that start from a completely clean/fresh IRIS database instance. These tests validate the complete setup orchestration including schema creation, data ingestion, and pipeline initialization. Clean IRIS tests are essential for:
- Validating auto-setup and orchestration capabilities
- Testing schema migration and upgrade procedures
- Ensuring reproducible deployment scenarios
- Verifying framework self-sufficiency in new environments

**Database Health Validation**: All test suites MUST verify IRIS health before execution using established connectivity tests from `evaluation_framework/test_iris_connectivity.py`.

**Container Management Best Practices**:
- **Avoid Port Conflicts**: Never force-stop existing IRIS containers; use dynamic port allocation instead
- **Image Selection**: Check `docker images | grep iris` to identify available licensed IRIS images before attempting pulls
- **Configuration Updates**: Modify docker-compose files to use available images and dynamic ports rather than creating new containers
- **Environment Variables**: Set `IRIS_HOST=localhost`, `IRIS_PORT=<discovered_port>`, `IRIS_USERNAME=_SYSTEM`, `IRIS_PASSWORD=SYS` for testing

Performance tests required for enterprise scale scenarios (10K+ documents) MUST execute against IRIS with actual vector operations and RAGAS evaluation.

**Rationale**: RAG systems are fundamentally dependent on IRIS database for document storage, vector embeddings, and search operations. Testing without live database connections provides false validation and cannot detect real-world integration failures, performance issues, or data consistency problems.

### IV. Performance & Enterprise Scale

All pipelines MUST support incremental indexing and concurrent operations. Vector operations MUST be optimized for IRIS database capabilities. Memory usage MUST be monitored and bounded. Performance benchmarks required for 1K, 10K document scenarios with RAGAS evaluation.

**Rationale**: Enterprise RAG applications must handle large document corpora efficiently without degrading user experience.

### V. Production Readiness Standards

All components MUST include structured logging, error handling, and health checks. Configuration MUST be externalized via environment variables and YAML. Docker deployment MUST be supported with comprehensive monitoring capabilities.

**Rationale**: Production RAG systems require operational excellence for reliability, debugging, and maintenance.

### VI. Explicit Error Handling (NON-NEGOTIABLE)

Domain errors only: no silent failures, every bug MUST be explicit. All error conditions MUST surface as clear exceptions with actionable messages. No swallowed exceptions or undefined behavior. Failed operations MUST provide specific context about what failed and why.

**Rationale**: RAG systems process critical knowledge; silent failures can lead to incorrect or missing information being returned to users, which is unacceptable in enterprise environments.

### VII. Standardized Database Interfaces

All database interactions MUST use proven, standardized utilities from the framework's SQL and vector helper modules. No ad-hoc database queries or direct IRIS API calls outside established patterns. New database patterns MUST be contributed back to shared utilities after validation.

**Rationale**: IRIS database interactions have complex edge cases and performance considerations. Hard-won fixes and optimizations must be systematized to prevent teams from rediscovering the same issues.

## Enterprise Requirements

Production deployments MUST include:

- Vector store persistence and backup procedures
- API rate limiting and authentication
- Observability stack (logging, metrics, tracing)
- Security scanning and vulnerability management
- Multi-environment configuration management (dev, staging, prod)

## Development Standards

**Package Management**: All Python projects MUST use uv for dependency management, virtual environment creation, and package installation. Traditional pip/virtualenv workflows are deprecated in favor of uv's superior performance and reliability.

Code MUST pass linting (black, isort, flake8, mypy) before commits. All public APIs MUST include comprehensive docstrings. Breaking changes MUST follow semantic versioning. Dependencies MUST be pinned and regularly updated for security.

Documentation MUST include quickstart guides, API references, and integration examples. Agent-specific guidance files (CLAUDE.md) MUST be maintained for AI development assistance.

## Governance

This constitution supersedes all other development practices. All pull requests MUST verify constitutional compliance. Complexity that violates simplicity principles MUST be justified in writing with alternatives considered.

### AI Architecting Principles

Development with AI tools MUST follow constraint-based architecture, not "vibecoding". Constitutional validation gates serve as constraint checklists that prevent repeating known bugs and design mistakes. Every bug fix MUST be captured as a new validation rule or enhanced guideline. AI development MUST work within established frameworks, patterns, and validation loops.

**Constraint Philosophy**: Less freedom = less chaos. Constraints are superpowers that prevent regression and ensure consistency.

Amendment procedure: Proposed changes require documentation of impact, team approval, and migration plan for affected components. Version increments follow semantic versioning based on change scope.

**Version**: 1.6.0 | **Ratified**: 2025-01-27 | **Last Amended**: 2025-09-28
