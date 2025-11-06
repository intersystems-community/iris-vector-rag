<!--
Sync Impact Report:
- Version change: 1.7.0 → 1.7.1 (PATCH)
- Version bump rationale: Enhanced Step 4 (PyPI Publishing) with detailed procedures
- List of modified principles: Principle VIII (Git & Release Workflow)
- Modified sections:
  * Step 4: PyPI Publishing - Added comprehensive publishing procedures
  * When to publish (semantic versioning guidance)
  * Version bumping procedure (files to update)
  * Build and validation steps (clean, build, test locally)
  * Upload procedure (secure token, verification)
  * Post-upload verification (install from PyPI, test imports)
- Removed sections: N/A
- Templates requiring updates:
  ✅ No template updates needed (workflow-specific, not template-affecting)
- Follow-up TODOs: None
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
- **Required IRIS Image**: ALL Docker Compose files MUST use `intersystemsdc/iris-community:2025.3.0EHAT.127.0-linux-arm64v8` (note: NOT iris-lockeddown)
- **Licensed IRIS Priority**: Use `intersystemsdc/iris-community:2025.3.0EHAT.127.0-linux-arm64v8` or similar licensed images when available
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

### VIII. Git & Release Workflow (NON-NEGOTIABLE)

All code changes MUST follow the exact dual-repository workflow to maintain separation between internal development and public releases:

**Step 1: Internal Repository (rag-templates)**
```bash
cd /Users/tdyar/ws/rag-templates
git add <changed-files>
git commit -m "<commit-message>"
git push git@gitlab.iscinternal.com:tdyar/rag-templates.git main
```

**Step 2: Sanitization Sync**
```bash
# From rag-templates directory
./scripts/sync_to_sanitized.sh
```

This script:
- Copies code to `../rag-templates-sanitized/`
- Applies redaction (replaces internal GitLab URLs → GitHub URLs, internal paths → community paths, internal emails → maintainer emails)
- Preserves git history for sanitized repository

**Step 3: Public Repository (iris-vector-rag)**
```bash
cd /Users/tdyar/ws/rag-templates-sanitized
git add <changed-files>
git commit -m "<commit-message>"
git push origin main  # iris-rag-templates repo
git push github main  # iris-vector-rag repo
```

**Step 4: PyPI Publishing (when applicable)**

**When to Publish:**
- New features added (MINOR version bump: 0.2.3 → 0.3.0)
- Bug fixes or patches (PATCH version bump: 0.2.3 → 0.2.4)
- Breaking changes (MAJOR version bump: 0.2.3 → 1.0.0)
- Skip if only documentation, tests, or internal changes

**Version Bumping:**
```bash
# Update version in BOTH files:
# 1. pyproject.toml (line 7): version = "0.2.4"
# 2. iris_rag/__init__.py (line 21): __version__ = "0.2.4"
```

**Build and Validate:**
```bash
cd /Users/tdyar/ws/rag-templates-sanitized

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build

# Validate distributions
python -m twine check dist/*

# Test installation locally
pip install --force-reinstall --no-deps dist/iris_vector_rag-*.whl

# Verify imports work
python -c "import iris_rag; print(f'iris_rag version: {iris_rag.__version__}'); from iris_rag import create_pipeline; print('✅ Package imports successfully')"
```

**Upload to PyPI:**
```bash
# Upload using secure token prompt
./upload_to_pypi.sh

# Verify upload succeeded
curl -s https://pypi.org/pypi/iris-vector-rag/json | python -c "import sys, json; data=json.load(sys.stdin); print(f\"Latest version: {data['info']['version']}\"); print(f\"Upload date: {list(data['releases'][data['info']['version']])[0]['upload_time']}\")"
```

**Post-Upload Verification:**
```bash
# Install from PyPI in fresh environment
pip install --upgrade iris-vector-rag

# Verify installation
python -c "import iris_rag; print(f'Installed version: {iris_rag.__version__}')"
```

**Critical Rules**:
- NEVER push internal repository code directly to public GitHub
- NEVER skip the sanitization step
- ALWAYS verify redaction worked (`grep -r "gitlab.iscinternal" ../rag-templates-sanitized/` should return nothing)
- ALWAYS test package installation before PyPI upload (`pip install dist/*.whl`)
- Public commits MUST NOT contain: internal URLs, internal email addresses, internal paths, proprietary information
- PyPI package name is `iris-vector-rag`, Python module name remains `iris_rag`

**Rationale**: The dual-repository workflow protects proprietary information while enabling open-source distribution. The sanitization step is non-negotiable to prevent accidental exposure of internal infrastructure details, employee information, or confidential development practices.

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

**Version**: 1.7.1 | **Ratified**: 2025-01-27 | **Last Amended**: 2025-11-06
