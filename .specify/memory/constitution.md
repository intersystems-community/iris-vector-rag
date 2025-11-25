# iris-vector-rag Constitution

<!--
Sync Impact Report - Version 1.1.0

Version Change: 1.0.0 → 1.1.0
Rationale: MINOR bump - added new Principle X (PyPI Publishing Standards)

Added Sections:
  - Principle X: PyPI Publishing Standards
  - Guidance on using twine instead of uv publish
  - Rationale for credential compatibility

Modified Principles: None

Removed Sections: None

Templates Requiring Updates:
  - ✅ CLAUDE.md (already contains PyPI workflow documentation)
  - ⚠️  Build/release documentation may need update if exists

Follow-up TODOs: None
-->

## Core Principles

### I. Library-First Design
Every feature starts as a standalone library component with a programmatic API before CLI/web/API exposure. Libraries must be self-contained, independently testable, and documented. Clear purpose required - no organizational-only libraries.

**Why**: Enables maximum reusability, testability, and flexibility for users integrating iris-vector-rag into their systems.

### II. .DAT Fixture-First Testing (NON-NEGOTIABLE)
Integration and E2E tests with ≥10 entities MUST use .DAT fixtures loaded via iris-devtools. Performance mandate: 100-200x faster than JSON fixtures (0.5-2s vs 39-75s for 100 entities).

**Why**: InterSystems IRIS binary format provides orders of magnitude faster test execution, enabling comprehensive integration testing without CI timeout issues.

**Enforcement**:
- All integration/E2E tests with ≥10 entities use `.DAT` fixtures
- Programmatic fixtures allowed for <10 entities or mocked components
- Use `@pytest.mark.dat_fixture("fixture-name")` decorator
- See `tests/fixtures/manager.py` for fixture management

### III. Test-First (TDD) (NON-NEGOTIABLE)
Contract tests written → User approved → Tests fail → Implement → Tests pass. Red-Green-Refactor cycle strictly enforced.

**Why**: Prevents scope creep, ensures requirements are testable, and validates implementation correctness.

**Enforcement**:
- Contract tests created before implementation
- All contract tests must fail initially (red phase)
- Implementation proceeds only after tests written and reviewed
- Integration tests added after contract tests pass

### IV. Backward Compatibility (NON-NEGOTIABLE)
All new features default to disabled. Zero breaking changes to existing public APIs.

**Why**: iris-vector-rag is a production library used by multiple teams. Breaking changes harm user trust and adoption.

**Enforcement**:
- All new features opt-in (disabled by default)
- Existing test suite must pass unchanged
- Configuration changes are additive only
- Deprecation warnings required 2 releases before removal

### V. InterSystems IRIS Integration
All vector storage uses IRIS native capabilities. No fallback to alternative databases. IRIS-specific features (vector search, SQL optimization, transactions) are first-class citizens.

**Why**: iris-vector-rag is built specifically for InterSystems IRIS. Supporting multiple backends dilutes focus and testing coverage.

**Enforcement**:
- Vector operations use IRIS native vector search
- Batch operations use IRIS transactions
- Test fixtures use IRIS .DAT format
- No "if iris_available" conditionals for core functionality

### VI. Performance Standards
- Query operations: <5ms overhead when features disabled
- Bulk operations: 10x+ speedup vs one-by-one (10K docs <10s)
- Monitoring: <5% overhead when enabled, 0% when disabled
- Test execution: Integration tests complete in <30s total

**Why**: Production RAG systems require sub-second query latency and efficient document indexing.

**Enforcement**:
- Performance benchmarks in contract tests
- CI fails if performance regressions detected
- pytest-benchmark for automated measurement

### VII. Observability
All operations emit structured logs. Critical paths instrumented with OpenTelemetry spans. Text I/O ensures debuggability.

**Why**: Production systems require visibility into failures, performance bottlenecks, and usage patterns.

**Enforcement**:
- OpenTelemetry integration for query/indexing operations
- Structured logging (JSON format) for all errors
- Clear error messages with actionable guidance

### VIII. Security-First
- SQL injection prevention via parameterized queries
- RBAC policy interface for access control
- Audit logging for permission decisions
- No secrets in logs or error messages

**Why**: RAG systems often handle sensitive documents requiring enterprise-grade security controls.

**Enforcement**:
- All metadata filters use parameterized SQL
- Contract tests verify SQL injection prevention
- RBAC permission checks before data access

### IX. Simplicity (YAGNI)
Start simple. Build for current requirements, not hypothetical future needs. Premature abstraction is technical debt.

**Why**: Unnecessary complexity slows development, increases bug surface area, and confuses users.

**Enforcement**:
- New abstractions require justification in design docs
- "We might need this later" is not justification
- Delete unused code immediately

### X. PyPI Publishing Standards
MUST use `twine` for publishing packages to PyPI. DO NOT use `uv publish`.

**Why**: `uv publish` does not properly read credentials from `.pypirc` files, causing authentication failures. `twine` is the standard Python packaging tool with mature credential handling.

**Enforcement**:
- Use `twine upload dist/*` for PyPI publishing
- Verify `.pypirc` configuration before publishing
- Build distributions with `uv build` (or `python -m build`)
- Never commit `.pypirc` or credentials to version control

**Publishing Workflow**:
```bash
# 1. Build distribution packages
uv build

# 2. Verify distributions created
ls -lh dist/

# 3. Publish to PyPI using twine (reads ~/.pypirc automatically)
twine upload dist/iris_vector_rag-*.whl dist/iris_vector_rag-*.tar.gz
```

## Testing Requirements

### Contract Tests (TDD)
- **Purpose**: Define expected behavior before implementation
- **Coverage**: ~8-12 tests per feature
- **Location**: `tests/contract/`
- **Format**: pytest with descriptive names (`test_custom_field_configuration`)

### Integration Tests
- **Purpose**: Validate end-to-end workflows with real IRIS database
- **Coverage**: ~10-15 tests per feature
- **Location**: `tests/integration/`
- **Requirement**: Use .DAT fixtures for ≥10 entities

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Coverage**: ~15-20 tests per module
- **Location**: `tests/unit/`
- **Scope**: Pure Python logic, mocked external dependencies

### Performance Benchmarks
- **Purpose**: Validate performance requirements
- **Tool**: pytest-benchmark
- **Location**: `tests/benchmarks/`
- **CI**: Fails on >10% regression

## Configuration Standards

### YAML Configuration
- **Format**: YAML with clear hierarchy
- **Location**: `iris_vector_rag/config/default_config.yaml`
- **Changes**: Additive only (no removals)
- **Documentation**: Every new key documented in comments

### Environment Variables
- **Prefix**: `IRIS_` for IRIS-specific settings
- **Format**: UPPER_SNAKE_CASE
- **Precedence**: Env vars override YAML config
- **Secrets**: Use env vars, never commit to version control

## Governance

### Constitution Authority
This constitution supersedes all other development practices. When conflicts arise, constitution principles take precedence.

### Amendments
- Amendments require documentation in `.specify/memory/constitution.md`
- Major changes require team discussion and approval
- Include migration plan for breaking changes

### Complexity Justification
Any addition that violates simplicity (Principle IX) requires explicit justification in design documents explaining why the complexity is necessary.

### Code Review Requirements
All PRs/reviews must verify:
- ✅ Tests written before implementation (TDD)
- ✅ .DAT fixtures used for integration tests ≥10 entities
- ✅ Backward compatibility maintained
- ✅ Performance benchmarks pass
- ✅ Constitution principles followed

**Version**: 1.1.0 | **Ratified**: 2025-11-22 | **Last Amended**: 2025-11-25
