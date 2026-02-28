# Contributing to iris-vector-rag

Thank you for your interest in contributing to iris-vector-rag! This document provides guidelines and workflows for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Git Workflow](#git-workflow)
- [Development Process](#development-process)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites
- Python 3.11+
- Docker Desktop (for IRIS database)
- Git
- uv (Python package manager)

### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/intersystems-community/iris-vector-rag.git
cd iris-vector-rag

# Setup environment
make setup-env    # Create .venv using uv
make install      # Install all dependencies
source .venv/bin/activate

# Start IRIS database
docker-compose up -d

# Initialize database
make setup-db
make load-data
```

### Verify Installation
```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/contract/       # Contract tests (TDD)
```

## Git Workflow

### Repository Structure

This project uses a **three-tier repository strategy** to balance private development with public collaboration:

```
origin (private)    → isc-tdyar/iris-vector-rag-private
fork (public)       → isc-tdyar/iris-vector-rag
upstream (community)→ intersystems-community/iris-vector-rag
```

**For Contributors**: You'll work with `upstream` (community repo) and your own fork.

### Contribution Workflows

#### 1. Standard Contribution (Recommended)

**For external contributors and most features:**

```bash
# 1. Fork the community repository on GitHub
# github.com/intersystems-community/iris-vector-rag → Fork

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/iris-vector-rag.git
cd iris-vector-rag

# 3. Add upstream remote
git remote add upstream https://github.com/intersystems-community/iris-vector-rag.git

# 4. Create feature branch
git checkout -b feature/your-feature-name

# 5. Make changes and commit
git add .
git commit -m "feat: your feature description"

# 6. Keep your fork up to date
git fetch upstream
git rebase upstream/main

# 7. Push to your fork
git push origin feature/your-feature-name

# 8. Create Pull Request on GitHub
# From: YOUR_USERNAME/iris-vector-rag:feature/your-feature-name
# To: intersystems-community/iris-vector-rag:main
```

#### 2. Internal Development (Maintainer Workflow)

**For core maintainers with access to private repository:**

```bash
# Daily private work
git commit -am "feat: experimental feature"
git push origin main  # Push to private repo

# When ready to share publicly
git checkout -b public/feature-name
git cherry-pick <commit-hash>  # Select specific commits
git push fork public/feature-name

# Create PR: fork → upstream
```

#### 3. Emergency Release (Maintainers Only)

**Rare cases when immediate public sync is needed:**

```bash
# Sync all repositories (requires write access to all)
git push origin main && git push fork main && git push upstream main
```

### Branch Naming Conventions

- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates
- `refactor/*` - Code refactoring
- `test/*` - Test improvements
- `public/*` - Public-facing feature branches (maintainers)
- `release/*` - Release preparation branches

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions or improvements
- `chore`: Build process or auxiliary tool changes
- `perf`: Performance improvements

**Examples:**
```bash
feat(pipelines): add ColBERT late interaction retrieval

Implement PyLateColBERTPipeline with maxsim scoring for improved
relevance matching on multi-token queries.

Closes #123
```

```bash
fix(embeddings): resolve DP-442038 model loading overhead

Add Python embedding cache layer to prevent IRIS from reloading
embedding models on every INSERT operation.

Performance: 1405x speedup (20 min → 0.85s for 1746 documents)
```

## Development Process

### Test-Driven Development (TDD)

**Required**: All new features must follow TDD approach per `.specify/memory/constitution.md` Principle III.

1. **Write Contract Tests First**
   ```bash
   # Create contract tests (8-12 tests)
   touch tests/contract/test_your_feature_contract.py

   # Tests should FAIL initially (red phase)
   pytest tests/contract/test_your_feature_contract.py
   ```

2. **Implement Feature**
   ```bash
   # Write implementation
   touch iris_vector_rag/your_module.py

   # Tests should PASS (green phase)
   pytest tests/contract/test_your_feature_contract.py
   ```

3. **Add Integration Tests**
   ```bash
   # Create integration tests (10-15 tests)
   touch tests/integration/test_your_feature_integration.py

   # Use .DAT fixtures for ≥10 entities
   pytest tests/integration/test_your_feature_integration.py
   ```

4. **Refactor**
   ```bash
   # Clean up implementation
   # All tests must still pass
   pytest tests/
   ```

### .DAT Fixture Requirements

**Mandatory**: Integration/E2E tests with ≥10 entities MUST use .DAT fixtures (Constitution Principle II).

**Why**: 100-200x faster than JSON fixtures (0.5-2s vs 39-75s for 100 entities)

```python
import pytest

@pytest.mark.dat_fixture("medical-graphrag-20")
def test_with_fixture():
    # Fixture automatically loaded before test
    # Database contains 21 entities, 15 relationships
    pass
```

**Fixture Commands:**
```bash
make fixture-list                     # List available fixtures
make fixture-info FIXTURE=name        # Get fixture details
make fixture-load FIXTURE=name        # Load fixture manually
make fixture-create FIXTURE=name      # Create new fixture
```

### Code Quality Standards

```bash
# Format code (required before commit)
black .
isort .

# Lint code
flake8 .
mypy iris_vector_rag/

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=iris_vector_rag --cov-report=html
```

### Performance Requirements

Per Constitution Principle VI:
- **Query operations**: <5ms overhead when features disabled
- **Bulk operations**: 10x+ speedup vs one-by-one (10K docs <10s)
- **Monitoring**: <5% overhead when enabled, 0% when disabled
- **Test execution**: Integration tests complete in <30s total

## Testing Requirements

### Test Suite Structure

```
tests/
├── unit/           # Component-level tests (15-20 per module)
├── contract/       # TDD contract tests (8-12 per feature)
├── integration/    # Cross-component tests (10-15 per feature)
├── e2e/            # Full pipeline workflows
└── benchmarks/     # Performance benchmarks (pytest-benchmark)
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific suite
pytest tests/unit/
pytest tests/contract/
pytest tests/integration/

# Specific test file
pytest tests/contract/test_iris_embedding_contract.py

# Specific test
pytest tests/contract/test_iris_embedding_contract.py::test_cache_hit_rate

# With verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=iris_vector_rag --cov-report=html

# Parallel execution (faster)
pytest tests/ -n auto
```

### Backend Mode Testing

Test with both Community and Enterprise IRIS editions:

```bash
# Community Edition (1 connection)
IRIS_BACKEND_MODE=community pytest tests/

# Enterprise Edition (999 connections)
IRIS_BACKEND_MODE=enterprise pytest tests/
```

## Pull Request Process

### Before Submitting

1. **Verify all tests pass**
   ```bash
   pytest tests/
   ```

2. **Check code formatting**
   ```bash
   black --check .
   isort --check .
   flake8 .
   ```

3. **Verify backward compatibility**
   ```bash
   # All existing tests must still pass
   pytest tests/integration/ tests/e2e/
   ```

4. **Update documentation**
   - Update CLAUDE.md if adding new features
   - Update README.md if changing public API
   - Add docstrings to all new functions/classes

5. **Review Constitution compliance**
   - [ ] TDD approach followed (contract tests first)
   - [ ] .DAT fixtures used for integration tests ≥10 entities
   - [ ] Backward compatible (no breaking changes)
   - [ ] Performance benchmarks pass
   - [ ] Security best practices followed

### PR Description Template

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Contract tests added (TDD)
- [ ] Integration tests added
- [ ] .DAT fixtures used where required
- [ ] All tests passing
- [ ] Performance benchmarks passing

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests written before implementation (TDD)
- [ ] .DAT fixtures used for ≥10 entities
- [ ] Backward compatibility maintained
- [ ] Documentation updated
- [ ] Constitution principles followed

## Related Issues
Closes #<issue_number>
```

### Review Process

1. Automated CI checks run (tests, linting, coverage)
2. Code review by maintainers
3. Approval required from at least one maintainer
4. Merge to main branch

## Release Process

### Version Bumping

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist (Maintainers)

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml  # version = "0.5.x"

# 2. Update CHANGELOG.md
vim CHANGELOG.md

# 3. Commit changes
git commit -am "chore: bump version to 0.5.x"

# 4. Build distribution packages
uv build

# 5. Verify distributions
ls -lh dist/

# 6. Publish to PyPI (use twine, not uv publish)
twine upload dist/iris_vector_rag-*.whl dist/iris_vector_rag-*.tar.gz

# 7. Create git tag
git tag -a v0.5.x -m "Release v0.5.x"

# 8. Push to all repositories
git push origin main
git push fork main
git push upstream main
git push --tags
```

### PyPI Publishing Standards

**Mandatory**: Use `twine` for PyPI publishing (Constitution Principle X).

```bash
# Build distributions
uv build

# Publish using twine (reads ~/.pypirc automatically)
twine upload dist/iris_vector_rag-*.whl dist/iris_vector_rag-*.tar.gz
```

**Never use `uv publish`** - it does not properly read `.pypirc` credentials.

## Questions or Problems?

- **Issues**: https://github.com/intersystems-community/iris-vector-rag/issues
- **Discussions**: https://github.com/intersystems-community/iris-vector-rag/discussions
- **Documentation**: See `CLAUDE.md` for development guidance
- **Constitution**: See `.specify/memory/constitution.md` for core principles

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
