# Contributing to RAG Templates Framework

Welcome to the RAG Templates Framework! We're excited that you're interested in contributing. This guide will help you get started with contributing to our project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code Review Process](#code-review-process)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful, inclusive, and collaborative in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (for containerized development)
- VS Code (recommended) with Python extension

### Project Overview

The RAG Templates Framework provides a comprehensive solution for building Retrieval-Augmented Generation applications with:

- **Memory Integration**: Mem0 for persistent conversation memory
- **MCP Server Support**: Model Context Protocol server integration
- **IRIS Database**: InterSystems IRIS for vector storage
- **Workspace Integration**: VS Code workspace pattern support
- **Configuration Management**: Environment-aware configuration system

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rag-templates.git
cd rag-templates

# Add the original repository as upstream
git remote add upstream https://github.com/ORIGINAL_OWNER/rag-templates.git
```

### 2. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# See docs/setup/ for detailed configuration guides
```

### 4. Database Setup (Optional)

```bash
# Start IRIS database with Docker
docker-compose up -d iris

# Or use the licensed version
docker-compose -f docker-compose.licensed.yml up -d
```

### 5. Verify Setup

```bash
# Run basic tests to verify setup
python -m pytest tests/unit/ -v

# Run configuration validation
python scripts/test_mcp_validation.py
```

## Development Workflow

### 1. Create Feature Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Keep commits atomic and focused
- Write descriptive commit messages
- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed

### 3. Test Changes

```bash
# Run full test suite
python -m pytest

# Run specific test categories
python -m pytest -m unit        # Unit tests only
python -m pytest -m integration # Integration tests only
python -m pytest -m e2e         # End-to-end tests only

# Check code coverage
python -m pytest --cov=src --cov-report=html

# Run linting and formatting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### 4. Update Documentation

```bash
# Build documentation locally
cd docs/
mkdocs serve

# Check for broken links and accessibility
python scripts/ci/validate-docs.py
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some project-specific guidelines:

- **Line Length**: 88 characters (Black default)
- **Import Sorting**: Use isort with profile "black"
- **Type Hints**: Required for all public functions and classes
- **Docstrings**: Google style for all public modules, classes, and functions

### Code Quality Tools

All code must pass these tools (automatically checked in CI):

```bash
# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Linting
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Complexity checking
radon cc src/ -a -nb
```

### Architecture Principles

- **Clean Architecture**: Separate concerns into distinct layers
- **Dependency Injection**: Use configuration abstractions
- **Modular Design**: Keep files under 500 lines
- **Error Handling**: Comprehensive error handling with proper logging
- **Security First**: Never hardcode secrets or sensitive data
- **Testability**: Design for easy testing and mocking

### File Organization

```
src/
├── config/          # Configuration management
├── data/            # Data models and database
├── mcp/             # MCP server integration
├── memory/          # Memory management (Mem0)
├── templates/       # Template generation
└── utils/           # Shared utilities

tests/
├── unit/            # Unit tests
├── integration/     # Integration tests
├── e2e/             # End-to-end tests
└── fixtures/        # Test data and fixtures
```

## Testing

### Test Categories

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance and scalability

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestExampleClass:
    """Test class with proper structure."""
    
    def test_should_return_expected_value_when_valid_input(self):
        """Test method with descriptive name."""
        # Arrange
        input_value = "test"
        expected = "expected_result"
        
        # Act
        result = example_function(input_value)
        
        # Assert
        assert result == expected
    
    @pytest.mark.integration
    def test_integration_scenario(self):
        """Integration test example."""
        # Test integration between components
        pass
    
    @pytest.mark.asyncio
    async def test_async_function(self):
        """Async test example."""
        result = await async_function()
        assert result is not None
```

### Test Requirements

- **Coverage**: Minimum 80% line coverage for new code
- **Assertions**: Use descriptive assertion messages
- **Mocking**: Mock external dependencies in unit tests
- **Cleanup**: Use fixtures for setup/teardown
- **Markers**: Use pytest markers to categorize tests

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run specific markers
python -m pytest -m "unit and not slow"

# Run in parallel
python -m pytest -n auto

# Run with verbose output
python -m pytest -v --tb=short
```

## Documentation

### Documentation Types

- **API Documentation**: Docstrings for all public APIs
- **User Guides**: Step-by-step instructions in `docs/guides/`
- **Architecture**: System design in `docs/architecture/`
- **Setup Guides**: Environment setup in `docs/setup/`

### Documentation Standards

- Use Markdown for all documentation
- Include code examples that work
- Test all code examples in CI
- Keep examples up to date
- Include both happy path and error scenarios

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build and serve locally
cd docs/
mkdocs serve

# Build for production
mkdocs build
```

## Submitting Changes

### Pull Request Process

1. **Update Your Branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run Pre-submission Checks**
   ```bash
   # Run full test suite
   python -m pytest
   
   # Run linting
   black --check src/ tests/
   flake8 src/ tests/
   mypy src/
   
   # Check security
   bandit -r src/
   ```

3. **Push and Create PR**
   ```bash
   git push origin your-feature-branch
   # Create pull request on GitHub
   ```

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Use the PR template provided
- **Size**: Keep PRs focused and reasonably sized
- **Tests**: Include tests for all new functionality
- **Documentation**: Update docs for user-facing changes
- **Breaking Changes**: Clearly mark and document breaking changes

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] CI checks passing
- [ ] Security considerations addressed

## Code Review Process

### Review Criteria

- **Functionality**: Does the code work as intended?
- **Design**: Is the code well-designed and fits the architecture?
- **Complexity**: Is the code more complex than necessary?
- **Tests**: Are there adequate, well-designed tests?
- **Naming**: Are names clear and descriptive?
- **Comments**: Are comments useful and necessary?
- **Documentation**: Is documentation updated appropriately?

### Review Guidelines

- **Be Constructive**: Focus on the code, not the person
- **Explain Why**: Provide reasoning for suggestions
- **Suggest Alternatives**: Offer specific improvements
- **Acknowledge Good Work**: Highlight well-written code
- **Be Timely**: Review PRs promptly

### Addressing Feedback

- **Be Responsive**: Address feedback promptly
- **Ask Questions**: Clarify unclear feedback
- **Test Changes**: Ensure fixes work correctly
- **Update Documentation**: Reflect any design changes

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

### Release Workflow

1. **Create Release Branch**
   ```bash
   git checkout -b release/v1.2.0
   ```

2. **Update Version and Changelog**
   - Update version in `setup.py` and `pyproject.toml`
   - Update `docs/docs/docs/CHANGELOG.md` with release notes

3. **Final Testing**
   ```bash
   # Run full test suite
   python -m pytest
   
   # Run performance benchmarks
   python scripts/ci/run-benchmarks.py
   ```

4. **Create Release PR**
   - Submit PR to main branch
   - Wait for approval and CI checks

5. **Tag and Release**
   - Merge release PR
   - Tag release: `git tag v1.2.0`
   - Push tag: `git push origin v1.2.0`
   - GitHub Actions will handle the rest

### Release Responsibilities

- **Maintainers**: Create and manage releases
- **Contributors**: Ensure changes are documented
- **Reviewers**: Verify release readiness

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Request Comments**: Code-specific discussions

### Documentation Resources

- **Setup Guides**: `docs/setup/`
- **Architecture Docs**: `docs/architecture/`
- **API Reference**: Generated from docstrings
- **Examples**: `examples/` directory

### Common Issues

- **Environment Setup**: Check `docs/setup/` guides
- **Test Failures**: Ensure dependencies are installed
- **Import Errors**: Verify virtual environment activation
- **Configuration Issues**: Check `.env` file setup

### Reporting Issues

When reporting issues, please include:

- Python version and operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Relevant configuration (without secrets)

### Feature Requests

For feature requests, please describe:

- The problem you're trying to solve
- Your proposed solution
- Alternative solutions considered
- Implementation considerations

---

Thank you for contributing to the RAG Templates Framework! Your contributions help make this project better for everyone.