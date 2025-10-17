# Research: Fix Critical Issues from Analyze Command

**Feature**: 026-fix-critical-issues
**Research Date**: 2025-10-04

## 1. Pytest Hook Architecture for Coverage Warnings

### Research Question
How to implement pytest hooks that run after coverage collection to display warnings without failing tests?

### Findings

**pytest_terminal_summary hook**:
- Runs after all tests complete but before final output
- Has access to terminalreporter object for custom output
- Can access coverage data if pytest-cov plugin is active
- Called with `terminalreporter`, `exitstatus`, and `config` parameters

**Coverage data access**:
- Coverage.py stores data in `.coverage` file
- Can be loaded using `coverage.Coverage()` API
- Module-level coverage available via `coverage.get_data()`
- Works with pytest-cov's `--cov` options

**Warning display options**:
- `terminalreporter.write_line()` for custom output
- `warnings.warn()` with custom warning classes
- `terminalreporter.section()` for formatted sections
- Color coding: yellow for warnings, red for critical

### Decision
Use `pytest_terminal_summary` hook with direct coverage API access to display threshold warnings after test execution.

### Rationale
- Non-invasive: doesn't affect test pass/fail status
- Flexible: full control over output formatting
- Compatible: works with existing pytest-cov workflow
- Visible: appears prominently in test summary

### Alternatives Considered
1. **pytest_collection_modifyitems**: Too early, coverage not collected yet
2. **Custom pytest-cov reporter**: Would replace existing coverage output
3. **Post-test script**: Requires separate command, breaks workflow

## 2. Error Message Validation Patterns

### Research Question
How to automatically validate that test error messages follow a three-part structure (what/why/action)?

### Findings

**Error message interception**:
- `pytest_assertrepr_compare` customizes assertion messages
- `pytest_exception_interact` intercepts exceptions
- `sys.excepthook` for global exception handling
- pytest's `_pytest.assertion.rewrite` module for AST manipulation

**Validation approaches**:
- Regex patterns to detect message structure
- NLP for semantic analysis (overkill for this use case)
- Template matching with configurable rules
- AST analysis of assertion statements

**Three-part structure detection**:
```python
# Example pattern:
# What: "Test X failed"
# Why: "Expected Y but got Z"
# Action: "Check configuration at..."
pattern = r"(.*failed.*|.*error.*)\s+(expected.*but.*|because.*)\s+(check.*|verify.*|ensure.*)"
```

### Decision
Implement `pytest_exception_interact` hook with configurable regex patterns to validate error message structure.

### Rationale
- Catches all test failures and errors
- Allows validation without modifying test code
- Configurable patterns for different test types
- Can suggest improvements in real-time

### Alternatives Considered
1. **Static analysis**: Would miss runtime-generated messages
2. **AST rewriting**: Too complex, may break existing tests
3. **Post-test analysis**: Delayed feedback, less actionable

## 3. TDD Compliance Validation Approaches

### Research Question
How to verify contract tests existed in a failing state before implementation using git history?

### Findings

**Git history analysis with GitPython**:
```python
import git
repo = git.Repo('.')
# Find commits where test was added
commits = list(repo.iter_commits(paths='tests/contract/test_*.py'))
# Check test results in those commits
```

**Test state detection**:
- Parse pytest output from git commits
- Look for test file creation followed by implementation
- Check CI logs if available (GitHub Actions API)
- Analyze commit messages for TDD patterns

**Contract test identification**:
- File naming: `test_*_contract.py`
- Decorator: `@pytest.mark.contract`
- Directory location: `tests/contract/`
- Import patterns from contracts module

### Decision
Use GitPython to analyze commit history, checking test files were created before implementation files and initially had failing status.

### Rationale
- Accurate: Git history is source of truth
- Automated: Can run in CI/CD pipeline
- Flexible: Works with any git workflow
- Educational: Shows TDD compliance gaps

### Alternatives Considered
1. **Manual checklist**: Not scalable, prone to human error
2. **Pre-commit hooks only**: Misses historical violations
3. **CI-only validation**: Doesn't help during development

## 4. Critical Module Detection Strategies

### Research Question
How should the system identify which modules are "critical" for the 80% coverage threshold?

### Findings

**Configuration approaches**:
```ini
# .coveragerc
[coverage:run]
critical_modules =
    iris_rag/pipelines/
    iris_rag/storage/
    iris_rag/validation/

[coverage:report]
fail_under = 60
critical_fail_under = 80
```

**Convention-based detection**:
- Directory patterns: `*/core/*`, `*/critical/*`
- File naming: `*_critical.py`, `*_essential.py`
- Module docstrings: `"""CRITICAL: ..."""`
- Decorator marking: `@critical_module`

**Threshold customization**:
- Global defaults: 60% overall, 80% critical
- Per-module overrides in config
- Environment variable overrides for CI
- Team-specific configuration files

### Decision
Use `.coveragerc` configuration with directory patterns to identify critical modules, allowing per-module threshold overrides.

### Rationale
- Explicit: Clear which modules are critical
- Flexible: Easy to adjust thresholds
- Standard: Uses existing coverage.py config
- Maintainable: Single source of truth

### Alternatives Considered
1. **Hard-coded lists**: Not maintainable
2. **Decorator-based**: Requires code changes
3. **AI/heuristic detection**: Too unpredictable

## 5. Task Mapping Validation

### Research Question
How to extract requirement IDs from markdown and validate all have corresponding tasks?

### Findings

**Requirement ID extraction**:
```python
# Regex for FR-XXX pattern
requirement_pattern = r'\*\*FR-(\d{3})\*\*:?\s*(.+)'
# Parse spec.md to build requirement list
requirements = re.findall(requirement_pattern, spec_content)
```

**Task linking strategies**:
- Explicit references: "Task addresses FR-001"
- Implicit mapping: Parse task descriptions
- Comment annotations: `# Implements: FR-001`
- Task ID naming: `T001-FR-001`

**Gap reporting formats**:
```markdown
## Coverage Gaps
- FR-007: No tasks found
- FR-013: No tasks found
- Edge Case "database failure": No test task
```

### Decision
Use regex-based extraction for requirements and explicit task references, generating markdown gap reports.

### Rationale
- Simple: Regex handles current format well
- Traceable: Explicit links prevent ambiguity
- Actionable: Clear gap reports drive fixes
- Compatible: Works with existing markdown

### Alternatives Considered
1. **AST parsing**: Overkill for markdown
2. **ML-based matching**: Too complex, unreliable
3. **Manual mapping**: Error-prone, not scalable

## Summary of Decisions

1. **Coverage Warnings**: pytest_terminal_summary hook with coverage API
2. **Error Validation**: pytest_exception_interact with regex patterns
3. **TDD Compliance**: GitPython commit history analysis
4. **Critical Modules**: .coveragerc configuration patterns
5. **Task Mapping**: Regex extraction with explicit references

These decisions prioritize simplicity, compatibility with existing tools, and actionable feedback for developers.