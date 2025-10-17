# Contract: TDD Compliance Validation System

**Contract ID**: TDD-001
**Version**: 1.0.0
**Component**: scripts/validate_tdd_compliance.py

## Purpose

Define the behavior of the TDD compliance validator that verifies contract tests existed in a failing state before implementation, ensuring adherence to Test-Driven Development principles.

## Requirements

This contract implements:
- **FR-010**: Validate contract tests failed before implementation
- **FR-011**: Check git history for initial test state
- **FR-012**: Report TDD violations with details
- **FR-013**: Run as part of development workflow

## Interface Definition

### Command Line Interface

```bash
# Validate all contract tests
python scripts/validate_tdd_compliance.py

# Validate specific test file
python scripts/validate_tdd_compliance.py tests/contract/test_foo_contract.py

# CI mode with strict checking
python scripts/validate_tdd_compliance.py --ci --fail-on-violations

# Generate detailed report
python scripts/validate_tdd_compliance.py --report tdd_compliance.json
```

### Python API

```python
from scripts.validate_tdd_compliance import TDDValidator

validator = TDDValidator(repo_path='.')
results = validator.validate_all_contract_tests()
report = validator.generate_report(results)
```

### Configuration

```yaml
# .tdd-validation.yml
contract_test_patterns:
  - "tests/contract/test_*_contract.py"
  - "tests/**/test_*_contract.py"

implementation_patterns:
  - "src/**/*.py"
  - "iris_rag/**/*.py"

exclude_patterns:
  - "tests/helpers/**"
  - "**/conftest.py"

validation_rules:
  max_time_between_test_and_impl: 86400  # 24 hours in seconds
  require_failing_commit: true
  check_ci_logs: false  # Future feature
```

## Behavior Specifications

### REQ-1: Contract Test Discovery

**Given** a repository with contract tests
**When** scanning for test files
**Then** identify all matching contract test patterns

**Validation**:
```python
def find_contract_tests(repo_path):
    tests = []
    for pattern in config['contract_test_patterns']:
        tests.extend(glob.glob(pattern, recursive=True))
    return tests

# Expected: Find all test_*_contract.py files
assert 'tests/contract/test_coverage_contract.py' in tests
```

### REQ-2: Git History Analysis

**Given** a contract test file
**When** analyzing git history
**Then** find initial commit and test state

**Process**:
```python
def analyze_test_history(test_file):
    repo = git.Repo('.')

    # Find when test was added
    commits = list(repo.iter_commits(paths=test_file))
    initial_commit = commits[-1]  # Oldest commit

    # Check if test failed initially
    repo.git.checkout(initial_commit.hexsha)
    result = run_pytest(test_file)

    return ContractTestState(
        test_file=test_file,
        initial_commit=initial_commit.hexsha,
        initial_state='FAIL' if result.failed else 'PASS'
    )
```

### REQ-3: Implementation Detection

**Given** a contract test
**When** searching for implementation
**Then** find corresponding implementation files

**Matching Strategy**:
```python
def find_implementation(test_file):
    # Extract component name from test
    # test_coverage_contract.py -> coverage
    component = extract_component_name(test_file)

    # Search for implementation files
    impl_files = []
    for pattern in config['implementation_patterns']:
        candidates = glob.glob(f"**/*{component}*.py")
        impl_files.extend(filter_implementations(candidates))

    return impl_files
```

### REQ-4: TDD Compliance Check

**Given** test and implementation history
**When** validating TDD compliance
**Then** verify test failed before implementation

**Validation Logic**:
```python
def check_tdd_compliance(test_state, impl_commits):
    # Test must exist and fail before implementation
    if test_state.initial_state != 'FAIL':
        return ContractTestState(
            compliance_status='VIOLATION',
            violation_reason='Test never failed initially'
        )

    # Implementation must come after test
    test_timestamp = get_commit_timestamp(test_state.initial_commit)
    impl_timestamp = min(get_commit_timestamp(c) for c in impl_commits)

    if impl_timestamp <= test_timestamp:
        return ContractTestState(
            compliance_status='VIOLATION',
            violation_reason='Implementation exists before test'
        )

    return ContractTestState(compliance_status='COMPLIANT')
```

### REQ-5: Violation Reporting

**Given** TDD violations found
**When** generating report
**Then** provide actionable details

**Report Format**:
```markdown
# TDD Compliance Report

## Summary
- Total Contract Tests: 10
- Compliant: 7 (70%)
- Violations: 3 (30%)

## Violations

### ❌ tests/contract/test_coverage_contract.py
- **Issue**: Test never failed initially
- **Test Added**: commit abc123 (2024-01-15)
- **First Pass**: commit abc123 (same commit)
- **Recommendation**: This suggests test and implementation were written together

### ❌ tests/contract/test_error_contract.py
- **Issue**: Implementation predates test
- **Implementation**: commit def456 (2024-01-10)
- **Test Added**: commit ghi789 (2024-01-12)
- **Recommendation**: Tests should be written before implementation
```

## Error Handling

### ERR-1: Git Repository Issues

**Given** invalid git repository
**When** running validation
**Then** provide clear error message

```
Error: Not a git repository. TDD validation requires git history.
Run 'git init' or check you're in the correct directory.
```

### ERR-2: Test Execution Failures

**Given** test fails to run in historical commit
**When** checking initial state
**Then** mark as UNKNOWN with explanation

```python
try:
    result = run_pytest(test_file)
except Exception as e:
    return ContractTestState(
        initial_state='UNKNOWN',
        violation_reason=f'Could not run test: {str(e)}'
    )
```

### ERR-3: Missing Dependencies

**Given** historical commit missing dependencies
**When** running tests
**Then** attempt minimal test execution

```python
# Run with minimal imports
pytest_args = ['-p', 'no:warnings', '--tb=no', '--no-header']
```

## Integration Points

### With CI/CD

```yaml
# .github/workflows/tdd-check.yml
- name: Check TDD Compliance
  run: |
    python scripts/validate_tdd_compliance.py --ci
  continue-on-error: true  # Warning only initially

- name: Comment on PR
  if: failure()
  uses: actions/github-script@v6
  with:
    script: |
      github.issues.createComment({
        issue_number: context.issue.number,
        body: 'TDD violations detected. See workflow logs.'
      })
```

### With Pre-commit

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: tdd-compliance
      name: Check TDD Compliance
      entry: python scripts/validate_tdd_compliance.py
      language: system
      files: 'tests/contract/.*\.py$'
```

## Contract Tests

```python
# tests/contract/test_tdd_validation_contract.py

def test_TDD001_find_contract_tests():
    """Verify contract test discovery works."""
    validator = TDDValidator('.')
    tests = validator.find_contract_tests()
    assert len(tests) > 0
    assert all('contract' in t for t in tests)

def test_TDD001_detect_violation_never_failed():
    """Verify detection of tests that never failed."""
    # Create test that passes initially
    # Run validator
    # Assert violation detected

def test_TDD001_detect_violation_impl_first():
    """Verify detection of implementation-first anti-pattern."""
    # Create implementation then test
    # Run validator
    # Assert violation detected

def test_TDD001_compliant_workflow():
    """Verify compliant TDD workflow passes."""
    # Create failing test
    # Commit
    # Add implementation
    # Run validator
    # Assert compliant

def test_TDD001_report_generation():
    """Verify report contains actionable information."""
    # Run validation
    # Generate report
    # Assert contains all required sections
```

## Performance Requirements

- Full repository scan < 10 seconds for 100 contract tests
- Git history analysis < 100ms per test file
- Memory usage < 100MB for large repositories

## Security Considerations

- Read-only git operations
- No code execution beyond pytest
- Sanitize file paths in reports
- No network access required

## Output Examples

### Console Output (Default)
```
Validating TDD Compliance...
Found 15 contract tests

✅ tests/contract/test_api_contract.py - COMPLIANT
✅ tests/contract/test_auth_contract.py - COMPLIANT
❌ tests/contract/test_cache_contract.py - VIOLATION: Never failed
⚠️  tests/contract/test_db_contract.py - UNKNOWN: Historical test failed

Summary: 12/15 compliant (80%)
```

### JSON Report
```json
{
  "summary": {
    "total_tests": 15,
    "compliant": 12,
    "violations": 2,
    "unknown": 1,
    "compliance_rate": 0.8
  },
  "violations": [
    {
      "test_file": "tests/contract/test_cache_contract.py",
      "violation_type": "never_failed",
      "initial_commit": "abc123",
      "details": "Test passed on first commit"
    }
  ]
}
```

## Future Extensions

1. **CI Log Analysis**: Check GitHub Actions logs for test results
2. **Incremental Checking**: Only validate changed tests
3. **Auto-Fix**: Generate failing test stubs
4. **IDE Integration**: Real-time TDD compliance feedback