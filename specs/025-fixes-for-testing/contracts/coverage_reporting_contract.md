# Contract: Coverage Reporting Requirements

**Feature**: 025-fixes-for-testing
**Date**: 2025-10-03

## Requirements

### REQ-1: Coverage Configuration
- `.coveragerc` must exclude test files from coverage calculations
- Source includes: `iris_rag`, `common`
- Omit patterns: `*/tests/*`, `*/test_*`, `*/__pycache__/*`

### REQ-2: Coverage Targets
- Overall coverage: >= 60%
- Critical modules (pipelines, storage, validation): >= 80%
- HTML reports generated in `htmlcov/`

### REQ-3: Coverage Reporting
- Show missing line numbers
- Report precision: 1 decimal place
- Include branch coverage when applicable

## Contract Test

```python
def test_coverage_configuration():
    """Validate .coveragerc excludes test files."""
    import configparser
    config = configparser.ConfigParser()
    config.read(".coveragerc")

    omit = config.get("coverage:run", "omit")
    assert "*/tests/*" in omit
    assert "*/test_*" in omit
```

## Success Criteria
- Coverage reports generated successfully
- Test files not included in coverage calculations
- Coverage targets enforced (60% overall, 80% critical)
