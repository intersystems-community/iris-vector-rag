# Quickstart: Test Coverage Enhancement Implementation

This quickstart guide validates the coverage enhancement feature through end-to-end scenarios derived from the user stories in [spec.md](./spec.md).

## Prerequisites

```bash
# Environment setup
source .venv/bin/activate
pip install pytest pytest-cov pytest-asyncio coverage

# IRIS database running
docker-compose up -d
make setup-db

# Baseline coverage measurement
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest tests/ --cov=iris_rag --cov=common --cov-report=term
```

## Test Scenario 1: Overall Coverage Validation

**User Story**: Development team needs at least 60% test coverage across the codebase

```bash
# Run comprehensive test suite with coverage
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest tests/ \
  --cov=iris_rag \
  --cov=common \
  --cov-report=term-missing \
  --cov-report=html \
  --maxfail=5

# Validate: Coverage percentage ≥ 60%
# Expected: Overall coverage meets 60% threshold
# HTML report generated in htmlcov/ directory
```

**Acceptance Criteria**:
- [ ] Overall coverage shows ≥60% in terminal output
- [ ] Detailed module-level metrics displayed
- [ ] HTML report contains uncovered line identification

## Test Scenario 2: Critical Module Coverage Validation

**User Story**: Critical core modules must achieve 80% coverage with prioritization

```bash
# Test critical modules specifically
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest \
  tests/unit/test_configuration_coverage.py \
  tests/unit/test_validation_coverage.py \
  tests/unit/test_pipeline_coverage.py \
  tests/unit/test_services_coverage.py \
  tests/unit/test_storage_coverage.py \
  --cov=iris_rag.config \
  --cov=iris_rag.validation \
  --cov=iris_rag.pipelines \
  --cov=iris_rag.services \
  --cov=iris_rag.storage \
  --cov-report=term

# Validate: Each critical module ≥ 80%
```

**Acceptance Criteria**:
- [ ] iris_rag.config coverage ≥80%
- [ ] iris_rag.validation coverage ≥80%
- [ ] iris_rag.pipelines coverage ≥80%
- [ ] iris_rag.services coverage ≥80%
- [ ] iris_rag.storage coverage ≥80%

## Test Scenario 3: Developer Feedback Loop

**User Story**: Developers receive immediate feedback on coverage impact

```bash
# Simulate development workflow
# 1. Make code change in iris_rag/config/manager.py
echo "# test change" >> iris_rag/config/manager.py

# 2. Run targeted coverage test
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest \
  tests/unit/test_configuration_coverage.py \
  --cov=iris_rag.config \
  --cov-report=term-missing

# 3. Revert change
git checkout -- iris_rag/config/manager.py
```

**Acceptance Criteria**:
- [ ] Coverage feedback provided within 30 seconds
- [ ] Uncovered lines clearly identified
- [ ] Coverage delta shown for changes

## Test Scenario 4: CI/CD Integration Validation

**User Story**: CI/CD pipeline enforces coverage requirements automatically

```bash
# Simulate CI/CD coverage enforcement
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest tests/ \
  --cov=iris_rag \
  --cov=common \
  --cov-report=xml \
  --cov-fail-under=60

# Expected: Process exits with code 0 if ≥60%, code 2 if <60%
echo "Exit code: $?"
```

**Acceptance Criteria**:
- [ ] Exit code 0 when coverage ≥60%
- [ ] Exit code 2 when coverage <60%
- [ ] XML report generated for CI integration

## Test Scenario 5: Performance Validation

**User Story**: Coverage analysis completes within 5 minutes with acceptable overhead

```bash
# Measure coverage analysis timing
time PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest tests/ \
  --cov=iris_rag \
  --cov=common \
  --cov-report=term

# Compare with baseline test execution
time PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest tests/
```

**Acceptance Criteria**:
- [ ] Coverage analysis completes in ≤300 seconds (5 minutes)
- [ ] Test execution overhead ≤2x baseline time
- [ ] Memory usage remains reasonable

## Test Scenario 6: Legacy Module Handling

**User Story**: Legacy modules receive differentiated coverage targets

```bash
# Test legacy module exemptions
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest \
  tests/unit/test_legacy_modules.py \
  --cov=iris_rag \
  --cov-report=term

# Validate: Legacy modules show reduced targets
```

**Acceptance Criteria**:
- [ ] Legacy modules identified in coverage report
- [ ] Reduced coverage targets applied appropriately
- [ ] Exemption justifications documented

## Test Scenario 7: Coverage Trend Tracking

**User Story**: Monthly coverage improvement tracking for milestone reporting

```bash
# Generate coverage trend data
PYTHONPATH=/Users/tdyar/ws/rag-templates python -c "
from iris_rag.testing.coverage_tracker import CoverageTrend
trend = CoverageTrend()
print(trend.generate_monthly_report())
"
```

**Acceptance Criteria**:
- [ ] Monthly trend data available
- [ ] Coverage delta calculations accurate
- [ ] Milestone achievement tracking functional

## Integration Test Scenario

**User Story**: Complete coverage workflow from development to reporting

```bash
# End-to-end coverage workflow
./scripts/run_complete_coverage_workflow.sh

# Expected workflow:
# 1. Run all tests with coverage
# 2. Generate multiple report formats
# 3. Validate coverage targets
# 4. Update trend tracking
# 5. Generate dashboard report
```

**Acceptance Criteria**:
- [ ] All test categories execute successfully
- [ ] Coverage reports generated in multiple formats
- [ ] Target validation passes for overall and critical modules
- [ ] Trend data updated correctly
- [ ] Dashboard shows current status

## Validation Checklist

### Coverage Targets Met
- [ ] Overall coverage ≥60%
- [ ] Critical modules (config, validation, pipelines, services, storage) ≥80%
- [ ] Legacy modules meet reduced targets with justification

### Performance Requirements
- [ ] Coverage analysis ≤5 minutes
- [ ] Test execution overhead ≤2x baseline
- [ ] Memory usage acceptable

### Quality Requirements
- [ ] No flaky tests introduced
- [ ] Deterministic test results
- [ ] Meaningful coverage (not just line coverage)

### Integration Requirements
- [ ] CI/CD pipeline integration functional
- [ ] Multiple report formats generated
- [ ] Monthly trend tracking operational

## Troubleshooting

### Common Issues

**Coverage below 60%**:
```bash
# Identify gaps
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest tests/ \
  --cov=iris_rag --cov-report=html
# Open htmlcov/index.html to identify uncovered code
```

**Tests timing out**:
```bash
# Run with timeout and reduced parallelism
PYTHONPATH=/Users/tdyar/ws/rag-templates python -m pytest tests/ \
  --timeout=300 --cov=iris_rag
```

**IRIS database connection issues**:
```bash
# Verify database status
docker ps | grep iris
make docker-logs | grep iris
```

This quickstart validates the complete coverage enhancement feature through systematic scenario testing that mirrors the user stories from the specification.