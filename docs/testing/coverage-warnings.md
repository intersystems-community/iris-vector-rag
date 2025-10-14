# Coverage Warning System

The coverage warning system helps maintain code quality by alerting developers when test coverage falls below configured thresholds. Unlike coverage failures, these warnings don't fail the test suite but provide visibility into areas needing attention.

## Overview

The system uses a pytest plugin (`tests/plugins/coverage_warnings.py`) that:
- Monitors code coverage during test execution
- Warns when modules fall below threshold (60% default)
- Enforces higher standards for critical modules (80%)
- Displays warnings after tests complete without failing the build

## Configuration

### Setting Critical Module Patterns

Critical modules require 80% coverage instead of the default 60%. Configure these in `.coveragerc`:

```ini
[coverage:critical_modules]
patterns =
    iris_rag/pipelines/
    iris_rag/storage/
    iris_rag/validation/
    my_app/core/
    my_app/security/
```

Patterns can be:
- Directory paths (ending with `/`) - matches all files in directory
- Glob patterns - for specific file matching
- Module paths - for exact matches

### Threshold Configuration

Currently, thresholds are hardcoded in the plugin:
- Standard modules: 60%
- Critical modules: 80%

To modify thresholds, edit `tests/plugins/coverage_warnings.py`:

```python
# In collect_coverage_warnings function
threshold = 80.0 if is_critical else 60.0  # Modify these values
```

## Usage

### Running Tests with Coverage Warnings

The plugin activates automatically when running tests with coverage:

```bash
# Standard pytest with coverage
pytest --cov=iris_rag --cov=common

# The warnings appear after test results
```

### Example Output

```
========================= Coverage Warnings =========================
WARNING: Coverage below threshold - iris_rag/utils/helpers.py: 45.2% < 60.0% [WARNING]
WARNING: Coverage below threshold - iris_rag/pipelines/basic.py: 72.1% < 80.0% [CRITICAL]
WARNING: Coverage below threshold - common/database.py: 58.9% < 60.0% [WARNING]

2 critical modules below 80% coverage!
Total modules with low coverage: 3
```

### Interpreting Warnings

- **[WARNING]**: Standard module below 60% threshold
- **[CRITICAL]**: Critical module below 80% threshold
- Modules are sorted with critical warnings first
- Shows current coverage vs required threshold

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run tests with coverage
  run: |
    pytest --cov=myapp --cov-report=term-missing
    # Warnings will appear in the output but won't fail the build
```

### Capturing Warnings Programmatically

```python
# In a custom script
import subprocess
result = subprocess.run(
    ["pytest", "--cov=myapp"],
    capture_output=True,
    text=True
)
# Parse result.stdout for "Coverage Warnings" section
```

## Troubleshooting

### No Warnings Displayed

1. Ensure coverage is enabled: `pytest --cov=your_package`
2. Check if `.coverage` file exists after test run
3. Verify plugin is registered in `pytest.ini`

### Incorrect Module Detection

1. Check patterns in `.coveragerc` match your module structure
2. Ensure test files are excluded from coverage
3. Verify Python path is set correctly

### Performance Issues

For large codebases:
1. Use `--cov-report=` (empty) to skip report generation
2. Add `omit` patterns in `.coveragerc` for third-party code
3. Consider running coverage warnings only in CI

## Best Practices

1. **Start with Lower Thresholds**: Begin with achievable goals and increase over time
2. **Focus on Critical Paths**: Prioritize coverage for business-critical modules
3. **Regular Reviews**: Review warnings weekly and address the most critical first
4. **Team Agreement**: Establish team consensus on thresholds and critical modules
5. **Gradual Improvement**: Use warnings to track progress without blocking development

## Example: Adding to Existing Project

1. Copy the plugin:
   ```bash
   cp tests/plugins/coverage_warnings.py your_project/tests/plugins/
   ```

2. Update `.coveragerc`:
   ```ini
   [coverage:critical_modules]
   patterns =
       your_app/api/
       your_app/auth/
   ```

3. Register in `pytest.ini`:
   ```ini
   plugins =
       tests.plugins.coverage_warnings
   ```

4. Run tests and review warnings:
   ```bash
   pytest --cov=your_app
   ```

## FAQ

**Q: Why warnings instead of failures?**
A: Warnings provide visibility without blocking development, allowing gradual improvement.

**Q: Can I make warnings fail the build?**
A: Yes, modify the plugin to set `exitstatus=1` in `pytest_terminal_summary` when warnings exist.

**Q: How do I suppress warnings temporarily?**
A: Set verbose level: `pytest -q` or modify the plugin to check a config option.

**Q: Can I set per-file thresholds?**
A: Not currently, but the plugin can be extended to read per-file configuration.