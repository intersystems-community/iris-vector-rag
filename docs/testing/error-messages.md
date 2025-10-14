# Error Message Best Practices

Well-crafted error messages are crucial for maintainable test suites. This guide explains the three-part structure for error messages and how to use the error message validator plugin.

## The Three-Part Structure

Every test failure message should answer three questions:

1. **WHAT failed?** - Clear identification of the failure
2. **WHY did it fail?** - The specific reason or difference
3. **What ACTION should be taken?** - How to fix or investigate

## Examples

### ✅ Good Error Messages

```python
def test_user_authentication():
    response = auth_service.login("testuser", "wrongpass")
    assert response.status_code == 200, (
        f"Test test_user_authentication failed: Authentication error.\n"
        f"Expected status 200 for user 'testuser' but got {response.status_code} (Unauthorized).\n"
        f"Check credentials in test_data.json and ensure auth service is running."
    )

def test_data_processing():
    result = process_data(input_data)
    assert len(result) == expected_count, (
        f"Test test_data_processing failed: Incorrect output size.\n"
        f"Expected {expected_count} records after processing but got {len(result)}.\n"
        f"Verify input data format and check filter conditions in process_data()."
    )

def test_api_response_time():
    elapsed = measure_api_call()
    assert elapsed < 2.0, (
        f"Test test_api_response_time failed: Performance degradation.\n"
        f"API call took {elapsed:.2f}s, exceeding 2.0s threshold.\n"
        f"Check database queries, enable caching, or investigate network latency."
    )
```

### ❌ Bad Error Messages

```python
# Too vague
assert result, "Test failed"

# Missing action
assert value == 42, f"Expected 42 but got {value}"

# No context
assert False, "Something went wrong"

# Missing why
assert response.ok, "Bad response"
```

## Using the Error Message Validator

The validator plugin automatically checks error messages during test failures.

### How It Works

1. Intercepts test failures via `pytest_exception_interact` hook
2. Analyzes error message structure
3. Provides feedback if message lacks required parts
4. Suggests improvements without failing tests

### Example Validator Output

```
════════════════════════════════════════════════════════════════
ERROR MESSAGE VALIDATION WARNING

Test: test_connection_timeout
Original Error: Connection failed

Error message needs improvement:
  - Add what failed (e.g., 'Test X failed: ...')
  - Explain why it failed (e.g., 'Expected X but got Y')
  - Add what to do to fix it (e.g., 'Check that...' or 'Ensure...')

Example of a good error message:
  "Test test_user_login failed: Authentication error.
   Expected successful login for user 'testuser' but got 401 Unauthorized.
   Check credentials in test_data.json and ensure test API is running."
════════════════════════════════════════════════════════════════
```

### Disabling Validation

For tests with intentionally simple messages:

```python
@pytest.mark.good_errors
def test_with_simple_message():
    # Validator skips tests marked with good_errors
    assert True
```

## Writing Better Error Messages

### 1. Start with Test Name

Always begin with which test failed:

```python
f"Test {test_name} failed: {failure_type}."
```

### 2. Include Actual vs Expected

Be specific about the discrepancy:

```python
f"Expected {expected_value} but got {actual_value}"
f"Expected response in {timeout}s but timed out after {elapsed}s"
f"Expected list with {n} items but got {len(actual)}"
```

### 3. Provide Actionable Guidance

Help the reader fix the issue:

- "Check configuration file at {path}"
- "Ensure service X is running on port {port}"
- "Verify test data includes required fields"
- "Investigate recent changes to {module}"

### 4. Include Relevant Context

Add details that aid debugging:

```python
assert user.is_active, (
    f"Test test_user_activation failed: User not active.\n"
    f"User '{user.email}' (ID: {user.id}) is not active after activation.\n"
    f"Check activation email was sent, verify token is valid, "
    f"and ensure activation endpoint processes correctly."
)
```

## Common Patterns

### API/HTTP Tests

```python
assert response.status_code == expected_status, (
    f"Test {test_name} failed: Unexpected HTTP status.\n"
    f"Expected {expected_status} for {method} {url} but got {response.status_code}.\n"
    f"Check API endpoint configuration, authentication, and request payload."
)
```

### Database Tests

```python
assert record is not None, (
    f"Test {test_name} failed: Record not found.\n"
    f"No record with ID {record_id} in {table_name} table.\n"
    f"Verify test data setup, check for deletion in other tests, "
    f"and ensure transaction isolation."
)
```

### Performance Tests

```python
assert metric < threshold, (
    f"Test {test_name} failed: Performance threshold exceeded.\n"
    f"{metric_name} was {metric:.2f} but threshold is {threshold:.2f}.\n"
    f"Profile the operation, check for N+1 queries, review recent changes, "
    f"or consider adjusting threshold if legitimate."
)
```

### Validation Tests

```python
assert validation_result.is_valid, (
    f"Test {test_name} failed: Validation error.\n"
    f"Validation failed with errors: {validation_result.errors}.\n"
    f"Review input data against schema, check field requirements, "
    f"and verify validation rules match specifications."
)
```

## Configuration

### Custom Validation Patterns

```python
from tests.plugins.error_message_validator import configure_validation

configure_validation({
    "what_pattern": r"(?:test.*failed|error|exception):",
    "why_pattern": r"(?:expected|got|but|because)",
    "action_pattern": r"(?:check|verify|ensure|review)"
})
```

### Integration with CI

The validator provides warnings without failing builds, making it safe for gradual adoption:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest -v
    # Error message warnings appear but don't affect exit code
```

## Tips for Success

1. **Be Specific**: Include actual values, not just "value mismatch"
2. **Think Like a Debugger**: What would you want to know when this fails?
3. **Avoid Technical Jargon**: Write for developers who might be new to the codebase
4. **Test Your Messages**: Intentionally break tests to see if messages are helpful
5. **Keep It Concise**: Aim for 2-4 lines total
6. **Use Formatting**: Line breaks and f-strings improve readability

## Gradual Adoption

1. Start with new tests following the pattern
2. Update existing tests when they fail
3. Run error validator to identify improvement areas
4. Share good examples in code reviews
5. Celebrate improvements in test maintainability

Remember: Good error messages save debugging time and reduce frustration. The investment in writing them pays off quickly.