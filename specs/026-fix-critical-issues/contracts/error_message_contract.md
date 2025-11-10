# Contract: Error Message Validation System

**Contract ID**: ERR-001
**Version**: 1.0.0
**Component**: tests/plugins/error_message_validator.py

## Purpose

Define the behavior of the pytest plugin that validates test error messages follow a three-part structure (what failed, why it failed, suggested action) to ensure actionable feedback.

## Requirements

This contract implements:
- **FR-006**: Error messages must have three components
- **FR-007**: Include relevant context (test name, assertions, values)
- **FR-008**: Consistent format across test types
- **FR-009**: Validate message quality

## Interface Definition

### Plugin Registration

```python
# pytest_plugins in conftest.py
pytest_plugins = ["tests.plugins.error_message_validator"]
```

### Hook Implementation

```python
def pytest_exception_interact(node, call, report):
    """Validate error message structure when test fails."""
    # 1. Extract error message
    # 2. Validate three-part structure
    # 3. Log validation result
    # 4. Suggest improvements if needed
```

### Configuration

```yaml
# pyproject.toml or pytest.ini
[tool.pytest.error_validation]
enabled = true
strict = false  # Don't fail on invalid messages
patterns = [
    # What patterns
    "failed", "error", "assertion", "exception",
    # Why patterns
    "expected .* but got", "because", "due to", "since",
    # Action patterns
    "check", "verify", "ensure", "fix", "update"
]
```

## Behavior Specifications

### REQ-1: Error Message Extraction

**Given** a test fails with an exception
**When** pytest_exception_interact hook is called
**Then** extract the complete error message

**Validation**:
```python
def extract_error_message(report):
    if hasattr(report, 'longrepr'):
        return str(report.longrepr)
    return str(report.exception)
```

### REQ-2: Three-Part Structure Detection

**Given** an error message
**When** validating structure
**Then** identify what/why/action components

**Good Example**:
```
AssertionError: Pipeline configuration validation failed.
Expected 'llm_model' in config but got None.
Check your config.yaml includes required 'llm_model' field.
```

**Components**:
- What: "Pipeline configuration validation failed"
- Why: "Expected 'llm_model' in config but got None"
- Action: "Check your config.yaml includes required 'llm_model' field"

### REQ-3: Validation Rules

**Given** error message components
**When** applying validation
**Then** check each part meets criteria

```python
def validate_message(message):
    validation = ErrorMessageValidation(test_name=node.nodeid)

    # What: Must describe the failure
    validation.has_what = any(pattern in message.lower()
                            for pattern in ['failed', 'error'])

    # Why: Must explain cause
    validation.has_why = any(pattern in message.lower()
                           for pattern in ['expected', 'because', 'but got'])

    # Action: Must suggest next step
    validation.has_action = any(pattern in message.lower()
                              for pattern in ['check', 'verify', 'ensure'])

    return validation
```

### REQ-4: Context Inclusion

**Given** a test failure
**When** validating context
**Then** ensure key information is present

**Required Context**:
- Test name (from node.nodeid)
- Assertion details (expected vs actual)
- Relevant variables/state

**Example with Context**:
```
test_vector_store.py::test_similarity_search FAILED
AssertionError: Vector search returned wrong results.
Expected 3 documents with similarity > 0.8, but got 0.
Check vector embedding dimensions match (expected: 384, actual: 768).
Current index: 'test_index', documents: 0, vectors: 0
```

### REQ-5: Improvement Suggestions

**Given** an invalid error message
**When** validation fails
**Then** provide specific suggestions

```python
def generate_suggestions(validation):
    suggestions = []

    if not validation.has_what:
        suggestions.append("Add clear statement of what failed")

    if not validation.has_why:
        suggestions.append("Explain why the test failed with expected vs actual")

    if not validation.has_action:
        suggestions.append("Suggest what to check or how to fix")

    return suggestions
```

**Output Example**:
```
⚠️  Error Message Validation Failed: test_pipeline.py::test_config
    ❌ Missing: Why it failed
    ❌ Missing: Suggested action

    Current message: "AssertionError"

    Suggestions:
    - Explain why the test failed with expected vs actual
    - Suggest what to check or how to fix

    Better example:
    "Configuration validation failed.
     Expected 'api_key' in environment but not found.
     Set OPENAI_API_KEY environment variable."
```

## Error Handling

### ERR-1: Malformed Exceptions

**Given** exception with no message
**When** extracting error text
**Then** use exception type and traceback

### ERR-2: Non-Assertion Errors

**Given** non-assertion error (e.g., ImportError)
**When** validating
**Then** apply relaxed rules appropriate to error type

### ERR-3: Custom Exception Classes

**Given** custom exception with attributes
**When** extracting message
**Then** include all relevant attributes in validation

## Integration Points

### With pytest Assertions

- Enhance assert statements with messages
- Suggest pytest.fail() with descriptive messages
- Compatible with pytest-assume

### With CI/CD

- Validation results in test metadata
- Summary report of message quality
- Trends over time

## Contract Tests

```python
# tests/contract/test_error_message_contract.py

def test_ERR001_three_part_detection():
    """Verify three-part structure is detected correctly."""
    good_message = """
    Test failed: Database connection error.
    Expected connection to localhost:5432 but got timeout.
    Check PostgreSQL is running and accepting connections.
    """
    validation = validate_message(good_message)
    assert validation.has_what
    assert validation.has_why
    assert validation.has_action

def test_ERR001_missing_components():
    """Verify missing components are identified."""
    bad_message = "AssertionError"
    validation = validate_message(bad_message)
    assert not validation.has_why
    assert not validation.has_action
    assert len(validation.suggestions) == 2

def test_ERR001_context_extraction():
    """Verify context is properly extracted."""
    # Create failing test
    # Capture error message
    # Assert context includes test name, values

def test_ERR001_suggestion_generation():
    """Verify helpful suggestions are generated."""
    # Test with various invalid messages
    # Assert suggestions are specific and actionable

def test_ERR001_non_failing_behavior():
    """Verify validator doesn't fail tests."""
    # Run with invalid messages
    # Assert tests still fail for original reason
```

## Performance Requirements

- Validation must add < 10ms per test failure
- Memory overhead < 1MB for 1000 test failures
- No impact on passing tests

## Security Considerations

- No code execution in message analysis
- Sanitize output to prevent injection
- No sensitive data in suggestions

## Configuration Examples

### Strict Mode (CI/CD)
```yaml
[tool.pytest.error_validation]
enabled = true
strict = true  # Fail tests with poor messages
min_components = 3
```

### Development Mode
```yaml
[tool.pytest.error_validation]
enabled = true
strict = false
show_examples = true
log_to_file = "error_validation.log"
```

### Custom Patterns
```yaml
[tool.pytest.error_validation]
what_patterns = ["failed", "error", "broke", "issue"]
why_patterns = ["because", "due to", "caused by", "expected .* got"]
action_patterns = ["please", "try", "consider", "should", "must"]
```

## Future Extensions

1. **ML-Based Analysis**: Use NLP for semantic validation
2. **Team Standards**: Custom rules per team
3. **Auto-Fix**: Suggest message rewrites
4. **IDE Integration**: Real-time validation while writing tests