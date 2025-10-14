# TDD Compliance Workflow

This guide explains how to follow Test-Driven Development (TDD) practices with contract tests and use the compliance validation tools to ensure proper TDD workflow.

## TDD Principles for Contract Tests

Contract tests define the expected behavior of your system before implementation. They must:

1. Be written before the implementation
2. Fail initially (proving they test something)
3. Pass only after correct implementation
4. Remain as regression tests

## The TDD Workflow

### Step 1: Write Failing Contract Tests

Create contract tests that define expected behavior:

```python
# tests/contract/test_payment_contract.py
import pytest

class TestPaymentContract:
    @pytest.mark.contract
    def test_payment_processing_contract(self):
        """Contract: Payment processor should validate and process payments."""
        from payment_system import PaymentProcessor  # Will fail - not implemented yet

        processor = PaymentProcessor()
        result = processor.process_payment(
            amount=100.00,
            currency="USD",
            card_number="4111111111111111"
        )

        assert result.success is True
        assert result.transaction_id is not None
        assert result.status == "completed"
```

### Step 2: Commit Failing Tests

This is crucial for TDD compliance:

```bash
# Run tests to verify they fail
pytest tests/contract/test_payment_contract.py  # Should fail

# Commit the failing tests
git add tests/contract/test_payment_contract.py
git commit -m "Add failing contract tests for payment processing"
```

### Step 3: Implement Features

Now implement the code to make tests pass:

```python
# payment_system.py
class PaymentProcessor:
    def process_payment(self, amount, currency, card_number):
        # Implementation here
        ...
```

### Step 4: Verify Tests Pass

```bash
pytest tests/contract/test_payment_contract.py  # Should now pass
git add payment_system.py
git commit -m "Implement payment processing to satisfy contracts"
```

## Using the TDD Compliance Validator

The `validate_tdd_compliance.py` script ensures contract tests followed proper TDD workflow.

### Running Validation

```bash
# Basic validation
python scripts/validate_tdd_compliance.py

# For CI/CD (fails on violations)
python scripts/validate_tdd_compliance.py --fail-on-violations

# JSON output for parsing
python scripts/validate_tdd_compliance.py --json
```

### Example Output

```
======================================================================
TDD Compliance Report
======================================================================

Total Contract Tests: 12
Compliant Tests: 10
VIOLATIONS FOUND: 2

======================================================================
VIOLATIONS DETAIL
======================================================================

❌ tests/contract/test_auth_contract.py
   Violation: never_failed
   Details: Contract test was already passing at introduction (abc123)

❌ tests/contract/test_cache_contract.py
   Violation: never_failed
   Details: Contract test was already passing at introduction (def456)

======================================================================
COMPLIANT TESTS: 10
======================================================================

✅ tests/contract/test_payment_contract.py
   Initial State: failing

✅ tests/contract/test_user_contract.py
   Initial State: error

... and 8 more compliant tests

======================================================================
Compliance Rate: 83.3%
======================================================================
```

## Common Violations and Fixes

### Violation: "never_failed"

**Problem**: Test was passing when first committed.

**Fix**:
1. Reset to before implementation
2. Verify test fails without implementation
3. Re-commit the failing test
4. Then add implementation

### Violation: "test_not_found"

**Problem**: Can't find test in git history.

**Fix**:
1. Ensure test file is tracked by git
2. Check file path and name
3. Verify git history is available

## CI/CD Integration

### GitHub Actions

```yaml
name: TDD Compliance Check

on:
  pull_request:
    paths:
      - 'tests/contract/**'

jobs:
  tdd-compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history needed

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install GitPython

      - name: Check TDD compliance
        run: |
          python scripts/validate_tdd_compliance.py --fail-on-violations
```

### Pre-commit Hook

Use the provided `.pre-commit-config.yaml`:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Now TDD validation runs on contract test commits
```

## Best Practices

### 1. Use Descriptive Test Names

```python
@pytest.mark.contract
def test_user_registration_validates_email_format(self):
    """Contract: User registration must validate email format."""
    # Not just "test_registration"
```

### 2. Write Complete Contracts

Include all expected behavior:

```python
@pytest.mark.contract
def test_api_rate_limiting_contract(self):
    """Contract: API should enforce rate limits."""
    # Test the limit
    for i in range(100):
        response = make_request()
        assert response.status_code == 200

    # Test exceeding limit
    response = make_request()
    assert response.status_code == 429
    assert "Retry-After" in response.headers
```

### 3. Document Implementation Links

```python
@pytest.mark.contract
@pytest.mark.implementation_commit("abc123def")
def test_feature_contract(self):
    """Contract implemented in commit abc123def."""
    pass
```

### 4. Group Related Contracts

```python
class TestUserContractSuite:
    """All contracts related to user management."""

    @pytest.mark.contract
    def test_user_creation_contract(self):
        pass

    @pytest.mark.contract
    def test_user_authentication_contract(self):
        pass
```

## Workflow Example: New Feature

1. **Receive requirement**: "Add email notification on order completion"

2. **Write contract test**:
   ```python
   @pytest.mark.contract
   def test_order_completion_sends_email(self):
       order = create_test_order()
       order.complete()

       assert email_service.send_called
       assert email_service.last_email.to == order.customer.email
       assert "order completed" in email_service.last_email.subject
   ```

3. **Verify test fails**:
   ```bash
   pytest tests/contract/test_order_contract.py::test_order_completion_sends_email
   # ModuleNotFoundError or AttributeError - Good!
   ```

4. **Commit failing test**:
   ```bash
   git add tests/contract/test_order_contract.py
   git commit -m "Contract: Orders should send completion emails"
   ```

5. **Implement feature**:
   - Add email service integration
   - Modify order.complete() method
   - Test manually if needed

6. **Verify contract passes**:
   ```bash
   pytest tests/contract/test_order_contract.py::test_order_completion_sends_email
   # Test passes!
   ```

7. **Commit implementation**:
   ```bash
   git add order_system.py email_integration.py
   git commit -m "Send email notifications on order completion"
   ```

8. **Validate TDD compliance**:
   ```bash
   python scripts/validate_tdd_compliance.py
   # Should show your test as compliant
   ```

## FAQ

**Q: What if I need to modify a contract test?**
A: Modify it like any test, but ensure it still tests the contract, not implementation details.

**Q: Can I skip TDD for bug fixes?**
A: Write a failing test that reproduces the bug first, then fix it.

**Q: What about integration tests?**
A: TDD principles apply - write failing integration contracts before implementing integrations.

**Q: How do I handle external dependencies?**
A: Mock them in contract tests, but write separate integration contracts for real services.

## Summary

TDD compliance ensures:
- Requirements are testable and clear
- Tests actually test something meaningful
- Implementation matches requirements
- Changes don't break contracts

The validation tools help enforce these practices without being overly restrictive.