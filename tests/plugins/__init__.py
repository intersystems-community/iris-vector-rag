"""Pytest plugins for testing framework compliance."""

# Plugin modules to be discovered
pytest_plugins = [
    "tests.plugins.coverage_warnings",
    "tests.plugins.error_message_validator",
    "tests.plugins.tdd_compliance",
    "tests.plugins.contract_test_marker",
]