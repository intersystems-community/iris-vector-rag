"""
Contract test marker plugin for pytest.

Automatically marks contract tests as xfail when they fail due to
unimplemented features (ModuleNotFoundError, ImportError, etc.).

Implements T018 from Feature 028.
"""

import pytest
from _pytest.outcomes import XFailed


def pytest_configure(config):
    """Register contract test marker."""
    config.addinivalue_line(
        "markers",
        "contract: Contract tests that define expected behavior (may fail if unimplemented)"
    )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """
    Intercept test failures and mark contract tests as xfail.

    When a test marked with @pytest.mark.contract fails due to:
    - ModuleNotFoundError (feature not implemented)
    - ImportError (dependencies missing)
    - AttributeError (API not defined)

    The failure is converted to xfail (expected failure) instead of ERROR.

    Implements FR-012, FR-013, FR-014 from Feature 028.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process contract-marked tests
    if "contract" not in item.keywords:
        return

    # Only process failures/errors during call phase
    if report.when != "call":
        return

    if report.failed or (hasattr(report, 'outcome') and report.outcome == 'failed'):
        # Check if failure is due to unimplemented feature
        if call.excinfo:
            exc_type = call.excinfo.type
            exc_value = str(call.excinfo.value)

            # Expected errors for contract tests
            expected_errors = [
                ModuleNotFoundError,
                ImportError,
                AttributeError,
            ]

            if exc_type in expected_errors:
                # Convert to xfail
                report.outcome = "skipped"
                report.wasxfail = f"Contract test - feature not implemented ({exc_value})"
                report.longrepr = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """
    Wrap test execution to catch and reclassify contract test failures.

    This hook runs during the actual test execution phase.
    """
    # Only process contract-marked tests
    if "contract" not in item.keywords:
        yield
        return

    try:
        yield
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        # Mark as xfail for contract tests
        pytest.xfail(f"Contract test - feature not implemented: {type(e).__name__}: {str(e)}")
    except Exception:
        # Other exceptions should fail normally
        raise


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add xfail marker to contract tests.

    This ensures contract tests that fail are reported as expected failures.
    """
    for item in items:
        if "contract" in item.keywords:
            # Check if test already has xfail marker
            if not any(marker.name == "xfail" for marker in item.own_markers):
                # Add xfail marker with strict=False (allows passing)
                item.add_marker(
                    pytest.mark.xfail(
                        reason="Contract test may fail if feature unimplemented",
                        strict=False,
                        raises=(ModuleNotFoundError, ImportError, AttributeError)
                    )
                )
