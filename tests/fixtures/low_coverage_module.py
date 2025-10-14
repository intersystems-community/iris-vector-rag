"""Module with intentionally low coverage for testing coverage warnings.

This module is designed to have multiple branches and functions
where only a small portion is covered by tests.
"""


def barely_tested_function(x):
    """Function with multiple branches but minimal test coverage."""
    if x == 1:
        return 1  # This branch is tested
    elif x == 2:
        return 4  # Not tested
    elif x == 3:
        return 9  # Not tested
    elif x == 4:
        return 16  # Not tested
    else:
        return x * x  # Not tested


def completely_untested_function():
    """This function is never called in tests."""
    result = 0
    for i in range(10):
        result += i * i
    return result


class UntestedClass:
    """This class is never instantiated in tests."""

    def __init__(self):
        self.value = 42

    def untested_method(self):
        """Never called."""
        return self.value * 2

    def another_untested_method(self, x, y):
        """Also never called."""
        if x > y:
            return x - y
        else:
            return y - x


# This ensures we have ~25% coverage when only barely_tested_function(1) is called