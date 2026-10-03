"""Pytest plugins for testing framework compliance.

Plugins load from the `pytest_plugins` list in tests/conftest.py. This file used
to carry its own `pytest_plugins` list, which pytest never reads — a package
__init__ is not a conftest — so naming a plugin here does nothing.
"""
