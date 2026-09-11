"""The plugins under tests/plugins/ are actually loaded — or gone.

They were listed in pytest.ini under a `plugins =` key, which pytest has no such
ini option for, and again in tests/plugins/__init__.py, which pytest never reads
because a package __init__ is not a conftest. So all four had never run: their
`pytest_terminal_summary`, `pytest_exception_interact` and report-header hooks
were dead code that looked wired up.

Three of them only add advisory output, and they are loaded from
tests/conftest.py now. The fourth, contract_test_marker, is deliberately gone
rather than loaded: it added `xfail(strict=False)` to every test marked
`contract` and rewrote ImportError/AttributeError failures into skips, which
would have turned most of the contract suite into results nobody reads.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ADVISORY_PLUGINS = [
    "tests.plugins.coverage_warnings",
    "tests.plugins.error_message_validator",
    "tests.plugins.tdd_compliance",
]


@pytest.mark.parametrize("name", ADVISORY_PLUGINS)
def test_the_advisory_plugins_are_registered(pytestconfig, name):
    assert pytestconfig.pluginmanager.hasplugin(name), (
        f"{name} is not loaded, so its hooks never run. Plugins load from the "
        f"`pytest_plugins` list in tests/conftest.py; a `plugins =` key in an "
        f"ini file or a `pytest_plugins` list in tests/plugins/__init__.py does "
        f"nothing."
    )


def test_the_contract_xfail_plugin_is_gone():
    assert importlib.util.find_spec("tests.plugins.contract_test_marker") is None, (
        "tests/plugins/contract_test_marker.py is back. It marks every "
        "`contract` test xfail and converts ImportError/AttributeError failures "
        "into skips, so a broken import in the library reads as an expected "
        "failure. If contract tests need to tolerate unimplemented features, "
        "mark the individual test, do not blanket the suite."
    )


def test_critical_coverage_thresholds_apply_without_a_coveragerc(tmp_path, monkeypatch):
    """The 80% critical threshold has to survive having no .coveragerc.

    load_critical_patterns() read `.coveragerc` relative to the working
    directory and returned an empty list when the file was missing, which put
    its default patterns below an unconditional early return. This repo keeps
    coverage config in pyproject.toml and has no .coveragerc, so every module
    was measured against the plain 60% threshold.
    """
    from tests.plugins.coverage_warnings import load_critical_patterns

    monkeypatch.chdir(tmp_path)
    assert not Path(".coveragerc").exists()

    patterns = load_critical_patterns()

    assert patterns, "no critical patterns without a .coveragerc"
    assert all(p.startswith("iris_vector_rag/") for p in patterns), (
        f"critical patterns name a package that does not exist: {patterns}. "
        f"The distribution is iris_vector_rag; iris_rag was the pre-rename name."
    )
