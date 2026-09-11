"""The pytest config in the repo must be the config pytest actually reads.

`pytest.ini` used to carry a `[tool:pytest]` header. That header is only valid in
`setup.cfg`; in a `pytest.ini` file pytest requires `[pytest]`. Because a
`pytest.ini` is treated as the config file whether or not it holds a readable
section, every option in it was ignored AND it masked the working
`[tool.pytest.ini_options]` block in `pyproject.toml`. The session ran with no
`--strict-markers`, no `filterwarnings = error` and no `asyncio_mode = auto`
while the repo looked like it had all three.

These tests read the live `pytest.Config`, so they fail if the settings stop
reaching pytest again — whatever the file on disk claims.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Markers used by tests in this repo. With --strict-markers an unregistered one
# is an error, so every marker here must be declared in the live config.
REQUIRED_MARKERS = [
    "unit",
    "integration",
    "e2e",
    "slow",
    "contract",
    "requires_docker",
    "requires_database",
    "requires_llm",
    "requires_vector_store",
    "requires_backend_mode",
    "benchmark",
    "security",
    "smoke",
    "mcp",
]


def test_pytest_reads_a_config_file_that_is_not_masking_another(pytestconfig):
    """No config file may be picked up only to be ignored section-by-section."""
    inifile = pytestconfig.inipath

    assert inifile is not None, (
        "pytest found no config file at all, so testpaths, markers and "
        "filterwarnings come from pytest defaults. Declare them in "
        "pyproject.toml under [tool.pytest.ini_options]."
    )
    if inifile.name == "pytest.ini":
        header = inifile.read_text().lstrip().splitlines()[0]
        assert header == "[pytest]", (
            f"pytest.ini opens with {header!r}. A pytest.ini needs a [pytest] "
            f"section; [tool:pytest] is setup.cfg syntax, so pytest reads none "
            f"of this file while still ignoring pyproject.toml."
        )


def test_strict_markers_is_on(pytestconfig):
    """A typo'd marker must fail the run, not silently select nothing."""
    assert pytestconfig.option.strict_markers, (
        "--strict-markers is not in effect, so a misspelled @pytest.mark "
        "registers as an unknown mark and the test it guards never runs "
        "under that selection. Add it to addopts."
    )


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_every_marker_this_repo_uses_is_registered(pytestconfig, marker):
    """--strict-markers turns an undeclared marker into a collection error."""
    declared = {
        line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")
    }

    assert marker in declared, (
        f"tests use @pytest.mark.{marker} but the live config does not declare "
        f"it; with --strict-markers that is a collection error. Add it to the "
        f"markers list in pyproject.toml."
    )


def test_warnings_are_errors_by_default(pytestconfig):
    """A new DeprecationWarning must fail a run rather than scroll past."""
    filters = pytestconfig.getini("filterwarnings")

    assert filters, (
        "filterwarnings is empty, so warnings only print. A dependency "
        "deprecating an API we call cannot fail CI."
    )
    assert filters[0] == "error", (
        f"the first filterwarnings entry is {filters[0]!r}, not 'error'. "
        f"Ignores must come after the blanket error to narrow it, not before."
    )


def test_asyncio_mode_is_auto(pytestconfig):
    """Async tests must run; strict mode skips the ones without a marker."""
    assert pytestconfig.getini("asyncio_mode") == "auto", (
        "asyncio_mode is not 'auto', so every `async def` test without an "
        "explicit @pytest.mark.asyncio is skipped instead of run."
    )


def test_only_one_pytest_config_file_exists():
    """Two config files means one of them is a lie about how tests run."""
    pytest_ini = REPO_ROOT / "pytest.ini"
    pyproject = REPO_ROOT / "pyproject.toml"

    if pytest_ini.exists():
        assert "[tool.pytest.ini_options]" not in pyproject.read_text(), (
            "pytest.ini and [tool.pytest.ini_options] both exist. pytest.ini "
            "wins and pyproject.toml is ignored, so the two drift apart "
            "silently. Keep the settings in one file."
        )
