"""The REST API (iris_vector_rag.api) is removed (feature retire-rest-api, removal half).

ADR 0001 / ticket G1: deprecated in 0.15, deleted in 0.16. The code and its tests
live on at the git tag ``archive/rest-api-v1``; the design pack under
``docs/archived/rest-api/`` stays. Findings AP-01 to AP-24 are closed by deletion.
"""

import importlib.machinery
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TAG = "archive/rest-api-v1"
# Requirements only the REST API needed.
API_ONLY = {
    "fastapi",
    "uvicorn",
    "redis",
    "bcrypt",
    "python-jose",
    "email-validator",
    "python-multipart",
    "websockets",
}
# Places that may still name the removed package: history and design records.
HISTORY = (
    "CHANGELOG.md",
    "RELEASE_NOTES_v0.15.0.md",
    "docs/archived/",
    "docs/adr/",
    "docs/remediation/",
    "docs/reports/",
    "specs/",
    "tests/unit/test_rest_api_removal.py",
)
MENTION = re.compile(r"iris_vector_rag[./]api\b|tests_api|tests/integration/api")


def _name(requirement: str) -> str:
    return re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].lower()


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_api_package_does_not_import():
    """Looked up on the checkout's own package path: an editable install of another
    checkout can map ``iris_vector_rag.api`` to a tree that still has it."""
    import iris_vector_rag

    spec = importlib.machinery.PathFinder.find_spec(
        "iris_vector_rag.api", list(iris_vector_rag.__path__)
    )
    assert spec is None


@pytest.mark.parametrize(
    "path",
    [
        "iris_vector_rag/api",
        "tests_api",
        "tests/integration/api",
        "config/api_config.yaml",
        "tests/unit/test_rest_api_deprecation.py",
    ],
)
def test_removed_paths_are_gone(path):
    assert not (ROOT / path).exists(), path


def test_api_extra_is_gone_and_no_extra_carries_its_requirements():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "api" not in extras
    for name, reqs in extras.items():
        leaked = API_ONLY & {_name(r) for r in reqs}
        assert not leaked, f"extra {name!r} still carries {leaked}"
    base = {_name(r) for r in _pyproject()["project"]["dependencies"]}
    assert not (API_ONLY & base)


def test_pyproject_has_no_api_filterwarnings_entry():
    text = (ROOT / "pyproject.toml").read_text()
    assert "iris_vector_rag\\\\.api" not in text
    assert "the REST API" not in text


def test_workflows_do_not_install_api_requirements():
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = wf.read_text()
        for needle in ("bcrypt", "email-validator", "--extra api", "[api"):
            assert needle not in text, f"{wf.name} still mentions {needle!r}"
        yaml.safe_load(text)


def test_makefile_has_no_api_targets():
    targets = re.findall(r"^([A-Za-z0-9_.-]+):", (ROOT / "Makefile").read_text(), re.M)
    assert not [t for t in targets if t.startswith("api")], targets
    assert "tests_api" not in (ROOT / "Makefile").read_text()


def test_design_pack_stays_and_names_the_tag():
    pack = ROOT / "docs" / "archived" / "rest-api"
    assert (pack / "README.md").is_file()
    assert (pack / "version-2-must-do-differently.md").is_file()
    assert TAG in (pack / "README.md").read_text()


def test_nothing_outside_history_mentions_the_removed_package():
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    offenders = []
    for rel in files:
        if rel.startswith(HISTORY) or not (ROOT / rel).is_file():
            continue
        try:
            text = (ROOT / rel).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if MENTION.search(text):
            offenders.append(rel)
    assert not offenders, offenders


def test_changelog_announces_the_removal():
    unreleased = (ROOT / "CHANGELOG.md").read_text().split("\n## ", 2)[1]
    assert "iris_vector_rag.api" in unreleased and TAG in unreleased
