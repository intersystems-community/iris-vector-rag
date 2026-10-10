"""The ontology package is removed (feature ontology-retirement, removal half).

Findings OE-08, OE-09, OE-11 and OE-17 are closed by deletion; the last version of
the package is at the git tag ``archive/ontology-v1`` and the design history is in
``docs/archived/ontology/``. A leftover ``ontology:`` section in a user's config
still loads and changes nothing.
"""

import importlib.machinery
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.services.entity_extraction import EntityExtractionService

ROOT = Path(__file__).resolve().parents[2]
TAG = "archive/ontology-v1"
# Places that may still name the removed package: history and design records.
HISTORY = (
    "CHANGELOG.md",
    "RELEASE_NOTES_v0.15.0.md",
    "docs/archived/",
    "docs/adr/",
    "docs/remediation/",
    "docs/reports/",
    "specs/",
    "tests/unit/test_ontology_removal.py",
)
MENTION = re.compile(r"iris_vector_rag[./]ontology\b|iris_rag[./]ontology\b")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "default", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_ontology_package_does_not_import():
    """Looked up on the checkout's own package path: an editable install of another
    checkout can map ``iris_vector_rag.ontology`` to a tree that still has it."""
    import iris_vector_rag

    spec = importlib.machinery.PathFinder.find_spec(
        "iris_vector_rag.ontology", list(iris_vector_rag.__path__)
    )
    assert spec is None


def test_entity_extraction_imports_without_ontology():
    result = _run("""
        import sys
        import iris_vector_rag.services.entity_extraction
        loaded = [m for m in sys.modules if m.startswith("iris_vector_rag.ontology")]
        print("LOADED:" + ",".join(loaded))
        """)
    assert result.returncode == 0, result.stderr
    assert "LOADED:iris" not in result.stdout


@pytest.mark.parametrize(
    "path",
    [
        "iris_vector_rag/ontology",
        "tests/test_ontology_support.py",
        "tests/test_ontology_loader.py",
        "docs/ontology_integration_guide.md",
        "docs/architecture/graphrag_ontology_integration_analysis.md",
    ],
)
def test_removed_paths_are_gone(path):
    assert not (ROOT / path).exists(), path


def test_default_config_has_no_ontology_section():
    data = yaml.safe_load(
        (ROOT / "iris_vector_rag" / "config" / "default_config.yaml").read_text()
    )
    assert "ontology" not in data
    assert "ontology_integration" not in data.get("graphrag", {})
    text = (ROOT / "iris_vector_rag" / "config" / "default_config.yaml").read_text()
    assert "ontology" not in text.lower()


def test_pyproject_has_no_ontology_filterwarnings_entry():
    assert "ontology" not in (ROOT / "pyproject.toml").read_text().lower()


def test_archived_design_history_is_kept():
    pack = ROOT / "docs" / "archived" / "ontology"
    assert (pack / "ontology_integration_guide.md").is_file()
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
    assert "iris_vector_rag.ontology" in unreleased and TAG in unreleased


def _extractor(ontology_section):
    cm = ConfigurationManager()
    cm._config = {
        "entity_extraction": {"method": "llm_basic", "confidence_threshold": 0.6},
        "ontology": ontology_section,
    }
    return EntityExtractionService(
        config_manager=cm,
        llm_func=lambda prompt: '[{"text": "Alice Smith", "type": "PERSON"}]',
    )


@pytest.mark.parametrize(
    "section",
    [
        {},
        {"enabled": False},
        {"enabled": True, "reasoning": {"enable_inference": True}},
        {
            "enabled": True,
            "sources": [{"type": "example", "concepts": []}],
            "auto_detect_domain": True,
        },
    ],
)
def test_ontology_config_section_does_not_change_behaviour(section):
    text = "Alice Smith works at Acme Corp in Boston."
    baseline = _extractor({}).extract_entities_basic(text)
    got = _extractor(section).extract_entities_basic(text)
    assert [(e.text, e.entity_type, e.confidence) for e in got] == [
        (e.text, e.entity_type, e.confidence) for e in baseline
    ]
    assert [e.text for e in got] == ["Alice Smith"]


def test_extractor_holds_no_ontology_state():
    ex = _extractor({"enabled": True})
    assert not hasattr(ex, "ontology_plugin")
    assert not hasattr(ex, "reasoner")
