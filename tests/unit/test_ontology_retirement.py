"""Ontology retirement (decouple and deprecate), finding group OE-08/09/11/17 + OS-27.

Entity extraction must not depend on ``iris_vector_rag.ontology``; importing the
ontology package warns once; the ``ontology:`` config section is a no-op.
"""

import subprocess
import sys
import textwrap

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.services.entity_extraction import EntityExtractionService


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "default", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_entity_extraction_import_does_not_import_ontology():
    result = _run("""
        import sys
        import iris_vector_rag.services.entity_extraction
        loaded = [m for m in sys.modules if m.startswith("iris_vector_rag.ontology")]
        print("LOADED:" + ",".join(loaded))
        """)
    assert result.returncode == 0, result.stderr
    assert "LOADED:\n" in result.stdout + "\n", result.stdout
    assert "LOADED:iris" not in result.stdout


def test_importing_ontology_package_warns_exactly_once():
    result = _run("""
        import iris_vector_rag.ontology
        import iris_vector_rag.ontology.reasoner
        import iris_vector_rag.ontology.plugins
        import importlib
        importlib.import_module("iris_vector_rag.ontology")
        """)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("DeprecationWarning") == 1, result.stderr
    assert "iris_vector_rag.ontology" in result.stderr
    assert "deprecated" in result.stderr


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
    baseline = _extractor({}).extract_with_ontology(text)
    got = _extractor(section).extract_with_ontology(text)
    assert [(e.text, e.entity_type, e.confidence) for e in got] == [
        (e.text, e.entity_type, e.confidence) for e in baseline
    ]
    assert [e.text for e in got] == ["Alice Smith"]


def test_extractor_holds_no_ontology_state():
    ex = _extractor({"enabled": True})
    assert not hasattr(ex, "ontology_plugin")
    assert not hasattr(ex, "reasoner")
