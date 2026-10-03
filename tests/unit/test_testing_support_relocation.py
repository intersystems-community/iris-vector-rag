"""VM-16: test-only harness code does not ship in the wheel.

The backend-mode harness and the fake session driver live in
`tests/support/`. Only the test embedder belongs in `iris_vector_rag/testing/`.
"""

import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "iris_vector_rag"

MOVED = [
    "backend_manager",
    "connection_pool",
    "exceptions",
    "validators",
    "fake_driver",
    "backend_modes",
]


@pytest.mark.parametrize("module", MOVED)
def test_module_is_importable_from_tests_support(module):
    importlib.import_module(f"tests.support.{module}")


@pytest.mark.parametrize("module", MOVED)
def test_module_is_not_in_the_package(module):
    assert not (PACKAGE / "testing" / f"{module}.py").exists()
    assert not (PACKAGE / "config" / f"{module}.py").exists()


def test_package_testing_directory_holds_no_harness():
    stray = sorted(
        p.name
        for p in (PACKAGE / "testing").glob("*.py")
        if p.name not in ("__init__.py", "embedder.py")
    )
    assert stray == []


def test_production_code_imports_only_the_test_embedder_from_the_testing_package():
    """`embeddings.provider: stub` is the one way production code reaches test support."""
    pattern = re.compile(
        r"iris_vector_rag\.testing|from iris_vector_rag import testing"
    )
    sanctioned = re.compile(
        r"(iris_vector_rag\.testing\.embedder|from \.\.testing\.embedder import)"
    )
    offenders = [
        str(p.relative_to(REPO))
        for p in PACKAGE.rglob("*.py")
        if "testing" not in p.relative_to(PACKAGE).parts
        and pattern.search(sanctioned.sub("", p.read_text()))
    ]
    assert offenders == []
