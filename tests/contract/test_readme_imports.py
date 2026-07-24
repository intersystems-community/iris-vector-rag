"""Contract test for US8: README quickstart imports resolve (Feature 065, FR-017, SC-006).

Extracts every `import`/`from ... import` statement referencing the package from
README.md and verifies each resolves on the installed package. This catches the
class of defect where docs referenced the old `iris_rag` module (ModuleNotFoundError
on a user's first copy-paste).
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

# Match import statements that reference the project package by either the correct
# name or the historical (wrong) name, so a regression to `iris_rag` is caught.
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(iris_vector_rag[\w.]*)\s+import\s+[\w,*\s()]+"
    r"|import\s+(iris_vector_rag[\w.]*)"
    r"|from\s+(iris_rag[\w.]*)\s+import\s+[\w,*\s()]+"
    r"|import\s+(iris_rag[\w.]*))\s*$",
    re.MULTILINE,
)


def _readme_import_statements():
    text = README.read_text(encoding="utf-8")
    statements = []
    for line in text.splitlines():
        stripped = line.strip()
        if IMPORT_RE.match(stripped):
            statements.append(stripped)
    return statements


def test_readme_has_import_examples():
    assert README.exists(), "README.md must exist"
    assert _readme_import_statements(), "expected at least one package import in README"


@pytest.mark.parametrize("statement", _readme_import_statements())
def test_readme_import_resolves(statement):
    """Every package import in the README must execute without ModuleNotFoundError."""
    assert "iris_rag." not in statement and not statement.endswith("iris_rag"), (
        f"README references the non-existent 'iris_rag' package: {statement!r} "
        f"(should be 'iris_vector_rag')"
    )
    try:
        exec(compile(statement, "<readme>", "exec"), {})
    except ModuleNotFoundError as e:  # pragma: no cover - failure path
        pytest.fail(f"README import failed: {statement!r} -> {e}")
