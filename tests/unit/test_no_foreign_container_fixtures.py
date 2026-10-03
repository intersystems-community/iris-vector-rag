"""OS-28: the test suite must never attach to a container this project does not own.

`iris_database_config` and `iris_test_session` attached to `los-iris` (belongs to
productivity-framework), ran DROP TABLE cleanup there, and returned a Mock on any failure.
"""

import ast
import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent

ATTACH_FOREIGN = re.compile(r"""attach\(\s*["'](?!iris-vector-rag-iris)[\w-]+["']""")


def _fixture_names(path: Path) -> set:
    tree = ast.parse(path.read_text())
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_los_iris_fixtures_are_gone():
    names = _fixture_names(TESTS / "conftest.py")
    assert "iris_database_config" not in names
    assert "iris_test_session" not in names


def test_no_test_attaches_to_a_container_by_a_hard_coded_foreign_name():
    offenders = [
        str(p.relative_to(REPO))
        for p in TESTS.rglob("*.py")
        if p != Path(__file__).resolve() and ATTACH_FOREIGN.search(p.read_text())
    ]
    assert offenders == []


def test_conftest_never_yields_a_mock_in_place_of_a_database():
    source = (TESTS / "conftest.py").read_text()
    assert "los-iris" not in source
    assert '"mock": True' not in source
