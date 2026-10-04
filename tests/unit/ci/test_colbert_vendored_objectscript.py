"""Only the ObjectScript that exists nowhere else ships, and it ships in the wheel.

Findings OS-11, OS-12, OS-13 and OS-14 (colbert-vendored-objectscript). Thirteen of the
fourteen vendored `sp/*.cls` were copies of classes iris-vector-graph provides (several
stale); only `RAG.ColBERTSearch` is this project's own. The wheel test builds the real
artifact, because package data is invisible until someone installs the result.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SP = ROOT / "iris_vector_rag" / "pipelines" / "colbert_iris" / "sp"


def _ivg_class_names() -> set[str]:
    """Class names iris-vector-graph ships, read from the installed distribution."""
    import importlib.util

    spec = importlib.util.find_spec("iris_vector_graph")
    if spec is None or not spec.submodule_search_locations:
        pytest.fail("iris-vector-graph is a dependency and must be installed")
    names = set()
    for cls in Path(list(spec.submodule_search_locations)[0]).rglob("*.cls"):
        for line in cls.read_text(errors="ignore").splitlines():
            if line.startswith("Class "):
                names.add(line.split()[1])
                break
    return names


def _declared_class(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("Class "):
            return line.split()[1]
    raise AssertionError(f"{path} declares no class")


def test_only_colbertsearch_is_vendored():
    assert sorted(p.name for p in SP.glob("*.cls")) == ["ColBERTSearch.cls"]


def test_no_vendored_class_is_one_ivg_provides():
    provided = _ivg_class_names()
    for cls in SP.glob("*.cls"):
        assert _declared_class(cls) not in provided, cls.name


def test_user_exec_is_not_vendored():
    assert not list(SP.rglob("UserExec*")) and not list(SP.rglob("*Exec*.cls"))


def test_colbertsearch_builds_no_sql_from_doc_ids_or_counts():
    """OS-11: an apostrophe in a doc_id broke or injected into the query."""
    text = (SP / "ColBERTSearch.cls").read_text()
    assert "'{d}'" not in text
    assert "int(nProbe)" in text and "int(topK)" in text


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    src = tmp_path_factory.mktemp("src")
    out = tmp_path_factory.mktemp("wheel")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(ROOT / name, src / name)
    shutil.copytree(
        ROOT / "iris_vector_rag",
        src / "iris_vector_rag",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools import build_meta; "
            "print(build_meta.build_wheel(sys.argv[1]))",
            str(out),
        ],
        cwd=src,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    (path,) = out.glob("*.whl")
    return zipfile.ZipFile(path)


def test_wheel_ships_exactly_colbertsearch_cls(wheel):
    cls = sorted(n for n in wheel.namelist() if n.endswith(".cls"))
    assert cls == ["iris_vector_rag/pipelines/colbert_iris/sp/ColBERTSearch.cls"]


def test_wheel_does_not_carry_user_exec(wheel):
    assert not [n for n in wheel.namelist() if "Exec" in n]
