"""The wheel ships what the code reads, and every name the project declares exists.

Findings OS-06, OS-09, OS-18, OS-24, CM-12 and OS-20 (ci-honesty stage 2). Static
checks parse pyproject.toml, the Makefile, CLAUDE.md and the workflows; the wheel
test builds the real artifact and reads it, because a packaging mistake is
invisible until someone installs the result.
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _extras_named(text: str) -> set[str]:
    """Every extra named in a `.[a,b]` / `".[a,b]"` install spec in `text`."""
    names: set[str] = set()
    for group in re.findall(r"\.\[([^\]]+)\]", text):
        names.update(n.strip() for n in group.split(","))
    return names


def _requirement_names(reqs) -> set[str]:
    return {re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0].lower() for r in reqs}


# -- extras and dev dependencies (OS-06, OS-09) ------------------------------------


def test_makefile_install_names_only_declared_extras(pyproject):
    declared = set(pyproject["project"]["optional-dependencies"])
    named = _extras_named((ROOT / "Makefile").read_text())
    assert named, "Makefile install no longer names an extra; update this test"
    assert named <= declared, f"Makefile names undeclared extras {named - declared}"


def test_workflows_install_only_declared_extras(pyproject):
    declared = set(pyproject["project"]["optional-dependencies"])
    for wf in WORKFLOWS.glob("*.yml"):
        missing = _extras_named(wf.read_text()) - declared
        assert not missing, f"{wf.name} installs undeclared extras {missing}"


def test_dev_extra_carries_pytest_timeout(pyproject):
    """`make test-unit` and the workflows pass --timeout; `pip install -e .[dev]` must
    be enough to accept it."""
    dev = _requirement_names(pyproject["project"]["optional-dependencies"]["dev"])
    assert "pytest-timeout" in dev


def test_all_extra_carries_pytest_timeout(pyproject):
    allx = _requirement_names(pyproject["project"]["optional-dependencies"]["all"])
    assert "pytest-timeout" in allx


# -- classifiers (OS-18) ------------------------------------------------------------


def test_no_python_310_classifier(pyproject):
    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert (
        "Programming Language :: Python :: 3.10"
        not in pyproject["project"]["classifiers"]
    )


# -- console scripts (OS-24) --------------------------------------------------------


def test_every_console_script_points_at_something_real(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert "iris-vector-rag-mcp" in scripts
    for name, target in scripts.items():
        module, _, attr = target.partition(":")
        mod = importlib.import_module(module)
        assert hasattr(mod, attr), f"{name} -> {target}: no attribute {attr}"


# -- the config file the manager reads (CM-12) --------------------------------------


def test_pipelines_yaml_lives_inside_the_package():
    assert (ROOT / "iris_vector_rag" / "config" / "pipelines.yaml").is_file()
    assert not (ROOT / "config" / "pipelines.yaml").exists()


def test_pipeline_requirements_come_from_the_packaged_file():
    from iris_vector_rag.config.manager import ConfigurationManager

    packaged = yaml.safe_load(
        (ROOT / "iris_vector_rag" / "config" / "pipelines.yaml").read_text()
    )
    graphrag = next(p for p in packaged["pipelines"] if p["name"].lower() == "graphrag")
    got = ConfigurationManager().get_pipeline_requirements("graphrag")
    assert got["schema_requirements"] == graphrag["schema_requirements"]
    assert got["schema_requirements"], "GraphRAG requirements must not be empty"


def test_a_missing_pipelines_file_is_an_error(monkeypatch, tmp_path):
    """The lookup used to return `enabled: True` defaults when it found nothing."""
    from iris_vector_rag.config import manager

    monkeypatch.setattr(manager, "PIPELINES_CONFIG_PATH", tmp_path / "absent.yaml")
    with pytest.raises(FileNotFoundError):
        manager.ConfigurationManager().get_pipeline_requirements("graphrag")


# -- the built wheel ----------------------------------------------------------------


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    """Build the wheel from a copy of the sources, so the checkout stays clean.

    `setuptools.build_meta` is called directly: setuptools >= 70.1 brings its own
    `bdist_wheel`, and `python -m build --no-isolation` would demand the separate
    `wheel` package that a dev environment does not always have.
    """
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


def test_wheel_contains_pipelines_yaml(wheel):
    assert "iris_vector_rag/config/pipelines.yaml" in wheel.namelist()


def test_wheel_contains_the_mcp_schema_files(wheel):
    assert any(
        n.startswith("iris_vector_rag/mcp/") and n.endswith(".json")
        for n in wheel.namelist()
    )


def test_wheel_declares_the_console_script_and_only_it(wheel):
    (entry,) = [n for n in wheel.namelist() if n.endswith("entry_points.txt")]
    text = wheel.read(entry).decode()
    assert "iris-vector-rag-mcp = iris_vector_rag.mcp.__main__:main" in text
    assert "rag-templates" not in text


def test_wheel_metadata_has_no_python_310_classifier(wheel):
    (meta,) = [n for n in wheel.namelist() if n.endswith("METADATA")]
    assert "Python :: 3.10" not in wheel.read(meta).decode()


def test_wheel_declares_pytest_timeout_in_dev_extra(wheel):
    (meta,) = [n for n in wheel.namelist() if n.endswith("METADATA")]
    assert re.search(
        r"Requires-Dist: pytest-timeout.*extra == ['\"]dev['\"]",
        wheel.read(meta).decode(),
    )


# -- CLAUDE.md names only targets that exist (OS-20) --------------------------------


def test_claude_md_make_targets_exist():
    targets = set(
        re.findall(r"^([A-Za-z0-9_-]+):", (ROOT / "Makefile").read_text(), re.M)
    )
    referenced = set(
        re.findall(r"\bmake ([a-z][a-z0-9_-]*)", (ROOT / "CLAUDE.md").read_text())
    )
    assert referenced, "CLAUDE.md names no make target; update this test"
    assert (
        referenced <= targets
    ), f"CLAUDE.md names missing targets {referenced - targets}"


def test_claude_md_keeps_the_declaration_lines_scripts_parse():
    text = (ROOT / "CLAUDE.md").read_text()
    assert re.search(r"^IRIS_CONTAINER: iris-vector-rag-iris$", text, re.M)
    assert re.search(r"^IRIS_PORT: 51972$", text, re.M)


# -- removed packages stay out of the wheel (retire-rest-api, ontology-retirement) --


@pytest.mark.parametrize("package", ["api", "ontology"])
def test_wheel_ships_no_removed_package(wheel, package):
    leaked = [
        n for n in wheel.namelist() if n.startswith(f"iris_vector_rag/{package}/")
    ]
    assert not leaked, leaked
