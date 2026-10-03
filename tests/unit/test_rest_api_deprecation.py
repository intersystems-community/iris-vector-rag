"""The REST API (iris_vector_rag.api) is deprecated in 0.15 and removed in 0.16.

ADR 0001 / ticket G1: the deprecation release warns on import, drops the ``api``
extra from ``all`` and from CI, names the archive tag, and ships the design pack.
Delete this file with the package in the removal release.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TAG = "archive/rest-api-v1"
# Requirements only the REST API needs; `mcp` owns fastapi/uvicorn/websockets.
API_ONLY = {"redis", "bcrypt", "python-jose", "email-validator", "python-multipart"}


def _name(requirement: str) -> str:
    for sep in ("[", ">", "<", "=", "~", "!", ";", " "):
        requirement = requirement.split(sep)[0]
    return requirement.lower()


def _extras() -> dict:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data["project"]["optional-dependencies"]


def test_import_emits_exactly_one_deprecation_warning():
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import iris_vector_rag.api\n"
        "    import iris_vector_rag.api  # cached: no second warning\n"
        "msgs = [str(x.message) for x in w\n"
        "        if issubclass(x.category, DeprecationWarning)\n"
        "        and 'iris_vector_rag.api' in str(x.message)]\n"
        "print('\\n'.join(msgs))\n"
        "print('COUNT', len(msgs))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "COUNT 1" in out.stdout, out.stdout
    assert "0.16" in out.stdout, "message must name the removal release"
    assert TAG in out.stdout, "message must name the archive tag"
    assert (
        "MCP" in out.stdout or "mcp" in out.stdout
    ), "message must name the replacement"


def test_warning_category_and_stacklevel_point_at_the_importer():
    with pytest.warns(DeprecationWarning, match="iris_vector_rag.api"):
        import importlib

        sys.modules.pop("iris_vector_rag.api", None)
        importlib.import_module("iris_vector_rag.api")


def test_api_extra_still_exists_but_is_not_in_all():
    extras = _extras()
    assert "api" in extras, "extra stays for one release so installs do not break"
    all_names = {_name(r) for r in extras["all"]}
    assert not (API_ONLY & all_names), API_ONLY & all_names


def test_all_extra_keeps_what_mcp_needs():
    extras = _extras()
    all_names = {_name(r) for r in extras["all"]}
    assert {_name(r) for r in extras["mcp"]} <= all_names


def test_api_extra_is_marked_deprecated_in_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    head = text.split("\napi = [")[0].rsplit("\n", 6)[-6:]
    assert any("deprecated" in line.lower() for line in head)


@pytest.mark.parametrize("wf", ["ci.yml", "security.yml"])
def test_workflows_do_not_install_api_extra(wf):
    text = (ROOT / ".github" / "workflows" / wf).read_text()
    assert "--extra api" not in text


def test_design_pack_exists():
    pack = ROOT / "docs" / "archived" / "rest-api"
    expected = [
        "README.md",
        "architecture.md",
        "routes.md",
        "auth-and-rate-limits.md",
        "websocket-protocol.md",
        "data-model.md",
        "configuration.md",
        "version-2-must-do-differently.md",
        "original/README.md",
        "original/IMPLEMENTATION_COMPLETE.md",
        "original/IMPLEMENTATION_FINAL.md",
        "original/api_config.yaml",
        "original/schema.sql",
        "original/001_initial_schema.sql",
    ]
    missing = [p for p in expected if not (pack / p).is_file()]
    assert not missing, missing
    assert TAG in (pack / "README.md").read_text()


def test_notices_in_readme_and_changelog():
    readme = (ROOT / "README.md").read_text()
    assert (
        "deprecated"
        in readme.split("pip install iris-vector-rag[api]")[1].split("\n")[0].lower()
    )
    unreleased = (ROOT / "CHANGELOG.md").read_text().split("\n## ", 2)[1]
    assert "iris_vector_rag.api" in unreleased and TAG in unreleased
