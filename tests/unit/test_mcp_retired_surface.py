"""ADR 0002: what the rebuild retired stays gone, and the one entry point exists."""

import importlib
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "path",
    [
        "tools/nodejs",
        "Dockerfile.mcp",
        "scripts/docker/docker-entrypoint-mcp.sh",
        "config/mcp_config.yaml",
        "iris_vector_rag/mcp/bridge.py",
        "iris_vector_rag/mcp/cli.py",
        "iris_vector_rag/mcp/config.py",
    ],
)
def test_retired_file_is_gone(path):
    assert not (ROOT / path).exists(), path


@pytest.mark.parametrize("module", ["bridge", "cli", "config", "iris_llm_handler"])
def test_retired_module_does_not_import(module):
    with pytest.raises(ImportError):
        importlib.import_module(f"iris_vector_rag.mcp.{module}")


def test_no_authentication_machinery_remains():
    for py in (ROOT / "iris_vector_rag" / "mcp").glob("*.py"):
        text = py.read_text()
        assert "MCP_AUTH_MODE" not in text, py
        assert "api_key" not in text, py


def test_one_console_script_starts_the_stdio_server():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert (
        project["scripts"]["iris-vector-rag-mcp"] == "iris_vector_rag.mcp.__main__:main"
    )


def test_mcp_extra_is_the_official_sdk_only():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    names = {r.split(">")[0].split("=")[0].split("[")[0].strip() for r in extras["mcp"]}
    assert names == {"mcp"}


def test_tool_schema_file_ships_in_the_wheel():
    text = (ROOT / "pyproject.toml").read_text()
    assert "mcp/*.json" in text


def test_docs_use_the_resolvers_variable_names():
    text = (ROOT / "docs" / "MCP_INTEGRATION.md").read_text()
    assert "IRIS_USERNAME" in text
    for line in text.splitlines():
        if re.search(r"IRIS_USER(?![A-Z])", line):
            assert "deprecated" in line, line
    assert "Production-Ready" not in text
    assert "iris-vector-rag-mcp" in text


def test_entry_point_target_is_callable():
    from iris_vector_rag.mcp.__main__ import main

    assert callable(main)
