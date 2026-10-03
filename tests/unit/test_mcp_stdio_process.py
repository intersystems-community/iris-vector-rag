"""The real process over real stdio: `python -m iris_vector_rag.mcp` and the console script."""

import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the `mcp` extra provides the official SDK")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


async def _handshake(command, args):
    params = StdioServerParameters(command=command, args=args, cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            return init, {t.name for t in tools.tools}


async def test_module_entry_point_completes_the_handshake_without_touching_iris():
    init, names = await _handshake(sys.executable, ["-m", "iris_vector_rag.mcp"])
    assert init.serverInfo.name == "iris-vector-rag"
    assert len(names) == 6 and "rag_health_check" in names


async def test_console_script_name_resolves_to_the_same_server():
    script = shutil.which("iris-vector-rag-mcp", path=str(Path(sys.executable).parent))
    if script is None:
        pytest.skip(
            "console script not installed in this environment (needs pip install -e .)"
        )
    _, names = await _handshake(script, [])
    assert len(names) == 6
