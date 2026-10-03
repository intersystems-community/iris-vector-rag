"""Entry point: ``iris-vector-rag-mcp`` or ``python -m iris_vector_rag.mcp``.

Starts the stdio MCP server. There are no subcommands and no options; the IRIS
connection comes from the environment (see docs/MCP_INTEGRATION.md).
"""

import logging
import sys


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        import anyio

        from iris_vector_rag.mcp.server import serve
    except ImportError as exc:
        print(
            f"iris-vector-rag-mcp needs the MCP SDK ({exc}). "
            'Install it with: pip install "iris-vector-rag[mcp]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    anyio.run(serve)


if __name__ == "__main__":
    main()
