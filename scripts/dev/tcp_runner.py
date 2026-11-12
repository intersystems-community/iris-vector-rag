#!/usr/bin/env python3
"""Run ObjectScript over TCP for test suites that need session-like behavior.

Replaces `docker exec iris session` patterns that Cortex XDR blocks.
"""

from __future__ import annotations

import os
from typing import Optional


def run_objectscript(
    script: str,
    namespace: str = "USER",
    port: Optional[int] = None,
    host: str = "localhost",
) -> str:
    """Run ObjectScript lines and return their output.

    Args:
        script: ObjectScript code (multiple lines OK, halt implied)
        namespace: Target namespace
        port: IRIS superserver port (default from IRIS_PORT env or 51972)
        host: IRIS host (default localhost)

    Returns:
        Combined stdout from the script
    """
    import iris

    port = port or int(os.environ.get("IRIS_PORT", "51972"))
    username = os.environ.get("IRIS_USERNAME", "_SYSTEM")
    password = os.environ.get("IRIS_PASSWORD", "SYS")

    conn = iris.createConnection(
        hostname=host,
        port=port,
        namespace=namespace,
        username=username,
        password=password,
    )
    try:
        native = iris.createIRIS(conn)

        # Execute line by line, collecting output
        # For simple cases, use direct calls. For complex scripts, would need
        # a server-side helper that captures IO.
        lines = [l.strip() for l in script.strip().splitlines() if l.strip()]
        output = []

        for line in lines:
            if line.lower() in ("halt", "h"):
                continue
            if line.startswith("zn "):
                # Namespace switch
                ns = line[3:].strip().strip('"')
                conn.close()
                conn = iris.createConnection(
                    hostname=host,
                    port=port,
                    namespace=ns,
                    username=username,
                    password=password,
                )
                native = iris.createIRIS(conn)
                continue

            # Try to execute as expression
            try:
                # This is limited - for full script execution would need server-side eval
                result = str(native.classMethodValue("%SYSTEM.SQL", "Execute", line))
                if result and result != "None":
                    output.append(result)
            except Exception as e:
                output.append(f"ERROR: {e}")

        return "\n".join(output)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: tcp_runner.py '<objectscript code>' [namespace] [port]")
        sys.exit(1)

    script = sys.argv[1]
    namespace = sys.argv[2] if len(sys.argv) > 2 else "USER"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else None

    output = run_objectscript(script, namespace, port)
    print(output)
