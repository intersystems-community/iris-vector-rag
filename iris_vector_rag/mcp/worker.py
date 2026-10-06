"""The one thread that talks to IRIS on behalf of the MCP server.

A session gives every thread its own connection, and a Community IRIS licenses only a
handful at once (8 licence units). ``asyncio.to_thread`` runs each call on whichever
executor thread is free, so N calls in flight meant N connections, each kept until its
thread ended, even though the handlers run one query at a time. Routing every call that
touches IRIS through this single-thread executor gives the server one connection in all.
The executor outlives any one event loop, so a server rebuilt per test (or per client)
reuses it too.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-mcp-iris")


async def run_on_iris_thread(func: Callable[..., T], *args: Any) -> T:
    """Run ``func(*args)`` on the IRIS thread without blocking the event loop."""
    return await asyncio.get_running_loop().run_in_executor(_executor, func, *args)
