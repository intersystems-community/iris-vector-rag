"""The ``rag_health_check`` tool: measured facts only.

Three checks, all real, none invented:

* IRIS reachable: run ``SELECT 1`` through the session the connection-settings
  resolver points at, and time it.
* Each pipeline constructible: build it (its first use would do the same).
* Embedder loadable: embed one string. With no fake vectors (embedding-integrity)
  a model that cannot load raises here.

A failure is reported with the exception's type and message; it is never
replaced by a healthy-looking value.
"""

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry
from iris_vector_rag.mcp.worker import run_on_iris_thread

IrisProbe = Callable[[], Awaitable[float]]
EmbedderProbe = Callable[[], Awaitable[int]]

_MAX_ERROR = 300


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR]


async def probe_iris() -> float:
    """Milliseconds for ``SELECT 1`` against the resolved IRIS. Raises if unreachable."""

    def run() -> float:
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.session import session_for

        session = session_for(ConfigurationManager())
        started = time.perf_counter()
        row = session.one("SELECT 1")
        elapsed = (time.perf_counter() - started) * 1000
        if row is None or row[0] != 1:
            raise RuntimeError(f"SELECT 1 returned {row!r}")
        return round(elapsed, 1)

    return await run_on_iris_thread(run)


async def probe_embedder() -> int:
    """Dimension of a real embedding. Raises if the embedder cannot load."""

    def run() -> int:
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.embeddings.manager import EmbeddingManager

        vector = EmbeddingManager(ConfigurationManager()).embed_text("health check")
        return len(vector)

    return await asyncio.to_thread(run)


async def check_health(
    registry: TechniqueHandlerRegistry,
    *,
    include_details: bool = False,
    iris_probe: Optional[IrisProbe] = None,
    embedder_probe: Optional[EmbedderProbe] = None,
) -> Dict[str, Any]:
    """Run the three checks. ``status`` is unavailable, degraded or healthy."""
    iris_probe = iris_probe or probe_iris
    embedder_probe = embedder_probe or probe_embedder

    database: Dict[str, Any]
    try:
        database = {"connected": True, "response_time_ms": await iris_probe()}
    except Exception as exc:
        database = {"connected": False, "error": _describe(exc)}

    embedder: Dict[str, Any]
    try:
        embedder = {"loadable": True, "dimension": await embedder_probe()}
    except Exception as exc:
        embedder = {"loadable": False, "error": _describe(exc)}

    pipelines: Dict[str, Any] = {}
    for name, handler in registry.get_all_handlers().items():
        try:
            await run_on_iris_thread(handler.construct)
            pipelines[name] = {"constructible": True}
        except Exception as exc:
            pipelines[name] = {"constructible": False, "error": _describe(exc)}

    if not database["connected"]:
        status = "unavailable"
    elif not embedder["loadable"] or not all(
        p["constructible"] for p in pipelines.values()
    ):
        status = "degraded"
    else:
        status = "healthy"

    report: Dict[str, Any] = {
        "status": status,
        "database": database,
        "embedder": embedder,
    }
    if include_details or status != "healthy":
        report["pipelines"] = pipelines
    return report
