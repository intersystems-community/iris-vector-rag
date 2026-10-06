"""Every IRIS-touching MCP call runs on one worker thread.

A session hands each thread its own connection, and a Community IRIS licenses only a
handful at once. The handlers already run one query at a time (a lock), so queries that
each ran on a fresh executor thread took a connection apiece for no gain: five queries in
flight held five licences, and the sixth connect in the process was refused. One worker
thread means one connection, however many calls are in flight.
"""

import asyncio
import threading

import pytest

pytest.importorskip("mcp", reason="the `mcp` extra provides the official SDK")


def _run(coro):
    """Run a coroutine on its own event loop, in a thread of its own.

    `asyncio.run` on the main thread resets the main thread's loop, which drops the one
    pytest-asyncio made and sets off an unraisable-exception warning in whichever test
    the garbage collector reaches next.
    """
    outcome = {}

    def target():
        outcome["value"] = asyncio.run(coro)

    worker = threading.Thread(target=target)
    worker.start()
    worker.join()
    return outcome["value"]


class ThreadRecordingPipeline:
    def __init__(self):
        self.threads = []

    def query(self, query, **params):
        self.threads.append(threading.get_ident())
        return {"answer": "a", "retrieved_documents": [], "execution_time": 0.0}


def test_concurrent_queries_share_one_thread():
    from iris_vector_rag.mcp.technique_handlers import TechniqueHandler

    pipeline = ThreadRecordingPipeline()
    handler = TechniqueHandler("basic", lambda: pipeline)

    async def run():
        await asyncio.gather(*(handler.execute(f"q{i}", {}) for i in range(8)))

    _run(run())

    assert len(pipeline.threads) == 8
    assert len(set(pipeline.threads)) == 1


def test_queries_from_separate_event_loops_share_the_thread():
    """Each pytest-asyncio test has its own loop; the worker must outlive them."""
    from iris_vector_rag.mcp.technique_handlers import TechniqueHandler

    pipeline = ThreadRecordingPipeline()
    handler = TechniqueHandler("basic", lambda: pipeline)

    for _ in range(3):
        _run(handler.execute("q", {}))

    assert len(set(pipeline.threads)) == 1


def test_handlers_of_different_techniques_share_the_thread():
    from iris_vector_rag.mcp.technique_handlers import TechniqueHandler

    first, second = ThreadRecordingPipeline(), ThreadRecordingPipeline()

    async def run():
        await asyncio.gather(
            TechniqueHandler("basic", lambda: first).execute("q", {}),
            TechniqueHandler("crag", lambda: second).execute("q", {}),
        )

    _run(run())

    assert first.threads == second.threads


def test_the_health_probe_and_the_health_construct_use_the_same_thread():
    from iris_vector_rag.mcp.technique_handlers import TechniqueHandler
    from iris_vector_rag.mcp.worker import run_on_iris_thread

    pipeline = ThreadRecordingPipeline()
    handler = TechniqueHandler("basic", lambda: pipeline)

    async def run():
        probe = await run_on_iris_thread(threading.get_ident)
        await handler.execute("q", {})
        return probe

    probe = _run(run())
    assert pipeline.threads == [probe]
