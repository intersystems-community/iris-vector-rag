"""Shared fixtures for contract tests.

Contract tests validate infrastructure and API contracts.
"""

# Contract tests that need IRIS fixtures should import directly
# to avoid circular dependencies with pytest plugin system

import pytest

#: The width the tests' table declares, and the wrong one a query is made to have.
DECLARED_WIDTH = 384
WRONG_WIDTH = 768


@pytest.fixture
def wrong_width_queries(mocker):
    """Make a pipeline embed its queries at 768 dimensions against a 384 table.

    Patches the embedder the search really calls. BasicRAG and Rerank search
    through the store, which embeds the query with its `embedder`; CRAG embeds
    the query itself with `embedding_func` and hands the store the vector. The
    store refuses a vector whose width is not the column's, with
    `DimensionMismatchError`.
    """

    def apply(pipeline):
        vector = [0.1] * WRONG_WIDTH
        if hasattr(pipeline, "_initial_retrieval"):
            mocker.patch.object(
                pipeline, "embedding_func", lambda texts: [vector for _ in texts]
            )
        else:
            mocker.patch.object(pipeline.vector_store, "embedder", lambda text: vector)

    return apply


@pytest.fixture
def unreachable_iris(mocker):
    """Make every connection a pipeline's store asks for fail, as a down IRIS does."""

    def apply(pipeline):
        mocker.patch.object(
            pipeline.vector_store.connection_manager,
            "get_connection",
            side_effect=ConnectionError("IRIS connection refused (simulated)"),
        )

    return apply


def assert_names_the_mismatch(exc):
    """The error gives both widths and the fix, not just "dimension"."""
    msg = str(exc)
    assert str(WRONG_WIDTH) in msg and str(DECLARED_WIDTH) in msg, msg
    assert "migrate the column" in msg, msg
