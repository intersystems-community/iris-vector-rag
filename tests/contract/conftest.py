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


@pytest.fixture(scope="session")
def ivg_deployed_for_contracts(iris_target):
    """Deploy IVG tables in USER once per contract test session.

    Idempotent: ivg skips the class deploy when the installed set is unchanged.
    """
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager
    from iris_vector_rag.storage.schema_manager import SchemaManager

    config = ConfigurationManager()
    manager = SchemaManager(ConnectionManager(config), config)
    result = manager.ensure_iris_vector_graph_tables("graphrag")
    assert not result.error_messages, result.error_messages
    return manager


@pytest.fixture
def hybrid_on_known_corpus(ivg_deployed_for_contracts, iris_target):
    """HybridGraphRAG pipeline with a small known corpus for contract tests.

    The graph is deployed but empty, so all legs fall back to vector search over
    RAG.SourceDocuments. The corpus has 3 distinctive documents:
    - kc-axolotl: Mexican salamander regeneration
    - kc-basalt: Volcanic rock formation
    - kc-hanseatic: Medieval trading league

    After the test, the known documents are deleted from RAG.SourceDocuments.
    """
    from iris_vector_rag import create_pipeline
    from iris_vector_rag.core.models import Document

    # Stub LLM function for tests that don't need answer generation
    def stub_llm(prompt, **kwargs):
        return "This is a stub answer for testing purposes."

    pipeline = create_pipeline(
        "graphrag", validate_requirements=False, llm_func=stub_llm
    )

    # Known corpus with distinctive topics
    known_docs = [
        Document(
            id="kc-axolotl",
            page_content="The axolotl, a Mexican salamander, can regrow lost limbs, including bone, cartilage, muscles, and nerves. This remarkable regenerative ability makes it valuable for medical research.",
        ),
        Document(
            id="kc-basalt",
            page_content="Basalt is a dark volcanic rock formed from rapidly cooled lava. It is the most common rock type in the Earth's oceanic crust and covers much of the ocean floor.",
        ),
        Document(
            id="kc-hanseatic",
            page_content="The Hanseatic League was a medieval commercial and defensive confederation of merchant guilds and market towns in Northwestern and Central Europe. It dominated Baltic maritime trade for three centuries.",
        ),
    ]

    # Add documents to the store
    pipeline.vector_store.add_documents(known_docs)

    yield pipeline

    # Cleanup: delete known documents
    from tests.support.iris_environment import connect

    connection = connect(iris_target)
    try:
        cursor = connection.cursor()
        # By doc_id: %ID is the row id, so a delete on it removed nothing and the
        # corpus piled up across runs.
        cursor.execute(
            "DELETE FROM RAG.SourceDocuments WHERE doc_id IN (?, ?, ?)",
            ["kc-axolotl", "kc-basalt", "kc-hanseatic"],
        )
        connection.commit()
    finally:
        connection.close()
