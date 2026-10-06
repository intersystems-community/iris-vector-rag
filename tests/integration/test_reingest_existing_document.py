"""Re-ingesting a document that is already stored updates it quietly.

The store used to INSERT first and rely on the failure of a duplicate: iris-vector-graph
logged `insert_vector failed: <LIST ERROR> Incorrect list format ...` at ERROR level for
every re-ingested document, although the row was then updated and the call succeeded.
Real IRIS, real embedder (the suite's `iris_target`).
"""

import logging

from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document


def _count(pipeline, doc_id):
    cursor = pipeline.connection_manager.get_connection().cursor()
    try:
        cursor.execute(
            "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id = ?", [doc_id]
        )
        return cursor.fetchone()[0]
    finally:
        cursor.close()


def test_second_ingest_updates_without_an_error_log(caplog):
    pipeline = create_pipeline("graphrag", validate_requirements=False)
    doc_id = "reingest-probe"
    first = Document(id=doc_id, page_content="first text about insulin", metadata={})
    second = Document(id=doc_id, page_content="second text about insulin", metadata={})

    pipeline.load_documents(documents=[first])
    caplog.set_level(logging.INFO)
    caplog.clear()
    result = pipeline.load_documents(documents=[second])

    assert result["documents_loaded"] == 1
    errors = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR and "insert_vector" in r.getMessage()
    ]
    assert errors == []
    assert _count(pipeline, doc_id) == 1
    stored = pipeline.vector_store.fetch_documents_by_ids([doc_id])
    assert "second text" in stored[0].page_content
