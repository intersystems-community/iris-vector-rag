"""
iris-vector-rag retriever backed by langchain-intersystems IRISVectorStore.

Wraps IRISVectorStore as a LangChain BaseRetriever so it can be dropped
into any iris-vector-rag pipeline that accepts a retriever.

Status: STUB — pending langchain-intersystems PyPI release and spike results.
See docs/LANGCHAIN_INTERSYSTEMS_SPIKE.md for the research questions that must
be answered before this adapter is production-ready.

Usage (once langchain-intersystems is on PyPI):
    from iris_vector_rag.contrib.langchain_intersystems import IRISRAGRetriever

    retriever = IRISRAGRetriever(
        embeddings=OpenAIEmbeddings(),
        connect_kwargs={
            'hostname': 'localhost',
            'port': 1972,
            'namespace': 'USER',
            'username': '_SYSTEM',
            'password': 'SYS',
        },
        collection_name='my_docs',
        search_kwargs={'k': 5, 'filter': {'category': 'radiology'}}
    )
    pipeline = BasicRAGPipeline(retriever=retriever, llm=...)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langchain_intersystems import IRISVectorStore, Predicate, SimilarityMetric  # noqa: F401
    _LANGCHAIN_INTERSYSTEMS_AVAILABLE = True
except ImportError:
    _LANGCHAIN_INTERSYSTEMS_AVAILABLE = False

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    _LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    _LANGCHAIN_CORE_AVAILABLE = False


def _check_dependencies() -> None:
    if not _LANGCHAIN_CORE_AVAILABLE:
        raise ImportError(
            "langchain-core is required. Install with: pip install langchain-core"
        )
    if not _LANGCHAIN_INTERSYSTEMS_AVAILABLE:
        raise ImportError(
            "langchain-intersystems is required but not yet on PyPI. "
            "Install the wheel from the READY 2026 hackathon kit:\n"
            "  pip install ready2026-hackathon/demos/langchain-vectorstore/"
            "dist/langchain_intersystems-0.0.1-py3-none-any.whl\n"
            "See docs/LANGCHAIN_INTERSYSTEMS_SPIKE.md for spike instructions."
        )


class IRISRAGRetriever(BaseRetriever):
    """
    LangChain BaseRetriever backed by IRISVectorStore.

    Bridges the official InterSystems LangChain integration into the
    iris-vector-rag pipeline framework. Supports the full IRISVectorStore
    metadata filtering API via search_kwargs['filter'].

    Args:
        embeddings: Any LangChain Embeddings object (OpenAI, Ollama, HuggingFace, etc.)
        connect_kwargs: DB-API connection dict with keys:
            hostname, port, namespace, username, password
        collection_name: IRIS SQL table name for this collection
        search_kwargs: Passed to similarity_search_with_score. Keys:
            k (int): number of results, default 4
            filter (dict): IRISVectorStore Predicate filter expression
            score_threshold (float): minimum similarity score (optional)
        similarity_metric: SimilarityMetric.COSINE (default) or DOT_PRODUCT/EUCLIDEAN
        replace_collection: Drop and recreate collection on init (default False)
    """

    embeddings: Any
    connect_kwargs: dict[str, Any]
    collection_name: str
    search_kwargs: dict[str, Any]
    similarity_metric: Any  # SimilarityMetric enum — typed as Any for import-optional
    replace_collection: bool

    _store: Any  # IRISVectorStore instance, lazily initialized

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        embeddings: Any,
        connect_kwargs: dict[str, Any],
        collection_name: str = "iris_rag",
        search_kwargs: dict[str, Any] | None = None,
        similarity_metric: Any = None,
        replace_collection: bool = False,
    ) -> None:
        _check_dependencies()
        super().__init__(
            embeddings=embeddings,
            connect_kwargs=connect_kwargs,
            collection_name=collection_name,
            search_kwargs=search_kwargs or {"k": 4},
            similarity_metric=similarity_metric or SimilarityMetric.COSINE,
            replace_collection=replace_collection,
        )
        self._store = None

    def _get_store(self) -> "IRISVectorStore":
        if self._store is None:
            logger.debug(
                "Initializing IRISVectorStore collection=%s host=%s:%s",
                self.collection_name,
                self.connect_kwargs.get("hostname", "localhost"),
                self.connect_kwargs.get("port", 1972),
            )
            self._store = IRISVectorStore(
                self.embeddings,
                connect_kwargs=self.connect_kwargs,
                collection_name=self.collection_name,
                replace_collection=self.replace_collection,
                similarity_metric=self.similarity_metric,
            )
        return self._store

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun",
    ) -> list["Document"]:
        store = self._get_store()
        k = self.search_kwargs.get("k", 4)
        filter_expr = self.search_kwargs.get("filter", None)
        score_threshold = self.search_kwargs.get("score_threshold", None)

        results = store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_expr,
        )

        docs = []
        for doc, score in results:
            if score_threshold is not None and score < score_threshold:
                continue
            # Attach retrieval score to metadata for pipeline observability
            doc.metadata["_retrieval_score"] = round(float(score), 6)
            docs.append(doc)

        logger.debug("IRISRAGRetriever returned %d docs for query=%r", len(docs), query[:60])
        return docs

    def add_documents(self, documents: list["Document"]) -> list[str]:
        """Index documents into the IRIS vector store."""
        return self._get_store().add_documents(documents)

    @classmethod
    def from_documents(
        cls,
        documents: list["Document"],
        embeddings: Any,
        connect_kwargs: dict[str, Any],
        collection_name: str = "iris_rag",
        **kwargs: Any,
    ) -> "IRISRAGRetriever":
        """Create retriever and index documents in one call."""
        retriever = cls(
            embeddings=embeddings,
            connect_kwargs=connect_kwargs,
            collection_name=collection_name,
            replace_collection=True,
            **kwargs,
        )
        retriever.add_documents(documents)
        return retriever


__all__ = ["IRISRAGRetriever", "Predicate", "SimilarityMetric"]
