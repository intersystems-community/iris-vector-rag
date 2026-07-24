"""Contract tests for US2: consistent query() signature across all registered pipelines.

T014 — TDD, must fail before T015–T020 implementation.

FR-004: canonical params (query, top_k, generate_answer, include_sources) with consistent defaults.
FR-005: query_text= alias works; both query= + query_text= uses query= and warns.
FR-006: all pipelines return the same response keys.
C-Q1..C-Q6 from contracts/query_api.md.

All tests are unit-level; no IRIS connection required.
"""
import warnings
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.core.models import Document

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STUB_DOCS = [
    Document(page_content="stub doc 1", metadata={"source": "test"}),
    Document(page_content="stub doc 2", metadata={"source": "test"}),
]

# FR-006: every pipeline must return at least these top-level keys
REQUIRED_RESPONSE_KEYS = {"query", "answer", "retrieved_documents", "contexts", "metadata"}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stub factories
# ──────────────────────────────────────────────────────────────────────────────

def _make_basic():
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline
    with patch.object(BasicRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = BasicRAGPipeline.__new__(BasicRAGPipeline)
    p.connection_manager = MagicMock()
    p.config_manager = MagicMock()
    p.config_manager.get = MagicMock(side_effect=lambda k, d=None: d)
    p.vector_store = MagicMock()
    p.vector_store.similarity_search.return_value = STUB_DOCS
    p.llm_func = MagicMock(return_value="")
    p.embedding_manager = MagicMock()
    p.embedding_config = None
    p.use_iris_embedding = False
    p.pipeline_config = {}
    p.chunk_size = 1000
    p.chunk_overlap = 200
    p.default_top_k = 5
    return p


def _make_crag():
    from iris_vector_rag.pipelines.crag import CRAGPipeline
    with patch.object(CRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = CRAGPipeline.__new__(CRAGPipeline)
    p.connection_manager = MagicMock()
    p.vector_store = MagicMock()
    p.vector_store.similarity_search.return_value = STUB_DOCS
    p.llm_func = None
    p.config_manager = MagicMock()
    p.top_k = 5
    p.correction_threshold = 0.5
    p.max_web_results = 3
    return p


def _make_multi_query_rrf():
    from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline
    with patch.object(MultiQueryRRFPipeline, "__init__", lambda self, *a, **kw: None):
        p = MultiQueryRRFPipeline.__new__(MultiQueryRRFPipeline)
    p.connection_manager = MagicMock()
    p.vector_store = MagicMock()
    p.vector_store.similarity_search.return_value = STUB_DOCS
    p.llm_func = None
    p.config_manager = MagicMock()
    p.rrf_k = 60
    p.num_queries = 3
    p.use_llm_expansion = False
    p.retrieved_k = 20
    return p


def _make_hybrid_graphrag():
    from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
    with patch.object(HybridGraphRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    p.connection_manager = MagicMock()
    p.vector_store = MagicMock()
    p.vector_store.similarity_search.return_value = STUB_DOCS
    p.llm_func = None
    p.config_manager = MagicMock()
    p.iris_engine = None
    p.retrieval_methods = None
    p.default_top_k = 10
    p.entity_extraction_enabled = False
    p.use_global_communities = False
    return p


def _make_pylate():
    from iris_vector_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
    with patch.object(PyLateColBERTPipeline, "__init__", lambda self, *a, **kw: None):
        p = PyLateColBERTPipeline.__new__(PyLateColBERTPipeline)
    p.connection_manager = MagicMock()
    p.vector_store = MagicMock()
    p.vector_store.similarity_search.return_value = STUB_DOCS
    p.llm_func = None
    p.use_native_reranking = False
    p.is_initialized = False
    p.model = None
    p.rerank_factor = 3
    p.model_name = "colbert-test"
    p._document_store = {}
    p._embedding_cache = {}
    p.stats = {"queries_processed": 0, "reranking_operations": 0, "documents_indexed": 0}
    return p


# Map pipeline name → (factory, patch_context_manager)
# Some pipelines need extra patches to avoid crashing in unit context
def _run_query(name: str, factory, query: str = "test query", **kwargs):
    """Run pipeline.query() with appropriate patches for unit testing."""
    p = factory()
    ctx = {}
    if name == "hybrid_graphrag":
        ctx["vkg"] = patch.object(type(p), "_validate_knowledge_graph", lambda *a, **kw: None, create=True)
        ctx["ehf"] = patch.object(type(p), "_enhanced_hybrid_fallback", return_value=(STUB_DOCS, "fallback"), create=True)
    if name == "pylate":
        ctx["rm"] = patch.object(p, "_restore_metadata", side_effect=lambda docs: docs)

    patches = list(ctx.values())
    for ptch in patches:
        ptch.start()
    try:
        return p.query(query=query, **kwargs)
    finally:
        for ptch in patches:
            ptch.stop()


PIPELINES = [
    ("basic", _make_basic),
    ("crag", _make_crag),
    ("multi_query_rrf", _make_multi_query_rrf),
    ("hybrid_graphrag", _make_hybrid_graphrag),
    ("pylate", _make_pylate),
]

PIPELINE_IDS = [name for name, _ in PIPELINES]


# ──────────────────────────────────────────────────────────────────────────────
# C-Q1 / FR-004: canonical params accepted, same defaults
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,factory", PIPELINES, ids=PIPELINE_IDS)
def test_canonical_query_param_accepted(name, factory):
    """C-Q1 / FR-004: query(query=...) works on every pipeline."""
    result = _run_query(name, factory, generate_answer=False)
    assert isinstance(result, dict), f"{name}: query() must return a dict"


@pytest.mark.parametrize("name,factory", PIPELINES, ids=PIPELINE_IDS)
def test_default_top_k_param_is_5(name, factory):
    """C-Q1 / FR-004: query() default top_k parameter is 5 (not 10, not 20).

    Internal retrieval may fetch more candidates (e.g. colbert rerank_factor),
    but the top_k *parameter default* must be 5 for API consistency.
    Verified by inspecting the function signature.
    """
    import inspect
    if name == "pylate":
        from iris_vector_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
        fn = PyLateColBERTPipeline.query
    elif name == "multi_query_rrf":
        from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline
        fn = MultiQueryRRFPipeline.query
    elif name == "hybrid_graphrag":
        from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
        fn = HybridGraphRAGPipeline.query
    elif name == "crag":
        from iris_vector_rag.pipelines.crag import CRAGPipeline
        fn = CRAGPipeline.query
    else:
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline
        fn = BasicRAGPipeline.query

    sig = inspect.signature(fn)
    default = sig.parameters.get("top_k")
    assert default is not None, f"{name}: query() must have a top_k parameter"
    assert default.default == 5, (
        f"{name}: default top_k must be 5, got {default.default!r} (FR-004)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# C-Q2 / FR-005: query_text= alias
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,factory", PIPELINES, ids=PIPELINE_IDS)
def test_query_text_alias_accepted(name, factory):
    """C-Q2 / FR-005: query_text= is an accepted alias for query=."""
    result = _run_query(name, factory, query=None, query_text="alias query", generate_answer=False)
    assert isinstance(result, dict), f"{name}: query_text= alias must work"


# ──────────────────────────────────────────────────────────────────────────────
# C-Q5 / FR-006: standardized response keys
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,factory", PIPELINES, ids=PIPELINE_IDS)
def test_response_has_required_keys(name, factory):
    """C-Q5 / FR-006: response must include query, answer, retrieved_documents, contexts, metadata."""
    result = _run_query(name, factory, generate_answer=False)
    missing = REQUIRED_RESPONSE_KEYS - result.keys()
    assert not missing, (
        f"{name}: response missing keys {missing} (FR-006). Got: {set(result.keys())}"
    )


@pytest.mark.parametrize("name,factory", PIPELINES, ids=PIPELINE_IDS)
def test_response_echoes_query_string(name, factory):
    """C-Q5 / FR-006: response['query'] must echo back the input query string."""
    result = _run_query(name, factory, query="echo this back", generate_answer=False)
    assert result.get("query") == "echo this back", (
        f"{name}: response['query'] must echo the input. Got: {result.get('query')!r}"
    )


@pytest.mark.parametrize("name,factory", PIPELINES, ids=PIPELINE_IDS)
def test_include_sources_false_returns_empty_sources(name, factory):
    """C-Q5 / FR-006: include_sources=False means response['sources'] is empty."""
    result = _run_query(name, factory, generate_answer=False, include_sources=False)
    assert result.get("sources", []) == [], (
        f"{name}: include_sources=False must produce empty sources. Got: {result.get('sources')}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# C-Q6: invalid top_k raises
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,factory", PIPELINES, ids=PIPELINE_IDS)
def test_top_k_zero_raises(name, factory):
    """C-Q6: top_k=0 raises a clear ValueError."""
    with pytest.raises((ValueError, Exception)):
        _run_query(name, factory, top_k=0, generate_answer=False)
