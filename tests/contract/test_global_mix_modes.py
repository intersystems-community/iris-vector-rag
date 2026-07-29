"""Contract tests for global and mix retrieval modes — Feature 081."""

from unittest.mock import MagicMock, patch


# ─── Global mode contracts ────────────────────────────────────────────────────


def test_global_mode_registered_with_correct_prerequisites():
    """RetrievalMode.get_mode('global') has prerequisites knowledge_graph + relation_embeddings."""
    from iris_vector_rag.retrieval.modes import get_mode

    mode = get_mode("global")
    assert "knowledge_graph" in mode.requires
    assert "relation_embeddings" in mode.requires


def test_check_prerequisites_global_no_kg_raises_named_error():
    """check_prerequisites('global') with no KG tables raises RetrievalPrerequisiteError naming knowledge_graph."""
    from iris_vector_rag.retrieval.modes import check_prerequisites, RetrievalPrerequisiteError
    import pytest

    conn = MagicMock()
    with patch(
        "iris_vector_rag.retrieval.modes._check_knowledge_graph_available",
        return_value=False,
    ):
        with pytest.raises(RetrievalPrerequisiteError) as exc_info:
            check_prerequisites("global", connection=conn)
        assert exc_info.value.missing == "knowledge_graph"


def test_retrieve_global_empty_index_returns_degraded_result():
    """_retrieve_global() when count_embedded()==0 returns degraded=True result, no exception."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    engine.keyword_extractor = MagicMock()
    engine.keyword_extractor.extract.return_value = (["macro theme"], [])

    mock_store = MagicMock()
    mock_store.count_embedded.return_value = 0
    mock_store.search.return_value = []

    opts = QueryOptions(query="test", retrieval="global", top_k=5)

    with patch(
        "iris_vector_rag.retrieval.engine.RelationEmbeddingStore",
        return_value=mock_store,
    ):
        result = engine._retrieve_global(opts)

    assert isinstance(result, dict), "Expected dict result"
    assert result["metadata"]["degraded"] is True
    assert result["metadata"].get("degradation_reason"), "Expected non-empty degradation_reason"


def test_retrieve_engine_global_dispatches_to_retrieve_global():
    """RetrievalEngine.retrieve(opts) with retrieval='global' calls _retrieve_global."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    opts = QueryOptions(query="test", retrieval="global", top_k=3)

    with patch.object(engine, "_retrieve_global", return_value={"retrieved_documents": [], "metadata": {}}) as mock_g, \
         patch("iris_vector_rag.retrieval.modes.check_prerequisites"):
        engine.retrieve(opts)

    mock_g.assert_called_once_with(opts)


def test_retrieve_global_result_has_keyword_metadata():
    """_retrieve_global() result contains high_level_keywords and degraded in metadata."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    engine.keyword_extractor = MagicMock()
    engine.keyword_extractor.extract.return_value = (["systemic risk"], [])

    mock_store = MagicMock()
    mock_store.count_embedded.return_value = 0
    mock_store.search.return_value = []

    opts = QueryOptions(query="What are systemic risks?", retrieval="global", top_k=3)

    with patch("iris_vector_rag.retrieval.engine.RelationEmbeddingStore", return_value=mock_store):
        result = engine._retrieve_global(opts)

    meta = result["metadata"]
    assert "high_level_keywords" in meta
    assert "degraded" in meta


def test_retrieve_global_partial_keywords_marks_degraded():
    """_retrieve_global() when only low_kws non-empty (high empty) → degraded=True."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    engine.keyword_extractor = MagicMock()
    # high_kws empty, low_kws non-empty
    engine.keyword_extractor.extract.return_value = ([], ["detail_entity"])

    mock_store = MagicMock()
    mock_store.count_embedded.return_value = 5
    mock_store.search.return_value = []

    opts = QueryOptions(query="query", retrieval="global", top_k=3)

    with patch("iris_vector_rag.retrieval.engine.RelationEmbeddingStore", return_value=mock_store):
        result = engine._retrieve_global(opts)

    assert result["metadata"]["degraded"] is True


# ─── Mix mode contracts ───────────────────────────────────────────────────────


def test_mix_mode_registered_with_correct_prerequisites_and_fusion():
    """RetrievalMode.get_mode('mix') has prerequisites and fusion='rrf'."""
    from iris_vector_rag.retrieval.modes import get_mode

    mode = get_mode("mix")
    assert "knowledge_graph" in mode.requires
    assert "relation_embeddings" in mode.requires
    assert mode.fusion == "rrf"


def test_retrieve_engine_mix_dispatches_to_retrieve_mix():
    """RetrievalEngine.retrieve(opts) with retrieval='mix' calls _retrieve_mix."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    opts = QueryOptions(query="test", retrieval="mix", top_k=3)

    with patch.object(engine, "_retrieve_mix", return_value={"retrieved_documents": [], "metadata": {}}) as mock_m, \
         patch("iris_vector_rag.retrieval.modes.check_prerequisites"):
        engine.retrieve(opts)

    mock_m.assert_called_once_with(opts)


def test_mix_no_weights_fusion_method_is_rrf():
    """_retrieve_mix() without weights → metadata fusion_method=='rrf'."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    engine.keyword_extractor = MagicMock()
    engine.keyword_extractor.extract.return_value = (["theme"], ["entity"])
    engine.vector_store.search_by_text = MagicMock(return_value=[])

    mock_store = MagicMock()
    mock_store.count_embedded.return_value = 0
    mock_store.search.return_value = []

    opts = QueryOptions(query="test", retrieval="mix", top_k=3)

    with patch("iris_vector_rag.retrieval.engine.RelationEmbeddingStore", return_value=mock_store):
        result = engine._retrieve_mix(opts)

    assert result["metadata"]["fusion_method"] == "rrf"


def test_mix_with_weights_fusion_method_is_weighted_score():
    """_retrieve_mix() with weights → metadata fusion_method=='weighted_score'."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    engine.keyword_extractor = MagicMock()
    engine.keyword_extractor.extract.return_value = (["theme"], ["entity"])
    engine.vector_store.search_by_text = MagicMock(return_value=[])

    mock_store = MagicMock()
    mock_store.count_embedded.return_value = 0
    mock_store.search.return_value = []

    opts = QueryOptions(query="test", retrieval="mix", top_k=3, weights={"relation": 0.6, "vector": 0.4})

    with patch("iris_vector_rag.retrieval.engine.RelationEmbeddingStore", return_value=mock_store):
        result = engine._retrieve_mix(opts)

    assert result["metadata"]["fusion_method"] == "weighted_score"


def test_mix_docs_have_retrieval_source_metadata():
    """Each doc from _retrieve_mix() has metadata.retrieval_source."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.retrieval.engine import _make_doc
    from iris_vector_rag.core.query_options import QueryOptions
    from iris_vector_rag.core.models import Document

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    engine.keyword_extractor = MagicMock()
    engine.keyword_extractor.extract.return_value = (["theme"], ["entity"])

    doc_a = Document(id="a", page_content="doc a", metadata={})
    doc_b = Document(id="b", page_content="doc b", metadata={})
    engine.vector_store.search_by_text = MagicMock(return_value=[doc_a])

    mock_store = MagicMock()
    mock_store.count_embedded.return_value = 1
    mock_store.search.return_value = [{"relationship_id": "r1", "source_entity_id": "e1",
                                       "target_entity_id": "e2", "relationship_type": "CAUSES",
                                       "score": 0.8}]

    opts = QueryOptions(query="test", retrieval="mix", top_k=5)

    with patch("iris_vector_rag.retrieval.engine.RelationEmbeddingStore", return_value=mock_store):
        result = engine._retrieve_mix(opts)

    for doc in result["retrieved_documents"]:
        assert "retrieval_source" in doc.metadata


def test_mix_basic_pipeline_no_kg_raises_prerequisite_error():
    """pipeline.query with retrieval='mix' on basic pipeline (no KG) raises RetrievalPrerequisiteError."""
    from iris_vector_rag.retrieval.modes import check_prerequisites, RetrievalPrerequisiteError
    import pytest

    with patch(
        "iris_vector_rag.retrieval.modes._check_knowledge_graph_available",
        return_value=False,
    ):
        with pytest.raises(RetrievalPrerequisiteError) as exc_info:
            check_prerequisites("mix")
        assert exc_info.value.missing == "knowledge_graph"
