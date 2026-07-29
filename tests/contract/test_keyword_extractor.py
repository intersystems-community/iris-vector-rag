"""Contract tests for KeywordExtractor — Feature 081, Phase 4."""

from unittest.mock import MagicMock


def _make_extractor(llm_response=""):
    from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor

    llm_func = MagicMock(return_value=llm_response)
    return KeywordExtractor(llm_func=llm_func, language="English"), llm_func


def test_extract_valid_json_returns_tuple_of_lists():
    """extract() with valid LLM JSON returns (high_kws, low_kws) both lists."""
    payload = '{"high_level_keywords": ["systemic risk", "financial stability"], "low_level_keywords": ["Basel III", "capital ratio"]}'
    extractor, _ = _make_extractor(payload)
    high, low = extractor.extract("What are the systemic risks?")
    assert isinstance(high, list)
    assert isinstance(low, list)
    assert "systemic risk" in high
    assert "Basel III" in low


def test_extract_malformed_json_returns_empty_no_exception():
    """extract() with malformed LLM JSON returns ([], []) without raising."""
    extractor, _ = _make_extractor("this is not json")
    high, low = extractor.extract("query")
    assert high == []
    assert low == []


def test_extract_llm_exception_returns_empty_no_exception():
    """extract() when LLM raises returns ([], []) without propagating."""
    from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor

    def boom(prompt):
        raise TimeoutError("LLM timed out")

    extractor = KeywordExtractor(llm_func=boom)
    high, low = extractor.extract("query")
    assert high == []
    assert low == []


def test_extract_markdown_fenced_json_is_stripped():
    """extract() strips ```json / ``` fences before parsing."""
    payload = '```json\n{"high_level_keywords": ["theme"], "low_level_keywords": ["detail"]}\n```'
    extractor, _ = _make_extractor(payload)
    high, low = extractor.extract("query")
    assert "theme" in high
    assert "detail" in low


def test_extraction_model_attribute():
    """extraction_model attribute reflects the model name passed at construction."""
    from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor

    extractor = KeywordExtractor(llm_func=MagicMock(), model_name="gpt-4o-mini")
    assert extractor.model_name == "gpt-4o-mini"


# ─── US3: tunability / model-routing ──────────────────────────────────────────


def test_pre_supplied_keywords_skips_extractor_call():
    """_get_or_extract_keywords uses opts fields when pre-supplied, skipping LLM."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock())
    mock_extractor = MagicMock()
    engine.keyword_extractor = mock_extractor

    opts = QueryOptions(
        query="test",
        high_level_keywords=["pre-high"],
        low_level_keywords=["pre-low"],
    )
    high, low = engine._get_or_extract_keywords(opts)

    mock_extractor.extract.assert_not_called()
    assert high == ["pre-high"]
    assert low == ["pre-low"]


def test_custom_keyword_extractor_routes_to_cheap_llm():
    """Setting engine.keyword_extractor routes to that extractor, not default."""
    from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor

    cheap_llm = MagicMock(return_value='{"high_level_keywords":["cheap"],"low_level_keywords":[]}')
    expensive_llm = MagicMock()
    extractor = KeywordExtractor(llm_func=cheap_llm, model_name="cheap-model")

    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock())
    engine.keyword_extractor = extractor

    opts = QueryOptions(query="test query")
    high, low = engine._get_or_extract_keywords(opts)

    cheap_llm.assert_called_once()
    expensive_llm.assert_not_called()
    assert "cheap" in high


def test_extraction_model_in_global_result_metadata():
    """extraction_model in global result metadata reflects configured extractor."""
    from iris_vector_rag.retrieval.engine import RetrievalEngine
    from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor
    from iris_vector_rag.core.query_options import QueryOptions

    engine = RetrievalEngine(vector_store=MagicMock(), connection=MagicMock())
    extractor = KeywordExtractor(llm_func=MagicMock(
        return_value='{"high_level_keywords":[],"low_level_keywords":[]}'
    ), model_name="gpt-4o-mini")
    engine.keyword_extractor = extractor

    mock_store = MagicMock()
    mock_store.count_embedded.return_value = 0
    mock_store.search.return_value = []

    opts = QueryOptions(query="test", retrieval="global", top_k=3)

    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "iris_vector_rag.retrieval.engine.RelationEmbeddingStore",
        return_value=mock_store,
    ):
        result = engine._retrieve_global(opts)

    assert result["metadata"]["extraction_model"] == "gpt-4o-mini"
