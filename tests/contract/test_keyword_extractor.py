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
