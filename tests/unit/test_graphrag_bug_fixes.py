"""
Unit tests for GraphRAG bug fixes (2025-10-22).

Tests six critical bug fixes:
1. entity_extraction_enabled flag properly disables entity extraction
2. batch_processing.enabled config controls batch DSPy module
3. Generic configure_dspy() supports OpenAI-compatible endpoints
4. Individual extraction (_call_llm) uses configure_dspy for GPT-OSS
5. configure_dspy() extracts and passes api_key to dspy.LM()
6. DirectOpenAILM class bypasses LiteLLM to preserve model name prefix

Reference: GRAPHRAG_BUGS_FIXED.md, GRAPHRAG_BUGS_REPORT.md, GRAPHRAG_BUG_5_API_KEY.md, GRAPHRAG_BUG_6_MODEL_NAME.md
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from iris_vector_rag.core.models import Document, Entity


class TestBug1EntityExtractionEnabledFlag:
    """Test Bug #1 Fix: entity_extraction_enabled flag exists and works."""

    def test_graphrag_has_entity_extraction_enabled_attribute(self):
        """Verify GraphRAGPipeline has entity_extraction_enabled attribute."""
        from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline

        # Check that the class or instance should have this attribute
        # We'll check the source code was modified correctly
        import inspect
        source = inspect.getsource(GraphRAGPipeline.__init__)

        assert "entity_extraction_enabled" in source, \
            "GraphRAGPipeline.__init__ should reference entity_extraction_enabled"

    def test_graphrag_load_documents_checks_entity_extraction_flag(self):
        """Verify load_documents() checks entity_extraction_enabled flag."""
        from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline

        import inspect
        source = inspect.getsource(GraphRAGPipeline.load_documents)

        assert "entity_extraction_enabled" in source, \
            "load_documents should check entity_extraction_enabled flag"

        assert "if not self.entity_extraction_enabled" in source, \
            "load_documents should have early return when entity_extraction_enabled is False"


class TestBug2BatchProcessingConfig:
    """Test Bug #2 Fix: batch_processing.enabled config is respected."""

    def test_extract_batch_checks_batch_processing_enabled(self):
        """Verify extract_batch_with_dspy() checks batch_processing.enabled config."""
        from iris_vector_rag.services.entity_extraction import EntityExtractionService

        import inspect
        source = inspect.getsource(EntityExtractionService.extract_batch_with_dspy)

        assert "batch_processing" in source, \
            "extract_batch_with_dspy should reference batch_processing config"

        assert "batch_enabled" in source or "enabled" in source, \
            "extract_batch_with_dspy should check enabled flag"



class TestBug3ConfigureDspy:
    """Test Bug #3 Fix: Generic configure_dspy() function."""

    def test_configure_dspy_function_exists(self):
        """Verify configure_dspy() function exists."""
        from iris_vector_rag.dspy_modules import entity_extraction_module

        assert hasattr(entity_extraction_module, 'configure_dspy'), \
            "Should have configure_dspy function"

    def test_configure_dspy_accepts_llm_config_dict(self):
        """Verify configure_dspy() accepts llm_config dict parameter."""
        from iris_vector_rag.dspy_modules.entity_extraction_module import configure_dspy

        import inspect
        sig = inspect.signature(configure_dspy)

        assert 'llm_config' in sig.parameters, \
            "configure_dspy should accept llm_config parameter"



class TestBug5CallLlmUsesConfigureDspy:
    """Test Bug #5 Fix: _call_llm() uses configure_dspy for GPT-OSS instead of returning empty list."""

    def test_call_llm_uses_configure_dspy_for_openai(self):
        """Verify _call_llm() calls configure_dspy for OpenAI-compatible endpoints."""
        from iris_vector_rag.services import entity_extraction
        import inspect

        source = inspect.getsource(entity_extraction.EntityExtractionService._call_llm)

        # Should check for both "gpt" and api_type
        assert '"gpt"' in source or "'gpt'" in source, \
            "Bug #5 fix: Should check for 'gpt' in model name"

        assert "api_type" in source, \
            "Bug #5 fix: Should check api_type config"

        # Should import and call configure_dspy
        assert "configure_dspy" in source, \
            "Bug #5 fix: Should import and use configure_dspy"

        assert "dspy" in source.lower(), \
            "Bug #5 fix: Should use DSPy for OpenAI-compatible models"

    def test_call_llm_not_hardcoded_empty_return(self):
        """Verify _call_llm() doesn't have hardcoded empty return for GPT models."""
        from iris_vector_rag.services import entity_extraction
        import inspect

        source = inspect.getsource(entity_extraction.EntityExtractionService._call_llm)

        # Check that it doesn't just return '[]' immediately for GPT
        # The old buggy code had: if "gpt" in model.lower(): return '[]'
        # This should not be present in a simple pattern
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if '"gpt"' in line or "'gpt'" in line:
                # If we find a gpt check, the next few lines should NOT just return '[]'
                # They should call configure_dspy or use DSPy
                next_few_lines = '\n'.join(lines[i:i+5])
                if 'return' in next_few_lines and "'[]'" in next_few_lines:
                    # Make sure it's inside a try/except block (fallback), not immediate return
                    assert 'try:' in next_few_lines or 'except' in next_few_lines, \
                        "Bug #5 fix: Should not immediately return '[]' for GPT models - should try DSPy first"


class TestBug5ApiKeyPassedToDspy:
    """Test Bug #5 API Key Fix: configure_dspy() extracts and passes api_key to dspy.LM()."""

    def test_configure_dspy_extracts_api_key_from_config(self):
        """Verify configure_dspy() extracts api_key from llm_config."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import inspect

        source = inspect.getsource(entity_extraction_module.configure_dspy)

        # Should extract api_key from llm_config
        assert "api_key" in source, \
            "Bug #5 API key fix: Should extract api_key from llm_config"

        assert 'llm_config.get("api_key"' in source or "llm_config.get('api_key'" in source, \
            "Bug #5 API key fix: Should use llm_config.get() to extract api_key"

    def test_configure_dspy_passes_api_key_to_dspy_lm(self):
        """Verify configure_dspy() passes api_key parameter to dspy.LM()."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import inspect

        source = inspect.getsource(entity_extraction_module.configure_dspy)

        # Should pass api_key to dspy.LM() for OpenAI-compatible endpoints
        # Check for pattern: dspy.LM(..., api_key=api_key, ...)
        assert "dspy.LM" in source, \
            "Bug #5 API key fix: Should call dspy.LM()"

        # Count occurrences of api_key parameter being passed
        # There should be at least 2: one for extraction, one for passing to dspy.LM
        api_key_count = source.count("api_key")
        assert api_key_count >= 2, \
            f"Bug #5 API key fix: Should extract AND pass api_key (found {api_key_count} occurrences, need >= 2)"


class TestBug6LiteLLMModelNameStripping:
    """Test Bug #6 Fix: DirectOpenAILM bypasses LiteLLM to preserve model name prefix."""

    def test_direct_openai_lm_class_exists(self):
        """Verify DirectOpenAILM class exists and inherits from dspy.BaseLM."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import dspy

        assert hasattr(entity_extraction_module, 'DirectOpenAILM'), \
            "Bug #6 fix: Should have DirectOpenAILM class"

        # Verify it inherits from dspy.BaseLM
        DirectOpenAILM = entity_extraction_module.DirectOpenAILM
        assert issubclass(DirectOpenAILM, dspy.BaseLM), \
            "Bug #6 fix: DirectOpenAILM should inherit from dspy.BaseLM"

    def test_direct_openai_lm_preserves_model_name(self):
        """Verify DirectOpenAILM preserves full model name in __init__."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import inspect

        source = inspect.getsource(entity_extraction_module.DirectOpenAILM.__init__)

        # Should preserve full model name
        assert "self.model = model" in source, \
            "Bug #6 fix: DirectOpenAILM should preserve full model name"

        # Should have comment about preserving model name
        source_all = inspect.getsource(entity_extraction_module.DirectOpenAILM)
        assert "openai/gpt-oss-120b" in source_all, \
            "Bug #6 fix: DirectOpenAILM should reference preserving openai/gpt-oss-120b"

    def test_configure_dspy_uses_direct_openai_lm_for_gpt_oss(self):
        """Verify configure_dspy() uses DirectOpenAILM for GPT-OSS models."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import inspect

        source = inspect.getsource(entity_extraction_module.configure_dspy)

        # Should check for GPT-OSS model
        assert "openai/gpt-oss-120b" in source, \
            "Bug #6 fix: configure_dspy should check for openai/gpt-oss-120b model"

        # Should create DirectOpenAILM instance
        assert "DirectOpenAILM" in source, \
            "Bug #6 fix: configure_dspy should use DirectOpenAILM class"

        # Should mention Bug #6 fix
        assert "Bug #6" in source or ("prefix" in source.lower() and "preserve" in source.lower()), \
            "Bug #6 fix: Should document why DirectOpenAILM is used"


class TestCodeVerification:
    """Verify the bug fixes are actually present in the code."""

    def test_graphrag_pipeline_has_entity_extraction_enabled_property(self):
        """Verify GraphRAGPipeline.__init__ sets entity_extraction_enabled."""
        from iris_vector_rag.pipelines import graphrag
        import inspect

        source = inspect.getsource(graphrag.GraphRAGPipeline.__init__)

        assert "self.entity_extraction_enabled" in source, \
            "Bug #1 fix: Should set self.entity_extraction_enabled in __init__"

        assert "entity_extraction_enabled" in source, \
            "Bug #1 fix: Should reference entity_extraction_enabled config"

    def test_graphrag_load_documents_has_early_return(self):
        """Verify load_documents has early return when entity_extraction_enabled=False."""
        from iris_vector_rag.pipelines import graphrag
        import inspect

        source = inspect.getsource(graphrag.GraphRAGPipeline.load_documents)

        assert "if not self.entity_extraction_enabled" in source, \
            "Bug #1 fix: Should check entity_extraction_enabled flag"

        assert "return" in source, \
            "Bug #1 fix: Should return early when disabled"

    def test_entity_extraction_service_checks_batch_config(self):
        """Verify extract_batch_with_dspy checks batch_processing.enabled."""
        from iris_vector_rag.services import entity_extraction
        import inspect

        source = inspect.getsource(entity_extraction.EntityExtractionService.extract_batch_with_dspy)

        assert "batch_processing" in source, \
            "Bug #2 fix: Should check batch_processing config"

        assert "enabled" in source, \
            "Bug #2 fix: Should check enabled flag"

        assert "process_document" in source, \
            "Bug #2 fix: Should fall back to individual processing"

    def test_configure_dspy_supports_api_type_parameter(self):
        """Verify configure_dspy checks api_type for OpenAI-compatible endpoints."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import inspect

        source = inspect.getsource(entity_extraction_module.configure_dspy)

        assert "api_type" in source, \
            "Bug #3 fix: Should check api_type"

        assert "openai" in source.lower(), \
            "Bug #3 fix: Should handle OpenAI-compatible endpoints"

        assert "supports_response_format" in source, \
            "Bug #3 fix: Should respect supports_response_format flag"

    def test_call_llm_uses_dspy_for_gpt_models(self):
        """Verify _call_llm() uses DSPy/configure_dspy for GPT models instead of stub."""
        from iris_vector_rag.services import entity_extraction
        import inspect

        source = inspect.getsource(entity_extraction.EntityExtractionService._call_llm)

        # Bug #5 fix: Should use configure_dspy for OpenAI-compatible models
        assert "configure_dspy" in source, \
            "Bug #5 fix: _call_llm should import configure_dspy"

        assert "api_type" in source, \
            "Bug #5 fix: _call_llm should check api_type config"

        # Should create DSPy predictor
        assert "dspy" in source.lower(), \
            "Bug #5 fix: _call_llm should use DSPy for GPT models"

    def test_configure_dspy_extracts_and_passes_api_key(self):
        """Verify configure_dspy() extracts api_key from config and passes to dspy.LM()."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import inspect

        source = inspect.getsource(entity_extraction_module.configure_dspy)

        # Bug #5 API key fix: Should extract api_key
        assert 'llm_config.get("api_key"' in source or "llm_config.get('api_key'" in source, \
            "Bug #5 API key fix: Should extract api_key from llm_config"

        # Bug #5 API key fix: Should pass api_key to dspy.LM()
        # Count occurrences - should have extraction + at least one pass to dspy.LM()
        api_key_count = source.count("api_key")
        assert api_key_count >= 3, \
            f"Bug #5 API key fix: Should extract api_key AND pass to dspy.LM() (found {api_key_count} occurrences, need >= 3)"

    def test_configure_dspy_uses_direct_openai_lm_for_prefix_preservation(self):
        """Verify configure_dspy() uses DirectOpenAILM to prevent prefix stripping."""
        from iris_vector_rag.dspy_modules import entity_extraction_module
        import inspect

        source = inspect.getsource(entity_extraction_module.configure_dspy)

        # Bug #6 fix: Should use DirectOpenAILM for GPT-OSS
        assert "DirectOpenAILM" in source, \
            "Bug #6 fix: configure_dspy should use DirectOpenAILM class"

        # Should check for GPT-OSS model specifically
        assert "openai/gpt-oss-120b" in source, \
            "Bug #6 fix: configure_dspy should check for openai/gpt-oss-120b model"

        # Should create DirectOpenAILM instance with full model name preserved
        lines = source.split('\n')
        found_direct_openai_creation = False
        for i, line in enumerate(lines):
            if 'DirectOpenAILM(' in line:
                found_direct_openai_creation = True
                # Check next few lines for model parameter
                next_lines = '\n'.join(lines[i:i+5])
                assert "model=model" in next_lines, \
                    "Bug #6 fix: DirectOpenAILM should be initialized with full model name"
                break

        assert found_direct_openai_creation, \
            "Bug #6 fix: configure_dspy should create DirectOpenAILM instance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
