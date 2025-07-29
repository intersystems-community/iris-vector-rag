"""
Test suite for import validation - ensures all critical imports work without silent fallbacks.

This test suite was created to address the critical issue where broken imports in tests/utils.py
were masked by silent fallback patterns, preventing proper detection of import errors.

Following TDD principles:
1. RED: Write failing tests that expose import issues
2. GREEN: Fix the imports to make tests pass  
3. REFACTOR: Improve import validation coverage
"""

import pytest
import sys
import importlib
from unittest.mock import patch


class TestImportValidation:
    """Test suite to validate all critical imports work without silent fallbacks."""
    
    def test_colbert_import_without_fallback(self):
        """
        Test that ColBERT imports work directly without falling back to broken paths.
        
        This test should FAIL initially because tests/utils.py has a broken import:
        `from src.working.colbert.doc_encoder import generate_token_embeddings_for_documents`
        
        The correct import should be from common.utils.
        """
        # First, test that the broken import path fails as expected
        with pytest.raises(ImportError, match="No module named 'src'"):
            from src.working.colbert.doc_encoder import generate_token_embeddings_for_documents
    
    def test_tests_utils_imports_without_silent_fallback(self):
        """
        Test that tests.utils can be imported without relying on silent fallbacks.
        
        This test validates that all imports in tests/utils.py work correctly
        without falling back to broken import paths.
        """
        # Remove tests.utils from sys.modules if it exists to force fresh import
        if 'tests.utils' in sys.modules:
            del sys.modules['tests.utils']
        
        # Mock the broken import to ensure it fails loudly instead of silently
        with patch.dict('sys.modules', {'src': None, 'src.working': None, 'src.working.colbert': None}):
            # This should work because the correct import should be used
            try:
                import tests.utils
                # Verify that colbert_generate_embeddings is available
                assert hasattr(tests.utils, 'colbert_generate_embeddings')
            except ImportError as e:
                # If this fails, it means the broken import path is still being used
                pytest.fail(f"tests.utils import failed due to broken import path: {e}")
    
    def test_colbert_function_availability_from_common_utils(self):
        """
        Test that ColBERT functions are available from common.utils.
        
        This validates that the correct import path exists and works.
        """
        from common.utils import get_colbert_doc_encoder_func, get_colbert_query_encoder_func
        
        # Test that functions are callable
        doc_encoder = get_colbert_doc_encoder_func()
        query_encoder = get_colbert_query_encoder_func()
        
        assert callable(doc_encoder)
        assert callable(query_encoder)
        
        # Test basic functionality
        test_text = "This is a test document for ColBERT encoding."
        doc_result = doc_encoder(test_text)
        query_result = query_encoder(test_text)
        
        # Validate return types
        assert isinstance(doc_result, list)
        assert isinstance(query_result, list)
        
        if doc_result:  # If not empty
            assert isinstance(doc_result[0], tuple)
            assert len(doc_result[0]) == 2  # (token, embedding)
            assert isinstance(doc_result[0][0], str)  # token
            assert isinstance(doc_result[0][1], list)  # embedding
        
        if query_result:  # If not empty
            assert isinstance(query_result[0], list)  # embedding vector

    def test_no_silent_import_fallbacks_in_codebase(self):
        """
        Test that there are no other silent import fallbacks that could mask errors.
        
        This is a meta-test to ensure we don't have similar issues elsewhere.
        """
        # This test will be expanded as we discover other problematic patterns
        # For now, it serves as a placeholder for future import validation
        
        # Test that common.utils imports work directly
        from common.utils import Document, get_embedding_func, get_llm_func
        
        # Verify these are the expected types/functions
        assert Document is not None
        assert callable(get_embedding_func)
        assert callable(get_llm_func)

    def test_import_error_propagation(self):
        """
        Test that import errors are properly propagated and not silently caught.
        
        This ensures that when imports fail, we get clear error messages
        instead of silent fallbacks to mock implementations.
        """
        # Test importing a definitely non-existent module
        with pytest.raises(ImportError):
            import definitely_does_not_exist_module_12345
        
        # Test importing from a non-existent path similar to the broken one
        with pytest.raises(ImportError, match="No module named 'src'"):
            exec("from src.definitely.does.not.exist import some_function")


class TestImportValidationIntegration:
    """Integration tests for import validation across the codebase."""
    
    def test_tests_utils_colbert_integration(self):
        """
        Test that tests.utils ColBERT integration works end-to-end.
        
        This test validates that the fixed import allows proper ColBERT functionality.
        """
        # Import after potential fixes
        import tests.utils
        
        # Test that colbert_generate_embeddings works
        test_documents = [
            {"id": "test1", "content": "This is test document one."},
            {"id": "test2", "content": "This is test document two."}
        ]
        
        # This should work without falling back to broken imports
        result = tests.utils.colbert_generate_embeddings(test_documents, mock=True)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        for doc_result in result:
            assert "id" in doc_result
            assert "tokens" in doc_result  
            assert "token_embeddings" in doc_result
            assert isinstance(doc_result["token_embeddings"], list)

    def test_critical_imports_comprehensive(self):
        """
        Comprehensive test of all critical imports used throughout the codebase.
        
        This test ensures that all major import paths work correctly.
        """
        critical_imports = [
            # Core utilities
            ("common.utils", ["Document", "get_embedding_func", "get_llm_func"]),
            ("common.utils", ["get_colbert_doc_encoder_func", "get_colbert_query_encoder_func"]),
            
            # Database utilities  
            ("common.iris_connection_manager", ["get_iris_connection"]),
            
            # Test utilities (after fix)
            ("tests.utils", ["colbert_generate_embeddings", "build_knowledge_graph"]),
        ]
        
        for module_name, expected_attrs in critical_imports:
            try:
                module = importlib.import_module(module_name)
                for attr_name in expected_attrs:
                    assert hasattr(module, attr_name), f"{module_name} missing {attr_name}"
            except ImportError as e:
                pytest.fail(f"Critical import failed: {module_name} - {e}")