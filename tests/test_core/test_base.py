import pytest
import abc

def test_rag_pipeline_is_abstract_base_class():
    """Tests that RAGPipeline is an abstract base class."""
    try:
        from iris_rag.core.base import RAGPipeline
        assert issubclass(RAGPipeline, abc.ABC), "RAGPipeline should be an ABC."
    except ImportError:
        pytest.fail("Failed to import RAGPipeline from iris_rag.core.base")
    except AttributeError:
        pytest.fail("RAGPipeline class not found in iris_rag.core.base")

def test_rag_pipeline_has_abstract_methods():
    """Tests that RAGPipeline has the required abstract methods."""
    # This test will initially fail because RAGPipeline and its methods don't exist.
    # We'll create a placeholder that will still fail until abstract methods are defined.
    class ConcretePipeline: # Placeholder, not used for actual test logic once RAGPipeline exists
        pass

    try:
        from iris_rag.core.base import RAGPipeline
        # Check for abstract methods
        expected_abstract_methods = {'execute', 'load_documents', 'query'}
        actual_abstract_methods = RAGPipeline.__abstractmethods__
        assert actual_abstract_methods == expected_abstract_methods, \
            f"RAGPipeline abstract methods mismatch. Expected: {expected_abstract_methods}, Got: {actual_abstract_methods}"

        # Attempt to instantiate a concrete subclass without implementing abstract methods
        # This should raise a TypeError if RAGPipeline is correctly defined as an ABC
        # with those abstract methods.
        class IncompletePipeline(RAGPipeline):
            # No implementation of abstract methods
            # No implementation of abstract methods
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class IncompletePipeline without an implementation for abstract methods 'execute', 'load_documents', 'query'"):
            IncompletePipeline()

    except ImportError:
        pytest.fail("Failed to import RAGPipeline from iris_rag.core.base")
    except AttributeError: # Handles case where RAGPipeline exists but __abstractmethods__ doesn't (not an ABC) or other attributes are missing
        pytest.fail("RAGPipeline is not correctly defined as an ABC or abstract methods are missing.")