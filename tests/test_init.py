import pytest

# Attempt to import, will fail if not set up correctly
try:
    from iris_rag import create_pipeline, RAGPipeline
except ImportError:
    create_pipeline = None
    RAGPipeline = None # RAGPipeline should be available from iris_rag.core via iris_rag

def test_imports_from_package_init():
    """Tests that essential components can be imported from package root."""
    assert create_pipeline is not None, "Failed to import create_pipeline from iris_rag"
    assert RAGPipeline is not None, "Failed to import RAGPipeline from iris_rag (expected from core via __init__)"

def test_create_pipeline_unknown_pipeline():
    """Tests that create_pipeline raises ValueError for an unknown pipeline type."""
    if create_pipeline is None:
        pytest.fail("create_pipeline not imported")
    
    with pytest.raises(ValueError, match="Unknown pipeline type: unknown_dummy_pipeline"):
        create_pipeline(pipeline_type="unknown_dummy_pipeline")

def test_create_pipeline_basic():
    """Tests creating a basic pipeline (will fail without proper config but should not raise ValueError)."""
    if create_pipeline is None:
        pytest.fail("create_pipeline not imported")
    
    # This should not raise ValueError for unknown pipeline type
    # It may raise other errors due to missing config, but that's expected
    try:
        create_pipeline(pipeline_type="basic")
    except ValueError as e:
        if "Unknown pipeline type" in str(e):
            pytest.fail("Basic pipeline type should be recognized")
        # Other ValueErrors (like missing config) are acceptable for this test
    except Exception:
        # Other exceptions (like missing config) are acceptable for this test
        pass

# Future test:
# def test_create_pipeline_valid_pipeline():
#     """Tests creating a known (mocked) pipeline successfully."""
#     # This will require a mock concrete pipeline implementation and registration in create_pipeline
#     pass