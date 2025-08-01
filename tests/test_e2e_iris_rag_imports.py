def test_core_module_imports():
    """Tests imports from iris_rag.core module."""
    from iris_rag.core import base
    from iris_rag.core import connection
    from iris_rag.core import models
    assert base is not None, "iris_rag.core.base failed to import"
    assert connection is not None, "iris_rag.core.connection failed to import"
    assert models is not None, "iris_rag.core.models failed to import"

def test_embeddings_module_imports():
    """Tests imports from iris_rag.embeddings module."""
    from iris_rag.embeddings import manager as embedding_manager
    assert embedding_manager is not None, "iris_rag.embeddings.manager failed to import"

def test_pipelines_module_imports():
    """Tests imports from iris_rag.pipelines module."""
    from iris_rag.pipelines import basic as basic_pipeline
    # Assuming __init__.py might have factory functions or main classes
    import iris_rag.pipelines
    assert basic_pipeline is not None, "iris_rag.pipelines.basic failed to import"
    assert iris_rag.pipelines is not None, "iris_rag.pipelines package failed to import"

def test_storage_module_imports():
    """Tests imports from iris_rag.storage module."""
    from iris_rag.storage import enterprise_storage as iris_storage
    # Assuming __init__.py might have factory functions or main classes
    import iris_rag.storage
    assert iris_storage is not None, "iris_rag.storage.iris failed to import"
    assert iris_rag.storage is not None, "iris_rag.storage package failed to import"

def test_config_module_imports():
    """Tests imports from iris_rag.config module."""
    from iris_rag.config import manager as config_manager
    assert config_manager is not None, "iris_rag.config.manager failed to import"

def test_top_level_package_import():
    """Tests the top-level iris_rag package import."""
    import iris_rag
    assert iris_rag is not None, "iris_rag package failed to import"

def test_specific_class_function_imports():
    """Tests direct import of key classes and functions."""
    from iris_rag.core.connection import ConnectionManager
    assert ConnectionManager is not None

    from iris_rag.embeddings.manager import EmbeddingManager
    assert EmbeddingManager is not None

    from iris_rag.pipelines.basic import BasicRAGPipeline
    assert BasicRAGPipeline is not None

    from iris_rag.storage.enterprise_storage import IRISStorage
    assert IRISStorage is not None

    from iris_rag.config.manager import ConfigurationManager
    assert ConfigurationManager is not None

    from iris_rag.core.models import Document
    assert Document is not None

    # Example: Test for a factory function if one is expected at the package level
    # from iris_rag import create_pipeline
    # assert create_pipeline is not None

# To run these tests, navigate to the root directory and use:
# PYTHONPATH=. pytest tests/test_e2e_iris_rag_imports.py