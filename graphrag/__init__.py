# graphrag/__init__.py

from .pipeline import FixedGraphRAGPipeline

def create_graphrag_pipeline(iris_connector=None, llm_func=None):
    """
    Factory function to create a fixed GraphRAG pipeline that properly uses the knowledge graph.
    """
    from common.iris_connector import get_iris_connection
    from common.utils import get_embedding_func, get_llm_func
    
    if iris_connector is None:
        iris_connector = get_iris_connection()
    
    if llm_func is None:
        llm_func = get_llm_func()
    
    embedding_func = get_embedding_func()
    
    return FixedGraphRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )

# Backward compatibility alias
GraphRAGPipeline = FixedGraphRAGPipeline
