"""
Dimension utilities for backwards compatibility and LangChain integration.

This module provides helper functions that components can use to get vector dimensions
without importing the schema manager directly, maintaining simplicity for LangChain developers.
"""

import logging

logger = logging.getLogger(__name__)

# Global cache for dimension lookups to avoid repeated schema manager initialization
_dimension_cache = {}

def get_vector_dimension(table_name: str = "SourceDocuments", 
                        connection_manager=None, 
                        config_manager=None,
                        model_name: str = None) -> int:
    """
    Simple helper to get vector dimension for any table.
    
    This provides a lightweight API that doesn't require components to import
    or understand the schema manager directly.
    
    Args:
        table_name: Name of the table (SourceDocuments, DocumentTokenEmbeddings, etc.)
        connection_manager: Optional connection manager
        config_manager: Optional config manager
        model_name: Optional specific model name override
        
    Returns:
        Vector dimension for the table
    """
    cache_key = f"{table_name}:{model_name or 'default'}"
    
    # Return cached value if available
    if cache_key in _dimension_cache:
        return _dimension_cache[cache_key]
    
    # If managers are provided, use schema manager
    if connection_manager and config_manager:
        try:
            from iris_rag.storage.schema_manager import SchemaManager
            schema_manager = SchemaManager(connection_manager, config_manager)
            dimension = schema_manager.get_vector_dimension(table_name, model_name)
            _dimension_cache[cache_key] = dimension
            return dimension
        except Exception as e:
            logger.warning(f"Failed to get dimension from schema manager: {e}")
    
    # HARD FAIL if schema manager not available and no managers provided
    error_msg = f"CRITICAL: Cannot get dimension for {table_name} - schema manager required"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

def clear_dimension_cache():
    """Clear the dimension cache to force fresh lookups."""
    global _dimension_cache
    _dimension_cache.clear()
    logger.debug("Dimension cache cleared")