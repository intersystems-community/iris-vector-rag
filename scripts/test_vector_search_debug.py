#!/usr/bin/env python3
"""
Quick debug script to test vector search
"""

import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test the vector utilities
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.storage.vector_store_iris import IRISVectorStore
from common.utils import get_llm_func

try:
    # Setup
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    llm_func = get_llm_func()
    vector_store = IRISVectorStore(connection_manager, config_manager)

    # Load a simple document
    from iris_rag.core.models import Document
    docs = [Document(id="test1", page_content="This is a test document about diabetes.")]

    logger.info("Loading test document...")
    vector_store.add_documents(docs)
    logger.info("✅ Document loaded successfully")

    # Try a search
    logger.info("Attempting vector search...")
    results = vector_store.similarity_search("diabetes", k=1)

    if results:
        logger.info(f"✅ Search successful! Found {len(results)} results")
        for i, doc in enumerate(results):
            logger.info(f"  Result {i+1}: {doc.page_content[:100]}...")
    else:
        logger.warning("❌ Search returned no results")

except Exception as e:
    logger.error(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
