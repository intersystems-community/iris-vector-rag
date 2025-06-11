#!/usr/bin/env python3
"""
Debug script to investigate RAGAS context extraction issue.

This script will:
1. Run a single ColBERT query
2. Examine the exact structure of returned data
3. Test the context extraction logic
4. Identify why contexts are showing as numeric IDs instead of content
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required components
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.pipelines.registry import PipelineRegistry

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_colbert_pipeline_output():
    """Debug what the ColBERT pipeline actually returns."""
    
    # Initialize managers
    connection_manager = ConnectionManager()
    config_manager = ConfigurationManager()
    
    # Create LLM function for pipelines
    def create_llm_function():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1024
        )
        return lambda prompt: llm.invoke(prompt).content

    llm_func = create_llm_function()
    
    # Create embedding function
    from langchain_openai import OpenAIEmbeddings
    embedding_func = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store
    from iris_rag.storage.vector_store_iris import IRISVectorStore
    vector_store = IRISVectorStore(connection_manager, config_manager)
    
    # Setup framework dependencies
    framework_dependencies = {
        "connection_manager": connection_manager,
        "config_manager": config_manager,
        "llm_func": llm_func,
        "vector_store": vector_store
    }
    
    # Initialize dynamic loading services
    config_service = PipelineConfigService()
    module_loader = ModuleLoader()
    pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
    pipeline_registry = PipelineRegistry(pipeline_factory)
    
    # Register pipelines
    pipeline_registry.register_pipelines()
    
    # Get ColBERT pipeline
    colbert_pipeline = pipeline_registry.get_pipeline("ColBERTRAG")
    
    if not colbert_pipeline:
        logger.error("ColBERT pipeline not found!")
        return
    
    # Test query
    test_query = "What are the effects of metformin on type 2 diabetes?"
    
    logger.info(f"Testing ColBERT pipeline with query: {test_query}")
    
    # Execute pipeline
    try:
        result = colbert_pipeline.execute(test_query)
        
        logger.info("=== PIPELINE RESULT STRUCTURE ===")
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            # Check retrieved_documents
            retrieved_docs = result.get('retrieved_documents', [])
            logger.info(f"Retrieved documents count: {len(retrieved_docs)}")
            logger.info(f"Retrieved documents type: {type(retrieved_docs)}")
            
            if retrieved_docs:
                logger.info("=== FIRST RETRIEVED DOCUMENT ===")
                first_doc = retrieved_docs[0]
                logger.info(f"Document type: {type(first_doc)}")
                logger.info(f"Document attributes: {dir(first_doc) if hasattr(first_doc, '__dict__') else 'No attributes'}")
                
                if hasattr(first_doc, 'page_content'):
                    logger.info(f"Page content type: {type(first_doc.page_content)}")
                    logger.info(f"Page content length: {len(first_doc.page_content) if first_doc.page_content else 0}")
                    logger.info(f"Page content preview: {first_doc.page_content[:200] if first_doc.page_content else 'None'}...")
                
                if hasattr(first_doc, 'metadata'):
                    logger.info(f"Metadata: {first_doc.metadata}")
                
                if hasattr(first_doc, 'id'):
                    logger.info(f"Document ID: {first_doc.id}")
            
            # Test context extraction logic from RAGAS evaluation script
            logger.info("=== TESTING CONTEXT EXTRACTION LOGIC ===")
            contexts = result.get('retrieved_documents', result.get('contexts', []))
            context_strings = []
            
            if contexts:
                for i, ctx in enumerate(contexts):
                    logger.info(f"Context {i}: type={type(ctx)}")
                    
                    if hasattr(ctx, 'page_content'):
                        logger.info(f"  Has page_content: {type(ctx.page_content)}")
                        logger.info(f"  Page content preview: {ctx.page_content[:100] if ctx.page_content else 'None'}...")
                        context_strings.append(ctx.page_content)
                    elif isinstance(ctx, dict):
                        logger.info(f"  Is dict with keys: {list(ctx.keys())}")
                        content_val = ctx.get('content', ctx.get('text', ctx.get('page_content', str(ctx))))
                        logger.info(f"  Extracted content preview: {str(content_val)[:100]}...")
                        context_strings.append(str(content_val))
                    else:
                        logger.info(f"  Fallback to str: {str(ctx)[:100]}...")
                        context_strings.append(str(ctx))
            
            logger.info(f"Final context strings count: {len(context_strings)}")
            for i, ctx_str in enumerate(context_strings[:2]):  # Show first 2
                logger.info(f"Context string {i} preview: {ctx_str[:200]}...")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_colbert_pipeline_output()