#!/usr/bin/env python3
"""
Test all 7 RAG techniques in the complete enterprise system
"""

import sys
import os
# Add project root to sys.path
if '.' not in sys.path:
    sys.path.insert(0, '.')
if os.path.abspath('.') not in sys.path: # ensure absolute path is also there
    sys.path.insert(0, os.path.abspath('.'))

import logging
from typing import Dict, Any
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_technique(technique_name: str, pipeline_class, iris_connector, embedding_func, llm_func) -> Dict[str, Any]:
    """Test a single RAG technique"""
    try:
        logger.info(f"Testing {technique_name}...")
        
        # Initialize pipeline
        pipeline = pipeline_class(iris_connector, embedding_func, llm_func)
        
        # Test query
        test_query = "What are the main findings about cancer treatment?"
        
        # Run pipeline
        result = pipeline.run(test_query, top_k=3)
        
        # Validate result
        if not isinstance(result, dict):
            return {"status": "error", "message": "Result is not a dictionary"}
        
        if "answer" not in result:
            return {"status": "error", "message": "No answer in result"}
        
        if not result["answer"] or len(result["answer"].strip()) < 10:
            return {"status": "error", "message": "Answer too short or empty"}
        
        return {
            "status": "success",
            "answer_length": len(result["answer"]),
            "num_documents": len(result.get("retrieved_documents", [])),
            "metadata": result.get("metadata", {})
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def test_all_rag_techniques():
    """Test all 7 RAG techniques"""
    logger.info("Starting comprehensive RAG system test...")
    
    # Connect to database
    iris = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Define all 7 techniques
    techniques = [
        {
            "name": "BasicRAG V2",
            "module": "basic_rag.pipeline_v2",
            "class": "BasicRAGPipelineV2"
        },
        {
            "name": "CRAG V2", 
            "module": "crag.pipeline_v2",
            "class": "CRAGPipelineV2"
        },
        {
            "name": "ColBERT V2",
            "module": "colbert.pipeline_v2", 
            "class": "ColBERTPipelineV2"
        },
        {
            "name": "NodeRAG V2",
            "module": "noderag.pipeline_v2",
            "class": "NodeRAGPipelineV2"
        },
        {
            "name": "HyDE",
            "module": "hyde.pipeline",
            "class": "HyDEPipeline"
        },
        {
            "name": "Hybrid iFindRAG",
            "module": "hybrid_ifind_rag.pipeline",
            "class": "HybridiFindRAGPipeline"
        },
        {
            "name": "GraphRAG V2",
            "module": "graphrag.pipeline_v2",
            "class": "GraphRAGPipelineV2"
        }
    ]
    
    results = {}
    
    for technique in techniques:
        try:
            # Import the module and class
            module = __import__(technique["module"], fromlist=[technique["class"]])
            pipeline_class = getattr(module, technique["class"])
            
            # Test the technique
            result = test_technique(
                technique["name"], 
                pipeline_class, 
                iris, 
                embedding_func, 
                llm_func
            )
            
            results[technique["name"]] = result
            
            if result["status"] == "success":
                logger.info(f"✅ {technique['name']}: SUCCESS - Answer length: {result['answer_length']}, Documents: {result['num_documents']}")
            else:
                logger.error(f"❌ {technique['name']}: FAILED - {result['message']}")
                
        except Exception as e:
            results[technique["name"]] = {"status": "error", "message": f"Import/setup error: {str(e)}"}
            logger.error(f"❌ {technique['name']}: IMPORT ERROR - {str(e)}")
    
    # Summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    total = len(results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ENTERPRISE RAG SYSTEM TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Successful techniques: {successful}/{total}")
    logger.info(f"Success rate: {successful/total*100:.1f}%")
    
    # Detailed results
    for name, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        logger.info(f"{status_icon} {name}: {result['status'].upper()}")
        if result["status"] == "error":
            logger.info(f"   Error: {result['message']}")
    
    # Check data completeness
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA COMPLETENESS CHECK")
    logger.info(f"{'='*60}")
    
    cursor = iris.cursor()
    
    # Check all tables
    tables_to_check = [
        "SourceDocuments",
        "DocumentChunks", 
        "DocumentTokenEmbeddings",
        "Entities",
        "KnowledgeGraphNodes",
        "KnowledgeGraphEdges"
    ]
    
    for table in tables_to_check:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table}")
            count = cursor.fetchone()[0]
            logger.info(f"✅ {table}: {count:,} records")
        except Exception as e:
            logger.error(f"❌ {table}: Error - {e}")
    
    cursor.close()
    iris.close()
    
    return results

if __name__ == "__main__":
    test_all_rag_techniques()