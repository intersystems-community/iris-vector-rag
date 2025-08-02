"""
Comprehensive validation script for all RAG pipelines
Tests functionality, performance, and consistency
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import traceback

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.utils import get_embedding_func, get_llm_func # Updated import
from common.simplified_connection_manager import get_simplified_connection_manager # Updated import

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test queries for validation
TEST_QUERIES = [
    "What are the symptoms of diabetes?",
    "How is COVID-19 transmitted?",
    "What are the treatment options for hypertension?",
    "Explain the mechanism of action of statins",
    "What are the risk factors for cardiovascular disease?"
]

class PipelineValidator:
    """Validates RAG pipeline functionality and performance"""
    
    def __init__(self):
        """Initialize validator with common components"""
        self.embedding_func = get_embedding_func()
        self.llm_func = get_llm_func()
        self.connection_manager = get_simplified_connection_manager()
        self.results = {}
    
    def validate_pipeline(self, pipeline_name: str, pipeline_module: str) -> Dict[str, Any]:
        """
        Validate a single pipeline
        
        Args:
            pipeline_name: Name of the pipeline
            pipeline_module: Module path (e.g., 'basic_rag.pipeline')
            
        Returns:
            Validation results
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating {pipeline_name}")
        logger.info(f"{'='*50}")
        
        results = {
            "pipeline": pipeline_name,
            "status": "unknown",
            "import_success": False,
            "initialization_success": False,
            "query_results": [],
            "average_time": 0,
            "errors": []
        }
        
        try:
            # Import pipeline
            module = __import__(pipeline_module, fromlist=[''])
            
            # Find the pipeline class
            pipeline_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    attr_name.endswith('Pipeline') and 
                    attr_name != 'BaseRAGPipeline'):
                    pipeline_class = attr
                    break
            
            if not pipeline_class:
                raise ValueError(f"No pipeline class found in {pipeline_module}")
            
            results["import_success"] = True
            logger.info(f"✅ Import successful: {pipeline_class.__name__}")
            
            # Initialize pipeline
            if pipeline_name == "BasicRAG" and hasattr(module, 'BasicRAGPipeline'): # Changed "basic_rag" to "BasicRAG" to match key
                # Use the refactored version if available
                from iris_rag.pipelines.basic_refactored import BasicRAGPipeline # Updated import
                pipeline = BasicRAGPipeline(
                    embedding_func=self.embedding_func,
                    llm_func=self.llm_func
                )
            else:
                # Try different initialization patterns
                try:
                    # Pattern 1: With connection manager
                    pipeline = pipeline_class(
                        connection_manager=self.connection_manager,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                except:
                    try:
                        # Pattern 2: With iris_connector
                        from common.iris_connector import get_iris_connection # Updated import
                        pipeline = pipeline_class(
                            iris_connector=get_iris_connection(),
                            embedding_func=self.embedding_func,
                            llm_func=self.llm_func
                        )
                    except:
                        # Pattern 3: With individual functions
                        pipeline = pipeline_class(
                            embedding_func=self.embedding_func,
                            llm_func=self.llm_func
                        )
            
            results["initialization_success"] = True
            logger.info("✅ Initialization successful")
            
            # Test queries
            total_time = 0
            successful_queries = 0
            
            for i, query in enumerate(TEST_QUERIES[:3]):  # Test first 3 queries
                logger.info(f"\n📊 Testing query {i+1}: {query[:50]}...")
                
                query_result = {
                    "query": query,
                    "success": False,
                    "time": 0,
                    "num_documents": 0,
                    "answer_preview": "",
                    "error": None
                }
                
                try:
                    start_time = time.time()
                    result = pipeline.query(query, top_k=3)
                    elapsed_time = time.time() - start_time
                    
                    query_result["success"] = True
                    query_result["time"] = elapsed_time
                    query_result["num_documents"] = len(result.get("retrieved_documents", []))
                    query_result["answer_preview"] = result.get("answer", "")[:100] + "..."
                    
                    total_time += elapsed_time
                    successful_queries += 1
                    
                    logger.info(f"   ✅ Success - {elapsed_time:.2f}s, {query_result['num_documents']} docs")
                    
                except Exception as e:
                    query_result["error"] = str(e)
                    logger.error(f"   ❌ Failed: {e}")
                
                results["query_results"].append(query_result)
            
            # Calculate average time
            if successful_queries > 0:
                results["average_time"] = total_time / successful_queries
                results["status"] = "success" if successful_queries == len(TEST_QUERIES[:3]) else "partial"
            else:
                results["status"] = "failed"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            logger.error(f"❌ Pipeline validation failed: {e}")
            logger.debug(traceback.format_exc())
        
        return results
    
    def validate_all_pipelines(self) -> Dict[str, Any]:
        """Validate all RAG pipelines"""
        
        pipelines = [
            ("BasicRAG", "iris_rag.pipelines.basic"), # Updated path
            ("CRAG", "iris_rag.pipelines.crag"), # Updated path
            ("HyDE", "iris_rag.pipelines.hyde"), # Updated path
            ("ColBERT", "iris_rag.pipelines.colbert"), # Updated path (assuming ColBERTRAGPipeline)
            ("NodeRAG", "iris_rag.pipelines.noderag"), # Updated path
            ("GraphRAG", "iris_rag.pipelines.graphrag"), # Updated path
            ("Hybrid iFIND", "iris_rag.pipelines.hybrid_ifind") # Updated path
        ]
        
        all_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "pipelines": {},
            "summary": {
                "total": len(pipelines),
                "successful": 0,
                "partial": 0,
                "failed": 0
            }
        }
        
        for name, module in pipelines:
            result = self.validate_pipeline(name, module)
            all_results["pipelines"][name] = result
            
            # Update summary
            if result["status"] == "success":
                all_results["summary"]["successful"] += 1
            elif result["status"] == "partial":
                all_results["summary"]["partial"] += 1
            else:
                all_results["summary"]["failed"] += 1
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a validation report"""
        
        report = []
        report.append("# RAG Pipeline Validation Report")
        report.append(f"\nGenerated: {results['validation_timestamp']}")
        report.append("\n## Summary")
        
        summary = results['summary']
        report.append(f"- Total Pipelines: {summary['total']}")
        report.append(f"- ✅ Successful: {summary['successful']}")
        report.append(f"- ⚠️  Partial: {summary['partial']}")
        report.append(f"- ❌ Failed: {summary['failed']}")
        
        report.append("\n## Pipeline Details")
        
        for pipeline_name, pipeline_results in results['pipelines'].items():
            status_icon = {
                "success": "✅",
                "partial": "⚠️",
                "failed": "❌",
                "unknown": "❓"
            }[pipeline_results['status']]
            
            report.append(f"\n### {status_icon} {pipeline_name}")
            report.append(f"- Import: {'✅' if pipeline_results['import_success'] else '❌'}")
            report.append(f"- Initialization: {'✅' if pipeline_results['initialization_success'] else '❌'}")
            
            if pipeline_results['query_results']:
                report.append(f"- Average Query Time: {pipeline_results['average_time']:.2f}s")
                report.append("- Query Results:")
                
                for qr in pipeline_results['query_results']:
                    if qr['success']:
                        report.append(f"  - {qr['query'][:30]}... - {qr['time']:.2f}s, {qr['num_documents']} docs")
                    else:
                        report.append(f"  - {qr['query'][:30]}... - Failed: {qr['error']}")
            
            if pipeline_results['errors']:
                report.append("- Errors:")
                for error in pipeline_results['errors']:
                    report.append(f"  - {error}")
        
        return "\n".join(report)

def main():
    """Main validation entry point"""
    logger.info("🚀 Starting RAG Pipeline Validation")
    logger.info("=" * 50)
    
    # Create validator
    validator = PipelineValidator()
    
    # Run validation
    results = validator.validate_all_pipelines()
    
    # Generate report
    report = validator.generate_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    with open(f"validation_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save markdown report
    with open(f"validation_report_{timestamp}.md", 'w') as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE")
    print("=" * 50)
    print(f"\nResults saved to:")
    print(f"  - validation_results_{timestamp}.json")
    print(f"  - validation_report_{timestamp}.md")
    
    print(f"\nSummary:")
    print(f"  - Total: {results['summary']['total']}")
    print(f"  - Successful: {results['summary']['successful']}")
    print(f"  - Partial: {results['summary']['partial']}")
    print(f"  - Failed: {results['summary']['failed']}")
    
    # Exit with appropriate code
    if results['summary']['failed'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()