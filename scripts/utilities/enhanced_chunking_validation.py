"""
Enhanced Chunking System Validation Script

This script validates the enhanced chunking system at enterprise scale:
1. Tests all chunking strategies with 1000+ PMC documents
2. Integrates chunking with all 7 RAG techniques
3. Measures performance and quality metrics
4. Validates database storage and retrieval
5. Generates comprehensive performance reports
"""

import sys
import os
import json
import time
import logging
import statistics
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chunking.enhanced_chunking_service import EnhancedDocumentChunkingService # Path remains same
from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import

# Import all RAG techniques (these will likely need to change to class imports)
from iris_rag.pipelines.basic import BasicRAGPipeline # Changed to class
from iris_rag.pipelines.hyde import HyDERAGPipeline # Changed to class
from iris_rag.pipelines.crag import CRAGPipeline # Changed to class
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Changed to class
from iris_rag.pipelines.noderag import NodeRAGPipeline # Changed to class
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Changed to class
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Changed to class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_chunking_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedChunkingValidator:
    """Comprehensive validator for enhanced chunking system."""
    
    def __init__(self):
        self.embedding_model = get_embedding_model(mock=True)
        # Create a function wrapper for the model
        def embedding_func(texts):
            return self.embedding_model.embed_documents(texts)
        self.embedding_func = embedding_func
        self.chunking_service = EnhancedDocumentChunkingService(embedding_func=self.embedding_func)
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "chunking_strategies": {},
            "rag_integration": {},
            "performance_metrics": {},
            "quality_metrics": {},
            "scale_testing": {},
            "errors": []
        }
        
        # RAG techniques mapping
        self.rag_techniques = {
            "BasicRAG": BasicRAGPipeline,
            "HyDE": HyDERAGPipeline,
            "CRAG": CRAGPipeline,
            "ColBERT": ColBERTRAGPipeline,
            "NodeRAG": NodeRAGPipeline,
            "GraphRAG": GraphRAGPipeline,
            "HybridiFindRAG": HybridIFindRAGPipeline
        }
    
    def validate_chunking_strategies(self, sample_size: int = 100) -> Dict[str, Any]:
        """Validate all enhanced chunking strategies."""
        logger.info("🔍 Validating enhanced chunking strategies...")
        
        strategies = ["recursive", "semantic", "adaptive", "hybrid", "recursive_fast", "recursive_high_quality"]
        strategy_results = {}
        
        # Get sample documents
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute("""
                SELECT TOP ? doc_id, title, text_content
                FROM RAG.SourceDocuments_V2
                WHERE text_content IS NOT NULL
                AND LENGTH(text_content) BETWEEN 500 AND 5000
                ORDER BY RANDOM()
            """, (sample_size,))
            
            documents = cursor.fetchall()
            logger.info(f"Testing with {len(documents)} documents")
            
            for strategy in strategies:
                logger.info(f"Testing {strategy} strategy...")
                strategy_metrics = {
                    "documents_processed": 0,
                    "total_chunks": 0,
                    "processing_times": [],
                    "quality_scores": [],
                    "coherence_scores": [],
                    "biomedical_densities": [],
                    "token_counts": [],
                    "errors": []
                }
                
                for doc_id, title, text_content in documents[:20]:  # Test subset for detailed analysis
                    try:
                        start_time = time.time()
                        
                        # Chunk document
                        chunks = self.chunking_service.chunk_document(doc_id, text_content, strategy)
                        
                        processing_time = time.time() - start_time
                        strategy_metrics["processing_times"].append(processing_time * 1000)
                        
                        # Analyze quality
                        analysis = self.chunking_service.analyze_chunking_effectiveness(
                            doc_id, text_content, [strategy]
                        )
                        
                        if strategy in analysis["strategy_analysis"]:
                            metrics = analysis["strategy_analysis"][strategy]
                            if "error" not in metrics:
                                strategy_metrics["quality_scores"].append(metrics.get("quality_score", 0))
                                strategy_metrics["coherence_scores"].append(metrics.get("avg_semantic_coherence", 0))
                                strategy_metrics["biomedical_densities"].append(metrics.get("avg_biomedical_density", 0))
                                strategy_metrics["token_counts"].extend([chunk["chunk_metadata"] for chunk in chunks])
                        
                        strategy_metrics["documents_processed"] += 1
                        strategy_metrics["total_chunks"] += len(chunks)
                        
                    except Exception as e:
                        error_msg = f"Error processing {doc_id} with {strategy}: {e}"
                        logger.error(error_msg)
                        strategy_metrics["errors"].append(error_msg)
                
                # Calculate summary statistics
                if strategy_metrics["processing_times"]:
                    strategy_results[strategy] = {
                        "documents_processed": strategy_metrics["documents_processed"],
                        "total_chunks": strategy_metrics["total_chunks"],
                        "avg_processing_time_ms": statistics.mean(strategy_metrics["processing_times"]),
                        "avg_quality_score": statistics.mean(strategy_metrics["quality_scores"]) if strategy_metrics["quality_scores"] else 0,
                        "avg_coherence": statistics.mean(strategy_metrics["coherence_scores"]) if strategy_metrics["coherence_scores"] else 0,
                        "avg_biomedical_density": statistics.mean(strategy_metrics["biomedical_densities"]) if strategy_metrics["biomedical_densities"] else 0,
                        "chunks_per_document": strategy_metrics["total_chunks"] / max(1, strategy_metrics["documents_processed"]),
                        "error_count": len(strategy_metrics["errors"]),
                        "success_rate": (strategy_metrics["documents_processed"] - len(strategy_metrics["errors"])) / max(1, strategy_metrics["documents_processed"])
                    }
                    
                    logger.info(f"✅ {strategy}: {strategy_results[strategy]['success_rate']:.1%} success, "
                               f"{strategy_results[strategy]['avg_processing_time_ms']:.1f}ms avg, "
                               f"{strategy_results[strategy]['avg_quality_score']:.2f} quality")
        
        finally:
            cursor.close()
            connection.close()
        
        self.results["chunking_strategies"] = strategy_results
        return strategy_results
    
    def validate_rag_integration(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Validate integration with all 7 RAG techniques."""
        logger.info("🔗 Validating RAG integration with enhanced chunking...")
        
        if test_queries is None:
            test_queries = [
                "What are the main findings of this study?",
                "What methodology was used in the research?",
                "What are the clinical implications?",
                "What statistical methods were applied?",
                "What are the limitations of this study?"
            ]
        
        integration_results = {}
        
        # First, create some chunked documents
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Get test documents
            cursor.execute("""
                SELECT TOP 5 doc_id, text_content
                FROM RAG.SourceDocuments_V2
                WHERE text_content IS NOT NULL
                AND LENGTH(text_content) > 1000
                ORDER BY RANDOM()
            """)
            
            documents = cursor.fetchall()
            
            # Create chunks for test documents
            logger.info("Creating chunks for RAG integration testing...")
            for doc_id, text_content in documents:
                try:
                    chunks = self.chunking_service.chunk_document(doc_id, text_content, "adaptive")
                    self.chunking_service.store_chunks(chunks)
                    logger.info(f"Created {len(chunks)} chunks for {doc_id}")
                except Exception as e:
                    logger.error(f"Error creating chunks for {doc_id}: {e}")
            
            # Test each RAG technique
            for technique_name, technique_func in self.rag_techniques.items():
                logger.info(f"Testing {technique_name} with enhanced chunking...")
                
                technique_results = {
                    "queries_tested": 0,
                    "successful_queries": 0,
                    "response_times": [],
                    "document_counts": [],
                    "errors": []
                }
                
                for query in test_queries[:3]:  # Test subset for speed
                    try:
                        start_time = time.time()
                        
                        # Instantiate and Run RAG technique
                        pipeline_instance = technique_func(
                            iris_connector=get_iris_connection(), # Assuming constructor takes these
                            embedding_func=self.embedding_func   # Or they might be passed to run()
                        )
                        result = pipeline_instance.run(
                            query=query,
                            top_k=5
                        )
                        
                        response_time = time.time() - start_time
                        
                        # Validate result
                        if "answer" in result and "retrieved_documents" in result:
                            technique_results["successful_queries"] += 1
                            technique_results["response_times"].append(response_time * 1000)
                            technique_results["document_counts"].append(len(result["retrieved_documents"]))
                        
                        technique_results["queries_tested"] += 1
                        
                    except Exception as e:
                        error_msg = f"Error in {technique_name} with query '{query}': {e}"
                        logger.error(error_msg)
                        technique_results["errors"].append(error_msg)
                        technique_results["queries_tested"] += 1
                
                # Calculate summary
                if technique_results["queries_tested"] > 0:
                    integration_results[technique_name] = {
                        "success_rate": technique_results["successful_queries"] / technique_results["queries_tested"],
                        "avg_response_time_ms": statistics.mean(technique_results["response_times"]) if technique_results["response_times"] else 0,
                        "avg_documents_retrieved": statistics.mean(technique_results["document_counts"]) if technique_results["document_counts"] else 0,
                        "error_count": len(technique_results["errors"]),
                        "queries_tested": technique_results["queries_tested"]
                    }
                    
                    logger.info(f"✅ {technique_name}: {integration_results[technique_name]['success_rate']:.1%} success, "
                               f"{integration_results[technique_name]['avg_response_time_ms']:.0f}ms avg")
        
        finally:
            cursor.close()
            connection.close()
        
        self.results["rag_integration"] = integration_results
        return integration_results
    
    def validate_scale_performance(self, document_limit: int = 1000) -> Dict[str, Any]:
        """Validate performance at scale with 1000+ documents."""
        logger.info(f"📊 Validating scale performance with {document_limit} documents...")
        
        scale_results = {}
        
        # Test different strategies at scale
        strategies_to_test = ["adaptive", "recursive", "semantic"]
        
        for strategy in strategies_to_test:
            logger.info(f"Scale testing {strategy} strategy...")
            
            start_time = time.time()
            
            try:
                results = self.chunking_service.process_documents_at_scale(
                    limit=document_limit,
                    strategy_names=[strategy],
                    batch_size=100
                )
                
                total_time = time.time() - start_time
                
                scale_results[strategy] = {
                    "documents_processed": results["processed_documents"],
                    "chunks_created": results["total_chunks_created"],
                    "total_time_seconds": total_time,
                    "documents_per_second": results["performance_metrics"]["documents_per_second"],
                    "chunks_per_second": results["performance_metrics"]["chunks_per_second"],
                    "avg_quality_metrics": results["quality_metrics"],
                    "error_count": len(results["errors"]),
                    "memory_efficiency": "Good" if total_time < document_limit * 0.1 else "Needs optimization"
                }
                
                logger.info(f"✅ {strategy} scale test: {results['processed_documents']} docs, "
                           f"{results['total_chunks_created']} chunks, "
                           f"{results['performance_metrics']['documents_per_second']:.1f} docs/sec")
                
            except Exception as e:
                error_msg = f"Scale test failed for {strategy}: {e}"
                logger.error(error_msg)
                scale_results[strategy] = {"error": error_msg}
        
        self.results["scale_testing"] = scale_results
        return scale_results
    
    def validate_database_operations(self) -> Dict[str, Any]:
        """Validate database storage and retrieval operations."""
        logger.info("💾 Validating database operations...")
        
        db_results = {
            "storage_test": {},
            "retrieval_test": {},
            "schema_validation": {}
        }
        
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Test document for database operations
            cursor.execute("""
                SELECT TOP 1 doc_id, text_content
                FROM RAG.SourceDocuments_V2
                WHERE text_content IS NOT NULL
                AND LENGTH(text_content) > 500
            """)
            
            result = cursor.fetchone()
            if not result:
                db_results["error"] = "No suitable test document found"
                return db_results
            
            doc_id, text_content = result
            test_doc_id = f"test_enhanced_{doc_id}"
            
            # Storage test
            logger.info("Testing chunk storage...")
            start_time = time.time()
            
            chunks = self.chunking_service.chunk_document(test_doc_id, text_content, "adaptive")
            storage_success = self.chunking_service.store_chunks(chunks)
            
            storage_time = time.time() - start_time
            
            db_results["storage_test"] = {
                "success": storage_success,
                "chunks_stored": len(chunks),
                "storage_time_ms": storage_time * 1000
            }
            
            # Retrieval test
            logger.info("Testing chunk retrieval...")
            start_time = time.time()
            
            cursor.execute("""
                SELECT chunk_id, chunk_text, chunk_metadata, embedding_str
                FROM RAG.DocumentChunks
                WHERE doc_id = ?
                ORDER BY chunk_index
            """, (test_doc_id,))
            
            retrieved_chunks = cursor.fetchall()
            retrieval_time = time.time() - start_time
            
            db_results["retrieval_test"] = {
                "chunks_retrieved": len(retrieved_chunks),
                "retrieval_time_ms": retrieval_time * 1000,
                "data_integrity": len(retrieved_chunks) == len(chunks)
            }
            
            # Schema validation
            logger.info("Validating chunk metadata schema...")
            schema_valid = True
            metadata_errors = []
            
            for chunk_id, chunk_text, chunk_metadata, embedding_str in retrieved_chunks:
                try:
                    metadata = json.loads(chunk_metadata)
                    
                    # Check required fields
                    required_fields = ["chunk_metrics", "biomedical_optimized", "processing_time_ms"]
                    for field in required_fields:
                        if field not in metadata:
                            schema_valid = False
                            metadata_errors.append(f"Missing field {field} in {chunk_id}")
                    
                    # Check chunk metrics
                    if "chunk_metrics" in metadata:
                        metrics = metadata["chunk_metrics"]
                        required_metrics = ["token_count", "character_count", "sentence_count"]
                        for metric in required_metrics:
                            if metric not in metrics:
                                schema_valid = False
                                metadata_errors.append(f"Missing metric {metric} in {chunk_id}")
                
                except json.JSONDecodeError as e:
                    schema_valid = False
                    metadata_errors.append(f"Invalid JSON in {chunk_id}: {e}")
            
            db_results["schema_validation"] = {
                "valid": schema_valid,
                "errors": metadata_errors
            }
            
            # Cleanup
            cursor.execute("DELETE FROM RAG.DocumentChunks WHERE doc_id = ?", (test_doc_id,))
            connection.commit()
            
            logger.info(f"✅ Database operations: Storage {storage_success}, "
                       f"Retrieved {len(retrieved_chunks)}/{len(chunks)} chunks, "
                       f"Schema valid: {schema_valid}")
        
        except Exception as e:
            error_msg = f"Database validation error: {e}"
            logger.error(error_msg)
            db_results["error"] = error_msg
        
        finally:
            cursor.close()
            connection.close()
        
        self.results["database_operations"] = db_results
        return db_results
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        logger.info("📋 Generating performance report...")
        
        report = []
        report.append("=" * 80)
        report.append("ENHANCED CHUNKING SYSTEM VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Validation Date: {self.results['validation_timestamp']}")
        report.append("")
        
        # Chunking Strategies Summary
        if "chunking_strategies" in self.results:
            report.append("📊 CHUNKING STRATEGIES PERFORMANCE")
            report.append("-" * 50)
            
            strategies = self.results["chunking_strategies"]
            for strategy, metrics in strategies.items():
                report.append(f"\n{strategy.upper()}:")
                report.append(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
                report.append(f"  Avg Processing Time: {metrics.get('avg_processing_time_ms', 0):.1f}ms")
                report.append(f"  Avg Quality Score: {metrics.get('avg_quality_score', 0):.2f}")
                report.append(f"  Avg Coherence: {metrics.get('avg_coherence', 0):.2f}")
                report.append(f"  Chunks per Document: {metrics.get('chunks_per_document', 0):.1f}")
        
        # RAG Integration Summary
        if "rag_integration" in self.results:
            report.append("\n\n🔗 RAG INTEGRATION RESULTS")
            report.append("-" * 50)
            
            integration = self.results["rag_integration"]
            for technique, metrics in integration.items():
                report.append(f"\n{technique}:")
                report.append(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
                report.append(f"  Avg Response Time: {metrics.get('avg_response_time_ms', 0):.0f}ms")
                report.append(f"  Avg Documents Retrieved: {metrics.get('avg_documents_retrieved', 0):.1f}")
        
        # Scale Testing Summary
        if "scale_testing" in self.results:
            report.append("\n\n📈 SCALE PERFORMANCE RESULTS")
            report.append("-" * 50)
            
            scale = self.results["scale_testing"]
            for strategy, metrics in scale.items():
                if "error" not in metrics:
                    report.append(f"\n{strategy.upper()} (Scale Test):")
                    report.append(f"  Documents Processed: {metrics.get('documents_processed', 0):,}")
                    report.append(f"  Chunks Created: {metrics.get('chunks_created', 0):,}")
                    report.append(f"  Processing Rate: {metrics.get('documents_per_second', 0):.1f} docs/sec")
                    report.append(f"  Memory Efficiency: {metrics.get('memory_efficiency', 'Unknown')}")
        
        # Database Operations Summary
        if "database_operations" in self.results:
            report.append("\n\n💾 DATABASE OPERATIONS")
            report.append("-" * 50)
            
            db_ops = self.results["database_operations"]
            if "storage_test" in db_ops:
                storage = db_ops["storage_test"]
                report.append(f"\nStorage Test:")
                report.append(f"  Success: {storage.get('success', False)}")
                report.append(f"  Chunks Stored: {storage.get('chunks_stored', 0)}")
                report.append(f"  Storage Time: {storage.get('storage_time_ms', 0):.1f}ms")
            
            if "retrieval_test" in db_ops:
                retrieval = db_ops["retrieval_test"]
                report.append(f"\nRetrieval Test:")
                report.append(f"  Chunks Retrieved: {retrieval.get('chunks_retrieved', 0)}")
                report.append(f"  Data Integrity: {retrieval.get('data_integrity', False)}")
                report.append(f"  Retrieval Time: {retrieval.get('retrieval_time_ms', 0):.1f}ms")
        
        # Recommendations
        report.append("\n\n💡 RECOMMENDATIONS")
        report.append("-" * 50)
        
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"• {rec}")
        
        # Summary
        report.append("\n\n✅ VALIDATION SUMMARY")
        report.append("-" * 50)
        
        summary = self._generate_summary()
        for item in summary:
            report.append(f"• {item}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"enhanced_chunking_validation_report_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write(report_text)
        
        logger.info(f"📋 Report saved to {report_filename}")
        
        return report_text
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyze chunking strategy performance
        if "chunking_strategies" in self.results:
            strategies = self.results["chunking_strategies"]
            
            # Find best performing strategy
            best_strategy = max(strategies.items(), 
                              key=lambda x: x[1].get("success_rate", 0) * x[1].get("avg_quality_score", 0))
            
            recommendations.append(f"Recommended primary strategy: {best_strategy[0]} "
                                 f"(Success: {best_strategy[1].get('success_rate', 0):.1%}, "
                                 f"Quality: {best_strategy[1].get('avg_quality_score', 0):.2f})")
            
            # Check for slow strategies
            slow_strategies = [name for name, metrics in strategies.items() 
                             if metrics.get("avg_processing_time_ms", 0) > 1000]
            if slow_strategies:
                recommendations.append(f"Consider optimizing slow strategies: {', '.join(slow_strategies)}")
        
        # Analyze RAG integration
        if "rag_integration" in self.results:
            integration = self.results["rag_integration"]
            
            failed_techniques = [name for name, metrics in integration.items() 
                               if metrics.get("success_rate", 0) < 0.8]
            if failed_techniques:
                recommendations.append(f"Review integration issues with: {', '.join(failed_techniques)}")
        
        # Scale performance recommendations
        if "scale_testing" in self.results:
            scale = self.results["scale_testing"]
            
            slow_scale = [name for name, metrics in scale.items() 
                         if "error" not in metrics and metrics.get("documents_per_second", 0) < 1.0]
            if slow_scale:
                recommendations.append(f"Scale optimization needed for: {', '.join(slow_scale)}")
        
        if not recommendations:
            recommendations.append("All systems performing within acceptable parameters")
        
        return recommendations
    
    def _generate_summary(self) -> List[str]:
        """Generate validation summary."""
        summary = []
        
        # Count successful validations
        validations = 0
        successes = 0
        
        if "chunking_strategies" in self.results:
            validations += 1
            strategies = self.results["chunking_strategies"]
            if all(metrics.get("success_rate", 0) > 0.8 for metrics in strategies.values()):
                successes += 1
                summary.append("✅ Chunking strategies validation: PASSED")
            else:
                summary.append("❌ Chunking strategies validation: FAILED")
        
        if "rag_integration" in self.results:
            validations += 1
            integration = self.results["rag_integration"]
            if all(metrics.get("success_rate", 0) > 0.7 for metrics in integration.values()):
                successes += 1
                summary.append("✅ RAG integration validation: PASSED")
            else:
                summary.append("❌ RAG integration validation: FAILED")
        
        if "scale_testing" in self.results:
            validations += 1
            scale = self.results["scale_testing"]
            if all("error" not in metrics for metrics in scale.values()):
                successes += 1
                summary.append("✅ Scale performance validation: PASSED")
            else:
                summary.append("❌ Scale performance validation: FAILED")
        
        if "database_operations" in self.results:
            validations += 1
            db_ops = self.results["database_operations"]
            if (db_ops.get("storage_test", {}).get("success", False) and 
                db_ops.get("retrieval_test", {}).get("data_integrity", False)):
                successes += 1
                summary.append("✅ Database operations validation: PASSED")
            else:
                summary.append("❌ Database operations validation: FAILED")
        
        summary.append(f"\nOverall Success Rate: {successes}/{validations} ({successes/max(1,validations):.1%})")
        
        if successes == validations:
            summary.append("🎉 Enhanced chunking system ready for production deployment!")
        else:
            summary.append("⚠️  Some validations failed - review recommendations before deployment")
        
        return summary
    
    def run_full_validation(self, document_limit: int = 1000) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("🚀 Starting enhanced chunking system validation...")
        
        start_time = time.time()
        
        try:
            # Run all validation tests
            self.validate_chunking_strategies(sample_size=50)
            self.validate_rag_integration()
            self.validate_scale_performance(document_limit)
            self.validate_database_operations()
            
            # Generate report
            report = self.generate_performance_report()
            
            total_time = time.time() - start_time
            
            self.results["validation_summary"] = {
                "total_validation_time_seconds": total_time,
                "validation_completed": True,
                "report_generated": True
            }
            
            logger.info(f"✅ Validation completed in {total_time:.1f} seconds")
            
            return self.results
            
        except Exception as e:
            error_msg = f"Validation failed: {e}"
            logger.error(error_msg)
            self.results["validation_error"] = error_msg
            return self.results

def main():
    """Main function to run enhanced chunking validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Chunking System Validation")
    parser.add_argument("--documents", type=int, default=1000, 
                       help="Number of documents for scale testing (default: 1000)")
    parser.add_argument("--strategies-only", action="store_true",
                       help="Test only chunking strategies (skip RAG integration)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick validation with reduced document count")
    
    args = parser.parse_args()
    
    if args.quick:
        args.documents = 100
    
    print("🚀 Enhanced Chunking System Validation")
    print("=" * 50)
    
    validator = EnhancedChunkingValidator()
    
    if args.strategies_only:
        print("Running chunking strategies validation only...")
        validator.validate_chunking_strategies()
        validator.validate_database_operations()
        report = validator.generate_performance_report()
    else:
        print(f"Running full validation with {args.documents} documents...")
        results = validator.run_full_validation(args.documents)
    
    print("\n✅ Validation completed!")
    print("Check the generated report file for detailed results.")

if __name__ == "__main__":
    main()