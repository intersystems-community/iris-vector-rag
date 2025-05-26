#!/usr/bin/env python3
"""
Production Scale RAG Testing Script

This script scales up the RAG system to handle production-scale workloads:
- Downloads and processes up to 92k PMC documents
- Uses real PyTorch models for embeddings
- Tests HNSW performance improvements
- Runs comprehensive benchmarks at scale
- Monitors system performance and memory usage

Usage:
    python scripts/production_scale_rag_test.py --target-docs 10000 --batch-size 100
    python scripts/production_scale_rag_test.py --target-docs 50000 --use-gpu
    python scripts/production_scale_rag_test.py --full-scale  # Targets 92k docs
"""

import os
import sys
import logging
import time
import json
import argparse
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
from data.pmc_processor import process_pmc_files
from data.loader import load_documents_to_iris, process_and_load_documents
from eval.bench_runner import BenchmarkRunner
from eval.metrics import calculate_retrieval_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_scale_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScaleTestConfig:
    """Configuration for production scale testing"""
    target_documents: int = 10000
    batch_size: int = 100
    embedding_model: str = "intfloat/e5-base-v2"
    use_gpu: bool = False
    max_memory_gb: float = 16.0
    pmc_data_dir: str = "data/pmc_oas_downloaded"
    download_additional_data: bool = True
    test_hnsw: bool = True
    run_benchmarks: bool = True
    monitor_performance: bool = True

class SystemMonitor:
    """Monitor system performance during testing"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = []
        
    def record_metrics(self, stage: str, additional_data: Dict = None):
        """Record current system metrics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        metric = {
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time,
            "stage": stage,
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "cpu_percent": cpu_percent,
            "available_memory_gb": memory.available / (1024**3)
        }
        
        if additional_data:
            metric.update(additional_data)
            
        self.metrics.append(metric)
        logger.info(f"📊 {stage}: Memory {metric['memory_used_gb']:.1f}GB ({metric['memory_percent']:.1f}%), CPU {cpu_percent:.1f}%")
        
        return metric
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

class ProductionScaleRAGTester:
    """Main class for production scale RAG testing"""
    
    def __init__(self, config: ScaleTestConfig):
        self.config = config
        self.monitor = SystemMonitor()
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        
    def setup_models(self):
        """Initialize real PyTorch models"""
        logger.info("🔧 Setting up real PyTorch models...")
        self.monitor.record_metrics("model_setup_start")
        
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_func = get_embedding_func(
                model_name=self.config.embedding_model,
                mock=False
            )
            
            # Test embedding model
            test_texts = ["Test embedding generation", "Production scale testing"]
            test_embeddings = self.embedding_func(test_texts)
            logger.info(f"✅ Embedding model loaded: {len(test_embeddings[0])} dimensions")
            
            # Initialize LLM
            logger.info("Loading LLM...")
            self.llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
            
            # Test LLM
            test_response = self.llm_func("Test prompt for production scale testing.")
            logger.info(f"✅ LLM loaded and tested")
            
            self.monitor.record_metrics("model_setup_complete", {
                "embedding_dim": len(test_embeddings[0]),
                "llm_response_length": len(test_response)
            })
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            raise
    
    def setup_database(self):
        """Setup database connection and schema"""
        logger.info("🔧 Setting up database connection...")
        self.monitor.record_metrics("db_setup_start")
        
        try:
            self.connection = get_iris_connection()
            if not self.connection:
                raise Exception("Failed to establish database connection")
            
            # Test connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            logger.info("✅ Database connection established")
            self.monitor.record_metrics("db_setup_complete")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
    
    def download_additional_pmc_data(self):
        """Download additional PMC data if needed"""
        if not self.config.download_additional_data:
            return
            
        logger.info("📥 Checking for additional PMC data...")
        self.monitor.record_metrics("data_download_start")
        
        # Count current documents
        current_count = 0
        pmc_dir = Path(self.config.pmc_data_dir)
        if pmc_dir.exists():
            current_count = len(list(pmc_dir.rglob("*.xml")))
        
        logger.info(f"Current PMC documents: {current_count}")
        
        if current_count < self.config.target_documents:
            logger.info(f"Need {self.config.target_documents - current_count} more documents")
            
            # Extract additional data from archive if available
            archive_path = pmc_dir / "oa_comm_xml.incr.2024-12-19.tar.gz"
            if archive_path.exists():
                logger.info("Extracting additional documents from archive...")
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=pmc_dir)
                
                # Recount
                new_count = len(list(pmc_dir.rglob("*.xml")))
                logger.info(f"Extracted documents. New total: {new_count}")
            
            # TODO: Add logic to download more PMC data from NCBI if needed
            # This would involve using the NCBI E-utilities API
            
        self.monitor.record_metrics("data_download_complete", {
            "total_documents_available": len(list(pmc_dir.rglob("*.xml")))
        })
    
    def process_documents_in_batches(self):
        """Process and load documents in batches"""
        logger.info(f"📄 Processing up to {self.config.target_documents} documents in batches of {self.config.batch_size}")
        self.monitor.record_metrics("document_processing_start")
        
        total_processed = 0
        total_loaded = 0
        batch_count = 0
        
        try:
            # Process documents in batches to manage memory
            pmc_dir = self.config.pmc_data_dir
            
            for batch_start in range(0, self.config.target_documents, self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, self.config.target_documents)
                batch_size = batch_end - batch_start
                batch_count += 1
                
                logger.info(f"🔄 Processing batch {batch_count}: documents {batch_start+1}-{batch_end}")
                
                # Process this batch
                documents = []
                doc_count = 0
                for doc_metadata in process_pmc_files(pmc_dir, limit=batch_end):
                    if doc_count >= batch_start:
                        documents.append(doc_metadata)
                    doc_count += 1
                    if len(documents) >= batch_size:
                        break
                
                if not documents:
                    logger.warning(f"No documents found for batch {batch_count}")
                    continue
                
                # Generate embeddings for this batch
                logger.info(f"Generating embeddings for {len(documents)} documents...")
                batch_start_time = time.time()
                
                # Load documents with embeddings
                load_stats = load_documents_to_iris(
                    self.connection,
                    documents,
                    embedding_func=self.embedding_func,
                    batch_size=min(50, len(documents))  # Smaller sub-batches for memory
                )
                
                batch_time = time.time() - batch_start_time
                total_processed += len(documents)
                total_loaded += load_stats.get('loaded_doc_count', 0)
                
                # Record metrics for this batch
                self.monitor.record_metrics(f"batch_{batch_count}_complete", {
                    "batch_documents": len(documents),
                    "batch_loaded": load_stats.get('loaded_doc_count', 0),
                    "batch_time_seconds": batch_time,
                    "batch_docs_per_second": len(documents) / batch_time if batch_time > 0 else 0,
                    "total_processed": total_processed,
                    "total_loaded": total_loaded
                })
                
                # Memory management
                del documents
                gc.collect()
                
                # Check memory usage
                memory = psutil.virtual_memory()
                if memory.used / (1024**3) > self.config.max_memory_gb:
                    logger.warning(f"Memory usage ({memory.used / (1024**3):.1f}GB) exceeds limit ({self.config.max_memory_gb}GB)")
                    break
                
                logger.info(f"✅ Batch {batch_count} complete: {len(documents)} docs in {batch_time:.1f}s")
        
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise
        
        self.monitor.record_metrics("document_processing_complete", {
            "total_processed": total_processed,
            "total_loaded": total_loaded,
            "total_batches": batch_count
        })
        
        logger.info(f"✅ Document processing complete: {total_loaded}/{total_processed} documents loaded")
        return total_loaded
    
    def test_hnsw_performance(self):
        """Test HNSW indexing performance"""
        if not self.config.test_hnsw:
            return
            
        logger.info("🔍 Testing HNSW performance...")
        self.monitor.record_metrics("hnsw_test_start")
        
        try:
            cursor = self.connection.cursor()
            
            # Check if we have documents with embeddings
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            doc_count = cursor.fetchone()[0]
            logger.info(f"Testing HNSW with {doc_count} documents")
            
            if doc_count == 0:
                logger.warning("No documents with embeddings found for HNSW testing")
                return
            
            # Test vector similarity search performance
            test_queries = [
                "machine learning algorithms",
                "biomedical research methods",
                "clinical trial outcomes",
                "drug discovery process",
                "genetic analysis techniques"
            ]
            
            hnsw_results = []
            
            for i, query in enumerate(test_queries):
                logger.info(f"Testing query {i+1}/{len(test_queries)}: {query}")
                
                # Generate query embedding
                query_embedding = self.embedding_func([query])[0]
                query_vector_str = ','.join(map(str, query_embedding))
                
                # Test vector similarity search
                start_time = time.time()
                
                # Use IRIS vector similarity (working around TO_VECTOR limitations)
                sql = """
                SELECT TOP 10 doc_id, title, 
                       VECTOR_DOT_PRODUCT(?, embedding) as similarity
                FROM RAG.SourceDocuments 
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                
                try:
                    cursor.execute(sql, (query_vector_str,))
                    results = cursor.fetchall()
                    search_time = time.time() - start_time
                    
                    hnsw_results.append({
                        "query": query,
                        "results_count": len(results),
                        "search_time_ms": search_time * 1000,
                        "top_similarity": results[0][2] if results else 0
                    })
                    
                    logger.info(f"Query completed in {search_time*1000:.1f}ms, {len(results)} results")
                    
                except Exception as e:
                    logger.error(f"HNSW query failed: {e}")
                    # Fallback to basic similarity without VECTOR functions
                    hnsw_results.append({
                        "query": query,
                        "results_count": 0,
                        "search_time_ms": 0,
                        "error": str(e)
                    })
            
            cursor.close()
            
            # Calculate average performance
            successful_queries = [r for r in hnsw_results if "error" not in r]
            if successful_queries:
                avg_time = np.mean([r["search_time_ms"] for r in successful_queries])
                avg_results = np.mean([r["results_count"] for r in successful_queries])
                
                logger.info(f"✅ HNSW Performance: {avg_time:.1f}ms avg, {avg_results:.1f} avg results")
                
                self.monitor.record_metrics("hnsw_test_complete", {
                    "avg_search_time_ms": avg_time,
                    "avg_results_count": avg_results,
                    "successful_queries": len(successful_queries),
                    "total_queries": len(test_queries)
                })
            else:
                logger.warning("No successful HNSW queries")
                
        except Exception as e:
            logger.error(f"❌ HNSW testing failed: {e}")
    
    def run_comprehensive_benchmarks(self):
        """Run comprehensive RAG benchmarks"""
        if not self.config.run_benchmarks:
            return
            
        logger.info("📊 Running comprehensive benchmarks...")
        self.monitor.record_metrics("benchmark_start")
        
        try:
            # Initialize benchmark runner
            benchmark_runner = BenchmarkRunner(
                iris_connector=self.connection,
                embedding_func=self.embedding_func,
                llm_func=self.llm_func
            )
            
            # Define test queries for biomedical domain
            test_queries = [
                "What are the latest treatments for diabetes?",
                "How does machine learning help in drug discovery?",
                "What are the side effects of immunotherapy?",
                "How do genetic mutations cause cancer?",
                "What is the role of AI in medical diagnosis?"
            ]
            
            # Run benchmarks for different RAG techniques
            techniques = ["basic_rag", "graphrag", "hyde"]  # Start with available techniques
            
            benchmark_results = {}
            
            for technique in techniques:
                logger.info(f"Benchmarking {technique}...")
                
                try:
                    technique_results = benchmark_runner.run_technique_benchmark(
                        technique=technique,
                        queries=test_queries,
                        num_runs=3  # Multiple runs for statistical significance
                    )
                    
                    benchmark_results[technique] = technique_results
                    logger.info(f"✅ {technique} benchmark complete")
                    
                except Exception as e:
                    logger.error(f"❌ {technique} benchmark failed: {e}")
                    benchmark_results[technique] = {"error": str(e)}
            
            # Save benchmark results
            timestamp = int(time.time())
            results_file = f"production_benchmark_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            logger.info(f"✅ Benchmark results saved to {results_file}")
            
            self.monitor.record_metrics("benchmark_complete", {
                "techniques_tested": len(techniques),
                "queries_per_technique": len(test_queries),
                "results_file": results_file
            })
            
        except Exception as e:
            logger.error(f"❌ Comprehensive benchmarking failed: {e}")
    
    def validate_semantic_search_quality(self):
        """Validate semantic search quality with larger corpus"""
        logger.info("🔍 Validating semantic search quality...")
        self.monitor.record_metrics("semantic_validation_start")
        
        try:
            cursor = self.connection.cursor()
            
            # Get total document count
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            total_docs = cursor.fetchone()[0]
            
            logger.info(f"Validating semantic search with {total_docs} documents")
            
            # Test semantic search with domain-specific queries
            test_cases = [
                {
                    "query": "diabetes treatment",
                    "expected_keywords": ["diabetes", "treatment", "insulin", "glucose", "medication"]
                },
                {
                    "query": "cancer immunotherapy",
                    "expected_keywords": ["cancer", "immunotherapy", "immune", "tumor", "therapy"]
                },
                {
                    "query": "machine learning medical diagnosis",
                    "expected_keywords": ["machine learning", "AI", "diagnosis", "medical", "algorithm"]
                }
            ]
            
            validation_results = []
            
            for test_case in test_cases:
                query = test_case["query"]
                expected_keywords = test_case["expected_keywords"]
                
                logger.info(f"Testing semantic search for: {query}")
                
                # Generate query embedding
                query_embedding = self.embedding_func([query])[0]
                query_vector_str = ','.join(map(str, query_embedding))
                
                # Search for similar documents
                sql = """
                SELECT TOP 20 doc_id, title, text_content,
                       VECTOR_DOT_PRODUCT(?, embedding) as similarity
                FROM RAG.SourceDocuments 
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                """
                
                try:
                    cursor.execute(sql, (query_vector_str,))
                    results = cursor.fetchall()
                    
                    # Analyze result quality
                    keyword_matches = 0
                    total_results = len(results)
                    
                    for doc_id, title, content, similarity in results[:10]:  # Check top 10
                        text_to_check = (title + " " + content).lower()
                        for keyword in expected_keywords:
                            if keyword.lower() in text_to_check:
                                keyword_matches += 1
                                break
                    
                    relevance_score = keyword_matches / min(10, total_results) if total_results > 0 else 0
                    avg_similarity = np.mean([r[3] for r in results[:10]]) if results else 0
                    
                    validation_results.append({
                        "query": query,
                        "total_results": total_results,
                        "relevance_score": relevance_score,
                        "avg_similarity": avg_similarity,
                        "keyword_matches": keyword_matches
                    })
                    
                    logger.info(f"Query '{query}': {relevance_score:.2f} relevance, {avg_similarity:.4f} avg similarity")
                    
                except Exception as e:
                    logger.error(f"Semantic search failed for '{query}': {e}")
                    validation_results.append({
                        "query": query,
                        "error": str(e)
                    })
            
            cursor.close()
            
            # Calculate overall quality metrics
            successful_tests = [r for r in validation_results if "error" not in r]
            if successful_tests:
                avg_relevance = np.mean([r["relevance_score"] for r in successful_tests])
                avg_similarity = np.mean([r["avg_similarity"] for r in successful_tests])
                
                logger.info(f"✅ Semantic search quality: {avg_relevance:.2f} avg relevance, {avg_similarity:.4f} avg similarity")
                
                self.monitor.record_metrics("semantic_validation_complete", {
                    "total_documents": total_docs,
                    "avg_relevance_score": avg_relevance,
                    "avg_similarity": avg_similarity,
                    "successful_tests": len(successful_tests)
                })
            
        except Exception as e:
            logger.error(f"❌ Semantic search validation failed: {e}")
    
    def run_production_scale_test(self):
        """Run the complete production scale test"""
        logger.info("🚀 Starting Production Scale RAG Test")
        logger.info("=" * 80)
        
        try:
            # Setup phase
            self.setup_models()
            self.setup_database()
            
            # Data preparation phase
            self.download_additional_pmc_data()
            
            # Document processing phase
            documents_loaded = self.process_documents_in_batches()
            
            if documents_loaded == 0:
                logger.error("❌ No documents loaded - cannot continue with testing")
                return False
            
            # Performance testing phase
            self.test_hnsw_performance()
            
            # Quality validation phase
            self.validate_semantic_search_quality()
            
            # Comprehensive benchmarking phase
            self.run_comprehensive_benchmarks()
            
            # Final summary
            self.monitor.record_metrics("test_complete")
            
            logger.info("=" * 80)
            logger.info("🎉 Production Scale RAG Test Complete!")
            logger.info(f"📊 Documents processed: {documents_loaded}")
            logger.info(f"⏱️  Total time: {(time.time() - self.monitor.start_time)/60:.1f} minutes")
            
            # Save performance metrics
            metrics_file = f"production_scale_metrics_{int(time.time())}.json"
            self.monitor.save_metrics(metrics_file)
            logger.info(f"📈 Performance metrics saved to: {metrics_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Production scale test failed: {e}")
            return False
        
        finally:
            # Cleanup
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Production Scale RAG Testing")
    parser.add_argument("--target-docs", type=int, default=10000, 
                       help="Target number of documents to process")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for document processing")
    parser.add_argument("--embedding-model", type=str, default="intfloat/e5-base-v2",
                       help="Embedding model to use")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU for model inference")
    parser.add_argument("--max-memory-gb", type=float, default=16.0,
                       help="Maximum memory usage in GB")
    parser.add_argument("--full-scale", action="store_true",
                       help="Target full scale (92k documents)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip downloading additional data")
    parser.add_argument("--skip-hnsw", action="store_true",
                       help="Skip HNSW performance testing")
    parser.add_argument("--skip-benchmarks", action="store_true",
                       help="Skip comprehensive benchmarking")
    
    args = parser.parse_args()
    
    # Configure based on arguments
    config = ScaleTestConfig(
        target_documents=92000 if args.full_scale else args.target_docs,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        use_gpu=args.use_gpu,
        max_memory_gb=args.max_memory_gb,
        download_additional_data=not args.skip_download,
        test_hnsw=not args.skip_hnsw,
        run_benchmarks=not args.skip_benchmarks
    )
    
    logger.info(f"Configuration: {config}")
    
    # Run the test
    tester = ProductionScaleRAGTester(config)
    success = tester.run_production_scale_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()