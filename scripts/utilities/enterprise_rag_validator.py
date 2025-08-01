#!/usr/bin/env python3
"""
Enterprise RAG Validator - Single Parameterized Script

A clean, single script that validates all 7 RAG techniques at any scale.
No more boilerplate - just specify the number of documents you want.

Usage:
    python scripts/enterprise_rag_validator.py --docs 1000     # 1K validation
    python scripts/enterprise_rag_validator.py --docs 5000     # 5K validation  
    python scripts/enterprise_rag_validator.py --docs 50000    # 50K validation
    python scripts/enterprise_rag_validator.py --docs 92000    # 92K validation (maximum)
    
    # Additional options:
    python scripts/enterprise_rag_validator.py --docs 92000 --skip-ingestion
    python scripts/enterprise_rag_validator.py --docs 5000 --fast-mode
    python scripts/enterprise_rag_validator.py --docs 10000 --skip-colbert
"""

import os
import sys
import logging
import time
import json
import argparse
import psutil
import numpy as np
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import gc

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func, get_colbert_query_encoder_func, get_colbert_doc_encoder_func_adapted # Updated import
from data.loader_fixed import load_documents_to_iris # Path remains correct
from data.pmc_processor import process_pmc_files # Path remains correct

# Import all RAG pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_rag_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results from technique validation"""
    technique: str
    success: bool
    avg_time_ms: float
    avg_docs_retrieved: float
    success_rate: float
    total_queries: int
    error: Optional[str] = None

class SystemMonitor:
    """Simple system resource monitor"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start(self):
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        return self.metrics
        
    def _monitor_loop(self):
        while self.monitoring:
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'cpu_percent': cpu
                })
                if memory.percent > 90:
                    logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            time.sleep(5)

class EnterpriseRAGValidator:
    """Single validator for all RAG techniques at any scale"""
    
    def __init__(self, target_docs: int, fast_mode: bool = False):
        self.target_docs = target_docs
        self.fast_mode = fast_mode
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        self.monitor = SystemMonitor()
        
        # Scale-appropriate test queries
        if fast_mode:
            self.test_queries = [
                "What are diabetes treatments?",
                "How does AI help medical diagnosis?"
            ]
        else:
            self.test_queries = [
                "What are the latest treatments for diabetes?",
                "How does machine learning improve medical diagnosis?",
                "What are the mechanisms of cancer immunotherapy?",
                "How do genetic mutations contribute to disease development?",
                "What role does AI play in modern healthcare systems?",
                "What are cardiovascular disease prevention methods?",
                "How do neurological disorders affect brain function?",
                "What are infectious disease control strategies?"
            ]
    
    def setup(self) -> bool:
        """Setup database and models"""
        logger.info(f"🔧 Setting up for {self.target_docs:,} document validation...")
        
        try:
            # Database connection
            self.connection = get_iris_connection()
            if not self.connection:
                raise Exception("Failed to get database connection")
            
            # Check current document count
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
            current_docs = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE embedding IS NOT NULL")
            docs_with_embeddings = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"📊 Database: {current_docs:,} total docs, {docs_with_embeddings:,} with embeddings")
            
            # Setup models
            self.embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
            self.llm_func = get_llm_func(provider="stub")
            
            logger.info("✅ Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def ensure_documents(self, skip_ingestion: bool = False) -> bool:
        """Ensure we have enough documents for testing"""
        if skip_ingestion:
            logger.info("⏭️ Skipping document ingestion")
            return True
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
            current_count = cursor.fetchone()[0]
            cursor.close()
            
            if current_count >= self.target_docs:
                logger.info(f"✅ Already have {current_count:,} documents (target: {self.target_docs:,})")
                return True
            
            docs_needed = self.target_docs - current_count
            logger.info(f"📥 Need to ingest {docs_needed:,} more documents...")
            
            # Check available PMC data
            pmc_data_dir = "data/pmc_oas_downloaded"
            if not os.path.exists(pmc_data_dir):
                logger.error(f"❌ PMC data directory not found: {pmc_data_dir}")
                return False
            
            # Get available PMC files
            pmc_files = []
            for root, dirs, files in os.walk(pmc_data_dir):
                for file in files:
                    if file.endswith('.xml'):
                        pmc_files.append(os.path.join(root, file))
            
            logger.info(f"📁 Found {len(pmc_files):,} PMC files available")
            
            if len(pmc_files) < docs_needed:
                logger.warning(f"⚠️ Only {len(pmc_files):,} files available, need {docs_needed:,}")
            
            # Process files in batches
            batch_size = min(500, docs_needed // 10) if docs_needed > 1000 else 100
            files_to_process = pmc_files[:docs_needed]
            
            logger.info(f"🔄 Processing {len(files_to_process):,} files in batches of {batch_size}")
            
            self.monitor.start()
            total_processed = 0
            
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i+batch_size]
                batch_start = time.time()
                
                try:
                    for file_path in batch:
                        documents = process_pmc_files([file_path])
                        if documents:
                            load_result = load_documents_to_iris(
                                self.connection,
                                documents,
                                embedding_func=self.embedding_func,
                                batch_size=50
                            )
                            total_processed += load_result.get('loaded_doc_count', 0)
                    
                    batch_time = time.time() - batch_start
                    logger.info(f"✅ Batch {i//batch_size + 1}: {len(batch)} files in {batch_time:.1f}s")
                    
                    # Check if target reached
                    cursor = self.connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
                    current_count = cursor.fetchone()[0]
                    cursor.close()
                    
                    if current_count >= self.target_docs:
                        logger.info(f"🎯 Target reached: {current_count:,} documents")
                        break
                        
                    gc.collect()  # Memory cleanup
                    
                except Exception as e:
                    logger.error(f"❌ Batch processing error: {e}")
            
            self.monitor.stop()
            
            # Final verification
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
            final_count = cursor.fetchone()[0]
            cursor.close()
            
            success = final_count >= self.target_docs * 0.9  # 90% is acceptable
            if success:
                logger.info(f"✅ Document ingestion completed: {final_count:,} documents")
            else:
                logger.warning(f"⚠️ Partial ingestion: {final_count:,}/{self.target_docs:,} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {e}")
            return False
    
    def create_mock_colbert_encoder(self, embedding_dim: int = 128):
        """Create mock ColBERT encoder"""
        def mock_encoder(text: str) -> List[List[float]]:
            import numpy as np
            words = text.split()[:10]
            embeddings = []
            for word in words:
                np.random.seed(hash(word) % 10000)
                embedding = np.random.randn(embedding_dim)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding.tolist())
            return embeddings
        return mock_encoder
    
    def test_technique(self, pipeline, technique_name: str) -> ValidationResult:
        """Test a single RAG technique"""
        logger.info(f"🧪 Testing {technique_name}...")
        
        start_time = time.time()
        query_times = []
        query_docs = []
        success_count = 0
        error_msg = None
        
        try:
            for i, query in enumerate(self.test_queries):
                query_start = time.time()
                
                try:
                    if technique_name == "OptimizedColBERT":
                        result = pipeline.query(query, top_k=5, similarity_threshold=0.3)
                    else:
                        result = pipeline.query(query, top_k=5)
                    
                    query_time = (time.time() - query_start) * 1000  # Convert to ms
                    docs_found = len(result.get("retrieved_documents", []))
                    
                    query_times.append(query_time)
                    query_docs.append(docs_found)
                    success_count += 1
                    
                    if i == 0:  # Log first query details
                        logger.info(f"  First query: {query_time:.1f}ms, {docs_found} docs")
                        
                except Exception as e:
                    logger.warning(f"  Query {i+1} failed: {e}")
                    if not error_msg:
                        error_msg = str(e)
            
            avg_time = np.mean(query_times) if query_times else 0
            avg_docs = np.mean(query_docs) if query_docs else 0
            success_rate = success_count / len(self.test_queries)
            
            result = ValidationResult(
                technique=technique_name,
                success=success_count > 0,
                avg_time_ms=avg_time,
                avg_docs_retrieved=avg_docs,
                success_rate=success_rate,
                total_queries=len(self.test_queries),
                error=error_msg if success_count == 0 else None
            )
            
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {technique_name}: {avg_time:.1f}ms avg, {avg_docs:.1f} docs avg, {success_rate*100:.0f}% success")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {technique_name} failed completely: {e}")
            return ValidationResult(
                technique=technique_name,
                success=False,
                avg_time_ms=0,
                avg_docs_retrieved=0,
                success_rate=0,
                total_queries=len(self.test_queries),
                error=str(e)
            )
    
    def run_validation(self, skip_techniques: List[str] = None) -> Dict[str, Any]:
        """Run validation on all RAG techniques"""
        if skip_techniques is None:
            skip_techniques = []
            
        logger.info(f"🚀 Starting validation of all RAG techniques at {self.target_docs:,} document scale...")
        
        self.monitor.start()
        validation_start = time.time()
        results = []
        
        try:
            # Initialize pipelines
            pipelines = {}
            mock_colbert_encoder = self.create_mock_colbert_encoder(128)
            
            # BasicRAG
            if "BasicRAG" not in skip_techniques:
                try:
                    pipelines["BasicRAG"] = BasicRAGPipeline(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                except Exception as e:
                    logger.error(f"❌ BasicRAG initialization failed: {e}")
            
            # HyDE
            if "HyDE" not in skip_techniques:
                try:
                    pipelines["HyDE"] = HyDERAGPipeline(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                except Exception as e:
                    logger.error(f"❌ HyDE initialization failed: {e}")
            
            # CRAG
            if "CRAG" not in skip_techniques:
                try:
                    pipelines["CRAG"] = CRAGPipeline(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                except Exception as e:
                    logger.error(f"❌ CRAG initialization failed: {e}")
            
            # OptimizedColBERT
            if "OptimizedColBERT" not in skip_techniques:
                try:
                    pipelines["OptimizedColBERT"] = ColBERTRAGPipeline(
                        iris_connector=self.connection,
                        colbert_query_encoder_func=mock_colbert_encoder,
                        colbert_doc_encoder_func=mock_colbert_encoder,
                        llm_func=self.llm_func
                    )
                except Exception as e:
                    logger.error(f"❌ OptimizedColBERT initialization failed: {e}")
            
            # NodeRAG
            if "NodeRAG" not in skip_techniques:
                try:
                    pipelines["NodeRAG"] = NodeRAGPipeline(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                except Exception as e:
                    logger.error(f"❌ NodeRAG initialization failed: {e}")
            
            # GraphRAG
            if "GraphRAG" not in skip_techniques:
                try:
                    pipelines["GraphRAG"] = GraphRAGPipeline(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                except Exception as e:
                    logger.error(f"❌ GraphRAG initialization failed: {e}")
            
            # Hybrid iFind RAG
            if "HybridiFindRAG" not in skip_techniques:
                try:
                    pipelines["HybridiFindRAG"] = HybridIFindRAGPipeline(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                except Exception as e:
                    logger.error(f"❌ HybridiFindRAG initialization failed: {e}")
            
            logger.info(f"✅ Initialized {len(pipelines)} RAG pipelines")
            
            # Test each pipeline
            for technique_name, pipeline in pipelines.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing {technique_name}")
                logger.info('='*60)
                
                result = self.test_technique(pipeline, technique_name)
                results.append(result)
                
                # Memory cleanup between techniques
                gc.collect()
            
            monitoring_data = self.monitor.stop()
            total_time = time.time() - validation_start
            
            # Generate summary
            successful_techniques = [r for r in results if r.success]
            
            # Performance ranking (fastest to slowest)
            performance_ranking = sorted(
                [(r.technique, r.avg_time_ms) for r in successful_techniques],
                key=lambda x: x[1]
            )
            
            # Retrieval ranking (most to least documents)
            retrieval_ranking = sorted(
                [(r.technique, r.avg_docs_retrieved) for r in successful_techniques],
                key=lambda x: x[1], reverse=True
            )
            
            report = {
                "timestamp": time.time(),
                "target_documents": self.target_docs,
                "fast_mode": self.fast_mode,
                "total_validation_time_seconds": total_time,
                "techniques_tested": len(results),
                "successful_techniques": len(successful_techniques),
                "success_rate": len(successful_techniques) / len(results) if results else 0,
                "test_queries_count": len(self.test_queries),
                "performance_ranking": performance_ranking,
                "retrieval_ranking": retrieval_ranking,
                "detailed_results": [
                    {
                        "technique": r.technique,
                        "success": r.success,
                        "avg_time_ms": r.avg_time_ms,
                        "avg_docs_retrieved": r.avg_docs_retrieved,
                        "success_rate": r.success_rate,
                        "error": r.error
                    } for r in results
                ],
                "system_metrics": {
                    "peak_memory_gb": max([m['memory_gb'] for m in monitoring_data]) if monitoring_data else 0,
                    "avg_cpu_percent": np.mean([m['cpu_percent'] for m in monitoring_data]) if monitoring_data else 0
                }
            }
            
            return report
            
        except Exception as e:
            self.monitor.stop()
            logger.error(f"❌ Validation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def print_summary(self, report: Dict[str, Any]):
        """Print validation summary"""
        if "error" in report:
            logger.error(f"❌ Validation failed: {report['error']}")
            return
        
        logger.info("\n" + "="*80)
        logger.info(f"ENTERPRISE RAG VALIDATION SUMMARY - {report['target_documents']:,} DOCUMENTS")
        logger.info("="*80)
        
        logger.info(f"📊 Scale: {report['target_documents']:,} documents")
        logger.info(f"⏱️  Total time: {report['total_validation_time_seconds']:.1f}s")
        logger.info(f"✅ Success rate: {report['success_rate']*100:.1f}% ({report['successful_techniques']}/{report['techniques_tested']} techniques)")
        logger.info(f"🧪 Test queries: {report['test_queries_count']}")
        
        logger.info("\n🏆 Performance Ranking (fastest to slowest):")
        for i, (technique, avg_time) in enumerate(report["performance_ranking"], 1):
            logger.info(f"  {i}. {technique}: {avg_time:.1f}ms avg")
        
        logger.info("\n📄 Retrieval Ranking (most to least documents):")
        for i, (technique, avg_docs) in enumerate(report["retrieval_ranking"], 1):
            logger.info(f"  {i}. {technique}: {avg_docs:.1f} docs avg")
        
        logger.info(f"\n💾 System Resources:")
        logger.info(f"  Peak memory: {report['system_metrics']['peak_memory_gb']:.1f} GB")
        logger.info(f"  Avg CPU: {report['system_metrics']['avg_cpu_percent']:.1f}%")
        
        # Show any failures
        failed_techniques = [r for r in report['detailed_results'] if not r['success']]
        if failed_techniques:
            logger.info(f"\n❌ Failed Techniques:")
            for r in failed_techniques:
                logger.info(f"  {r['technique']}: {r['error']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enterprise RAG Validator - Single Parameterized Script")
    parser.add_argument("--docs", type=int, default=5000, help="Number of documents to validate with (default: 5000)")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip document ingestion phase")
    parser.add_argument("--fast-mode", action="store_true", help="Use fewer test queries for faster validation")
    parser.add_argument("--skip-colbert", action="store_true", help="Skip ColBERT technique")
    parser.add_argument("--skip-noderag", action="store_true", help="Skip NodeRAG technique")
    parser.add_argument("--skip-graphrag", action="store_true", help="Skip GraphRAG technique")
    
    args = parser.parse_args()
    
    # Validate document count
    if args.docs < 100:
        logger.error("❌ Minimum document count is 100")
        return
    
    if args.docs > 100000:
        logger.warning("⚠️ Document count > 100K may require significant resources")
    
    logger.info(f"🚀 Starting Enterprise RAG Validation for {args.docs:,} documents...")
    
    # Build skip list
    skip_techniques = []
    if args.skip_colbert:
        skip_techniques.append("OptimizedColBERT")
    if args.skip_noderag:
        skip_techniques.append("NodeRAG")
    if args.skip_graphrag:
        skip_techniques.append("GraphRAG")
    
    if skip_techniques:
        logger.info(f"⏭️ Skipping techniques: {', '.join(skip_techniques)}")
    
    try:
        # Initialize validator
        validator = EnterpriseRAGValidator(args.docs, args.fast_mode)
        
        # Setup
        if not validator.setup():
            logger.error("❌ Setup failed")
            return
        
        # Ensure documents
        if not validator.ensure_documents(args.skip_ingestion):
            logger.error("❌ Document preparation failed")
            return
        
        # Run validation
        report = validator.run_validation(skip_techniques)
        
        # Print summary
        validator.print_summary(report)
        
        # Save results
        timestamp = int(time.time())
        results_file = f"enterprise_rag_validation_{args.docs}docs_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        # Final status
        if "error" not in report and report.get("success_rate", 0) > 0.8:
            logger.info(f"🎉 Enterprise validation PASSED at {args.docs:,} document scale!")
        else:
            logger.warning(f"⚠️ Enterprise validation needs attention at {args.docs:,} document scale")
        
        if validator.connection:
            validator.connection.close()
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()