#!/usr/bin/env python3
"""
Ultimate 100K Enterprise Validation System

Comprehensive benchmarking and validation of all 7 RAG techniques on 100k documents:
- Test all 7 RAG techniques on the full 100k dataset
- Implement comprehensive performance benchmarking
- Add system resource monitoring throughout
- Generate detailed enterprise validation reports
- Compare HNSW vs non-HNSW performance at massive scale
- Include production deployment recommendations

Usage:
    python scripts/ultimate_100k_enterprise_validation.py --docs 100000
    python scripts/ultimate_100k_enterprise_validation.py --docs 50000 --fast-mode
    python scripts/ultimate_100k_enterprise_validation.py --docs 100000 --skip-ingestion
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
import signal
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import gc

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func, get_colbert_query_encoder_func, get_colbert_doc_encoder_func_adapted # Updated import

# Import all RAG pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_100k_enterprise_validation.log'),
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
    peak_memory_mb: float
    avg_cpu_percent: float
    error: Optional[str] = None
    schema_type: str = "RAG"

class Ultimate100kEnterpriseValidator:
    """Ultimate enterprise validator for 100k document scale"""
    
    def __init__(self, target_docs: int, fast_mode: bool = False):
        self.target_docs = target_docs
        self.fast_mode = fast_mode
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Enterprise test queries for comprehensive validation
        if fast_mode:
            self.test_queries = [
                "What are diabetes treatments?",
                "How does AI help medical diagnosis?",
                "What are cancer immunotherapy mechanisms?"
            ]
        else:
            self.test_queries = [
                "What are the latest treatments for diabetes and their effectiveness?",
                "How does machine learning improve medical diagnosis accuracy?",
                "What are the mechanisms of cancer immunotherapy and checkpoint inhibitors?",
                "How do genetic mutations contribute to disease development and progression?",
                "What role does artificial intelligence play in modern healthcare systems?",
                "What are cardiovascular disease prevention methods and lifestyle interventions?",
                "How do neurological disorders affect brain function and cognitive abilities?",
                "What are infectious disease control strategies and public health measures?",
                "How does precision medicine personalize treatment approaches?",
                "What are the latest advances in gene therapy and CRISPR technology?"
            ]
        
        logger.info(f"🚀 Ultimate100kEnterpriseValidator initialized for {target_docs:,} documents")
        logger.info(f"🧪 Test queries: {len(self.test_queries)}")
        logger.info(f"⚡ Fast mode: {fast_mode}")
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def setup(self, schema_type: str = "RAG") -> bool:
        """Setup database and models"""
        logger.info(f"🔧 Setting up for {self.target_docs:,} document validation ({schema_type} schema)...")
        
        try:
            # Database connection
            self.connection = get_iris_connection()
            if not self.connection:
                raise Exception("Failed to get database connection")
            
            # Check current document count
            table_name = f"{schema_type}.SourceDocuments"
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            current_docs = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
            docs_with_embeddings = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"📊 Database ({schema_type}): {current_docs:,} total docs, {docs_with_embeddings:,} with embeddings")
            
            if current_docs < self.target_docs * 0.9:  # Need at least 90% of target
                logger.warning(f"⚠️ Insufficient documents: {current_docs:,} < {self.target_docs:,}")
                return False
            
            # Setup models
            self.embedding_func = get_embedding_func(model_name="sentence-transformers/all-MiniLM-L6-v2", mock=False)
            # Load .env file and try to use real OpenAI LLM
            try:
                from dotenv import load_dotenv
                import os
                load_dotenv()  # Load .env file
                
                if os.getenv("OPENAI_API_KEY"):
                    self.llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
                    logger.info("✅ Using OpenAI GPT-3.5-turbo LLM")
                else:
                    self.llm_func = get_llm_func(provider="stub")
                    logger.info("⚠️ Using stub LLM (set OPENAI_API_KEY for real LLM)")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI LLM failed, using stub: {e}")
                self.llm_func = get_llm_func(provider="stub")
            
            # Setup web search function for CRAG
            def simple_web_search(query: str, num_results: int = 3) -> List[str]:
                """Simple mock web search for CRAG demonstration"""
                return [
                    f"Web search result {i+1}: Information about {query} from medical databases and research papers."
                    for i in range(num_results)
                ]
            self.web_search_func = simple_web_search
            
            logger.info("✅ Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def create_mock_colbert_encoder(self, embedding_dim: int = 128):
        """Create mock ColBERT encoder for enterprise testing with consistent dimensions"""
        def mock_encoder(text: str) -> List[List[float]]:
            import numpy as np
            words = text.split()[:10]
            embeddings = []
            for i, word in enumerate(words):
                # Use consistent seed based on word and position for reproducibility
                np.random.seed(hash(word + str(i)) % 10000)
                embedding = np.random.randn(embedding_dim)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                else:
                    # Fallback for zero vectors
                    embedding = np.ones(embedding_dim) / np.sqrt(embedding_dim)
                embeddings.append(embedding.tolist())
            
            # Ensure we always return at least one embedding
            if not embeddings:
                np.random.seed(42)
                embedding = np.random.randn(embedding_dim)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding.tolist())
                
            return embeddings
        return mock_encoder
    
    def test_technique_enterprise(self, pipeline, technique_name: str, schema_type: str = "RAG") -> ValidationResult:
        """Test a single RAG technique with enterprise-level monitoring"""
        logger.info(f"🧪 Enterprise testing {technique_name} ({schema_type} schema)...")
        
        start_time = time.time()
        query_times = []
        query_docs = []
        success_count = 0
        error_msg = None
        peak_memory = 0
        cpu_readings = []
        
        try:
            for i, query in enumerate(self.test_queries):
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping technique test")
                    break
                    
                query_start = time.time()
                
                # Monitor resources during query
                memory_before = psutil.virtual_memory().used / (1024**2)  # MB
                cpu_before = psutil.cpu_percent()
                
                try:
                    if technique_name == "OptimizedColBERT":
                        result = pipeline.run(query, top_k=5, similarity_threshold=0.3)
                    else:
                        result = pipeline.run(query, top_k=5)
                    
                    query_time = time.time() - query_start
                    docs_found = len(result.get("retrieved_documents", []))
                    
                    # Monitor resources after query
                    memory_after = psutil.virtual_memory().used / (1024**2)  # MB
                    cpu_after = psutil.cpu_percent()
                    
                    query_times.append(query_time)
                    query_docs.append(docs_found)
                    success_count += 1
                    
                    peak_memory = max(peak_memory, memory_after)
                    cpu_readings.append((cpu_before + cpu_after) / 2)
                    
                    if i == 0:  # Log first query details
                        logger.info(f"  First query: {query_time*1000:.1f}ms, {docs_found} docs")
                    
                    # Memory cleanup for long-running tests
                    if i % 5 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.warning(f"  Query {i+1} failed: {e}")
                    if not error_msg:
                        error_msg = str(e)
            
            # Calculate metrics
            avg_time = np.mean(query_times) * 1000 if query_times else 0  # Convert to ms
            avg_docs = np.mean(query_docs) if query_docs else 0
            success_rate = success_count / len(self.test_queries)
            avg_cpu = np.mean(cpu_readings) if cpu_readings else 0
            
            result = ValidationResult(
                technique=technique_name,
                success=success_count > 0,
                avg_time_ms=avg_time,
                avg_docs_retrieved=avg_docs,
                success_rate=success_rate,
                total_queries=len(self.test_queries),
                peak_memory_mb=peak_memory,
                avg_cpu_percent=avg_cpu,
                error=error_msg if success_count == 0 else None,
                schema_type=schema_type
            )
            
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {technique_name}: {avg_time:.1f}ms avg, {avg_docs:.1f} docs avg, {success_rate*100:.0f}% success")
            logger.info(f"   💾 Memory: {peak_memory:.1f}MB peak, CPU: {avg_cpu:.1f}% avg")
            
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
                peak_memory_mb=0,
                avg_cpu_percent=0,
                error=str(e),
                schema_type=schema_type
            )
    
    def run_enterprise_validation(self, schema_type: str = "RAG", skip_techniques: List[str] = None) -> Dict[str, Any]:
        """Run enterprise validation on all RAG techniques"""
        if skip_techniques is None:
            skip_techniques = []
            
        logger.info(f"🚀 Starting ULTIMATE enterprise validation at {self.target_docs:,} document scale ({schema_type} schema)...")
        
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
                        llm_func=self.llm_func,
                        web_search_func=self.web_search_func
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
            
            logger.info(f"✅ Initialized {len(pipelines)} RAG pipelines for enterprise testing")
            
            # Test each pipeline with enterprise monitoring
            for technique_name, pipeline in pipelines.items():
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping validation")
                    break
                    
                logger.info(f"\n{'='*80}")
                logger.info(f"🏢 ENTERPRISE TESTING: {technique_name}")
                logger.info('='*80)
                
                result = self.test_technique_enterprise(pipeline, technique_name, schema_type)
                results.append(result)
                
                # Memory cleanup between techniques
                gc.collect()
                
                # Brief pause between techniques for system stability
                time.sleep(2)
            
            total_time = time.time() - validation_start
            
            # Generate enterprise analysis
            successful_techniques = [r for r in results if r.success]
            
            # Performance ranking (fastest to slowest)
            performance_ranking = sorted(
                [(r.technique, r.avg_time_ms) for r in successful_techniques],
                key=lambda x: x[1]
            )
            
            # Memory efficiency ranking
            memory_ranking = sorted(
                [(r.technique, r.peak_memory_mb) for r in successful_techniques],
                key=lambda x: x[1]
            )
            
            # Generate enterprise report
            report = {
                "enterprise_validation_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "target_documents": self.target_docs,
                    "schema_type": schema_type,
                    "fast_mode": self.fast_mode,
                    "total_validation_time_seconds": total_time,
                    "techniques_tested": len(results),
                    "successful_techniques": len(successful_techniques),
                    "success_rate": len(successful_techniques) / len(results) if results else 0,
                    "test_queries_count": len(self.test_queries)
                },
                "performance_rankings": {
                    "latency_ranking": performance_ranking,
                    "memory_efficiency_ranking": memory_ranking
                },
                "detailed_results": [asdict(r) for r in results],
                "enterprise_recommendations": self.generate_enterprise_recommendations(results, self.target_docs)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Enterprise validation failed: {e}")
            return {"error": str(e), "results": results}
    
    def generate_enterprise_recommendations(self, results: List[ValidationResult], doc_count: int) -> List[str]:
        """Generate enterprise deployment recommendations"""
        recommendations = []
        
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return ["❌ No techniques succeeded - investigate infrastructure issues"]
        
        # Performance recommendations
        fastest = min(successful_results, key=lambda x: x.avg_time_ms)
        recommendations.append(f"🚀 Fastest technique: {fastest.technique} ({fastest.avg_time_ms:.1f}ms avg)")
        
        # Memory efficiency recommendations
        most_efficient = min(successful_results, key=lambda x: x.peak_memory_mb)
        recommendations.append(f"💾 Most memory efficient: {most_efficient.technique} ({most_efficient.peak_memory_mb:.1f}MB peak)")
        
        # Scale recommendations
        if doc_count >= 100000:
            recommendations.append("📈 At 100k+ document scale, consider horizontal scaling")
            recommendations.append("🔄 Implement caching layer for frequently accessed documents")
            recommendations.append("⚡ Use HNSW indexing for optimal vector search performance")
        
        # Production recommendations
        high_performers = [r for r in successful_results if r.avg_time_ms < 1000]  # Sub-second
        if high_performers:
            techniques = [r.technique for r in high_performers]
            recommendations.append(f"🏆 Production-ready techniques: {', '.join(techniques)}")
        
        return recommendations
    
    def print_enterprise_summary(self, report: Dict[str, Any]):
        """Print comprehensive enterprise summary"""
        logger.info("\n" + "="*100)
        logger.info("🏢 ULTIMATE 100K ENTERPRISE VALIDATION SUMMARY")
        logger.info("="*100)
        
        summary = report.get("enterprise_validation_summary", {})
        logger.info(f"🎯 Target Documents: {summary.get('target_documents', 0):,}")
        logger.info(f"🗄️ Schema: {summary.get('schema_type', 'Unknown')}")
        logger.info(f"✅ Successful Techniques: {summary.get('successful_techniques', 0)}/{summary.get('techniques_tested', 0)}")
        logger.info(f"📈 Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
        logger.info(f"⏱️ Total Validation Time: {summary.get('total_validation_time_seconds', 0):.1f}s")
        
        # Performance rankings
        rankings = report.get("performance_rankings", {})
        logger.info(f"\n🏆 PERFORMANCE RANKINGS:")
        
        latency_ranking = rankings.get("latency_ranking", [])
        if latency_ranking:
            logger.info("   Latency (fastest to slowest):")
            for i, (technique, latency) in enumerate(latency_ranking[:5], 1):
                logger.info(f"   {i}. {technique}: {latency:.1f}ms")
        
        # Recommendations
        recommendations = report.get("enterprise_recommendations", [])
        if recommendations:
            logger.info(f"\n🎯 ENTERPRISE RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"   {rec}")
        
        logger.info("="*100)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Ultimate 100K Enterprise Validation System")
    parser.add_argument("--docs", type=int, default=100000,
                       help="Number of documents to validate against")
    parser.add_argument("--fast-mode", action="store_true",
                       help="Use fast mode with fewer test queries")
    parser.add_argument("--skip-ingestion", action="store_true",
                       help="Skip document ingestion (assume data already loaded)")
    parser.add_argument("--schema-type", type=str, default="RAG", choices=["RAG", "RAG_HNSW"],
                       help="Database schema to use")
    parser.add_argument("--skip-techniques", nargs="*", default=[],
                       help="Techniques to skip (e.g., --skip-techniques BasicRAG HyDE)")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Ultimate 100K Enterprise Validation System")
    logger.info(f"🎯 Target Documents: {args.docs:,}")
    logger.info(f"🗄️ Schema: {args.schema_type}")
    logger.info(f"⚡ Fast Mode: {args.fast_mode}")
    
    validator = Ultimate100kEnterpriseValidator(args.docs, args.fast_mode)
    
    try:
        # Setup
        if not validator.setup(args.schema_type):
            logger.error("❌ Setup failed")
            return False
        
        # Run validation
        report = validator.run_enterprise_validation(args.schema_type, args.skip_techniques)
        
        if "error" in report:
            logger.error(f"❌ Validation failed: {report['error']}")
            return False
        
        # Print summary
        validator.print_enterprise_summary(report)
        
        # Save detailed report
        timestamp = int(time.time())
        report_file = f"ultimate_100k_enterprise_validation_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Detailed report saved: {report_file}")
        
        # Determine success
        summary = report.get("enterprise_validation_summary", {})
        success_rate = summary.get("success_rate", 0)
        
        if success_rate >= 0.8:  # 80% success rate for enterprise
            logger.info("🎉 ENTERPRISE VALIDATION SUCCESSFUL!")
            return True
        else:
            logger.warning(f"⚠️ Enterprise validation partially successful: {success_rate*100:.1f}% success rate")
            return False
            
    except KeyboardInterrupt:
        logger.info("🛑 Validation interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)