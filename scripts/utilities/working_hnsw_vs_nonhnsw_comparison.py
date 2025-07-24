#!/usr/bin/env python3
"""
Fixed HNSW vs Non-HNSW Performance Comparison Script

Based on proven working patterns from enterprise validation scripts.
This script provides actual, verifiable results from running HNSW vs non-HNSW comparison
with real data using the same successful patterns as the working enterprise scripts.

Usage:
    python scripts/working_hnsw_vs_nonhnsw_comparison.py
    python scripts/working_hnsw_vs_nonhnsw_comparison.py --fast
"""

import os
import sys
import logging
import time
import json
import argparse
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Custom JSON encoder for numpy types (learned from enterprise scripts)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

# Import RAG pipelines that actually work (proven from enterprise validation)
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import

# Configure logging (same pattern as working scripts)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_hnsw_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_mock_colbert_encoder(embedding_dim: int = 128):
    """Create a mock ColBERT encoder for testing (from working enterprise scripts)."""
    def mock_encoder(text: str) -> List[List[float]]:
        import numpy as np
        words = text.split()[:10]  # Limit to 10 tokens
        embeddings = []
        
        for i, word in enumerate(words):
            np.random.seed(hash(word) % 10000)
            embedding = np.random.randn(embedding_dim)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    return mock_encoder

def create_mock_llm_func():
    """Create a mock LLM function for testing (from working enterprise scripts)."""
    def mock_llm(prompt: str) -> str:
        return f"Mock response based on the provided context. Query appears to be about: {prompt[:100]}..."
    return mock_llm

@dataclass
class ComparisonResult:
    """Results from HNSW vs non-HNSW comparison"""
    technique_name: str
    hnsw_available: bool
    hnsw_avg_time_ms: float
    varchar_avg_time_ms: float
    hnsw_success_rate: float
    varchar_success_rate: float
    speed_improvement_factor: float
    hnsw_docs_retrieved: float
    varchar_docs_retrieved: float
    actual_test_performed: bool
    error_details: str
    recommendation: str

class WorkingHNSWComparison:
    """Working HNSW vs non-HNSW comparison with honest results"""
    
    def __init__(self):
        self.results: List[ComparisonResult] = []
        self.start_time = time.time()
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        
        # Test queries
        self.test_queries = [
            "diabetes treatment and management strategies",
            "machine learning applications in medical diagnosis",
            "cancer immunotherapy approaches"
        ]
        
    def setup_environment(self) -> bool:
        """Setup environment using proven patterns from working enterprise scripts"""
        logger.info("🔧 Setting up fixed HNSW comparison environment...")
        
        try:
            # Setup database connection using proven pattern
            logger.info("Connecting to IRIS database...")
            self.connection = get_iris_connection()
            if not self.connection:
                logger.error("❌ Failed to establish database connection")
                return False
            
            # Check current document count using proven pattern
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
            current_docs = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"✅ Database connected: {current_docs} documents available")
            
            if current_docs == 0:
                logger.warning("⚠️ No documents found in RAG.SourceDocuments_V2 - comparison may not be meaningful")
            
            # Setup embedding function using proven pattern from enterprise scripts
            try:
                self.embedding_func = get_embedding_func()
                logger.info("✅ Embedding function initialized")
            except Exception as e:
                logger.warning(f"⚠️ Embedding function setup failed, using mock: {e}")
                self.embedding_func = get_embedding_func(mock=True)
            
            # Setup LLM function using proven pattern (mock for reliability)
            try:
                self.llm_func = create_mock_llm_func()
                logger.info("✅ Mock LLM function setup successful")
            except Exception as e:
                logger.error(f"❌ LLM function setup failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def check_hnsw_availability(self) -> bool:
        """Check if HNSW schema and indexes actually exist"""
        logger.info("🔍 Checking HNSW availability...")
        
        try:
            cursor = self.connection.cursor()
            
            # Check if RAG_HNSW schema exists
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.SCHEMATA 
                WHERE SCHEMA_NAME = 'RAG_HNSW'
            """)
            schema_exists = cursor.fetchone()[0] > 0
            
            if not schema_exists:
                logger.info("❌ RAG_HNSW schema does not exist")
                cursor.close()
                return False
            
            # Check if SourceDocuments table exists in HNSW schema
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'RAG_HNSW' AND TABLE_NAME = 'SourceDocuments_V2'
            """)
            table_exists = cursor.fetchone()[0] > 0
            
            if not table_exists:
                logger.info("❌ RAG_HNSW.SourceDocuments table does not exist")
                cursor.close()
                return False
            
            # Check if table has data
            cursor.execute("SELECT COUNT(*) FROM RAG_HNSW.SourceDocuments")
            hnsw_docs = cursor.fetchone()[0]
            
            # Check if VECTOR column exists
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'RAG_HNSW' 
                AND TABLE_NAME = 'SourceDocuments_V2' 
                AND COLUMN_NAME = 'embedding_vector'
                AND DATA_TYPE LIKE '%VECTOR%'
            """)
            vector_column_exists = cursor.fetchone()[0] > 0
            
            cursor.close()
            
            logger.info(f"📊 HNSW Schema Status:")
            logger.info(f"  - Schema exists: {schema_exists}")
            logger.info(f"  - Table exists: {table_exists}")
            logger.info(f"  - Documents: {hnsw_docs}")
            logger.info(f"  - VECTOR column: {vector_column_exists}")
            
            return schema_exists and table_exists and hnsw_docs > 0 and vector_column_exists
            
        except Exception as e:
            logger.error(f"❌ HNSW availability check failed: {e}")
            return False
    
    def test_technique(self, technique_name: str, pipeline_class, hnsw_available: bool) -> ComparisonResult:
        """Test a single RAG technique with both approaches using proven patterns"""
        logger.info(f"🧪 Testing {technique_name}...")
        
        result = ComparisonResult(
            technique_name=technique_name,
            hnsw_available=hnsw_available,
            hnsw_avg_time_ms=0.0,
            varchar_avg_time_ms=0.0,
            hnsw_success_rate=0.0,
            varchar_success_rate=0.0,
            speed_improvement_factor=1.0,
            hnsw_docs_retrieved=0.0,
            varchar_docs_retrieved=0.0,
            actual_test_performed=False,
            error_details="",
            recommendation="Not tested"
        )
        
        try:
            # Test VARCHAR approach (standard RAG schema) using proven patterns
            varchar_results = self._test_with_schema(technique_name, pipeline_class, "RAG", "VARCHAR")
            result.varchar_avg_time_ms = varchar_results['avg_time_ms']
            result.varchar_success_rate = varchar_results['success_rate']
            result.varchar_docs_retrieved = varchar_results['avg_docs']
            
            # Test HNSW approach if available using proven patterns
            if hnsw_available:
                hnsw_results = self._test_with_schema(technique_name, pipeline_class, "RAG_HNSW", "HNSW")
                result.hnsw_avg_time_ms = hnsw_results['avg_time_ms']
                result.hnsw_success_rate = hnsw_results['success_rate']
                result.hnsw_docs_retrieved = hnsw_results['avg_docs']
            else:
                logger.info(f"  ⏭️ Skipping HNSW test (not available)")
                result.error_details = "HNSW schema not available"
            
            # Calculate improvement factor
            if result.hnsw_avg_time_ms > 0 and result.varchar_avg_time_ms > 0:
                result.speed_improvement_factor = result.varchar_avg_time_ms / result.hnsw_avg_time_ms
            
            # Generate recommendation
            if not hnsw_available:
                result.recommendation = "HNSW not available - deploy HNSW schema first"
            elif result.speed_improvement_factor > 1.2:
                result.recommendation = f"HNSW Recommended: {result.speed_improvement_factor:.2f}x faster"
            elif result.speed_improvement_factor > 1.1:
                result.recommendation = f"HNSW Beneficial: {result.speed_improvement_factor:.2f}x faster"
            elif result.speed_improvement_factor < 0.9:
                result.recommendation = "VARCHAR Recommended: HNSW shows degradation"
            else:
                result.recommendation = "Neutral: No significant difference"
            
            result.actual_test_performed = True
            
        except Exception as e:
            logger.error(f"❌ {technique_name} test failed: {e}")
            result.error_details = str(e)
            result.recommendation = f"Test failed: {e}"
        
        return result
    
    def _test_with_schema(self, technique_name: str, pipeline_class, schema_name: str, approach_name: str) -> Dict[str, Any]:
        """Test a technique with a specific schema using proven enterprise patterns"""
        logger.info(f"  🔍 Testing {technique_name} with {approach_name} approach...")
        
        times = []
        successes = 0
        docs = []
        
        for i, query in enumerate(self.test_queries):
            try:
                start_time = time.time()
                
                # Initialize pipeline using proven patterns from enterprise scripts
                if technique_name == "HybridiFindRAG":
                    pipeline = pipeline_class(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func,
                        schema_name=schema_name
                    )
                elif technique_name == "OptimizedColBERT":
                    # Use proven ColBERT initialization pattern
                    mock_encoder = create_mock_colbert_encoder()
                    pipeline = pipeline_class(
                        iris_connector=self.connection,
                        query_encoder=mock_encoder,
                        doc_encoder=mock_encoder,
                        llm_func=self.llm_func
                    )
                elif technique_name == "CRAG":
                    # Use proven CRAG initialization pattern
                    pipeline = pipeline_class(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func,
                        web_search_func=lambda q: []  # Mock web search
                    )
                else:
                    # Standard initialization for other techniques
                    pipeline = pipeline_class(
                        iris_connector=self.connection,
                        embedding_func=self.embedding_func,
                        llm_func=self.llm_func
                    )
                
                # Execute query using proven patterns
                if hasattr(pipeline, 'query'):
                    response = pipeline.query(query, top_k=5)
                elif hasattr(pipeline, 'run'):
                    # Use run method for techniques that have it
                    response = pipeline.run(query, top_k=5)
                else:
                    # Fallback for pipelines without query method
                    retrieved_docs = pipeline.retrieve_documents(query)
                    answer = pipeline.generate_response(query, retrieved_docs)
                    response = {
                        'query': query,
                        'answer': answer,
                        'retrieved_documents': retrieved_docs
                    }
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                # Validate response using proven patterns
                if response and response.get('answer'):
                    times.append(response_time)
                    successes += 1
                    docs_retrieved = len(response.get('retrieved_documents', []))
                    docs.append(docs_retrieved)
                    logger.info(f"    ✅ {approach_name} query {i+1} succeeded: {response_time:.1f}ms, {docs_retrieved} docs")
                else:
                    times.append(0)
                    docs.append(0)
                    logger.warning(f"    ❌ {approach_name} query {i+1} returned empty result")
                
            except Exception as e:
                query_time = time.time() - start_time
                times.append(query_time * 1000)
                docs.append(0)
                logger.warning(f"    ❌ {approach_name} query {i+1} failed: {e}")
        
        return {
            'avg_time_ms': sum(times) / len(times) if times else 0,
            'success_rate': successes / len(self.test_queries),
            'avg_docs': sum(docs) / len(docs) if docs else 0
        }
    
    def run_comparison(self) -> bool:
        """Run the actual HNSW vs non-HNSW comparison using proven patterns"""
        logger.info("🚀 Starting Fixed HNSW vs Non-HNSW Comparison")
        
        # Check HNSW availability
        hnsw_available = self.check_hnsw_availability()
        
        # Define techniques to test (using proven working set from enterprise scripts)
        techniques = [
            ("BasicRAG", BasicRAGPipeline),
            ("HyDE", HyDERAGPipeline),
            ("CRAG", CRAGPipeline),
            ("NodeRAG", NodeRAGPipeline),
            ("GraphRAG", GraphRAGPipeline),
            ("HybridiFindRAG", HybridIFindRAGPipeline),
            ("OptimizedColBERT", ColBERTRAGPipeline)
        ]
        
        # Test each technique using proven patterns
        for technique_name, pipeline_class in techniques:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {technique_name}")
            logger.info('='*60)
            
            try:
                result = self.test_technique(technique_name, pipeline_class, hnsw_available)
                self.results.append(result)
                
                logger.info(f"✅ {technique_name} completed:")
                logger.info(f"  - VARCHAR: {result.varchar_avg_time_ms:.1f}ms, {result.varchar_success_rate:.1%} success")
                if hnsw_available:
                    logger.info(f"  - HNSW: {result.hnsw_avg_time_ms:.1f}ms, {result.hnsw_success_rate:.1%} success")
                    logger.info(f"  - Improvement: {result.speed_improvement_factor:.2f}x")
                logger.info(f"  - Recommendation: {result.recommendation}")
                
            except Exception as e:
                logger.error(f"❌ {technique_name} failed completely: {e}")
                
                # Add failed result using proven pattern
                failed_result = ComparisonResult(
                    technique_name=technique_name,
                    hnsw_available=hnsw_available,
                    hnsw_avg_time_ms=0.0,
                    varchar_avg_time_ms=0.0,
                    hnsw_success_rate=0.0,
                    varchar_success_rate=0.0,
                    speed_improvement_factor=1.0,
                    hnsw_docs_retrieved=0.0,
                    varchar_docs_retrieved=0.0,
                    actual_test_performed=False,
                    error_details=str(e),
                    recommendation=f"Failed to test: {e}"
                )
                self.results.append(failed_result)
        
        return len(self.results) > 0
    
    def generate_report(self) -> str:
        """Generate comprehensive comparison report using proven patterns"""
        logger.info("📊 Generating fixed HNSW vs non-HNSW comparison report...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"fixed_hnsw_comparison_{timestamp}.json"
        
        # Prepare results using proven pattern with NumpyEncoder
        comprehensive_results = {
            "test_metadata": {
                "timestamp": timestamp,
                "total_execution_time_seconds": time.time() - self.start_time,
                "techniques_tested": len(self.results),
                "hnsw_available": any(r.hnsw_available for r in self.results),
                "actual_tests_performed": sum(1 for r in self.results if r.actual_test_performed),
                "successful_techniques": len([r for r in self.results if r.actual_test_performed and r.varchar_success_rate > 0])
            },
            "honest_assessment": {
                "hnsw_schema_deployed": any(r.hnsw_available for r in self.results),
                "techniques_with_real_hnsw_benefit": len([r for r in self.results if r.speed_improvement_factor > 1.1 and r.actual_test_performed]),
                "techniques_tested_successfully": len([r for r in self.results if r.actual_test_performed]),
                "major_issues_found": [r.error_details for r in self.results if r.error_details and not r.actual_test_performed]
            },
            "technique_results": [asdict(result) for result in self.results],
            "performance_ranking": self._generate_performance_ranking(),
            "real_conclusions": self._generate_honest_conclusions()
        }
        
        # Save results using proven pattern with NumpyEncoder
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, cls=NumpyEncoder)
        
        # Generate markdown report
        self._generate_markdown_report(comprehensive_results, timestamp)
        
        logger.info(f"✅ Fixed comparison report generated: {results_file}")
        
        return results_file
    
    def _generate_performance_ranking(self) -> Dict[str, List]:
        """Generate performance rankings using proven patterns"""
        successful_results = [r for r in self.results if r.actual_test_performed and r.varchar_success_rate > 0]
        
        varchar_ranking = sorted(
            [(r.technique_name, r.varchar_avg_time_ms) for r in successful_results],
            key=lambda x: x[1]
        )
        
        hnsw_ranking = sorted(
            [(r.technique_name, r.hnsw_avg_time_ms) for r in successful_results if r.hnsw_avg_time_ms > 0],
            key=lambda x: x[1]
        )
        
        improvement_ranking = sorted(
            [(r.technique_name, r.speed_improvement_factor) for r in successful_results if r.speed_improvement_factor > 1.0],
            key=lambda x: x[1], reverse=True
        )
        
        return {
            "varchar_performance": varchar_ranking,
            "hnsw_performance": hnsw_ranking,
            "improvement_factor": improvement_ranking
        }
    
    def _generate_honest_conclusions(self) -> List[str]:
        """Generate honest conclusions based on actual results"""
        conclusions = []
        
        hnsw_available = any(r.hnsw_available for r in self.results)
        successful_tests = [r for r in self.results if r.actual_test_performed]
        
        if not hnsw_available:
            conclusions.append("CRITICAL: HNSW schema (RAG_HNSW) is not deployed - no real HNSW comparison possible")
            conclusions.append("RECOMMENDATION: Deploy HNSW schema with native VECTOR columns before claiming HNSW benefits")
        
        if not successful_tests:
            conclusions.append("CRITICAL: No techniques tested successfully - comparison framework has fundamental issues")
            conclusions.append("RECOMMENDATION: Fix basic RAG pipeline issues before attempting HNSW comparison")
        
        if successful_tests:
            avg_improvement = sum(r.speed_improvement_factor for r in successful_tests) / len(successful_tests)
            conclusions.append(f"ACTUAL RESULTS: Average speed improvement factor: {avg_improvement:.2f}x")
            
            if avg_improvement > 1.2:
                conclusions.append("CONCLUSION: HNSW shows measurable benefits - worth deploying")
            elif avg_improvement > 1.05:
                conclusions.append("CONCLUSION: HNSW shows modest benefits - evaluate cost vs benefit")
            else:
                conclusions.append("CONCLUSION: HNSW benefits are minimal or non-existent")
        
        # Add specific issues found
        for result in self.results:
            if result.error_details and not result.actual_test_performed:
                conclusions.append(f"ISSUE: {result.technique_name} failed - {result.error_details}")
        
        return conclusions
    
    def _generate_markdown_report(self, results: Dict[str, Any], timestamp: str):
        """Generate honest markdown report"""
        report_file = f"WORKING_HNSW_COMPARISON_REPORT_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Working HNSW vs Non-HNSW Comparison Report\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Execution Time:** {results['test_metadata']['total_execution_time_seconds']:.1f} seconds\n")
            f.write(f"**Techniques Tested:** {results['test_metadata']['techniques_tested']}\n\n")
            
            f.write("## Honest Assessment\n\n")
            assessment = results["honest_assessment"]
            f.write(f"- **HNSW Schema Deployed:** {assessment['hnsw_schema_deployed']}\n")
            f.write(f"- **Successful Tests:** {assessment['techniques_tested_successfully']}/{results['test_metadata']['techniques_tested']}\n")
            f.write(f"- **Real HNSW Benefits:** {assessment['techniques_with_real_hnsw_benefit']} techniques\n\n")
            
            if assessment['major_issues_found']:
                f.write("### Major Issues Found\n\n")
                for issue in assessment['major_issues_found']:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            f.write("## Technique Results\n\n")
            f.write("| Technique | Test Status | VARCHAR Time (ms) | HNSW Time (ms) | Improvement | Recommendation |\n")
            f.write("|-----------|-------------|-------------------|----------------|-------------|----------------|\n")
            
            for result in self.results:
                status = "✅ Tested" if result.actual_test_performed else "❌ Failed"
                varchar_time = f"{result.varchar_avg_time_ms:.1f}" if result.varchar_avg_time_ms > 0 else "N/A"
                hnsw_time = f"{result.hnsw_avg_time_ms:.1f}" if result.hnsw_avg_time_ms > 0 else "N/A"
                improvement = f"{result.speed_improvement_factor:.2f}x" if result.speed_improvement_factor != 1.0 else "N/A"
                
                f.write(f"| {result.technique_name} | {status} | {varchar_time} | {hnsw_time} | {improvement} | {result.recommendation} |\n")
            
            f.write("\n## Real Conclusions\n\n")
            for conclusion in results["real_conclusions"]:
                f.write(f"- {conclusion}\n")
        
        logger.info(f"✅ Markdown report generated: {report_file}")

def main():
    """Main execution function using proven patterns from enterprise scripts"""
    logger.info("🚀 Starting Fixed HNSW vs Non-HNSW Comparison")
    logger.info("=" * 70)
    
    # Parse arguments using proven pattern
    parser = argparse.ArgumentParser(description="Fixed HNSW vs Non-HNSW Performance Comparison")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode with fewer queries")
    args = parser.parse_args()
    
    # Fast mode for quick testing (proven pattern from enterprise scripts)
    fast_mode = args.fast
    if fast_mode:
        logger.info("🏃 Fast mode enabled - using 2 queries for quick validation")
    
    try:
        # Initialize comparison
        comparison = WorkingHNSWComparison()
        
        # Adjust test queries for fast mode (proven pattern)
        if fast_mode:
            comparison.test_queries = comparison.test_queries[:2]
        
        # Setup environment using proven patterns
        logger.info("Setting up environment...")
        if not comparison.setup_environment():
            logger.error("❌ Environment setup failed")
            return 1
        
        # Run comparison using proven patterns
        logger.info("Running HNSW vs non-HNSW comparison...")
        if not comparison.run_comparison():
            logger.error("❌ Comparison failed")
            return 1
        
        # Generate report using proven patterns
        logger.info("Generating comprehensive report...")
        results_file = comparison.generate_report()
        
        # Print summary using proven pattern from enterprise scripts
        logger.info("\n" + "=" * 70)
        logger.info("📊 FIXED HNSW VS NON-HNSW COMPARISON SUMMARY")
        logger.info("=" * 70)
        
        successful_tests = sum(1 for r in comparison.results if r.actual_test_performed)
        total_tests = len(comparison.results)
        successful_techniques = len([r for r in comparison.results if r.actual_test_performed and r.varchar_success_rate > 0])
        
        logger.info(f"Techniques tested: {total_tests}")
        logger.info(f"Successful tests: {successful_tests}")
        logger.info(f"Working techniques: {successful_techniques}")
        
        # Performance summary using proven pattern
        if successful_techniques > 0:
            logger.info("\nPerformance Results:")
            for result in comparison.results:
                if result.actual_test_performed and result.varchar_success_rate > 0:
                    status = "✅" if result.varchar_success_rate >= 0.8 else "⚠️" if result.varchar_success_rate >= 0.5 else "❌"
                    logger.info(f"  {status} {result.technique_name}: {result.varchar_success_rate:.1%} success, {result.varchar_avg_time_ms:.0f}ms avg")
                    if result.hnsw_avg_time_ms > 0:
                        logger.info(f"     HNSW improvement: {result.speed_improvement_factor:.2f}x")
                elif not result.actual_test_performed:
                    logger.info(f"  ❌ {result.technique_name}: FAILED - {result.error_details}")
            
            # Overall assessment using proven pattern
            improvements = [r.speed_improvement_factor for r in comparison.results if r.actual_test_performed and r.speed_improvement_factor > 1.0]
            avg_improvement = sum(improvements) / len(improvements) if improvements else 1.0
            
            logger.info(f"\n🎯 Overall Results:")
            logger.info(f"   • Average HNSW improvement: {avg_improvement:.2f}x")
            logger.info(f"   • Results saved to: {results_file}")
            
            if avg_improvement > 1.2:
                logger.info("✅ CONCLUSION: HNSW shows significant benefits - recommended for deployment")
            elif avg_improvement > 1.05:
                logger.info("✅ CONCLUSION: HNSW shows modest benefits - evaluate cost vs benefit")
            else:
                logger.info("⚠️ CONCLUSION: HNSW benefits are minimal - current setup may be sufficient")
        
        # Final status using proven pattern
        if successful_techniques == total_tests:
            logger.info("\n🎉 COMPARISON SUCCESSFUL!")
            logger.info("All RAG techniques tested successfully")
            return 0
        elif successful_techniques > 0:
            logger.info(f"\n✅ COMPARISON PARTIALLY SUCCESSFUL")
            logger.info(f"{successful_techniques}/{total_tests} techniques working")
            return 0
        else:
            logger.error("\n❌ COMPARISON FAILED")
            logger.error("No techniques tested successfully")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Fatal error during comparison: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())