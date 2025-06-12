#!/usr/bin/env python3
"""
Vector Search Community vs Licensed Edition Comparison

This script compares Vector Search functionality between:
- InterSystems IRIS Licensed Edition 2025.1 (port 1972)
- InterSystems IRIS Community Edition 2025.1 (port 1974)

Tests include:
1. VECTOR data type support
2. HNSW index creation capabilities
3. Vector function availability (TO_VECTOR, VECTOR_COSINE, etc.)
4. Performance differences
5. Feature limitations
"""

import logging
import time
import json
import sys
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorTestResult:
    """Results from a vector search test"""
    test_name: str
    success: bool
    execution_time_ms: float
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = None

@dataclass
class EditionComparison:
    """Comparison results between editions"""
    licensed_results: List[VectorTestResult]
    community_results: List[VectorTestResult]
    feature_comparison: Dict[str, Dict[str, bool]]
    performance_comparison: Dict[str, Dict[str, float]]
    recommendations: List[str]

class IRISConnection:
    """IRIS database connection wrapper using our existing connector"""
    
    def __init__(self, port: int, edition_name: str):
        self.port = port
        self.edition_name = edition_name
        self.iris_connector = None
        
    def connect(self) -> bool:
        """Connect to IRIS database"""
        try:
            # Use our existing IRIS connector with custom port configuration
            config = {
                "hostname": "localhost",
                "port": self.port,
                "namespace": "USER",
                "username": "_SYSTEM",
                "password": "SYS"
            }
            self.iris_connector = get_iris_connection(config=config)
            logger.info(f"✅ Connected to {self.edition_name} on port {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to {self.edition_name} on port {self.port}: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute a query and return results"""
        if not self.iris_connector:
            raise Exception("Not connected to database")
            
        return self.iris_connector.execute_query(query, params)
    
    def close(self):
        """Close database connection"""
        if self.iris_connector:
            self.iris_connector.close()
            logger.info(f"🔌 Disconnected from {self.edition_name}")

class VectorSearchTester:
    """Comprehensive Vector Search testing framework"""
    
    def __init__(self):
        self.licensed_conn = IRISConnection(1972, "Licensed Edition")
        self.community_conn = IRISConnection(1974, "Community Edition")
        
    def test_basic_connection(self, conn: IRISConnection) -> VectorTestResult:
        """Test basic database connection"""
        start_time = time.time()
        
        try:
            cursor = conn.execute_query("SELECT $HOROLOG")
            result = cursor.fetchone()
            execution_time = (time.time() - start_time) * 1000
            
            return VectorTestResult(
                test_name="basic_connection",
                success=True,
                execution_time_ms=execution_time,
                result_data=result[0] if result else None
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return VectorTestResult(
                test_name="basic_connection",
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_vector_data_type(self, conn: IRISConnection) -> VectorTestResult:
        """Test VECTOR data type support"""
        start_time = time.time()
        
        try:
            # Try to create a table with VECTOR column
            test_table = f"test_vector_type_{int(time.time())}"
            
            cursor = conn.execute_query(f"DROP TABLE IF EXISTS {test_table}")
            cursor = conn.execute_query(f"""
                CREATE TABLE {test_table} (
                    id INTEGER PRIMARY KEY,
                    test_vector VECTOR(FLOAT, 384)
                )
            """)
            
            # Check the actual column type
            cursor = conn.execute_query(f"""
                SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{test_table}' AND COLUMN_NAME = 'test_vector'
            """)
            
            result = cursor.fetchone()
            actual_type = result[0] if result else "UNKNOWN"
            
            # Clean up
            conn.execute_query(f"DROP TABLE IF EXISTS {test_table}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return VectorTestResult(
                test_name="vector_data_type",
                success='VECTOR' in actual_type.upper(),
                execution_time_ms=execution_time,
                result_data=actual_type,
                additional_info={"actual_column_type": actual_type}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return VectorTestResult(
                test_name="vector_data_type",
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_to_vector_function(self, conn: IRISConnection) -> VectorTestResult:
        """Test TO_VECTOR function"""
        start_time = time.time()
        
        try:
            # Test TO_VECTOR with correct syntax
            cursor = conn.execute_query("SELECT TO_VECTOR('0.1,0.2,0.3', double) as result")
            result = cursor.fetchone()
            
            execution_time = (time.time() - start_time) * 1000
            
            return VectorTestResult(
                test_name="to_vector_function",
                success=result is not None,
                execution_time_ms=execution_time,
                result_data=str(result[0]) if result else None
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return VectorTestResult(
                test_name="to_vector_function",
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_vector_cosine_function(self, conn: IRISConnection) -> VectorTestResult:
        """Test VECTOR_COSINE function"""
        start_time = time.time()
        
        try:
            cursor = conn.execute_query("""
                SELECT VECTOR_COSINE(
                    TO_VECTOR('0.1,0.2,0.3', double),
                    TO_VECTOR('0.4,0.5,0.6', double)
                ) as similarity
            """)
            result = cursor.fetchone()
            
            execution_time = (time.time() - start_time) * 1000
            
            return VectorTestResult(
                test_name="vector_cosine_function",
                success=result is not None,
                execution_time_ms=execution_time,
                result_data=float(result[0]) if result else None
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return VectorTestResult(
                test_name="vector_cosine_function",
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_vector_dot_product_function(self, conn: IRISConnection) -> VectorTestResult:
        """Test VECTOR_DOT_PRODUCT function"""
        start_time = time.time()
        
        try:
            cursor = conn.execute_query("""
                SELECT VECTOR_DOT_PRODUCT(
                    TO_VECTOR('1.0,2.0,3.0', double),
                    TO_VECTOR('4.0,5.0,6.0', double)
                ) as dot_product
            """)
            result = cursor.fetchone()
            
            execution_time = (time.time() - start_time) * 1000
            
            return VectorTestResult(
                test_name="vector_dot_product_function",
                success=result is not None,
                execution_time_ms=execution_time,
                result_data=float(result[0]) if result else None
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return VectorTestResult(
                test_name="vector_dot_product_function",
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_hnsw_index_creation(self, conn: IRISConnection) -> VectorTestResult:
        """Test HNSW index creation"""
        start_time = time.time()
        
        try:
            test_table = f"test_hnsw_{int(time.time())}"
            
            # Create table with VECTOR column
            cursor = conn.execute_query(f"DROP TABLE IF EXISTS {test_table}")
            cursor = conn.execute_query(f"""
                CREATE TABLE {test_table} (
                    id INTEGER PRIMARY KEY,
                    test_vector VECTOR(FLOAT, 384)
                )
            """)
            
            # Try to create HNSW index
            index_name = f"idx_hnsw_{int(time.time())}"
            cursor = conn.execute_query(f"""
                CREATE INDEX {index_name} ON {test_table} (test_vector)
                AS HNSW(Distance='Cosine')
            """)
            
            # Verify index was created
            cursor = conn.execute_query(f"""
                SELECT INDEX_NAME FROM INFORMATION_SCHEMA.STATISTICS
                WHERE TABLE_NAME = '{test_table}' AND INDEX_NAME = '{index_name}'
            """)
            
            index_result = cursor.fetchone()
            
            # Clean up
            conn.execute_query(f"DROP TABLE IF EXISTS {test_table}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return VectorTestResult(
                test_name="hnsw_index_creation",
                success=index_result is not None,
                execution_time_ms=execution_time,
                result_data=index_result[0] if index_result else None,
                additional_info={"index_name": index_name}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return VectorTestResult(
                test_name="hnsw_index_creation",
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def test_vector_search_performance(self, conn: IRISConnection) -> VectorTestResult:
        """Test vector search performance with sample data"""
        start_time = time.time()
        
        try:
            test_table = f"test_perf_{int(time.time())}"
            
            # Create table and insert test data
            cursor = conn.execute_query(f"DROP TABLE IF EXISTS {test_table}")
            cursor = conn.execute_query(f"""
                CREATE TABLE {test_table} (
                    id INTEGER PRIMARY KEY,
                    test_vector VECTOR(FLOAT, 3)
                )
            """)
            
            # Insert test vectors
            test_vectors = [
                "1.0,0.0,0.0",
                "0.0,1.0,0.0", 
                "0.0,0.0,1.0",
                "0.5,0.5,0.0",
                "0.3,0.3,0.4"
            ]
            
            for i, vector_str in enumerate(test_vectors):
                conn.execute_query(f"""
                    INSERT INTO {test_table} (id, test_vector)
                    VALUES ({i+1}, TO_VECTOR('{vector_str}', double))
                """)
            
            # Perform vector search
            search_start = time.time()
            cursor = conn.execute_query(f"""
                SELECT id, VECTOR_COSINE(test_vector, TO_VECTOR('1.0,1.0,1.0', double)) as similarity
                FROM {test_table}
                ORDER BY similarity DESC
            """)
            
            results = cursor.fetchall()
            search_time = (time.time() - search_start) * 1000
            
            # Clean up
            conn.execute_query(f"DROP TABLE IF EXISTS {test_table}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return VectorTestResult(
                test_name="vector_search_performance",
                success=len(results) > 0,
                execution_time_ms=execution_time,
                result_data=len(results),
                additional_info={
                    "search_time_ms": search_time,
                    "results_count": len(results),
                    "top_similarity": float(results[0][1]) if results else None
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return VectorTestResult(
                test_name="vector_search_performance",
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def run_all_tests(self, conn: IRISConnection) -> List[VectorTestResult]:
        """Run all vector search tests on a connection"""
        logger.info(f"🧪 Running all tests on {conn.edition_name}...")
        
        tests = [
            self.test_basic_connection,
            self.test_vector_data_type,
            self.test_to_vector_function,
            self.test_vector_cosine_function,
            self.test_vector_dot_product_function,
            self.test_hnsw_index_creation,
            self.test_vector_search_performance
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func(conn)
                results.append(result)
                
                status = "✅" if result.success else "❌"
                logger.info(f"  {status} {result.test_name}: {result.execution_time_ms:.1f}ms")
                
                if result.error_message:
                    logger.warning(f"    Error: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"  ❌ {test_func.__name__} failed: {e}")
                results.append(VectorTestResult(
                    test_name=test_func.__name__,
                    success=False,
                    execution_time_ms=0,
                    error_message=str(e)
                ))
        
        return results
    
    def compare_editions(self) -> EditionComparison:
        """Compare Vector Search capabilities between editions"""
        logger.info("🔍 Starting Vector Search comparison between editions...")
        
        # Connect to both editions
        if not self.licensed_conn.connect():
            raise Exception("Failed to connect to Licensed Edition")
            
        if not self.community_conn.connect():
            raise Exception("Failed to connect to Community Edition")
        
        try:
            # Run tests on both editions
            licensed_results = self.run_all_tests(self.licensed_conn)
            community_results = self.run_all_tests(self.community_conn)
            
            # Create feature comparison
            feature_comparison = {}
            performance_comparison = {}
            
            for licensed, community in zip(licensed_results, community_results):
                test_name = licensed.test_name
                
                feature_comparison[test_name] = {
                    "licensed": licensed.success,
                    "community": community.success
                }
                
                performance_comparison[test_name] = {
                    "licensed_ms": licensed.execution_time_ms,
                    "community_ms": community.execution_time_ms
                }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(licensed_results, community_results)
            
            return EditionComparison(
                licensed_results=licensed_results,
                community_results=community_results,
                feature_comparison=feature_comparison,
                performance_comparison=performance_comparison,
                recommendations=recommendations
            )
            
        finally:
            self.licensed_conn.close()
            self.community_conn.close()
    
    def _generate_recommendations(self, licensed_results: List[VectorTestResult], 
                                community_results: List[VectorTestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Count successful tests
        licensed_success = sum(1 for r in licensed_results if r.success)
        community_success = sum(1 for r in community_results if r.success)
        
        total_tests = len(licensed_results)
        
        recommendations.append(f"Licensed Edition: {licensed_success}/{total_tests} tests passed")
        recommendations.append(f"Community Edition: {community_success}/{total_tests} tests passed")
        
        # Feature-specific recommendations
        for licensed, community in zip(licensed_results, community_results):
            test_name = licensed.test_name
            
            if licensed.success and not community.success:
                recommendations.append(f"❌ {test_name}: Only available in Licensed Edition")
            elif not licensed.success and community.success:
                recommendations.append(f"✅ {test_name}: Available in Community Edition only")
            elif licensed.success and community.success:
                recommendations.append(f"✅ {test_name}: Available in both editions")
            else:
                recommendations.append(f"❌ {test_name}: Not available in either edition")
        
        # Overall recommendation
        if community_success >= licensed_success * 0.8:
            recommendations.append("🎯 Community Edition provides good Vector Search support")
        else:
            recommendations.append("🎯 Licensed Edition required for full Vector Search capabilities")
        
        return recommendations

def generate_comparison_report(comparison: EditionComparison) -> str:
    """Generate a detailed comparison report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"VECTOR_SEARCH_COMMUNITY_VS_LICENSED_COMPARISON_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Vector Search: Community vs Licensed Edition Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Feature availability summary
        licensed_features = sum(1 for test, results in comparison.feature_comparison.items() 
                              if results["licensed"])
        community_features = sum(1 for test, results in comparison.feature_comparison.items() 
                               if results["community"])
        total_features = len(comparison.feature_comparison)
        
        f.write(f"- **Licensed Edition:** {licensed_features}/{total_features} features available\n")
        f.write(f"- **Community Edition:** {community_features}/{total_features} features available\n")
        f.write(f"- **Feature Parity:** {(community_features/licensed_features*100):.1f}%\n\n")
        
        f.write("## Feature Comparison\n\n")
        f.write("| Feature | Licensed | Community | Notes |\n")
        f.write("|---------|----------|-----------|-------|\n")
        
        for test_name, results in comparison.feature_comparison.items():
            licensed_status = "✅" if results["licensed"] else "❌"
            community_status = "✅" if results["community"] else "❌"
            
            # Find corresponding results for notes
            licensed_result = next((r for r in comparison.licensed_results if r.test_name == test_name), None)
            community_result = next((r for r in comparison.community_results if r.test_name == test_name), None)
            
            notes = ""
            if licensed_result and community_result:
                if licensed_result.success and not community_result.success:
                    notes = "Licensed only"
                elif not licensed_result.success and community_result.success:
                    notes = "Community only"
                elif licensed_result.success and community_result.success:
                    notes = "Both editions"
                else:
                    notes = "Neither edition"
            
            f.write(f"| {test_name.replace('_', ' ').title()} | {licensed_status} | {community_status} | {notes} |\n")
        
        f.write("\n## Performance Comparison\n\n")
        f.write("| Test | Licensed (ms) | Community (ms) | Difference |\n")
        f.write("|------|---------------|----------------|------------|\n")
        
        for test_name, perf in comparison.performance_comparison.items():
            licensed_time = perf["licensed_ms"]
            community_time = perf["community_ms"]
            
            if licensed_time > 0 and community_time > 0:
                diff_pct = ((community_time - licensed_time) / licensed_time) * 100
                diff_str = f"{diff_pct:+.1f}%"
            else:
                diff_str = "N/A"
            
            f.write(f"| {test_name.replace('_', ' ').title()} | {licensed_time:.1f} | {community_time:.1f} | {diff_str} |\n")
        
        f.write("\n## Detailed Test Results\n\n")
        
        f.write("### Licensed Edition Results\n\n")
        for result in comparison.licensed_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            f.write(f"- **{result.test_name}:** {status} ({result.execution_time_ms:.1f}ms)\n")
            if result.error_message:
                f.write(f"  - Error: {result.error_message}\n")
            if result.additional_info:
                for key, value in result.additional_info.items():
                    f.write(f"  - {key}: {value}\n")
        
        f.write("\n### Community Edition Results\n\n")
        for result in comparison.community_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            f.write(f"- **{result.test_name}:** {status} ({result.execution_time_ms:.1f}ms)\n")
            if result.error_message:
                f.write(f"  - Error: {result.error_message}\n")
            if result.additional_info:
                for key, value in result.additional_info.items():
                    f.write(f"  - {key}: {value}\n")
        
        f.write("\n## Recommendations\n\n")
        for recommendation in comparison.recommendations:
            f.write(f"- {recommendation}\n")
        
        f.write("\n## Technical Details\n\n")
        f.write("### Test Environment\n")
        f.write("- **Licensed Edition:** InterSystems IRIS 2025.1 (port 1972)\n")
        f.write("- **Community Edition:** InterSystems IRIS Community 2025.1 (port 1974)\n")
        f.write("- **Test Framework:** Python with pyodbc\n")
        f.write("- **Vector Dimensions:** 3-384 dimensions tested\n")
        f.write("- **Distance Metrics:** Cosine similarity, Dot product\n\n")
        
        f.write("### Key Findings\n")
        
        # Analyze key differences
        vector_type_licensed = comparison.feature_comparison.get("vector_data_type", {}).get("licensed", False)
        vector_type_community = comparison.feature_comparison.get("vector_data_type", {}).get("community", False)
        
        hnsw_licensed = comparison.feature_comparison.get("hnsw_index_creation", {}).get("licensed", False)
        hnsw_community = comparison.feature_comparison.get("hnsw_index_creation", {}).get("community", False)
        
        if vector_type_licensed and not vector_type_community:
            f.write("- VECTOR data type is only available in Licensed Edition\n")
        elif vector_type_licensed and vector_type_community:
            f.write("- VECTOR data type is available in both editions\n")
        
        if hnsw_licensed and not hnsw_community:
            f.write("- HNSW indexing is only available in Licensed Edition\n")
        elif hnsw_licensed and hnsw_community:
            f.write("- HNSW indexing is available in both editions\n")
        
    logger.info(f"📊 Comparison report saved to: {report_file}")
    return report_file

def main():
    """Main execution function"""
    logger.info("🚀 Starting Vector Search Community vs Licensed Edition Comparison")
    
    try:
        # Initialize tester
        tester = VectorSearchTester()
        
        # Run comparison
        comparison = tester.compare_editions()
        
        # Generate report
        report_file = generate_comparison_report(comparison)
        
        # Save raw results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"vector_search_comparison_results_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            # Convert dataclasses to dict for JSON serialization
            comparison_dict = {
                "licensed_results": [asdict(r) for r in comparison.licensed_results],
                "community_results": [asdict(r) for r in comparison.community_results],
                "feature_comparison": comparison.feature_comparison,
                "performance_comparison": comparison.performance_comparison,
                "recommendations": comparison.recommendations,
                "timestamp": timestamp
            }
            json.dump(comparison_dict, f, indent=2)
        
        logger.info(f"📊 Raw results saved to: {json_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("🎉 VECTOR SEARCH COMPARISON COMPLETED!")
        logger.info("="*80)
        
        licensed_success = sum(1 for r in comparison.licensed_results if r.success)
        community_success = sum(1 for r in comparison.community_results if r.success)
        total_tests = len(comparison.licensed_results)
        
        logger.info(f"📊 Licensed Edition: {licensed_success}/{total_tests} tests passed")
        logger.info(f"📊 Community Edition: {community_success}/{total_tests} tests passed")
        logger.info(f"📊 Feature Parity: {(community_success/licensed_success*100):.1f}%")
        
        logger.info(f"\n📄 Reports generated:")
        logger.info(f"   - Detailed report: {report_file}")
        logger.info(f"   - Raw data: {json_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)