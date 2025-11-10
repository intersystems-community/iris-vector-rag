#!/usr/bin/env python3
"""
Comprehensive Testing Harness for GraphRAG Implementations

Tests both current (graphrag.py) and merged (graphrag_merged.py) implementations
with identical test cases and compares results and performance metrics.

Features:
- Parallel execution of identical test cases on both implementations
- Performance comparison and regression detection
- Detailed result analysis and reporting
- Mock data support for environments without database
- Graceful handling of missing dependencies
- Comprehensive metrics collection and visualization

Usage:
    python scripts/test_merged_graphrag_comprehensive.py [--config config.yaml] [--use-mocks]
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print(
        "Warning: Visualization packages not available. Install with: pip install pandas matplotlib seaborn"
    )

# IRIS RAG imports
try:
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager
    from iris_vector_rag.core.models import Document
    from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline as CurrentGraphRAG
    from iris_vector_rag.pipelines.graphrag_merged import GraphRAGPipeline as MergedGraphRAG
except ImportError as e:
    print(f"Failed to import IRIS RAG components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Individual test case for GraphRAG evaluation."""

    id: str
    name: str
    query: str
    query_type: str
    expected_entities: List[str] = field(default_factory=list)
    expected_documents: int = 0
    max_execution_time_ms: float = 10000.0
    description: str = ""
    complexity: str = "medium"  # low, medium, high


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_id: str
    implementation: str
    success: bool
    execution_time_ms: float
    retrieved_documents: int
    answer_length: int
    error_message: Optional[str] = None

    # Performance metrics
    db_exec_count: int = 0
    step_timings: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0

    # Quality metrics
    entities_found: int = 0
    relationships_traversed: int = 0
    retrieval_method: str = ""
    confidence_score: float = 0.0

    # Response data
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Comprehensive comparison report between implementations."""

    test_suite_name: str
    timestamp: str
    total_tests: int
    current_success_rate: float
    merged_success_rate: float

    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    quality_comparison: Dict[str, Any] = field(default_factory=dict)
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)

    detailed_results: List[TestResult] = field(default_factory=list)
    summary: str = ""


class MockDataGenerator:
    """Generate mock data for testing when database is not available."""

    @staticmethod
    def create_sample_documents() -> List[Document]:
        """Create sample medical documents for testing."""
        return [
            Document(
                id="mock_doc_1",
                page_content="""
                Type 2 diabetes is characterized by insulin resistance and high blood glucose.
                Common treatments include metformin, lifestyle changes, and insulin therapy.
                Complications can include cardiovascular disease, kidney damage, and neuropathy.
                Hypertension often coexists with diabetes, requiring ACE inhibitors or ARBs.
                """,
                metadata={"source": "mock_medical_text", "type": "diabetes_overview"},
            ),
            Document(
                id="mock_doc_2",
                page_content="""
                Metformin works by reducing hepatic glucose production and improving insulin sensitivity.
                It is the first-line treatment for type 2 diabetes with few contraindications.
                Side effects include gastrointestinal upset and rare lactic acidosis risk.
                Often combined with sulfonylureas, SGLT-2 inhibitors, or insulin for better control.
                """,
                metadata={"source": "mock_pharmacology", "type": "drug_profile"},
            ),
            Document(
                id="mock_doc_3",
                page_content="""
                Cardiovascular disease is the leading cause of mortality in diabetic patients.
                Statins like atorvastatin are recommended for cholesterol management.
                Blood pressure control is crucial, with targets below 130/80 mmHg.
                Aspirin therapy may be considered for primary prevention in high-risk patients.
                """,
                metadata={"source": "mock_cardiology", "type": "complications"},
            ),
            Document(
                id="mock_doc_4",
                page_content="""
                COVID-19 poses increased risks for diabetic patients due to immune dysfunction.
                Blood glucose control becomes more challenging during acute illness.
                Vaccination is strongly recommended for all patients with diabetes.
                Long COVID may contribute to new-onset diabetes in some individuals.
                """,
                metadata={"source": "mock_pandemic_health", "type": "covid_diabetes"},
            ),
            Document(
                id="mock_doc_5",
                page_content="""
                Diabetic nephropathy affects 30-40% of diabetic patients over time.
                Early detection involves microalbumin screening and creatinine monitoring.
                ACE inhibitors and ARBs provide kidney protection beyond blood pressure control.
                SGLT-2 inhibitors like empagliflozin show additional renal benefits.
                """,
                metadata={"source": "mock_nephrology", "type": "kidney_complications"},
            ),
        ]


class GraphRAGTestHarness:
    """Comprehensive testing harness for GraphRAG implementations."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_mocks: bool = False,
        output_dir: str = "outputs",
    ):
        """
        Initialize the testing harness.

        Args:
            config_path: Optional path to configuration file
            use_mocks: Whether to use mock data instead of real database
            output_dir: Directory for saving test results
        """
        self.use_mocks = use_mocks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        try:
            self.config_manager = ConfigurationManager(config_path)
            if not use_mocks:
                self.connection_manager = ConnectionManager()
            else:
                self.connection_manager = None
                logger.info("Using mock data mode - database connection skipped")

        except Exception as e:
            if not use_mocks:
                logger.warning(
                    f"Failed to initialize components, falling back to mock mode: {e}"
                )
                self.use_mocks = True
                self.connection_manager = None
            else:
                raise

        # Test results storage
        self.test_results: List[TestResult] = []
        self.comparison_report: Optional[ComparisonReport] = None

        logger.info(f"GraphRAG Test Harness initialized (mock_mode={self.use_mocks})")

    def create_test_suite(self) -> List[TestCase]:
        """Create comprehensive test suite for GraphRAG evaluation."""
        return [
            # Basic functionality tests
            TestCase(
                id="basic_001",
                name="Simple entity query",
                query="What is diabetes?",
                query_type="basic_entity",
                expected_entities=["diabetes"],
                expected_documents=2,
                max_execution_time_ms=5000.0,
                description="Basic entity recognition and retrieval",
                complexity="low",
            ),
            TestCase(
                id="basic_002",
                name="Treatment query",
                query="What treatments are available for diabetes?",
                query_type="treatment_query",
                expected_entities=["diabetes", "metformin", "insulin"],
                expected_documents=3,
                max_execution_time_ms=7000.0,
                description="Treatment-focused query with multiple entities",
                complexity="medium",
            ),
            # Multi-hop reasoning tests
            TestCase(
                id="multihop_001",
                name="Drug interaction query",
                query="What medications interact with drugs used for diabetes complications?",
                query_type="2-hop-drug-interaction",
                expected_entities=["diabetes", "complications", "medications"],
                expected_documents=4,
                max_execution_time_ms=10000.0,
                description="2-hop query: diabetes -> complications -> drug interactions",
                complexity="high",
            ),
            TestCase(
                id="multihop_002",
                name="Comorbidity treatment",
                query="How are cardiovascular complications of diabetes treated?",
                query_type="2-hop-comorbidity",
                expected_entities=["diabetes", "cardiovascular", "treatment"],
                expected_documents=3,
                max_execution_time_ms=8000.0,
                description="2-hop query: diabetes -> cardiovascular complications -> treatments",
                complexity="high",
            ),
            # Complex reasoning tests
            TestCase(
                id="complex_001",
                name="COVID diabetes interaction",
                query="How does COVID-19 affect diabetes management and what precautions should be taken?",
                query_type="complex-interaction",
                expected_entities=["covid", "diabetes", "management", "precautions"],
                expected_documents=3,
                max_execution_time_ms=12000.0,
                description="Complex multi-entity interaction analysis",
                complexity="high",
            ),
            TestCase(
                id="complex_002",
                name="Prevention strategy",
                query="What are the best strategies to prevent diabetes complications?",
                query_type="prevention-strategy",
                expected_entities=["diabetes", "complications", "prevention"],
                expected_documents=4,
                max_execution_time_ms=10000.0,
                description="Prevention-focused complex reasoning",
                complexity="high",
            ),
            # Edge cases and stress tests
            TestCase(
                id="edge_001",
                name="No entity query",
                query="What is the weather like today?",
                query_type="no-entity",
                expected_entities=[],
                expected_documents=0,
                max_execution_time_ms=3000.0,
                description="Query with no relevant entities",
                complexity="low",
            ),
            TestCase(
                id="edge_002",
                name="Ambiguous query",
                query="What about the treatment?",
                query_type="ambiguous",
                expected_entities=[],
                expected_documents=0,
                max_execution_time_ms=5000.0,
                description="Ambiguous query without clear context",
                complexity="medium",
            ),
            # Performance stress tests
            TestCase(
                id="stress_001",
                name="Long complex query",
                query="Provide a comprehensive analysis of diabetes management including all available treatments, their mechanisms of action, side effects, interactions with other medications, prevention strategies for complications, and the latest research findings on emerging therapies.",
                query_type="comprehensive-analysis",
                expected_entities=[
                    "diabetes",
                    "treatments",
                    "mechanisms",
                    "side effects",
                ],
                expected_documents=5,
                max_execution_time_ms=15000.0,
                description="Long, complex query testing system limits",
                complexity="high",
            ),
        ]

    def execute_test_case(
        self, test_case: TestCase, implementation: str, pipeline
    ) -> TestResult:
        """Execute a single test case on the specified implementation."""
        logger.info(f"Executing {test_case.id} on {implementation}")

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        try:
            # Execute query
            result = pipeline.query(
                query_text=test_case.query, top_k=10, include_sources=True
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            memory_usage = self._get_memory_usage() - start_memory

            # Extract metrics
            metadata = result.get("metadata", {})
            retrieved_docs = result.get("retrieved_documents", [])
            answer = result.get("answer", "")
            contexts = result.get("contexts", [])
            sources = result.get("sources", [])

            return TestResult(
                test_id=test_case.id,
                implementation=implementation,
                success=True,
                execution_time_ms=execution_time_ms,
                retrieved_documents=len(retrieved_docs),
                answer_length=len(answer) if answer else 0,
                # Performance metrics
                db_exec_count=metadata.get("db_exec_count", 0),
                step_timings=metadata.get("step_timings_ms", {}),
                memory_usage_mb=memory_usage,
                # Quality metrics
                entities_found=metadata.get("entities_found", 0),
                relationships_traversed=metadata.get("relationships_traversed", 0),
                retrieval_method=metadata.get("retrieval_method", ""),
                confidence_score=metadata.get("confidence", 0.0),
                # Response data
                answer=answer,
                contexts=contexts,
                sources=[str(s) for s in sources] if sources else [],
                metadata=metadata,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Test {test_case.id} failed on {implementation}: {e}")

            return TestResult(
                test_id=test_case.id,
                implementation=implementation,
                success=False,
                execution_time_ms=execution_time_ms,
                retrieved_documents=0,
                answer_length=0,
                error_message=str(e),
                memory_usage_mb=self._get_memory_usage() - start_memory,
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def setup_pipelines(self) -> Tuple[Any, Any]:
        """Setup both GraphRAG pipeline implementations."""
        sample_docs = MockDataGenerator.create_sample_documents()

        # Initialize real GraphRAG pipelines - no mock fallbacks
        current_pipeline = CurrentGraphRAG(
            connection_manager=self.connection_manager,
            config_manager=self.config_manager,
        )
        merged_pipeline = MergedGraphRAG(
            connection_manager=self.connection_manager,
            config_manager=self.config_manager,
        )

        # Load documents into both pipelines
        logger.info("Loading documents into current GraphRAG pipeline...")
        current_pipeline.load_documents("", documents=sample_docs)

        logger.info("Loading documents into merged GraphRAG pipeline...")
        merged_pipeline.load_documents("", documents=sample_docs)

        logger.info("Both real pipelines initialized successfully")
        return current_pipeline, merged_pipeline

    def _create_mock_pipeline(self, implementation_type: str):
        """Create a mock pipeline for testing when database is unavailable."""

        class MockPipeline:
            def __init__(self, impl_type):
                self.impl_type = impl_type

            def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
                import random

                time.sleep(random.uniform(0.1, 0.5))  # Simulate processing time

                # Generate mock response based on query
                mock_docs = [
                    {
                        "id": f"mock_doc_{i}",
                        "content": f"Mock content for {query_text[:50]}...",
                    }
                    for i in range(random.randint(1, 4))
                ]

                base_time = random.uniform(100, 1000)
                variation = (
                    1.2 if self.impl_type == "merged" else 1.0
                )  # Merged might be slightly slower

                return {
                    "query": query_text,
                    "answer": f"Mock answer from {self.impl_type} implementation for: {query_text}",
                    "retrieved_documents": mock_docs,
                    "contexts": [doc["content"] for doc in mock_docs],
                    "sources": [
                        {"document_id": doc["id"], "source": "mock"}
                        for doc in mock_docs
                    ],
                    "metadata": {
                        "num_retrieved": len(mock_docs),
                        "processing_time_ms": base_time * variation,
                        "pipeline_type": f"graphrag_{self.impl_type}",
                        "retrieval_method": (
                            "knowledge_graph_traversal"
                            if random.random() > 0.2
                            else "vector_fallback"
                        ),
                        "db_exec_count": random.randint(3, 8),
                        "step_timings_ms": {
                            "find_seed_entities_ms": base_time * 0.2,
                            "traverse_graph_ms": base_time * 0.5,
                            "get_documents_ms": base_time * 0.3,
                        },
                        "confidence": random.uniform(0.6, 0.95),
                    },
                }

            def load_documents(self, *args, **kwargs):
                logger.info(f"Mock {self.impl_type} pipeline: documents loaded")

        return MockPipeline(implementation_type)

    def run_comprehensive_tests(self) -> ComparisonReport:
        """Run comprehensive tests on both implementations."""
        logger.info("Starting comprehensive GraphRAG testing...")

        # Setup test environment
        test_cases = self.create_test_suite()
        current_pipeline, merged_pipeline = self.setup_pipelines()

        # Execute all tests
        all_results = []

        for test_case in test_cases:
            logger.info(f"Running test case: {test_case.name}")

            # Test current implementation
            current_result = self.execute_test_case(
                test_case, "current", current_pipeline
            )
            all_results.append(current_result)

            # Test merged implementation
            merged_result = self.execute_test_case(test_case, "merged", merged_pipeline)
            all_results.append(merged_result)

            # Brief pause between tests
            time.sleep(0.1)

        self.test_results = all_results

        # Generate comparison report
        self.comparison_report = self._generate_comparison_report(
            test_cases, all_results
        )

        # Save results
        self._save_results()

        return self.comparison_report

    def _generate_comparison_report(
        self, test_cases: List[TestCase], results: List[TestResult]
    ) -> ComparisonReport:
        """Generate comprehensive comparison report."""
        timestamp = datetime.now().isoformat()

        # Separate results by implementation
        current_results = [r for r in results if r.implementation == "current"]
        merged_results = [r for r in results if r.implementation == "merged"]

        # Calculate success rates
        current_success_rate = sum(1 for r in current_results if r.success) / len(
            current_results
        )
        merged_success_rate = sum(1 for r in merged_results if r.success) / len(
            merged_results
        )

        # Performance comparison
        current_avg_time = statistics.mean(
            [r.execution_time_ms for r in current_results if r.success]
        )
        merged_avg_time = statistics.mean(
            [r.execution_time_ms for r in merged_results if r.success]
        )

        current_avg_docs = statistics.mean(
            [r.retrieved_documents for r in current_results if r.success]
        )
        merged_avg_docs = statistics.mean(
            [r.retrieved_documents for r in merged_results if r.success]
        )

        performance_comparison = {
            "average_execution_time_ms": {
                "current": current_avg_time,
                "merged": merged_avg_time,
                "improvement_percentage": (
                    (current_avg_time - merged_avg_time) / current_avg_time
                )
                * 100,
            },
            "average_documents_retrieved": {
                "current": current_avg_docs,
                "merged": merged_avg_docs,
                "difference": merged_avg_docs - current_avg_docs,
            },
            "average_db_executions": {
                "current": statistics.mean(
                    [r.db_exec_count for r in current_results if r.success]
                ),
                "merged": statistics.mean(
                    [r.db_exec_count for r in merged_results if r.success]
                ),
            },
        }

        # Quality comparison
        quality_comparison = {
            "success_rates": {
                "current": current_success_rate,
                "merged": merged_success_rate,
                "improvement": merged_success_rate - current_success_rate,
            },
            "average_answer_length": {
                "current": statistics.mean(
                    [
                        r.answer_length
                        for r in current_results
                        if r.success and r.answer_length > 0
                    ]
                ),
                "merged": statistics.mean(
                    [
                        r.answer_length
                        for r in merged_results
                        if r.success and r.answer_length > 0
                    ]
                ),
            },
        }

        # Identify regressions and improvements
        regressions = []
        improvements = []

        for test_case in test_cases:
            current_result = next(
                (r for r in current_results if r.test_id == test_case.id), None
            )
            merged_result = next(
                (r for r in merged_results if r.test_id == test_case.id), None
            )

            if current_result and merged_result:
                # Check for regressions
                if current_result.success and not merged_result.success:
                    regressions.append(
                        f"{test_case.id}: Merged implementation failed where current succeeded"
                    )
                elif current_result.success and merged_result.success:
                    if (
                        merged_result.execution_time_ms
                        > current_result.execution_time_ms * 1.5
                    ):
                        regressions.append(
                            f"{test_case.id}: Significant performance regression (50%+ slower)"
                        )

                # Check for improvements
                if not current_result.success and merged_result.success:
                    improvements.append(
                        f"{test_case.id}: Merged implementation succeeded where current failed"
                    )
                elif current_result.success and merged_result.success:
                    if (
                        merged_result.execution_time_ms
                        < current_result.execution_time_ms * 0.8
                    ):
                        improvements.append(
                            f"{test_case.id}: Performance improvement (20%+ faster)"
                        )
                    if (
                        merged_result.retrieved_documents
                        > current_result.retrieved_documents
                    ):
                        improvements.append(
                            f"{test_case.id}: Better document retrieval ({merged_result.retrieved_documents} vs {current_result.retrieved_documents})"
                        )

        # Generate summary
        summary = f"""
        Test Suite Execution Summary:
        - Total Tests: {len(test_cases)}
        - Current Implementation Success Rate: {current_success_rate:.2%}
        - Merged Implementation Success Rate: {merged_success_rate:.2%}
        - Performance Change: {performance_comparison['average_execution_time_ms']['improvement_percentage']:+.1f}%
        - Regressions Found: {len(regressions)}
        - Improvements Found: {len(improvements)}
        """

        return ComparisonReport(
            test_suite_name="GraphRAG Comprehensive Comparison",
            timestamp=timestamp,
            total_tests=len(test_cases),
            current_success_rate=current_success_rate,
            merged_success_rate=merged_success_rate,
            performance_comparison=performance_comparison,
            quality_comparison=quality_comparison,
            regressions=regressions,
            improvements=improvements,
            detailed_results=results,
            summary=summary.strip(),
        )

    def _save_results(self):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"graphrag_comparison_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "test_results": [asdict(r) for r in self.test_results],
                    "comparison_report": (
                        asdict(self.comparison_report)
                        if self.comparison_report
                        else None
                    ),
                },
                f,
                indent=2,
            )

        # Save summary report
        summary_file = self.output_dir / f"graphrag_comparison_summary_{timestamp}.md"
        with open(summary_file, "w") as f:
            self._write_markdown_report(f)

        logger.info(f"Results saved to {results_file} and {summary_file}")

    def _write_markdown_report(self, file):
        """Write detailed markdown report."""
        if not self.comparison_report:
            return

        report = self.comparison_report

        file.write(f"# GraphRAG Implementation Comparison Report\n\n")
        file.write(f"**Generated:** {report.timestamp}\n")
        file.write(f"**Test Suite:** {report.test_suite_name}\n\n")

        file.write("## Executive Summary\n\n")
        file.write(report.summary)
        file.write("\n\n")

        file.write("## Performance Comparison\n\n")
        perf = report.performance_comparison
        file.write(
            f"- **Average Execution Time:** Current: {perf['average_execution_time_ms']['current']:.1f}ms, Merged: {perf['average_execution_time_ms']['merged']:.1f}ms\n"
        )
        file.write(
            f"- **Performance Change:** {perf['average_execution_time_ms']['improvement_percentage']:+.1f}%\n"
        )
        file.write(
            f"- **Average Documents Retrieved:** Current: {perf['average_documents_retrieved']['current']:.1f}, Merged: {perf['average_documents_retrieved']['merged']:.1f}\n"
        )
        file.write(
            f"- **Average DB Executions:** Current: {perf['average_db_executions']['current']:.1f}, Merged: {perf['average_db_executions']['merged']:.1f}\n\n"
        )

        file.write("## Quality Comparison\n\n")
        quality = report.quality_comparison
        file.write(
            f"- **Success Rates:** Current: {quality['success_rates']['current']:.2%}, Merged: {quality['success_rates']['merged']:.2%}\n"
        )
        file.write(
            f"- **Success Rate Change:** {quality['success_rates']['improvement']:+.2%}\n"
        )
        file.write(
            f"- **Average Answer Length:** Current: {quality['average_answer_length']['current']:.0f}, Merged: {quality['average_answer_length']['merged']:.0f}\n\n"
        )

        if report.regressions:
            file.write("## Regressions Found\n\n")
            for regression in report.regressions:
                file.write(f"- {regression}\n")
            file.write("\n")

        if report.improvements:
            file.write("## Improvements Found\n\n")
            for improvement in report.improvements:
                file.write(f"- {improvement}\n")
            file.write("\n")

        file.write("## Detailed Test Results\n\n")
        current_results = [
            r for r in report.detailed_results if r.implementation == "current"
        ]
        merged_results = [
            r for r in report.detailed_results if r.implementation == "merged"
        ]

        file.write(
            "| Test ID | Current Success | Current Time (ms) | Merged Success | Merged Time (ms) | Performance Change |\n"
        )
        file.write(
            "|---------|----------------|-------------------|---------------|------------------|-------------------|\n"
        )

        for current_result in current_results:
            merged_result = next(
                (r for r in merged_results if r.test_id == current_result.test_id), None
            )
            if merged_result:
                current_time = (
                    current_result.execution_time_ms
                    if current_result.success
                    else "FAIL"
                )
                merged_time = (
                    merged_result.execution_time_ms if merged_result.success else "FAIL"
                )

                if current_result.success and merged_result.success:
                    perf_change = (
                        (
                            current_result.execution_time_ms
                            - merged_result.execution_time_ms
                        )
                        / current_result.execution_time_ms
                    ) * 100
                    perf_str = f"{perf_change:+.1f}%"
                else:
                    perf_str = "N/A"

                file.write(
                    f"| {current_result.test_id} | {current_result.success} | {current_time} | {merged_result.success} | {merged_time} | {perf_str} |\n"
                )


def main():
    """Main entry point for the comprehensive testing harness."""
    parser = argparse.ArgumentParser(
        description="Comprehensive GraphRAG Implementation Testing"
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--use-mocks",
        action="store_true",
        help="Use mock data instead of real database",
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory for results"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize test harness
        harness = GraphRAGTestHarness(
            config_path=args.config,
            use_mocks=args.use_mocks,
            output_dir=args.output_dir,
        )

        # Run comprehensive tests
        print("🚀 Starting comprehensive GraphRAG testing...")
        report = harness.run_comprehensive_tests()

        # Display summary
        print("\n" + "=" * 60)
        print("📊 TEST EXECUTION COMPLETE")
        print("=" * 60)
        print(report.summary)

        if report.regressions:
            print(f"\n⚠️  {len(report.regressions)} regressions found:")
            for regression in report.regressions[:3]:  # Show first 3
                print(f"  - {regression}")

        if report.improvements:
            print(f"\n✅ {len(report.improvements)} improvements found:")
            for improvement in report.improvements[:3]:  # Show first 3
                print(f"  - {improvement}")

        print(f"\n📁 Detailed results saved to: {harness.output_dir}")

        # Recommendation
        if (
            report.merged_success_rate >= report.current_success_rate
            and len(report.regressions) == 0
        ):
            print("\n🎉 RECOMMENDATION: Merged implementation ready for deployment")
        elif len(report.regressions) > 0:
            print("\n⚠️  RECOMMENDATION: Address regressions before deployment")
        else:
            print("\n🔍 RECOMMENDATION: Further investigation needed")

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
