#!/usr/bin/env python3
"""
COMPREHENSIVE END-TO-END PIPELINE VALIDATION
IRONCLAD VALIDATION FOR 4 ACTUAL PIPELINES

This test provides SIMPLE but BULLETPROOF validation of our 4 real pipelines:
- BasicRAG, CRAG, BasicRAGReranking, GraphRAG

NO FAKE RESULTS. NO STALE RESULTS. REAL METRICS ONLY.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from iris_rag.core.models import Document

# Use the ACTUAL pipeline implementations we verified
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Simple, clear metrics for pipeline validation."""

    pipeline_name: str
    initialization_success: bool
    initialization_time: float
    initialization_error: Optional[str]

    document_ingestion_success: bool
    documents_ingested: int
    ingestion_time: float
    ingestion_error: Optional[str]

    query_execution_success: bool
    query_time: float
    documents_retrieved: int
    answer_generated: bool
    answer_length: int
    query_error: Optional[str]

    overall_status: str  # "WORKING", "PARTIAL", "BROKEN"


class ComprehensivePipelineValidator:
    """
    IRONCLAD validator for all pipeline operations.

    Tests each pipeline through complete lifecycle:
    1. Initialization
    2. Document ingestion
    3. Query execution
    4. Performance measurement

    NO RAGAS until we fix the data quality issue.
    FOCUS ON BASIC FUNCTIONALITY FIRST.
    """

    def __init__(self):
        self.test_documents = self._create_test_documents()
        self.test_queries = [
            "What are the main causes of cardiovascular disease?",
            "How do vaccines work to prevent infectious diseases?",
            "What are the symptoms of diabetes?",
        ]
        self.results: Dict[str, PipelineMetrics] = {}

    def _create_test_documents(self) -> List[Document]:
        """Create real test documents with medical content."""
        documents = [
            Document(
                id="doc1",
                page_content="""
                Cardiovascular disease (CVD) remains one of the leading causes of death worldwide. 
                The primary risk factors include high blood pressure, high cholesterol, smoking, 
                diabetes, obesity, and lack of physical activity. Prevention strategies focus on 
                lifestyle modifications including regular exercise, healthy diet, and smoking cessation.
                """,
                metadata={"source": "test_cardiology.txt", "category": "cardiology"},
            ),
            Document(
                id="doc2",
                page_content="""
                Vaccines work by training the immune system to recognize and fight specific pathogens.
                They contain antigens that stimulate the production of antibodies without causing disease.
                Common vaccine types include live attenuated, inactivated, subunit, and mRNA vaccines.
                Vaccination has been crucial in preventing infectious diseases like polio, measles, and COVID-19.
                """,
                metadata={"source": "test_immunology.txt", "category": "immunology"},
            ),
            Document(
                id="doc3",
                page_content="""
                Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency.
                Common symptoms include increased thirst, frequent urination, unexplained weight loss,
                fatigue, and blurred vision. Risk factors include obesity, sedentary lifestyle,
                family history, and age over 45. Management involves blood glucose monitoring,
                dietary changes, exercise, and medication when necessary.
                """,
                metadata={"source": "test_diabetes.txt", "category": "endocrinology"},
            ),
            Document(
                id="doc4",
                page_content="""
                Antibiotic resistance occurs when bacteria evolve to survive exposure to antibiotics.
                This is a major public health threat that makes infections harder to treat.
                Causes include overuse and misuse of antibiotics in humans and animals.
                Prevention requires appropriate antibiotic prescribing, infection control,
                and development of new antibiotics and alternative treatments.
                """,
                metadata={
                    "source": "test_infectious_disease.txt",
                    "category": "infectious_disease",
                },
            ),
            Document(
                id="doc5",
                page_content="""
                Mental health disorders affect millions worldwide, including depression, anxiety,
                bipolar disorder, and schizophrenia. Treatment approaches include psychotherapy,
                medication, lifestyle changes, and social support. Early intervention and 
                reducing stigma are crucial for improving outcomes. Access to mental health
                services remains a significant challenge in many regions.
                """,
                metadata={"source": "test_mental_health.txt", "category": "psychiatry"},
            ),
        ]
        return documents

    def validate_pipeline(self, pipeline_class, pipeline_name: str) -> PipelineMetrics:
        """
        Validate a single pipeline completely.

        Args:
            pipeline_class: The pipeline class to test
            pipeline_name: Name for reporting

        Returns:
            Complete metrics for the pipeline
        """
        logger.info(f"🔍 VALIDATING {pipeline_name}")

        # Initialize metrics
        metrics = PipelineMetrics(
            pipeline_name=pipeline_name,
            initialization_success=False,
            initialization_time=0.0,
            initialization_error=None,
            document_ingestion_success=False,
            documents_ingested=0,
            ingestion_time=0.0,
            ingestion_error=None,
            query_execution_success=False,
            query_time=0.0,
            documents_retrieved=0,
            answer_generated=False,
            answer_length=0,
            query_error=None,
            overall_status="BROKEN",
        )

        # Test 1: Initialization
        try:
            start_time = time.time()
            pipeline = pipeline_class()
            metrics.initialization_time = time.time() - start_time
            metrics.initialization_success = True
            logger.info(
                f"✅ {pipeline_name} initialized in {metrics.initialization_time:.3f}s"
            )
        except Exception as e:
            metrics.initialization_error = str(e)
            logger.error(f"❌ {pipeline_name} initialization failed: {e}")
            return metrics

        # Test 2: Document Ingestion
        try:
            start_time = time.time()
            pipeline.load_documents("", documents=self.test_documents)
            metrics.ingestion_time = time.time() - start_time
            metrics.documents_ingested = len(self.test_documents)
            metrics.document_ingestion_success = True
            logger.info(
                f"✅ {pipeline_name} ingested {metrics.documents_ingested} docs in {metrics.ingestion_time:.3f}s"
            )
        except Exception as e:
            metrics.ingestion_error = str(e)
            logger.error(f"❌ {pipeline_name} document ingestion failed: {e}")
            return metrics

        # Test 3: Query Execution
        successful_queries = 0
        total_query_time = 0.0
        total_docs_retrieved = 0
        total_answer_length = 0

        for query in self.test_queries:
            try:
                start_time = time.time()
                result = pipeline.query(query, top_k=3)
                query_time = time.time() - start_time

                # Validate response structure
                assert isinstance(result, dict), "Query result must be a dictionary"
                assert (
                    "retrieved_documents" in result
                ), "Must return retrieved_documents"
                assert "answer" in result, "Must return answer"

                docs_retrieved = len(result.get("retrieved_documents", []))
                answer = result.get("answer", "")
                answer_length = len(str(answer)) if answer else 0

                total_query_time += query_time
                total_docs_retrieved += docs_retrieved
                total_answer_length += answer_length
                successful_queries += 1

                logger.info(
                    f"✅ {pipeline_name} query '{query[:30]}...': {docs_retrieved} docs, {answer_length} chars, {query_time:.3f}s"
                )

            except Exception as e:
                logger.error(f"❌ {pipeline_name} query failed '{query[:30]}...': {e}")
                if metrics.query_error is None:
                    metrics.query_error = str(e)

        # Calculate query metrics
        if successful_queries > 0:
            metrics.query_execution_success = True
            metrics.query_time = total_query_time / successful_queries
            metrics.documents_retrieved = total_docs_retrieved // successful_queries
            metrics.answer_generated = total_answer_length > 0
            metrics.answer_length = total_answer_length // successful_queries

        # Determine overall status
        if (
            metrics.initialization_success
            and metrics.document_ingestion_success
            and metrics.query_execution_success
        ):
            metrics.overall_status = "WORKING"
        elif metrics.initialization_success and (
            metrics.document_ingestion_success or metrics.query_execution_success
        ):
            metrics.overall_status = "PARTIAL"
        else:
            metrics.overall_status = "BROKEN"

        logger.info(f"🏁 {pipeline_name} OVERALL STATUS: {metrics.overall_status}")
        return metrics

    def validate_all_pipelines(self) -> Dict[str, PipelineMetrics]:
        """Validate all 4 ACTUAL pipelines."""
        pipelines_to_test = [
            (BasicRAGPipeline, "BasicRAG"),
            (CRAGPipeline, "CRAG"),
            (BasicRAGRerankingPipeline, "BasicRAGReranking"),
            (GraphRAGPipeline, "GraphRAG"),
        ]

        logger.info("🚀 STARTING COMPREHENSIVE PIPELINE VALIDATION")
        logger.info(
            f"📊 Testing {len(pipelines_to_test)} pipelines with {len(self.test_documents)} documents"
        )

        for pipeline_class, pipeline_name in pipelines_to_test:
            self.results[pipeline_name] = self.validate_pipeline(
                pipeline_class, pipeline_name
            )

        return self.results

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        working_pipelines = [
            name
            for name, metrics in self.results.items()
            if metrics.overall_status == "WORKING"
        ]
        partial_pipelines = [
            name
            for name, metrics in self.results.items()
            if metrics.overall_status == "PARTIAL"
        ]
        broken_pipelines = [
            name
            for name, metrics in self.results.items()
            if metrics.overall_status == "BROKEN"
        ]

        # Calculate performance statistics for working pipelines
        if working_pipelines:
            working_metrics = [self.results[name] for name in working_pipelines]
            avg_init_time = sum(m.initialization_time for m in working_metrics) / len(
                working_metrics
            )
            avg_ingestion_time = sum(m.ingestion_time for m in working_metrics) / len(
                working_metrics
            )
            avg_query_time = sum(m.query_time for m in working_metrics) / len(
                working_metrics
            )
            avg_docs_retrieved = sum(
                m.documents_retrieved for m in working_metrics
            ) / len(working_metrics)
        else:
            avg_init_time = avg_ingestion_time = avg_query_time = avg_docs_retrieved = 0

        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "pipelines_tested": len(self.results),
            "test_documents": len(self.test_documents),
            "test_queries": len(self.test_queries),
            "summary": {
                "working_pipelines": len(working_pipelines),
                "partial_pipelines": len(partial_pipelines),
                "broken_pipelines": len(broken_pipelines),
                "working_pipeline_names": working_pipelines,
                "partial_pipeline_names": partial_pipelines,
                "broken_pipeline_names": broken_pipelines,
            },
            "performance_averages": {
                "initialization_time": round(avg_init_time, 3),
                "ingestion_time": round(avg_ingestion_time, 3),
                "query_time": round(avg_query_time, 3),
                "documents_retrieved": round(avg_docs_retrieved, 1),
            },
            "detailed_results": {
                name: asdict(metrics) for name, metrics in self.results.items()
            },
        }

        return report


# Test Functions for pytest
def test_basic_rag_pipeline():
    """Test BasicRAG pipeline end-to-end."""
    validator = ComprehensivePipelineValidator()
    metrics = validator.validate_pipeline(BasicRAGPipeline, "BasicRAG")

    assert (
        metrics.initialization_success
    ), f"BasicRAG initialization failed: {metrics.initialization_error}"
    assert (
        metrics.document_ingestion_success
    ), f"BasicRAG ingestion failed: {metrics.ingestion_error}"
    assert (
        metrics.query_execution_success
    ), f"BasicRAG query failed: {metrics.query_error}"
    assert (
        metrics.overall_status == "WORKING"
    ), f"BasicRAG not working: {metrics.overall_status}"


def test_crag_pipeline():
    """Test CRAG pipeline end-to-end."""
    validator = ComprehensivePipelineValidator()
    metrics = validator.validate_pipeline(CRAGPipeline, "CRAG")

    assert (
        metrics.initialization_success
    ), f"CRAG initialization failed: {metrics.initialization_error}"
    assert (
        metrics.document_ingestion_success
    ), f"CRAG ingestion failed: {metrics.ingestion_error}"
    assert metrics.query_execution_success, f"CRAG query failed: {metrics.query_error}"
    assert (
        metrics.overall_status == "WORKING"
    ), f"CRAG not working: {metrics.overall_status}"


def test_basic_rag_reranking_pipeline():
    """Test BasicRAGReranking pipeline end-to-end."""
    validator = ComprehensivePipelineValidator()
    metrics = validator.validate_pipeline(
        BasicRAGRerankingPipeline, "BasicRAGReranking"
    )

    assert (
        metrics.initialization_success
    ), f"BasicRAGReranking initialization failed: {metrics.initialization_error}"
    assert (
        metrics.document_ingestion_success
    ), f"BasicRAGReranking ingestion failed: {metrics.ingestion_error}"
    assert (
        metrics.query_execution_success
    ), f"BasicRAGReranking query failed: {metrics.query_error}"
    assert (
        metrics.overall_status == "WORKING"
    ), f"BasicRAGReranking not working: {metrics.overall_status}"


def test_graph_rag_pipeline():
    """Test GraphRAG pipeline end-to-end."""
    validator = ComprehensivePipelineValidator()
    metrics = validator.validate_pipeline(GraphRAGPipeline, "GraphRAG")

    assert (
        metrics.initialization_success
    ), f"GraphRAG initialization failed: {metrics.initialization_error}"
    assert (
        metrics.document_ingestion_success
    ), f"GraphRAG ingestion failed: {metrics.ingestion_error}"
    assert (
        metrics.query_execution_success
    ), f"GraphRAG query failed: {metrics.query_error}"
    assert (
        metrics.overall_status == "WORKING"
    ), f"GraphRAG not working: {metrics.overall_status}"


def test_comprehensive_validation():
    """Test all pipelines comprehensively and generate report."""
    validator = ComprehensivePipelineValidator()
    validator.validate_all_pipelines()
    report = validator.generate_validation_report()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "validation_results"
    os.makedirs(output_dir, exist_ok=True)

    report_file = f"{output_dir}/comprehensive_pipeline_validation_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n🎯 COMPREHENSIVE VALIDATION COMPLETE")
    print(f"📊 Report saved to: {report_file}")
    print(f"✅ Working: {len(report['summary']['working_pipeline_names'])}")
    print(f"⚠️  Partial: {len(report['summary']['partial_pipeline_names'])}")
    print(f"❌ Broken: {len(report['summary']['broken_pipeline_names'])}")

    # Require at least some working pipelines
    assert (
        len(report["summary"]["working_pipeline_names"]) > 0
    ), "No working pipelines found!"

    return report


if __name__ == "__main__":
    # Run comprehensive validation
    validator = ComprehensivePipelineValidator()
    results = validator.validate_all_pipelines()
    report = validator.generate_validation_report()

    print(json.dumps(report, indent=2))
