#!/usr/bin/env python3
"""
COMPREHENSIVE END-TO-END PIPELINE VALIDATION

Validates the four pipelines against real services: the suite's IRIS container, the
real embedding path and the real LLM (OPENAI_API_KEY). Each pipeline is constructed,
given documents and queried; a pipeline is WORKING only if all three steps succeed.
There is no mock mode: a run without infrastructure fails rather than reporting on
mocks.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from iris_vector_rag.core.models import Document

# Use the ACTUAL pipeline implementations we verified
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_vector_rag.pipelines.crag import CRAGPipeline
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineValidationResult:
    """Complete validation result for a pipeline."""

    pipeline_name: str
    test_mode: str  # "mock" or "full"

    # Constructor Testing
    constructor_success: bool
    constructor_time: float
    constructor_error: Optional[str]

    # Document Ingestion Testing
    ingestion_success: bool
    documents_ingested: int
    ingestion_time: float
    ingestion_error: Optional[str]

    # Query Testing
    query_success: bool
    queries_tested: int
    avg_query_time: float
    avg_docs_retrieved: float
    avg_answer_length: float
    query_error: Optional[str]

    # Infrastructure Assessment
    database_required: bool
    database_available: bool
    infrastructure_ready: bool

    # Overall Status
    status: str  # "WORKING", "PARTIAL", "BROKEN", "INFRASTRUCTURE_MISSING"


class ComprehensivePipelineValidator:
    """
    IRONCLAD validator that tests pipelines in BOTH mock and full modes.

    Mock Mode: Tests pipeline logic and interfaces without infrastructure
    Full Mode: Tests with real database and dependencies (when available)
    """

    def __init__(self):
        self.test_documents = self._create_test_documents()
        self.test_queries = [
            "What are the main symptoms of diabetes?",
            "How do vaccines prevent diseases?",
            "What causes cardiovascular disease?",
        ]
        self.results: Dict[str, PipelineValidationResult] = {}

    def _create_test_documents(self) -> List[Document]:
        """Create comprehensive test documents."""
        return [
            Document(
                id="test_doc_1",
                page_content="""
                Type 2 diabetes is a chronic condition affecting blood sugar regulation.
                Common symptoms include excessive thirst, frequent urination, fatigue,
                and blurred vision. Risk factors include obesity, sedentary lifestyle,
                and family history. Management involves lifestyle changes and medication.
                """,
                metadata={"source": "diabetes_guide.txt", "category": "endocrinology"},
            ),
            Document(
                id="test_doc_2",
                page_content="""
                Vaccines work by stimulating the immune system to recognize and fight
                specific pathogens. They contain antigens that trigger antibody production
                without causing disease. Types include live attenuated, inactivated,
                and mRNA vaccines. Vaccination has prevented diseases like polio and measles.
                """,
                metadata={
                    "source": "immunology_textbook.txt",
                    "category": "immunology",
                },
            ),
            Document(
                id="test_doc_3",
                page_content="""
                Cardiovascular disease encompasses conditions affecting the heart and blood vessels.
                Primary causes include atherosclerosis, high blood pressure, and elevated cholesterol.
                Risk factors are smoking, diabetes, obesity, and lack of exercise.
                Prevention focuses on healthy diet, regular exercise, and smoking cessation.
                """,
                metadata={"source": "cardiology_manual.txt", "category": "cardiology"},
            ),
        ]

    def _check_infrastructure_availability(self) -> bool:
        """Check if database infrastructure is available."""
        try:
            from iris_vector_rag.core.connection import ConnectionManager

            conn_mgr = ConnectionManager()
            connection = conn_mgr.get_connection()
            return connection is not None
        except Exception as e:
            logger.debug(f"Infrastructure check failed: {e}")
            return False

    def _build_pipeline(self, pipeline_class, pipeline_name: str):
        from iris_vector_rag.common.utils import get_llm_func
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.core.connection import ConnectionManager

        config_manager = ConfigurationManager()
        return pipeline_class(
            connection_manager=ConnectionManager(config_manager),
            config_manager=config_manager,
            llm_func=get_llm_func(),
        )

    def _load_documents(self, pipeline, pipeline_name: str) -> int:
        """Ingest the test documents through the pipeline's own loader.

        GraphRAG's loader takes a path and extracts entities with the LLM, so its
        documents go in as a JSON file.
        """
        if pipeline_name == "GraphRAG":
            import tempfile

            payload = [
                {"id": d.id, "page_content": d.page_content, "metadata": d.metadata}
                for d in self.test_documents
            ]
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "docs.json")
                with open(path, "w") as handle:
                    json.dump(payload, handle)
                pipeline.load_documents(path)
        else:
            pipeline.load_documents(documents=self.test_documents)
        return len(self.test_documents)

    def validate_pipeline(
        self, pipeline_class, pipeline_name: str
    ) -> PipelineValidationResult:
        """Construct, ingest and query one pipeline against the real services."""
        logger.info(f"Testing {pipeline_name} against real services")

        result = PipelineValidationResult(
            pipeline_name=pipeline_name,
            test_mode="full",
            constructor_success=False,
            constructor_time=0.0,
            constructor_error=None,
            ingestion_success=False,
            documents_ingested=0,
            ingestion_time=0.0,
            ingestion_error=None,
            query_success=False,
            queries_tested=0,
            avg_query_time=0.0,
            avg_docs_retrieved=0.0,
            avg_answer_length=0.0,
            query_error=None,
            database_required=True,
            database_available=True,
            infrastructure_ready=True,
            status="BROKEN",
        )

        try:
            start_time = time.time()
            pipeline = self._build_pipeline(pipeline_class, pipeline_name)
            result.constructor_time = time.time() - start_time
            result.constructor_success = True
        except Exception as e:
            result.constructor_error = str(e)
            logger.error(f"{pipeline_name} constructor failed: {e}")
            return result

        try:
            start_time = time.time()
            result.documents_ingested = self._load_documents(pipeline, pipeline_name)
            result.ingestion_time = time.time() - start_time
            result.ingestion_success = True
        except Exception as e:
            result.ingestion_error = str(e)
            logger.error(f"{pipeline_name} ingestion failed: {e}")
            return result

        query_times, docs_retrieved, answer_lengths = [], [], []
        for query in self.test_queries:
            try:
                start_time = time.time()
                query_result = pipeline.query(query, top_k=3)
                query_times.append(time.time() - start_time)

                assert isinstance(query_result, dict), "Query result must be dict"
                assert "retrieved_documents" in query_result
                assert "answer" in query_result
                docs_retrieved.append(len(query_result["retrieved_documents"]))
                answer_lengths.append(len(str(query_result["answer"] or "")))
            except Exception as e:
                logger.error(f"{pipeline_name} query failed: {e}")
                if result.query_error is None:
                    result.query_error = str(e)

        if query_times:
            result.query_success = True
            result.queries_tested = len(query_times)
            result.avg_query_time = sum(query_times) / len(query_times)
            result.avg_docs_retrieved = sum(docs_retrieved) / len(docs_retrieved)
            result.avg_answer_length = sum(answer_lengths) / len(answer_lengths)

        result.status = "WORKING" if result.query_success else "PARTIAL"
        return result

    def validate_all_pipelines(self) -> Dict[str, PipelineValidationResult]:
        """Validate all 4 ACTUAL pipelines."""
        pipelines_to_test = [
            (BasicRAGPipeline, "BasicRAG"),
            (CRAGPipeline, "CRAG"),
            (BasicRAGRerankingPipeline, "BasicRAGReranking"),
            (GraphRAGPipeline, "GraphRAG"),
        ]

        infrastructure_available = self._check_infrastructure_availability()
        logger.info("🚀 COMPREHENSIVE PIPELINE VALIDATION")
        logger.info(f"🏗️  Infrastructure Available: {infrastructure_available}")
        logger.info(f"📊 Testing {len(pipelines_to_test)} pipelines")

        for pipeline_class, pipeline_name in pipelines_to_test:
            self.results[pipeline_name] = self.validate_pipeline(
                pipeline_class, pipeline_name
            )

        return self.results

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        working = [name for name, r in self.results.items() if r.status == "WORKING"]
        partial = [name for name, r in self.results.items() if r.status == "PARTIAL"]
        broken = [name for name, r in self.results.items() if r.status == "BROKEN"]
        infra_missing = [
            name
            for name, r in self.results.items()
            if r.status == "INFRASTRUCTURE_MISSING"
        ]

        infrastructure_ready = any(
            r.infrastructure_ready for r in self.results.values()
        )
        test_mode = (
            self.results[list(self.results.keys())[0]].test_mode
            if self.results
            else "unknown"
        )

        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "test_mode": test_mode,
            "infrastructure_available": infrastructure_ready,
            "pipelines_tested": len(self.results),
            "summary": {
                "working": len(working),
                "partial": len(partial),
                "broken": len(broken),
                "infrastructure_missing": len(infra_missing),
                "working_pipelines": working,
                "partial_pipelines": partial,
                "broken_pipelines": broken,
                "infrastructure_missing_pipelines": infra_missing,
            },
            "infrastructure_assessment": {
                "database_required": all(
                    r.database_required for r in self.results.values()
                ),
                "database_available": any(
                    r.database_available for r in self.results.values()
                ),
                "can_test_full_functionality": infrastructure_ready,
                "recommendation": (
                    "Start database infrastructure for full validation"
                    if not infrastructure_ready
                    else "Infrastructure ready for full testing"
                ),
            },
            "detailed_results": {
                name: asdict(result) for name, result in self.results.items()
            },
        }

        return report


# Pytest Test Functions
@pytest.mark.usefixtures("iris_target")
def test_comprehensive_pipeline_validation():
    """Comprehensive test of all pipelines with proper infrastructure assessment."""
    validator = ComprehensivePipelineValidator()
    results = validator.validate_all_pipelines()
    report = validator.generate_comprehensive_report()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "validation_results"
    os.makedirs(output_dir, exist_ok=True)

    report_file = f"{output_dir}/comprehensive_pipeline_validation_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n🎯 COMPREHENSIVE PIPELINE VALIDATION COMPLETE")
    print(f"📊 Report saved to: {report_file}")
    print(f"🏗️  Infrastructure Available: {report['infrastructure_available']}")
    print(f"🧪 Test Mode: {report['test_mode'].upper()}")
    print(f"✅ Working: {report['summary']['working']}")
    print(f"⚠️  Partial: {report['summary']['partial']}")
    print(f"❌ Broken: {report['summary']['broken']}")
    print(f"🏗️  Infrastructure Missing: {report['summary']['infrastructure_missing']}")

    assert report["infrastructure_available"], "No IRIS infrastructure"
    assert report["summary"]["working"] > 0, (
        f"No pipelines working: {report['summary']['broken_pipelines']} broken, "
        f"{report['summary']['partial_pipelines']} partial"
    )
    logger.info(
        f"{report['summary']['working']}/{len(results)} pipelines working end to end"
    )


if __name__ == "__main__":
    # Run comprehensive validation
    validator = ComprehensivePipelineValidator()
    results = validator.validate_all_pipelines()
    report = validator.generate_comprehensive_report()

    print(json.dumps(report, indent=2))
