"""
E2E Tests for PyLate ColBERT Pipeline

These tests verify the PyLate ColBERT pipeline works end-to-end with real
IRIS database connections and actual document processing (no mocks).
"""

import logging
import os
from pathlib import Path

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from common.utils import get_llm_func
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def pipeline_dependencies():
    """Create real dependencies for E2E testing."""
    # Configuration manager
    config_manager = ConfigurationManager()

    # Connection manager
    connection_manager = ConnectionManager(config_manager)

    # LLM function
    llm_func = get_llm_func()

    # Vector store
    vector_store = IRISVectorStore(connection_manager, config_manager)

    return {
        "config_manager": config_manager,
        "connection_manager": connection_manager,
        "llm_func": llm_func,
        "vector_store": vector_store
    }


@pytest.fixture(scope="module")
def sample_biomedical_documents():
    """Create sample biomedical documents for testing."""
    return [
        Document(
            page_content="Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels. "
            "Type 1 diabetes results from autoimmune destruction of pancreatic beta cells, while Type 2 diabetes "
            "involves insulin resistance and relative insulin deficiency.",
            metadata={"source": "diabetes.txt", "doc_id": "dm_001"}
        ),
        Document(
            page_content="Hypertension, or high blood pressure, is a major risk factor for cardiovascular disease. "
            "It is defined as systolic blood pressure ≥140 mmHg or diastolic blood pressure ≥90 mmHg. "
            "Treatment includes lifestyle modifications and antihypertensive medications.",
            metadata={"source": "hypertension.txt", "doc_id": "htn_001"}
        ),
        Document(
            page_content="Alzheimer's disease is a progressive neurodegenerative disorder and the most common cause of dementia. "
            "It is characterized by accumulation of amyloid-beta plaques and neurofibrillary tangles in the brain. "
            "Symptoms include memory loss, cognitive decline, and behavioral changes.",
            metadata={"source": "alzheimers.txt", "doc_id": "ad_001"}
        ),
        Document(
            page_content="COVID-19 is an infectious disease caused by SARS-CoV-2 coronavirus. "
            "Common symptoms include fever, cough, and shortness of breath. "
            "Severe cases can lead to acute respiratory distress syndrome and multi-organ failure.",
            metadata={"source": "covid19.txt", "doc_id": "covid_001"}
        ),
        Document(
            page_content="Cancer immunotherapy harnesses the immune system to fight cancer. "
            "Checkpoint inhibitors like PD-1 and CTLA-4 antibodies block immune checkpoint proteins, "
            "enabling T cells to recognize and destroy tumor cells more effectively.",
            metadata={"source": "immunotherapy.txt", "doc_id": "immuno_001"}
        ),
        Document(
            page_content="CRISPR-Cas9 is a revolutionary gene-editing technology that allows precise modification of DNA sequences. "
            "It uses a guide RNA to direct the Cas9 enzyme to specific genomic locations for targeted editing. "
            "Applications include treating genetic disorders and developing disease-resistant crops.",
            metadata={"source": "crispr.txt", "doc_id": "crispr_001"}
        ),
        Document(
            page_content="Heart failure is a clinical syndrome where the heart cannot pump sufficient blood to meet the body's needs. "
            "Common causes include coronary artery disease, hypertension, and cardiomyopathy. "
            "Management includes ACE inhibitors, beta-blockers, diuretics, and lifestyle changes.",
            metadata={"source": "heart_failure.txt", "doc_id": "hf_001"}
        ),
        Document(
            page_content="Parkinson's disease is a progressive movement disorder caused by degeneration of dopaminergic neurons "
            "in the substantia nigra. Key symptoms include tremor, rigidity, bradykinesia, and postural instability. "
            "Treatment focuses on dopamine replacement therapy with levodopa and dopamine agonists.",
            metadata={"source": "parkinsons.txt", "doc_id": "pd_001"}
        ),
        Document(
            page_content="Asthma is a chronic inflammatory airway disease characterized by reversible airflow obstruction. "
            "Triggers include allergens, exercise, cold air, and respiratory infections. "
            "Treatment involves inhaled corticosteroids and bronchodilators for symptom control.",
            metadata={"source": "asthma.txt", "doc_id": "asthma_001"}
        ),
        Document(
            page_content="Antibiotic resistance is a growing global health threat where bacteria evolve mechanisms to resist antimicrobial drugs. "
            "Key resistance mechanisms include enzymatic degradation, efflux pumps, and target site modifications. "
            "Strategies to combat resistance include antibiotic stewardship and development of novel antibiotics.",
            metadata={"source": "antibiotic_resistance.txt", "doc_id": "abr_001"}
        ),
    ]


@pytest.mark.e2e
class TestPyLateColBERTPipelineE2E:
    """End-to-end tests for PyLate ColBERT pipeline."""

    def create_test_pipeline(self, pipeline_dependencies):
        """Helper to create PyLate pipeline for testing."""
        return PyLateColBERTPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            llm_func=pipeline_dependencies["llm_func"],
            vector_store=pipeline_dependencies["vector_store"]
        )

    def test_pipeline_creation_fallback_mode(self, pipeline_dependencies):
        """Test creating PyLate pipeline (should work in fallback mode without PyLate library)."""
        try:
            pipeline = PyLateColBERTPipeline(
                connection_manager=pipeline_dependencies["connection_manager"],
                config_manager=pipeline_dependencies["config_manager"],
                llm_func=pipeline_dependencies["llm_func"],
                vector_store=pipeline_dependencies["vector_store"]
            )

            assert pipeline is not None
            assert pipeline.model is None  # Fallback mode
            assert pipeline.is_initialized is False
            assert pipeline.use_native_reranking is False

            logger.info("✓ PyLate pipeline created successfully in fallback mode")

        except Exception as e:
            pytest.fail(f"Failed to create PyLate pipeline: {e}")

    def test_document_loading_e2e(self, pipeline_dependencies, sample_biomedical_documents):
        """Test loading documents into PyLate pipeline (E2E with real vector store)."""
        pipeline = PyLateColBERTPipeline(
            connection_manager=pipeline_dependencies["connection_manager"],
            config_manager=pipeline_dependencies["config_manager"],
            llm_func=pipeline_dependencies["llm_func"],
            vector_store=pipeline_dependencies["vector_store"]
        )

        result = pipeline.load_documents(sample_biomedical_documents)

        assert result is not None
        # Standardized API returns documents_loaded, embeddings_generated, documents_failed
        assert "documents_loaded" in result
        assert result["documents_loaded"] == 10
        assert pipeline.stats["documents_indexed"] == 10

        logger.info(f"✓ Loaded {pipeline.stats['documents_indexed']} documents successfully")

    def test_query_execution_e2e(self, pipeline_dependencies, sample_biomedical_documents):
        """Test query execution with real document retrieval (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        # Load documents
        pipeline.load_documents(sample_biomedical_documents)

        # Execute query
        result = pipeline.query(
            "What are the symptoms of diabetes?",
            top_k=3
        )

        # Verify response structure
        assert "query" in result
        assert "answer" in result
        assert "retrieved_documents" in result
        assert "metadata" in result
        assert "execution_time" in result

        # Verify retrieved documents
        assert len(result["retrieved_documents"]) <= 3
        assert len(result["contexts"]) == len(result["retrieved_documents"])

        # Verify metadata
        assert result["metadata"]["pipeline_type"] == "colbert_pylate"
        assert result["metadata"]["reranked"] is False  # Fallback mode
        assert result["metadata"]["num_retrieved"] == len(result["retrieved_documents"])

        logger.info(f"✓ Query executed: {result['metadata']['num_retrieved']} docs retrieved")

    def test_multiple_queries_e2e(self, pipeline_dependencies, sample_biomedical_documents):
        """Test executing multiple queries on same loaded documents (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        pipeline.load_documents(sample_biomedical_documents)

        queries = [
            "What is hypertension and how is it treated?",
            "Explain CRISPR gene editing technology",
            "What are the causes of Parkinson's disease?",
        ]

        for i, query_text in enumerate(queries, 1):
            result = pipeline.query(query_text, top_k=2)

            assert result is not None
            assert "answer" in result
            assert len(result["retrieved_documents"]) <= 2

            # Verify stats tracking
            assert pipeline.stats["queries_processed"] == i

            logger.info(f"✓ Query {i}: {result['metadata']['num_retrieved']} docs retrieved")

    def test_pipeline_info_e2e(self, pipeline_dependencies):
        """Test getting pipeline information (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        info = pipeline.get_pipeline_info()

        assert info is not None
        assert info["pipeline_type"] == "colbert_pylate"
        assert "rerank_factor" in info
        assert "model_name" in info
        assert "use_native_reranking" in info
        assert "is_initialized" in info
        assert "stats" in info

        assert info["is_initialized"] is False  # Fallback mode
        assert info["use_native_reranking"] is False

        logger.info("✓ Pipeline info retrieved successfully")

    def test_query_with_custom_parameters_e2e(self, pipeline_dependencies, sample_biomedical_documents):
        """Test query with custom parameters (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        pipeline.load_documents(sample_biomedical_documents)

        # Test with different top_k
        result_k3 = pipeline.query("What is COVID-19?", top_k=3)
        assert len(result_k3["retrieved_documents"]) <= 3

        result_k1 = pipeline.query("What is COVID-19?", top_k=1)
        assert len(result_k1["retrieved_documents"]) <= 1

        # Test with include_sources=False
        result_no_sources = pipeline.query(
            "What is COVID-19?",
            top_k=2,
            include_sources=False
        )
        assert "sources" not in result_no_sources

        # Test with include_sources=True (default)
        result_with_sources = pipeline.query("What is COVID-19?", top_k=2)
        assert "sources" in result_with_sources

        logger.info("✓ Custom parameters work correctly")

    def test_empty_query_handling_e2e(self, pipeline_dependencies):
        """Test handling of queries with no loaded documents (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        # Query without loading documents
        result = pipeline.query("What is diabetes?", top_k=3)

        # Should return response with no documents
        assert result is not None
        assert "retrieved_documents" in result
        # Depending on implementation, may return empty list or none
        # Just verify it doesn't crash

        logger.info("✓ Empty query handled gracefully")

    def test_stats_tracking_e2e(self, pipeline_dependencies, sample_biomedical_documents):
        """Test statistics tracking across operations (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        # Initial stats
        assert pipeline.stats["queries_processed"] == 0
        assert pipeline.stats["documents_indexed"] == 0
        assert pipeline.stats["reranking_operations"] == 0

        # Load documents
        pipeline.load_documents(sample_biomedical_documents)
        assert pipeline.stats["documents_indexed"] == 10

        # Execute queries
        for i in range(5):
            pipeline.query("Test query", top_k=2)
            assert pipeline.stats["queries_processed"] == i + 1

        # Reranking should be 0 in fallback mode
        assert pipeline.stats["reranking_operations"] == 0

        logger.info("✓ Stats tracking verified")

    def test_document_metadata_preservation_e2e(self, pipeline_dependencies, sample_biomedical_documents):
        """Test that document metadata is preserved through pipeline (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        pipeline.load_documents(sample_biomedical_documents)

        result = pipeline.query("What is diabetes?", top_k=5)

        # Check that metadata is preserved in retrieved documents
        # At least one document should have our expected metadata fields
        found_metadata_docs = 0
        for doc in result["retrieved_documents"]:
            assert hasattr(doc, "metadata")
            if "source" in doc.metadata and "doc_id" in doc.metadata:
                found_metadata_docs += 1

        # At least one document should have preserved metadata from our sample docs
        assert found_metadata_docs > 0, "Expected at least one document with preserved metadata fields"

        logger.info("✓ Document metadata preserved")

    def test_contexts_match_documents_e2e(self, pipeline_dependencies, sample_biomedical_documents):
        """Test that contexts field matches retrieved documents (E2E)."""
        pipeline = self.create_test_pipeline(pipeline_dependencies)

        pipeline.load_documents(sample_biomedical_documents)

        result = pipeline.query("Explain gene editing", top_k=4)

        assert len(result["contexts"]) == len(result["retrieved_documents"])

        for i, (context, doc) in enumerate(zip(result["contexts"], result["retrieved_documents"])):
            assert context == doc.page_content, f"Context {i} doesn't match document {i}"

        logger.info("✓ Contexts match documents")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
