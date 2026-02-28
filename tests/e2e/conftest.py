"""
E2E Test Configuration and Fixtures

This module provides shared fixtures for true end-to-end testing that follow
the E2E test strategy requirements:
- NO MOCKS or stubs
- MUST use real IRIS database connection
- MUST use realistic PMC biomedical data
- MUST test complete workflows end-to-end
- MUST validate actual behavior and performance

Based on patterns from evaluation_framework/true_e2e_evaluation.py
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

from iris_vector_rag.common.utils import get_embedding_func, get_llm_func

# Import real framework components (NO MOCKS)
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

logger = logging.getLogger(__name__)
PROJECT_CONTAINER_NAME = "iris-vector-rag-tests"
IRIS_ENV_KEYS = [
    "RAG_DATABASE__IRIS__HOST",
    "RAG_DATABASE__IRIS__PORT",
    "RAG_DATABASE__IRIS__USERNAME",
    "RAG_DATABASE__IRIS__PASSWORD",
    "RAG_DATABASE__IRIS__NAMESPACE",
    "IRIS_HOST",
    "IRIS_PORT",
    "IRIS_USERNAME",
    "IRIS_PASSWORD",
    "IRIS_NAMESPACE",
]

# Load environment variables for E2E tests
load_dotenv()

# Try to import iris-devtester for automated infrastructure
has_devtester = False
try:
    from iris_devtester import IRISContainer as _IRISContainer

    has_devtester = True
except ImportError:
    _IRISContainer = None
    has_devtester = False


@pytest.fixture(scope="session", autouse=True)
def iris_infrastructure():
    """
    Ensure IRIS infrastructure is available for E2E tests.
    
    If IRIS_HOST is not set, attempts to use iris-devtester to start
    a named project-specific container.
    """
    if os.environ.get("IRIS_HOST"):
        logger.info(
            "Using existing IRIS infrastructure at %s",
            os.environ.get("IRIS_HOST"),
        )
        yield
        return

    if not has_devtester:
        pytest.skip(
            "E2E tests require iris-devtester or IRIS_HOST. Install with: pip install 'iris-devtester[all]'"
        )

    logger.info("Starting IRIS container for E2E tests via iris-devtester")
    original_env = {key: os.environ.get(key) for key in IRIS_ENV_KEYS}

    from iris_devtester import IRISContainer as RuntimeIRISContainer

    with RuntimeIRISContainer.community(
        name=PROJECT_CONTAINER_NAME
    ).with_preconfigured_password(
        "SYS"
    ) as iris:
        # Trigger iris-devtester auto-remediation for password change required
        try:
            iris.get_connection()
        except Exception as exc:
            logger.debug("iris-devtester get_connection() failed: %s", exc)

        config = iris.get_config()
        try:
            port = iris.get_exposed_port(1972)
        except Exception:
            port = config.port

        os.environ["RAG_DATABASE__IRIS__HOST"] = config.host
        os.environ["RAG_DATABASE__IRIS__PORT"] = str(port)
        os.environ["RAG_DATABASE__IRIS__USERNAME"] = config.username
        os.environ["RAG_DATABASE__IRIS__PASSWORD"] = config.password
        os.environ["RAG_DATABASE__IRIS__NAMESPACE"] = config.namespace

        # Also set legacy names for compatibility with older code
        os.environ["IRIS_HOST"] = config.host
        os.environ["IRIS_PORT"] = str(port)
        os.environ["IRIS_USERNAME"] = config.username
        os.environ["IRIS_PASSWORD"] = config.password
        os.environ["IRIS_NAMESPACE"] = config.namespace

        logger.info(
            "IRIS container '%s' started at %s:%s",
            PROJECT_CONTAINER_NAME,
            config.host,
            port,
        )
        try:
            yield
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@pytest.fixture(scope="session", autouse=True)
def e2e_reset_namespace():
    """
    Ensure a clean namespace for E2E runs using iris-devtester utilities.
    """
    if os.environ.get("SKIP_IRIS_TESTS", "false") != "false":
        pytest.skip("E2E tests require live IRIS (SKIP_IRIS_TESTS=false)")

    if not has_devtester:
        return

    from iris_devtester.connections import get_connection
    from tests.fixtures.idt_cleanup import reset_rag_schema

    conn = get_connection()
    try:
        try:
            reset_rag_schema(conn, schema="RAG", strict=False)
        except Exception as exc:
            logger.warning("Namespace reset skipped: %s", exc)
        yield
        try:
            reset_rag_schema(conn, schema="RAG", strict=False)
        except Exception as exc:
            logger.warning("Namespace reset skipped: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture(scope="session")
def e2e_config_manager() -> ConfigurationManager:
    """
    Create real ConfigurationManager for E2E tests.

    NO MOCKS - Uses actual configuration system.
    """
    logger.info("Creating real ConfigurationManager for E2E tests")
    return ConfigurationManager()


@pytest.fixture(scope="session")
def e2e_connection_manager(
    e2e_config_manager: ConfigurationManager,
) -> ConnectionManager:
    """
    Create real ConnectionManager for E2E tests.

    NO MOCKS - Uses actual IRIS database connections.
    """
    logger.info("Creating real ConnectionManager for E2E tests")
    return ConnectionManager()


@pytest.fixture(scope="session")
def e2e_iris_vector_store(
    e2e_connection_manager: ConnectionManager, e2e_config_manager: ConfigurationManager
) -> IRISVectorStore:
    """
    Create real IRISVectorStore for E2E tests.

    NO MOCKS - Uses actual IRIS vector storage with real embeddings.
    """
    logger.info("Creating real IRISVectorStore for E2E tests")
    return IRISVectorStore(
        connection_manager=e2e_connection_manager, config_manager=e2e_config_manager
    )


@pytest.fixture(scope="function")
def fresh_iris_vector_store(
    e2e_config_manager: ConfigurationManager,
) -> IRISVectorStore:
    """
    Create fresh IRISVectorStore for tests that need isolated connections.

    This fixture creates a new connection manager and vector store for each test
    to avoid connection closure issues with session-scoped fixtures.
    """
    logger.info("Creating fresh IRISVectorStore with new connection manager")
    connection_manager = ConnectionManager()
    return IRISVectorStore(
        connection_manager=connection_manager, config_manager=e2e_config_manager
    )


@pytest.fixture(scope="session")
def e2e_embedding_function():
    """
    Create real embedding function for E2E tests.

    NO MOCKS - Uses actual sentence-transformers embeddings.
    """
    logger.info(
        "Loading real embedding function: sentence-transformers/all-MiniLM-L6-v2"
    )
    return get_embedding_func(
        model_name_override="sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.fixture(scope="session")
def e2e_llm_function():
    """
    Create real LLM function for E2E tests.

    NO MOCKS - Uses actual OpenAI API calls.
    """
    logger.info("Initializing real OpenAI LLM")
    return get_llm_func(provider="openai", model_name="gpt-4o-mini")


@pytest.fixture(scope="module")
def pipeline_dependencies(e2e_config_manager, e2e_connection_manager):
    """
    Module-scoped pipeline dependencies for E2E tests.

    Amortizes setup cost across test module by reusing session fixtures.
    Per T005 requirements: module-scoped fixtures with shared session objects.
    """
    logger.info("Creating module-scoped pipeline dependencies")
    return {
        "config": e2e_config_manager,
        "connection": e2e_connection_manager,
        "embedding_func": get_embedding_func(
            model_name_override="sentence-transformers/all-MiniLM-L6-v2"
        ),
        "llm_func": get_llm_func(provider="openai", model_name="gpt-4o-mini"),
    }


@pytest.fixture(scope="function")
def e2e_database_cleanup(e2e_connection_manager: ConnectionManager):
    """
    Clean database before and after each E2E test for isolation.

    NO MOCKS - Uses real database operations for cleanup.
    """

    def _clear_test_data():
        """Clear test data from IRIS database.

        Per T012 requirements: Delete test documents using ID patterns.
        Patterns: e2e_test_%, test_%, doc1, doc2, doc3
        """
        try:
            from tests.fixtures.idt_cleanup import reset_rag_schema

            connection = e2e_connection_manager.get_connection()
            reset_rag_schema(connection, schema="RAG", strict=False)
            connection.close()
            logger.info("Cleared E2E test data via idt cleanup")
        except Exception as e:
            logger.warning(f"Failed to clear test data via idt cleanup: {e}")

    # Clean before test
    _clear_test_data()

    yield

    # Clean after test
    _clear_test_data()


@pytest.fixture(scope="session")
def e2e_pmc_documents() -> List[Document]:
    """
    Load real PMC biomedical documents for E2E testing.

    NO MOCKS - Uses actual PMC XML documents from data/sample_10_docs/
    """
    logger.info("Loading real PMC documents for E2E testing")

    pmc_documents_dir = Path("data/sample_10_docs")
    documents = []

    if not pmc_documents_dir.exists():
        logger.warning("PMC documents directory not found, using biomedical test data")
        # Fallback to realistic biomedical content if PMC files not available
        return _get_biomedical_test_documents()

    # Load actual PMC XML files
    xml_files = list(pmc_documents_dir.glob("*.xml"))

    if not xml_files:
        logger.warning("No PMC XML files found, using biomedical test data")
        return _get_biomedical_test_documents()

    for i, xml_file in enumerate(xml_files[:10]):  # Limit to 10 for E2E tests
        try:
            # Read XML content (simplified parsing for E2E tests)
            with open(xml_file, "r", encoding="utf-8") as f:
                xml_content = f.read()

            # Extract basic content from XML (this could be enhanced with proper XML parsing)
            # For E2E purposes, we use the raw XML as content to test real document handling
            document = Document(
                id=f"e2e_test_pmc_{xml_file.stem}",
                page_content=xml_content[:5000],  # Truncate for manageable test size
                metadata={
                    "source": "pmc_e2e_test",
                    "type": "biomedical",
                    "pmcid": xml_file.stem,
                    "topic": "biomedical_research",
                    "test_type": "e2e",
                },
            )
            documents.append(document)

        except Exception as e:
            logger.warning(f"Failed to load PMC document {xml_file}: {e}")

    if not documents:
        logger.warning("No PMC documents loaded, using biomedical test data")
        return _get_biomedical_test_documents()

    # Always include deterministic biomedical docs for relevance assertions (prepend)
    fallback_docs = _get_biomedical_test_documents()
    documents = fallback_docs + documents

    logger.info(f"Loaded {len(documents)} real PMC documents for E2E testing")
    return documents


def _get_biomedical_test_documents() -> List[Document]:
    """Fallback biomedical test documents if PMC files not available."""
    return [
        Document(
            id="e2e_test_cardio_001",
            page_content="""Cardiovascular disease remains the leading cause of mortality worldwide. 
            Major risk factors include hypertension, dyslipidemia, diabetes mellitus, obesity, smoking, 
            and sedentary lifestyle. Primary prevention strategies focus on lifestyle modifications 
            including regular physical activity, dietary interventions with emphasis on Mediterranean 
            diet patterns, smoking cessation, and weight management. Secondary prevention involves 
            optimal medical therapy with statins, ACE inhibitors, beta-blockers, and antiplatelet therapy.""",
            metadata={
                "source": "e2e_test",
                "type": "biomedical",
                "topic": "cardiovascular",
                "test_type": "e2e",
            },
        ),
        Document(
            id="e2e_test_diabetes_002",
            page_content="""Type 2 diabetes mellitus affects over 400 million people worldwide and is 
            characterized by insulin resistance and progressive beta-cell dysfunction. Management strategies 
            include lifestyle modifications, oral antidiabetic agents, and insulin therapy. Metformin remains 
            the first-line therapy for most patients due to its efficacy, safety profile, and cardiovascular 
            benefits. Additional agents include sulfonylureas, DPP-4 inhibitors, SGLT2 inhibitors, and 
            GLP-1 receptor agonists.""",
            metadata={
                "source": "e2e_test",
                "type": "biomedical",
                "topic": "diabetes",
                "test_type": "e2e",
            },
        ),
        Document(
            id="e2e_test_oncology_003",
            page_content="""Cancer immunotherapy has emerged as a revolutionary treatment approach that 
            harnesses the immune system to fight malignant cells. Checkpoint inhibitors targeting PD-1, 
            PD-L1, and CTLA-4 pathways have shown remarkable efficacy in various cancer types including 
            melanoma, lung cancer, and renal cell carcinoma. CAR-T cell therapy represents a breakthrough 
            in treating hematologic malignancies.""",
            metadata={
                "source": "e2e_test",
                "type": "biomedical",
                "topic": "oncology",
                "test_type": "e2e",
            },
        ),
        Document(
            id="e2e_test_neurology_004",
            page_content="""Alzheimer disease is the most common cause of dementia, characterized by 
            progressive cognitive decline and neurodegeneration. Pathological hallmarks include amyloid-beta 
            plaques, neurofibrillary tangles containing hyperphosphorylated tau protein, and neuronal loss. 
            Current FDA-approved treatments include cholinesterase inhibitors and NMDA receptor antagonist 
            memantine.""",
            metadata={
                "source": "e2e_test",
                "type": "biomedical",
                "topic": "neurology",
                "test_type": "e2e",
            },
        ),
        Document(
            id="e2e_test_respiratory_005",
            page_content="""COVID-19, caused by SARS-CoV-2, presents with a wide spectrum of clinical 
            manifestations from asymptomatic infection to severe acute respiratory distress syndrome. 
            Treatment protocols include supportive care, oxygen therapy, and antiviral medications such 
            as remdesivir and paxlovid. Corticosteroids like dexamethasone reduce mortality in severe cases.""",
            metadata={
                "source": "e2e_test",
                "type": "biomedical",
                "topic": "respiratory",
                "test_type": "e2e",
            },
        ),
    ]


@pytest.fixture(scope="session")
def e2e_biomedical_queries() -> List[Dict[str, Any]]:
    """
    Generate realistic biomedical queries for E2E testing.

    NO MOCKS - These are realistic queries that should retrieve real documents.
    """
    return [
        {
            "query": "What are the primary treatment approaches for cardiovascular disease?",
            "expected_topics": ["cardiovascular", "treatment"],
            "expected_keywords": ["therapy", "treatment", "prevention", "medication"],
        },
        {
            "query": "How is type 2 diabetes managed in clinical practice?",
            "expected_topics": ["diabetes", "management"],
            "expected_keywords": ["diabetes", "insulin", "metformin", "management"],
        },
        {
            "query": "What are the mechanisms of cancer immunotherapy?",
            "expected_topics": ["oncology", "immunotherapy"],
            "expected_keywords": ["immunotherapy", "cancer", "checkpoint", "CAR-T"],
        },
        {
            "query": "What are the pathological features of Alzheimer disease?",
            "expected_topics": ["neurology", "alzheimer"],
            "expected_keywords": ["alzheimer", "amyloid", "tau", "dementia"],
        },
        {
            "query": "How is COVID-19 treated with current protocols?",
            "expected_topics": ["respiratory", "covid"],
            "expected_keywords": ["covid", "treatment", "remdesivir", "dexamethasone"],
        },
    ]


@pytest.fixture(scope="function")
def e2e_performance_monitor():
    """
    Monitor performance metrics during E2E tests.

    NO MOCKS - Measures actual execution time and performance.
    """

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.metrics = {}

        def start_timer(self, operation: str):
            """Start timing an operation."""
            self.start_time = time.time()
            self.metrics[operation] = {"start_time": self.start_time}

        def end_timer(self, operation: str):
            """End timing an operation."""
            if operation in self.metrics and self.start_time:
                end_time = time.time()
                duration = end_time - self.start_time
                self.metrics[operation].update(
                    {"end_time": end_time, "duration_seconds": duration}
                )
                logger.info(f"E2E Performance - {operation}: {duration:.2f} seconds")
                return duration
            return 0

        def get_metrics(self) -> Dict[str, Any]:
            """Get all collected metrics."""
            return self.metrics

    return PerformanceMonitor()


@pytest.fixture(scope="function")
def e2e_document_validator():
    """
    Validate document lifecycle operations in E2E tests.

    NO MOCKS - Performs real validation against database state.
    Uses fresh connection manager to avoid connection closure issues.
    """

    class DocumentValidator:
        def __init__(self):
            # Create fresh connection manager for each validator instance
            self.connection_manager = ConnectionManager()

        def validate_document_ingestion(
            self, document_ids: List[str]
        ) -> Dict[str, Any]:
            """Validate documents were properly ingested."""
            try:
                connection = self.connection_manager.get_connection()
                cursor = connection.cursor()

                results = {}
                for doc_id in document_ids:
                    # Try both schema formats: new (doc_id, text_content) and old (id, content)
                    try:
                        cursor.execute(
                            "SELECT doc_id, text_content, embedding FROM RAG.SourceDocuments WHERE doc_id = ?",
                            [doc_id],
                        )
                        row = cursor.fetchone()
                    except Exception:
                        # Fallback to simple schema (id, content)
                        cursor.execute(
                            "SELECT id, content, embedding FROM RAG.SourceDocuments WHERE id = ?",
                            [doc_id],
                        )
                        row = cursor.fetchone()

                    if row:
                        results[doc_id] = {
                            "found": True,
                            "has_content": bool(row[1]),
                            "has_embedding": bool(row[2]),
                            "content_length": len(row[1]) if row[1] else 0,
                        }
                    else:
                        results[doc_id] = {"found": False}

                cursor.close()
                connection.close()

                return results

            except Exception as e:
                logger.error(f"Failed to validate document ingestion: {e}")
                return {}

        def validate_vector_search_results(
            self, results: List[Any], min_expected: int = 1
        ) -> bool:
            """Validate vector search returned meaningful results."""
            if len(results) < min_expected:
                return False

            # Check that results contain actual document content
            for result in results:
                if hasattr(result, "page_content") and result.page_content:
                    return True
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Handle (Document, score) tuples
                    doc, score = result[0], result[1]
                    if hasattr(doc, "page_content") and doc.page_content and score > 0:
                        return True

            return False

    return DocumentValidator()


# E2E test markers and configuration
pytestmark = pytest.mark.e2e


def pytest_configure(config):
    """Configure E2E-specific test markers."""
    config.addinivalue_line(
        "markers", "true_e2e: mark test as true end-to-end with real infrastructure"
    )
    config.addinivalue_line(
        "markers", "iris_required: mark test as requiring real IRIS database"
    )
    config.addinivalue_line(
        "markers", "pmc_data: mark test as using PMC biomedical data"
    )


def pytest_runtest_setup(item):
    """Setup for E2E tests - verify required environment."""
    if "e2e" in item.keywords:
        # Verify IRIS connection is available or can be provisioned
        if not os.environ.get("IRIS_HOST") and not has_devtester:
            pytest.skip("E2E tests require IRIS_HOST environment variable or iris-devtester")

        # Log that we're running a true E2E test
        logger.info(f"Running TRUE E2E test: {item.name}")
