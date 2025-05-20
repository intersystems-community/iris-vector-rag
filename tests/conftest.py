# tests/conftest.py
# Common test fixtures for RAG template tests.

import pytest
import numpy as np
from typing import Dict, List, Any, Callable, Optional
import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Changed from INFO to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.utils import Document
from common.utils import get_iris_connector, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection

# Import standardized mocks from the mocks module
from tests.mocks.db import MockIRISConnector, MockIRISCursor
from tests.mocks.models import (
    mock_embedding_func, 
    mock_llm_func,
    mock_colbert_doc_encoder,
    mock_colbert_query_encoder
)

# Import real data fixtures
from tests.fixtures.real_data import (
    real_iris_available,
    real_data_available,
    use_real_data,
    iris_connection
)

# --- Fixtures for Real Dependencies (for e2e metrics tests) ---

# --- Updated Real Connection Fixture ---

@pytest.fixture(scope="session")
def iris_connection_real():
    """
    Provides a real connection to the running IRIS Docker container.
    Attempts connection using environment variables.
    Manages connection setup and teardown.
    
    Note: This fixture exists for backward compatibility. New tests should
    use the iris_connection fixture with the use_real_data flag.
    """
    print("\nFixture: Attempting to establish real IRIS connection...")
    
    # Use our new connection function
    conn = get_iris_connection(use_mock=False)
    
    if conn:
        print("Fixture: Real IRIS connection established.")
        yield conn
        try:
            print("\nFixture: Closing real IRIS connection.")
            conn.close()
        except Exception as e:
            print(f"Fixture: Error closing IRIS connection - {e}")
    else:
        print("Fixture: Failed to establish real IRIS connection. Yielding None.")
        # Yield None to allow tests to proceed and potentially be skipped
        yield None


@pytest.fixture(scope="session")
def embedding_model_fixture():
    """
    Loads and provides the actual embedding function.
    """
    print("\nFixture: Loading real embedding model function...")
    embed_func = get_embedding_func()
    # The get_embedding_func itself handles model loading and returns a callable.
    # It also prints success/failure messages.
    yield embed_func

@pytest.fixture(scope="session")
def llm_client_fixture():
    """
    Initializes and provides the actual LLM function.
    """
    print("\nFixture: Initializing real LLM client function...")
    llm_func = get_llm_func()
    # The get_llm_func itself handles client initialization and returns a callable.
    # It also prints success/failure messages.
    yield llm_func

@pytest.fixture(scope="session")
def colbert_query_encoder_fixture():
    """
    Provides a function that generates token-level embeddings for ColBERT queries.
    This is a temporary placeholder implementation until we have a proper ColBERT encoder.
    """
    print("\nFixture: Initializing ColBERT query encoder function...")
    
    # For now, we'll use the regular embedding function but wrap it to return multiple embeddings
    # (one per token) instead of a single embedding
    embedding_func = get_embedding_func()
    
    def colbert_query_encoder(text):
        # Simple tokenization by splitting on spaces
        tokens = text.split()
        if not tokens:
            tokens = [text]  # If no spaces, use the whole text as one token
            
        # Limit tokens to avoid excessive processing
        tokens = tokens[:20]
        
        # Get embeddings for each token
        try:
            token_embeddings = embedding_func(tokens)
            print(f"Generated {len(token_embeddings)} token embeddings for ColBERT query")
            return token_embeddings
        except Exception as e:
            print(f"Error generating ColBERT token embeddings: {e}")
            # Return dummy embeddings (3 tokens with 10-dim vectors)
            return [[0.1] * 10 for _ in range(3)]
    
    yield colbert_query_encoder

# --- Fixture for Evaluation Dataset ---

@pytest.fixture(scope="session")
def evaluation_dataset() -> List[Dict[str, Any]]:
    """
    Loads the shared evaluation dataset from sample_queries.json.
    """
    print("\nFixture: Loading evaluation dataset from eval/sample_queries.json")
    # Construct the full path to the JSON file relative to the project root (where pytest runs)
    # Assuming conftest.py is in tests/ and sample_queries.json is in eval/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_file_path = os.path.join(project_root, "eval", "sample_queries.json")
    
    try:
        with open(json_file_path, 'r') as f:
            dataset = json.load(f)
        print(f"Successfully loaded {len(dataset)} queries from {json_file_path}")
        return dataset
    except FileNotFoundError:
        print(f"Error: Evaluation dataset file not found at {json_file_path}")
        return [] # Return empty list or raise error
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return []

# --- Fixtures for Mocks (for unit tests) ---

@pytest.fixture
def mock_iris_connector():
    """Provides a standardized mock IRIS connector."""
    print("\nFixture: Providing standardized mock IRIS connector")
    return MockIRISConnector()

@pytest.fixture
def mock_embedding_func(mocker):
    """Provides a mock embedding function."""
    print("\nFixture: Providing mock embedding function")
    # Create a mock that wraps our standardized mock function
    mock_embed = mocker.Mock(side_effect=mock_embedding_func)
    return mock_embed

@pytest.fixture
def mock_llm_func(mocker):
    """Provides a mock LLM function."""
    print("\nFixture: Providing mock LLM function")
    # Create a mock that wraps our standardized mock function
    mock_llm = mocker.Mock(side_effect=mock_llm_func)
    return mock_llm

@pytest.fixture
def mock_colbert_doc_encoder(mocker):
    """Provides a mock ColBERT document encoder function."""
    print("\nFixture: Providing mock ColBERT document encoder function")
    # Create a mock that wraps our standardized mock function
    mock_encoder = mocker.Mock(side_effect=mock_colbert_doc_encoder)
    return mock_encoder


@pytest.fixture
def mock_web_search_func(mocker):
    """Provides a mock web search function (for CRAG)."""
    print("\nFixture: Providing mock web search function")
    mock_search = mocker.Mock()
    mock_search.return_value = ["Mock web result 1", "Mock web result 2"]
    return mock_search

@pytest.fixture
def mock_graph_lib(mocker):
    """Provides a mock graph library (for NodeRAG/GraphRAG)."""
    print("\nFixture: Providing mock graph library")
    mock_lib = mocker.Mock()
    # Mock specific graph functions as needed (e.g., mock_lib.Graph.return_value = mocker.Mock())
    return mock_lib

# --- Testcontainer Fixtures (for isolated testing with real data) ---

# Register custom pytest markers
def pytest_configure(config):
    """Register custom pytest markers"""
    config.addinivalue_line("markers", "force_testcontainer: mark test to force use of testcontainer")
    config.addinivalue_line("markers", "force_mock: mark test to force use of mock connection")
    config.addinivalue_line("markers", "force_real: mark test to force use of real connection")

@pytest.fixture(scope="session")
def iris_testcontainer():
    """
    Session-scoped IRIS testcontainer fixture.
    
    Creates an ephemeral IRIS container for testing and handles cleanup.
    Should be used with tests requiring a real database but isolated from
    the production environment.
    """
    logger.info("Starting IRIS testcontainer...")
    
    # Check if testcontainers module is available
    try:
        from testcontainers.iris import IRISContainer
        import sqlalchemy
    except ImportError:
        logger.error("testcontainers-iris package not installed. Install with: pip install testcontainers-iris")
        pytest.skip("testcontainers-iris not installed")
        return None
    
    # Use latest iris-community image by default or from environment variable
    is_arm64 = os.uname().machine == 'arm64'
    default_image = "intersystemsdc/iris-community:latest"
    image = os.environ.get("IRIS_DOCKER_IMAGE", default_image)
    
    logger.info(f"Creating IRIS testcontainer with image: {image} on {'ARM64' if is_arm64 else 'x86_64'}")
    
    # Create and start container
    container = IRISContainer(image)
    
    try:
        container.start()
        
        # Manually create connection URL to work around bug in testcontainers-iris
        host = container.get_container_host_ip()
        port = container.get_exposed_port(container.port)
        username = container.username
        password = container.password
        namespace = container.namespace
        
        connection_url = f"iris://{username}:{password}@{host}:{port}/{namespace}"
        
        # Store connection URL on the container for later use
        container.connection_url = connection_url
        
        logger.info(f"IRIS testcontainer started. Connection URL: {connection_url}")
        
        yield container
        
    finally:
        # Stop container when tests are done
        logger.info("Stopping IRIS testcontainer...")
        container.stop()
        logger.info("IRIS testcontainer stopped")

@pytest.fixture(scope="session")
def iris_testcontainer_connection(iris_testcontainer):
    """
    Session-scoped connection to IRIS testcontainer.
    
    Creates a SQLAlchemy connection to the testcontainer and initializes the database schema.
    """
    if iris_testcontainer is None:
        pytest.skip("IRIS testcontainer not available")
        return None
        
    try:
        import sqlalchemy
        from common.db_init import initialize_database
        
        # Create SQLAlchemy connection using the URL we manually created
        connection_url = iris_testcontainer.connection_url
        engine = sqlalchemy.create_engine(connection_url)
        connection = engine.connect().connection
        
        # Initialize database schema
        logger.info("Initializing database schema in testcontainer")
        initialize_database(connection, force_recreate=True)
        
        yield connection
        
        # Close connection with better error handling
        try:
            connection.close()
            engine.dispose()
            logger.info("Closed connection to IRIS testcontainer")
        except Exception as e:
            logger.warning(f"Note: Exception during connection close (can be ignored): {e}")
            
    except Exception as e:
        logger.error(f"Failed to create connection to IRIS testcontainer: {e}")
        pytest.skip(f"Failed to create connection to IRIS testcontainer: {e}")
        yield None

@pytest.fixture(scope="session")
def iris_with_pmc_data(iris_testcontainer_connection):
    """
    Session-scoped fixture with preloaded PMC data.
    
    Loads a sample of PMC documents into the testcontainer database.
    """
    if iris_testcontainer_connection is None:
        pytest.skip("IRIS testcontainer connection not available")
        return None
    
    # Import the load_pmc_documents function
    from tests.utils import load_pmc_documents
    
    # Process a configurable set of PMC documents
    logger.info("Loading PMC documents into testcontainer")
    
    # Check for document count first (takes priority), then fall back to PMC limit
    if os.environ.get('TEST_DOCUMENT_COUNT'):
        limit = int(os.environ.get('TEST_DOCUMENT_COUNT'))
        logger.info(f"Using TEST_DOCUMENT_COUNT={limit}")
        
        # When using a large document count, log warning and collect performance metrics
        if limit >= 1000:
            logger.info(f"Large-scale test with {limit} documents - collecting performance metrics")
            os.environ["COLLECT_PERFORMANCE_METRICS"] = "true"
    else:
        limit = int(os.environ.get('TEST_PMC_LIMIT', '30'))
        logger.info(f"Using TEST_PMC_LIMIT={limit}")
    
    # Log warning for large document counts
    if limit > 500:
        logger.warning(f"Processing {limit} documents may take a significant amount of time")
    
    try:
        # For large-scale tests, use the optimized loading function if available
        if limit >= 500 and 'tests.utils_large_scale' in sys.modules:
            from tests.utils_large_scale import load_pmc_documents_large_scale
            metrics = load_pmc_documents_large_scale(
                connection=iris_testcontainer_connection,
                limit=limit,
                pmc_dir="data/pmc_oas_downloaded",
                batch_size=50
            )
            doc_count = metrics["document_count"]
            logger.info(f"Loaded {doc_count} PMC documents into testcontainer using optimized loader")
            logger.info(f"Performance: {metrics['docs_per_second']:.2f} docs/sec, peak memory: {metrics['peak_memory_mb']:.1f} MB")
        else:
            # Use the standard loader for smaller document sets
            doc_count = load_pmc_documents(
                connection=iris_testcontainer_connection,
                limit=limit,
                pmc_dir="data/pmc_oas_downloaded"
            )
            logger.info(f"Loaded {doc_count} PMC documents into testcontainer")
        
        # NEVER skip the test, even if no documents were loaded
        # Let the test decide how to handle this case
        if doc_count == 0:
            logger.warning("No PMC documents were loaded - tests will run with empty database")
            
        # Return the connection with data loaded
        yield iris_testcontainer_connection
        
    except Exception as e:
        logger.error(f"Failed to load PMC data into testcontainer: {e}")
        pytest.skip(f"Failed to load PMC data: {e}")
        yield None

@pytest.fixture
def real_embedding_model():
    """
    Get a real embedding model for working with real data.
    
    This fixture returns a real embedding model by default to ensure
    that tests run with realistic embeddings. A mock model is only used
    if explicitly requested or in CI environments.
    
    Returns:
        A real embedding model unless mock is explicitly requested
    """
    from common.embedding_utils import get_embedding_model
    
    # Default to real embeddings
    use_mock = False
    
    # Only use mock embeddings if explicitly requested or in CI
    if os.environ.get('USE_MOCK_EMBEDDINGS', '').lower() in ('true', '1', 'yes'):
        logger.info("Using mock embeddings as requested via USE_MOCK_EMBEDDINGS")
        use_mock = True
    elif os.environ.get('CI', '').lower() in ('true', '1', 'yes'):
        logger.info("Using mock embeddings for CI environment")
        use_mock = True
    
    logger.info(f"Using {'mock' if use_mock else 'real'} embedding model")
    model = get_embedding_model(mock=use_mock)
    return model

@pytest.fixture
def use_testcontainer(request):
    """
    Flag to indicate whether to use a testcontainer.
    
    Checks for appropriate markers to determine if a testcontainer should be used.
    """
    # Force using testcontainer if the marker is present
    if request.node.get_closest_marker("force_testcontainer"):
        return True
    
    # Don't use testcontainer if force_mock is present
    if request.node.get_closest_marker("force_mock"):
        return False
    
    # Default: Use testcontainer if TEST_IRIS environment variable is set
    return os.environ.get('TEST_IRIS', '').lower() in ('true', '1', 'yes')

@pytest.fixture
def iris_connection_auto(use_testcontainer, iris_testcontainer_connection, request):
    """
    Flexible IRIS connection fixture that automatically chooses the best connection.
    
    This fixture provides the appropriate connection based on the test's requirements:
    - If force_testcontainer marker is present, uses testcontainer
    - If force_mock marker is present, uses mock
    - If use_real_data marker is present, uses real or testcontainer
    - Otherwise falls back based on availability
    
    Args:
        use_testcontainer: Boolean indicating whether to use testcontainer
        iris_testcontainer_connection: Testcontainer connection from fixture
        request: Pytest request object
        
    Returns:
        An IRIS connection (testcontainer, real, or mock)
    """
    from common.iris_connector import get_iris_connection
    
    # Check for markers
    force_mock = request.node.get_closest_marker("force_mock")
    force_real = request.node.get_closest_marker("force_real")
    
    # If force_mock, always use mock
    if force_mock:
        logger.info("Using mock connection due to force_mock marker")
        return get_iris_connection(use_mock=True)
    
    # If force_real, try real connection or skip
    if force_real:
        conn = get_iris_connection(use_mock=False, use_testcontainer=False)
        if not conn:
            pytest.skip("Test requires real connection but none is available")
        return conn
    
    # If testcontainer should be used and is available
    if use_testcontainer and iris_testcontainer_connection:
        logger.info("Using testcontainer connection")
        return iris_testcontainer_connection
    
    # Otherwise try in order: real > testcontainer > mock
    logger.info("Attempting to get the best available connection")
    return get_iris_connection()
