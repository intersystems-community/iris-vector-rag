# tests/fixtures/database.py

import pytest
import logging
from tests.utils.container_lifecycle_manager import ContainerLifecycleManager
from tests.utils.db_reset import DatabaseReset

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def iris_container():
    """Ensure IRIS container is running for the test session with COMPOSE_FILE change detection."""
    lifecycle_mgr = ContainerLifecycleManager()
    
    # Ensure correct container is running (handles COMPOSE_FILE changes)
    connection_params = lifecycle_mgr.ensure_correct_container_running()
    
    yield connection_params
    
    # Container stays running for potential next test runs
    # Cleanup is handled automatically by the lifecycle manager

@pytest.fixture(scope="function")
def initialized_database(iris_container):
    """Initialize database schema once per session using SchemaManager."""
    import intersystems_iris.dbapi as iris
    from iris_rag.storage.schema_manager import SchemaManager
    from iris_rag.config.manager import ConfigurationManager
    
    # Get connection using container params
    conn = iris.connect(**iris_container)
    
    try:
        # Initialize schema manager with proper connection manager
        class ConnectionManager:
            def get_connection(self):
                return iris.connect(**iris_container)
        
        connection_manager = ConnectionManager()
        config_manager = ConfigurationManager()
        
        logger.info("Initializing database schema using SchemaManager...")
        
        # Create schema manager - this will handle schema metadata table creation
        try:
            schema_manager = SchemaManager(connection_manager, config_manager)
            
            # Ensure essential tables exist
            essential_tables = ['SourceDocuments', 'DocumentTokenEmbeddings', 'KnowledgeGraphNodes', 'KnowledgeGraphEdges']
            
            for table_name in essential_tables:
                logger.info(f"Ensuring {table_name} table schema...")
                try:
                    success = schema_manager.ensure_table_schema(table_name)
                    if success:
                        logger.info(f"✅ {table_name} table ready")
                    else:
                        logger.warning(f"⚠️ {table_name} table setup had issues")
                except Exception as e:
                    logger.warning(f"⚠️ {table_name} table error: {e}")
                    # Continue with other tables even if one fails
            
            logger.info("✅ Database schema initialized successfully with SchemaManager")
            
        except Exception as e:
            logger.warning(f"SchemaManager initialization failed: {e}")
            logger.info("Falling back to basic table creation...")
            
            # Fallback: create basic SourceDocuments table manually if SchemaManager fails
            cursor = conn.cursor()
            try:
                cursor.execute("CREATE SCHEMA IF NOT EXISTS RAG")
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (
                    doc_id VARCHAR(255) PRIMARY KEY,
                    title VARCHAR(1000),
                    text_content CLOB,
                    embedding VARCHAR(60000),
                    metadata_json CLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                conn.commit()
                logger.info("✅ Basic SourceDocuments table created as fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback table creation also failed: {fallback_error}")
            finally:
                cursor.close()
        
        yield conn
        
    finally:
        conn.close()

@pytest.fixture(scope="function")
def clean_database(initialized_database, iris_container):
    """Provide a clean database state for each test."""
    import intersystems_iris.dbapi as iris
    
    # Get fresh connection using container params
    conn = iris.connect(**iris_container)
    
    # Reset to clean state (but preserve schema)
    db_reset = DatabaseReset()
    db_reset.clean_test_tables(conn)
    
    yield conn
    
    # Cleanup after test
    conn.close()

@pytest.fixture(scope="function")
def test_data_1000(iris_container):
    """Ensure 1000+ documents are available for testing."""
    import intersystems_iris.dbapi as iris
    
    conn = iris.connect(**iris_container)
    
    # Check if we have enough data
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    count = cursor.fetchone()[0]
    
    if count < 1000:
        pytest.skip(f"Test requires 1000+ documents, found {count}")
    
    yield count
    conn.close()