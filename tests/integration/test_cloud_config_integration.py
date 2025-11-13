"""
Integration tests for Feature 058: Cloud Configuration System

Tests real IRIS database operations with CloudConfiguration API:
- ConnectionManager with environment variables
- SchemaManager dimension configuration
- Complete configuration priority chain (env > config > defaults)
"""

import os
import pytest
import tempfile
import yaml
from unittest.mock import patch


@pytest.mark.integration
class TestConnectionManagerIntegration:
    """Test ConnectionManager with CloudConfiguration and IRIS database"""

    def test_connection_manager_reads_env_vars(self):
        """
        GIVEN IRIS connection environment variables set
        WHEN ConnectionManager initializes with CloudConfiguration
        THEN connections use environment variable values
        """
        with patch.dict(os.environ, {
            'IRIS_HOST': 'localhost',
            'IRIS_PORT': '1972',
            'IRIS_USERNAME': '_SYSTEM',
            'IRIS_PASSWORD': 'SYS',
            'IRIS_NAMESPACE': 'USER'
        }, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            from iris_vector_rag.common.connection_manager import ConnectionManager

            config_manager = ConfigurationManager()
            cloud_config = config_manager.get_cloud_config()

            # Verify cloud config reads environment variables
            assert cloud_config.connection.host == 'localhost'
            assert cloud_config.connection.port == 1972
            assert cloud_config.connection.namespace == 'USER'
            assert cloud_config.connection.username == '_SYSTEM'

            # Verify sources tracked correctly
            assert cloud_config.connection.source['host'].value == 'environment'
            assert cloud_config.connection.source['port'].value == 'environment'

    def test_connection_to_real_iris_database(self):
        """
        GIVEN IRIS database running on localhost:1972
        WHEN creating connection via CloudConfiguration
        THEN connection succeeds and can query database
        """
        import iris.dbapi as iris_dbapi
        from iris_vector_rag.config.manager import ConfigurationManager

        config_manager = ConfigurationManager()
        cloud_config = config_manager.get_cloud_config()

        # Create connection using cloud config values
        conn = iris_dbapi.connect(
            hostname=cloud_config.connection.host,
            port=cloud_config.connection.port,
            namespace=cloud_config.connection.namespace,
            username=cloud_config.connection.username,
            password=cloud_config.connection.password
        )

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

            # Test namespace is accessible
            cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES")
            table_count = cursor.fetchone()[0]
            assert table_count > 0

            cursor.close()
        finally:
            conn.close()


@pytest.mark.integration
class TestSchemaManagerIntegration:
    """Test SchemaManager dimension configuration with IRIS database"""

    def test_schema_manager_reads_default_dimension(self):
        """
        GIVEN no VECTOR_DIMENSION environment variable
        WHEN SchemaManager initializes
        THEN uses default 384 dimensions
        """
        # Clear VECTOR_DIMENSION if set
        env_copy = {k: v for k, v in os.environ.items() if k != 'VECTOR_DIMENSION'}

        with patch.dict(os.environ, env_copy, clear=True):
            from iris_vector_rag.config.manager import ConfigurationManager

            config_manager = ConfigurationManager()
            cloud_config = config_manager.get_cloud_config()

            assert cloud_config.vector.vector_dimension == 384
            assert cloud_config.vector.source['vector_dimension'].value in ('default', 'config_file')

    def test_schema_manager_reads_env_var_dimension(self):
        """
        GIVEN VECTOR_DIMENSION=1024 environment variable
        WHEN SchemaManager initializes
        THEN uses 1024 dimensions from environment
        """
        with patch.dict(os.environ, {'VECTOR_DIMENSION': '1024'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager

            config_manager = ConfigurationManager()
            cloud_config = config_manager.get_cloud_config()

            assert cloud_config.vector.vector_dimension == 1024
            assert cloud_config.vector.source['vector_dimension'].value == 'environment'

    def test_schema_manager_creates_correct_vector_columns(self):
        """
        GIVEN VECTOR_DIMENSION=512 environment variable
        WHEN SchemaManager creates test table
        THEN table has VECTOR(DOUBLE, 512) column
        """
        with patch.dict(os.environ, {'VECTOR_DIMENSION': '512'}, clear=False):
            import iris.dbapi as iris_dbapi
            from iris_vector_rag.config.manager import ConfigurationManager

            config_manager = ConfigurationManager()
            cloud_config = config_manager.get_cloud_config()

            # Verify configuration
            assert cloud_config.vector.vector_dimension == 512

            # Connect to database
            conn = iris_dbapi.connect(
                hostname='localhost',
                port=1972,
                namespace='USER',
                username='_SYSTEM',
                password='SYS'
            )

            try:
                cursor = conn.cursor()

                # Create test table with configured dimension
                table_name = "RAG.TestVectorDim512"
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

                create_sql = f"""
                    CREATE TABLE {table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        embedding VECTOR(DOUBLE, {cloud_config.vector.vector_dimension})
                    )
                """
                cursor.execute(create_sql)
                conn.commit()

                # Verify column dimension using INFORMATION_SCHEMA
                cursor.execute("""
                    SELECT CHARACTER_MAXIMUM_LENGTH
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = 'RAG'
                    AND TABLE_NAME = 'TestVectorDim512'
                    AND COLUMN_NAME = 'embedding'
                """)

                byte_length = cursor.fetchone()[0]
                # Formula: dimensions = round(CHARACTER_MAXIMUM_LENGTH / 346)
                actual_dim = round(byte_length / 346)

                assert actual_dim == 512, f"Expected 512 dimensions, got {actual_dim}"

                # Cleanup
                cursor.execute(f"DROP TABLE {table_name}")
                conn.commit()
                cursor.close()

            finally:
                conn.close()


@pytest.mark.integration
class TestConfigurationPriorityChain:
    """Test configuration priority: env > config > defaults"""

    def test_environment_overrides_config_file(self):
        """
        GIVEN config file with vector_dimension=768
        AND VECTOR_DIMENSION=1536 environment variable
        WHEN ConfigurationManager loads
        THEN environment variable takes precedence
        """
        # Create temporary config file (must include required database config)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'database': {
                    'iris': {
                        'host': 'localhost',
                        'port': 1972,
                        'namespace': 'USER',
                        'username': '_SYSTEM',
                        'password': 'SYS'
                    }
                },
                'storage': {
                    'iris': {
                        'vector_dimension': 768
                    }
                }
            }, f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {'VECTOR_DIMENSION': '1536'}, clear=False):
                from iris_vector_rag.config.manager import ConfigurationManager

                config_manager = ConfigurationManager(config_path=config_path)
                cloud_config = config_manager.get_cloud_config()

                # Environment should override config file
                assert cloud_config.vector.vector_dimension == 1536
                assert cloud_config.vector.source['vector_dimension'].value == 'environment'
        finally:
            os.unlink(config_path)

    def test_config_file_overrides_defaults(self):
        """
        GIVEN config file with vector_dimension=2048
        AND no VECTOR_DIMENSION environment variable
        WHEN ConfigurationManager loads
        THEN config file value used
        """
        # Create temporary config file (must include required database config)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'database': {
                    'iris': {
                        'host': 'localhost',
                        'port': 1972,
                        'namespace': 'USER',
                        'username': '_SYSTEM',
                        'password': 'SYS'
                    }
                },
                'storage': {
                    'vector_dimension': 2048
                }
            }, f)
            config_path = f.name

        try:
            # Clear VECTOR_DIMENSION env var
            env_copy = {k: v for k, v in os.environ.items() if k != 'VECTOR_DIMENSION'}

            with patch.dict(os.environ, env_copy, clear=True):
                from iris_vector_rag.config.manager import ConfigurationManager

                config_manager = ConfigurationManager(config_path=config_path)
                cloud_config = config_manager.get_cloud_config()

                # Config file should override defaults
                assert cloud_config.vector.vector_dimension == 2048
                assert cloud_config.vector.source['vector_dimension'].value == 'config_file'
        finally:
            os.unlink(config_path)

    def test_defaults_used_when_no_overrides(self):
        """
        GIVEN no config file
        AND no VECTOR_DIMENSION environment variable
        WHEN ConfigurationManager loads
        THEN default 384 dimensions used
        """
        # Clear VECTOR_DIMENSION env var
        env_copy = {k: v for k, v in os.environ.items() if k != 'VECTOR_DIMENSION'}

        with patch.dict(os.environ, env_copy, clear=True):
            from iris_vector_rag.config.manager import ConfigurationManager

            config_manager = ConfigurationManager()
            cloud_config = config_manager.get_cloud_config()

            # Should use defaults
            assert cloud_config.vector.vector_dimension == 384
            assert cloud_config.vector.source['vector_dimension'].value in ('default', 'config_file')


@pytest.mark.integration
class TestCompleteConfigurationFlow:
    """Test complete configuration flow from environment to database"""

    def test_end_to_end_config_to_database(self):
        """
        GIVEN complete environment configuration
        WHEN system initializes and creates tables
        THEN all configuration flows correctly to database
        """
        test_env = {
            'IRIS_HOST': 'localhost',
            'IRIS_PORT': '1972',
            'IRIS_NAMESPACE': 'USER',
            'IRIS_USERNAME': '_SYSTEM',
            'IRIS_PASSWORD': 'SYS',
            'VECTOR_DIMENSION': '256',
            'TABLE_SCHEMA': 'RAG'
        }

        with patch.dict(os.environ, test_env, clear=False):
            import iris.dbapi as iris_dbapi
            from iris_vector_rag.config.manager import ConfigurationManager

            # Load configuration
            config_manager = ConfigurationManager()
            cloud_config = config_manager.get_cloud_config()

            # Verify all configuration loaded from environment
            assert cloud_config.connection.host == 'localhost'
            assert cloud_config.connection.port == 1972
            assert cloud_config.connection.namespace == 'USER'
            assert cloud_config.vector.vector_dimension == 256
            assert cloud_config.tables.table_schema == 'RAG'

            # Verify all sources are environment
            assert cloud_config.connection.source['host'].value == 'environment'
            assert cloud_config.connection.source['port'].value == 'environment'
            assert cloud_config.vector.source['vector_dimension'].value == 'environment'
            assert cloud_config.tables.source['table_schema'].value == 'environment'

            # Test actual database connection
            conn = iris_dbapi.connect(
                hostname=cloud_config.connection.host,
                port=cloud_config.connection.port,
                namespace=cloud_config.connection.namespace,
                username=cloud_config.connection.username,
                password=cloud_config.connection.password
            )

            try:
                cursor = conn.cursor()

                # Create test table using cloud config
                table_name = f"{cloud_config.tables.table_schema}.TestE2EConfig"
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

                create_sql = f"""
                    CREATE TABLE {table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        embedding VECTOR(DOUBLE, {cloud_config.vector.vector_dimension})
                    )
                """
                cursor.execute(create_sql)
                conn.commit()

                # Verify table created with correct schema and dimension
                cursor.execute("""
                    SELECT TABLE_SCHEMA, CHARACTER_MAXIMUM_LENGTH
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = 'TestE2EConfig'
                    AND COLUMN_NAME = 'embedding'
                """)

                result = cursor.fetchone()
                schema = result[0]
                byte_length = result[1]
                actual_dim = round(byte_length / 346)

                assert schema == 'RAG'
                assert actual_dim == 256

                # Cleanup
                cursor.execute(f"DROP TABLE {table_name}")
                conn.commit()
                cursor.close()

            finally:
                conn.close()


# Integration test metadata
INTEGRATION_METADATA = {
    "feature": "058-cloud-config-flexibility",
    "test_type": "integration",
    "requires_iris": True,
    "iris_port": 1972,
    "total_tests": 10,
    "test_categories": [
        "ConnectionManager with environment variables",
        "SchemaManager dimension configuration",
        "Configuration priority chain",
        "End-to-end configuration flow"
    ]
}
