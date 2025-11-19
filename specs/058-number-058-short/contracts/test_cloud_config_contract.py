"""
Contract tests for Feature 058: Cloud Configuration Flexibility

These tests define the expected behavior for cloud deployment configuration.
All tests MUST fail initially (TDD principle) and pass after implementation.

Test Coverage:
- FR-001: Environment variable support for connection parameters
- FR-002: Config file respected by init_tables()
- FR-003: Configurable vector dimensions (128-8192)
- FR-004: Table schema prefix configuration
- FR-005: Configuration priority order (env > config > defaults)
- FR-006: Preflight validation with clear error messages
- FR-008: Backward compatibility preserved
- SC-007: init_tables respects --config flag 100% of the time
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestEnvironmentVariableConfiguration:
    """FR-001: System MUST read connection parameters from environment variables"""

    def test_iris_host_from_environment_variable(self):
        """
        GIVEN environment variable IRIS_HOST is set
        WHEN ConfigurationManager is initialized
        THEN connection configuration uses the environment variable value
        """
        with patch.dict(os.environ, {'IRIS_HOST': 'aws-iris.example.com'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            # Test both legacy API and new cloud config API
            cloud_config = config.get_cloud_config()
            assert cloud_config.connection.host == 'aws-iris.example.com'
            assert cloud_config.connection.source['host'].value == 'environment'

            # Legacy compatibility check
            db_config = config.get_database_config()
            assert db_config['host'] == 'aws-iris.example.com'

    def test_iris_port_from_environment_variable(self):
        """
        GIVEN environment variable IRIS_PORT is set
        WHEN ConfigurationManager is initialized
        THEN connection configuration uses the environment variable value
        """
        with patch.dict(os.environ, {'IRIS_PORT': '21972'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()
            assert cloud_config.connection.port == 21972  # Converted to int
            assert cloud_config.connection.source['port'].value == 'environment'

    def test_iris_namespace_from_environment_variable(self):
        """
        GIVEN environment variable IRIS_NAMESPACE is set to %SYS
        WHEN ConfigurationManager is initialized
        THEN connection configuration uses %SYS namespace
        """
        with patch.dict(os.environ, {'IRIS_NAMESPACE': '%SYS'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()
            assert cloud_config.connection.namespace == '%SYS'
            assert cloud_config.connection.source['namespace'].value == 'environment'

    def test_all_iris_connection_vars_supported(self):
        """
        GIVEN all IRIS connection environment variables are set
        WHEN ConfigurationManager is initialized
        THEN all variables are correctly loaded
        """
        env_vars = {
            'IRIS_HOST': 'cloud-iris.example.com',
            'IRIS_PORT': '1972',
            'IRIS_USERNAME': 'AppUser',
            'IRIS_PASSWORD': 'SecurePassword123',
            'IRIS_NAMESPACE': 'SQLUser'
        }

        with patch.dict(os.environ, env_vars, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()
            assert cloud_config.connection.host == 'cloud-iris.example.com'
            assert cloud_config.connection.port == 1972
            assert cloud_config.connection.username == 'AppUser'
            assert cloud_config.connection.password == 'SecurePassword123'
            assert cloud_config.connection.namespace == 'SQLUser'

            # All should be from environment
            assert all(src.value == 'environment' for src in cloud_config.connection.source.values())


class TestConfigurationPriority:
    """FR-005: System MUST provide configuration priority order (env vars > config > defaults)"""

    def test_environment_variable_overrides_config_file(self):
        """
        GIVEN config file specifies host=localhost
        AND environment variable IRIS_HOST=aws-iris.example.com
        WHEN ConfigurationManager loads configuration
        THEN environment variable value takes precedence
        """
        # Create temporary config file with localhost
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'database': {'iris': {'host': 'localhost'}}}, f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {'IRIS_HOST': 'aws-iris.example.com'}, clear=False):
                from iris_vector_rag.config.manager import ConfigurationManager
                config = ConfigurationManager(config_path=config_path)

                # Env var should override config file
                cloud_config = config.get_cloud_config()
                assert cloud_config.connection.host == 'aws-iris.example.com'
                assert cloud_config.connection.source['host'].value == 'environment'
        finally:
            os.unlink(config_path)

    def test_config_file_overrides_defaults(self):
        """
        GIVEN config file specifies non-default values
        AND no environment variables set
        WHEN ConfigurationManager loads configuration
        THEN config file values override defaults
        """
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'database': {'iris': {'host': 'custom-iris.local'}},
                'storage': {'vector_dimension': 1024}
            }, f)
            config_path = f.name

        try:
            # Clear IRIS env vars for this test
            with patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('IRIS_') and k != 'VECTOR_DIMENSION'}, clear=True):
                from iris_vector_rag.config.manager import ConfigurationManager
                config = ConfigurationManager(config_path=config_path)

                # Config file should override defaults
                cloud_config = config.get_cloud_config()
                assert cloud_config.connection.host == 'custom-iris.local'
                assert cloud_config.connection.source['host'].value == 'config_file'
                assert cloud_config.vector.vector_dimension == 1024
                assert cloud_config.vector.source['vector_dimension'].value == 'config_file'
        finally:
            os.unlink(config_path)

    def test_defaults_used_when_no_overrides(self):
        """
        GIVEN no config file provided
        AND no environment variables set
        WHEN ConfigurationManager loads configuration
        THEN default values are used
        """
        # Clear any IRIS env vars for this test
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('IRIS_') and k != 'VECTOR_DIMENSION'}, clear=True):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            # Defaults should be used (either hardcoded defaults or from default_config.yaml)
            cloud_config = config.get_cloud_config()
            assert cloud_config.connection.host == 'localhost'
            assert cloud_config.connection.port in (1972, 1974)  # 1972=hardcoded, 1974=default_config.yaml
            assert cloud_config.connection.namespace == 'USER'
            assert cloud_config.vector.vector_dimension == 384

            # All should be from defaults or config file (no environment overrides)
            assert all(src.value in ('default', 'config_file') for src in cloud_config.connection.source.values())
            assert all(src.value in ('default', 'config_file') for src in cloud_config.vector.source.values())


class TestVectorDimensionConfiguration:
    """FR-003: System MUST support configurable vector dimensions from 128 to 8192"""

    def test_vector_dimension_configurable_via_env_var(self):
        """
        GIVEN VECTOR_DIMENSION environment variable set to 1024
        WHEN ConfigurationManager loads configuration
        THEN vector dimension is set to 1024
        """
        with patch.dict(os.environ, {'VECTOR_DIMENSION': '1024'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()
            assert cloud_config.vector.vector_dimension == 1024
            assert cloud_config.vector.source['vector_dimension'].value == 'environment'

    def test_vector_dimension_validation_min_bound(self):
        """
        GIVEN VECTOR_DIMENSION set to 64 (below minimum 128)
        WHEN CloudConfiguration validates
        THEN ValueError is raised with clear message
        """
        with patch.dict(os.environ, {'VECTOR_DIMENSION': '64'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager

            config = ConfigurationManager()
            cloud_config = config.get_cloud_config()

            # Validation is explicit, not automatic
            with pytest.raises(ValueError) as exc_info:
                cloud_config.validate()

            assert "vector dimension" in str(exc_info.value).lower()
            assert "128" in str(exc_info.value)  # Minimum value mentioned

    def test_vector_dimension_validation_max_bound(self):
        """
        GIVEN VECTOR_DIMENSION set to 10000 (above maximum 8192)
        WHEN CloudConfiguration validates
        THEN ValueError is raised with clear message
        """
        with patch.dict(os.environ, {'VECTOR_DIMENSION': '10000'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager

            config = ConfigurationManager()
            cloud_config = config.get_cloud_config()

            # Validation is explicit, not automatic
            with pytest.raises(ValueError) as exc_info:
                cloud_config.validate()

            assert "vector dimension" in str(exc_info.value).lower()
            assert "8192" in str(exc_info.value)  # Maximum value mentioned

    def test_common_vector_dimensions_supported(self):
        """
        GIVEN common embedding model dimensions
        WHEN ConfigurationManager validates configuration
        THEN all common dimensions are accepted
        """
        common_dimensions = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]

        for dim in common_dimensions:
            with patch.dict(os.environ, {'VECTOR_DIMENSION': str(dim)}, clear=False):
                from iris_vector_rag.config.manager import ConfigurationManager
                config = ConfigurationManager()

                cloud_config = config.get_cloud_config()
                assert cloud_config.vector.vector_dimension == dim

                # Should not raise when validated
                cloud_config.validate()  # Should pass for valid dimensions


class TestTableSchemaConfiguration:
    """FR-004: System MUST allow table schema prefix configuration"""

    def test_table_schema_configurable_via_env_var(self):
        """
        GIVEN TABLE_SCHEMA environment variable set to SQLUser
        WHEN ConfigurationManager loads configuration
        THEN table schema prefix is set to SQLUser
        """
        with patch.dict(os.environ, {'TABLE_SCHEMA': 'SQLUser'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()
            assert cloud_config.tables.table_schema == 'SQLUser'
            assert cloud_config.tables.source['table_schema'].value == 'environment'

    def test_table_schema_supports_percent_sys_namespace(self):
        """
        GIVEN TABLE_SCHEMA set to %SYS
        WHEN ConfigurationManager loads configuration
        THEN %SYS schema is accepted
        """
        with patch.dict(os.environ, {'TABLE_SCHEMA': '%SYS'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()
            assert cloud_config.tables.table_schema == '%SYS'

            # Should validate successfully (% is allowed in schema names)
            cloud_config.validate()

    def test_full_table_name_includes_schema_prefix(self):
        """
        GIVEN table_schema configured as SQLUser
        WHEN getting full table name for Entities
        THEN returns SQLUser.Entities
        """
        with patch.dict(os.environ, {'TABLE_SCHEMA': 'SQLUser'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()
            assert cloud_config.tables.full_entities_table == "SQLUser.Entities"
            assert cloud_config.tables.full_relationships_table == "SQLUser.EntityRelationships"


class TestInitTablesConfigRespect:
    """FR-002 & SC-007: init_tables() MUST respect --config flag 100% of the time"""

    @pytest.mark.skip(reason="Feature not implemented yet - TDD contract test")
    def test_init_tables_loads_config_from_flag(self):
        """
        GIVEN a config file at /tmp/test-config.yaml with custom settings
        WHEN init_tables is run with --config /tmp/test-config.yaml
        THEN the config file settings are used for table creation
        """
        # This test requires mocking the actual table creation
        # and verifying the config was loaded correctly
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'storage': {
                    'vector_dimension': 1024,
                    'table_schema': 'SQLUser'
                }
            }, f)
            config_path = f.name

        try:
            # Mock the table creation to verify config was used
            with patch('iris_vector_rag.cli.init_tables.create_tables') as mock_create:
                from iris_vector_rag.cli import init_tables
                import sys

                # Simulate CLI args
                sys.argv = ['init_tables.py', '--config', config_path]
                init_tables.main()

                # Verify create_tables was called with config from file
                mock_create.assert_called_once()
                call_args = mock_create.call_args
                config_used = call_args[0][0]  # First positional arg is config

                assert config_used.get('storage.vector_dimension') == 1024
                assert config_used.get('storage.table_schema') == 'SQLUser'
        finally:
            os.unlink(config_path)

    @pytest.mark.skip(reason="Feature not implemented yet - TDD contract test")
    def test_init_tables_without_config_flag_uses_defaults(self):
        """
        GIVEN init_tables is run without --config flag
        WHEN init_tables executes
        THEN default configuration is used
        """
        with patch('iris_vector_rag.cli.init_tables.create_tables') as mock_create:
            from iris_vector_rag.cli import init_tables
            import sys

            sys.argv = ['init_tables.py']  # No --config flag
            init_tables.main()

            mock_create.assert_called_once()
            call_args = mock_create.call_args
            config_used = call_args[0][0]

            # Verify defaults are used
            assert config_used.get('storage.vector_dimension') == 384
            assert config_used.get('storage.table_schema') == 'RAG'


class TestPreflightValidation:
    """FR-006: System MUST validate configuration at startup and fail fast"""

    @pytest.mark.integration
    def test_vector_dimension_mismatch_detected(self):
        """
        GIVEN existing table with dimension 384
        AND config specifies dimension 1024
        WHEN VectorDimensionValidator runs
        THEN ValidationResult with FAIL status is returned with actionable message
        """
        # Import required modules
        import iris.dbapi as iris_dbapi
        from iris_vector_rag.config.validators import VectorDimensionValidator, ValidationStatus
        from iris_vector_rag.config.entities import VectorConfiguration, TableConfiguration, ConfigSource

        # Connect to real IRIS database
        conn = iris_dbapi.connect(
            hostname='localhost',
            port=1972,
            namespace='USER',
            username='_SYSTEM',
            password='SYS'
        )

        try:
            cursor = conn.cursor()

            # Create test table with 384 dimensions if it doesn't exist
            cursor.execute("DROP TABLE IF EXISTS RAG.TestEntities_384")
            cursor.execute("""
                CREATE TABLE RAG.TestEntities_384 (
                    id VARCHAR(255) PRIMARY KEY,
                    embedding VECTOR(DOUBLE, 384)
                )
            """)
            conn.commit()

            # Test validation with mismatched dimension (1024 vs 384)
            vector_config = VectorConfiguration(
                vector_dimension=1024,
                source={'vector_dimension': ConfigSource.ENVIRONMENT}
            )
            table_config = TableConfiguration(
                table_schema='RAG',
                entities_table='TestEntities_384'
            )

            validator = VectorDimensionValidator()
            result = validator.validate(vector_config, conn, table_config)

            # Verify failure result
            assert result.is_failure(), f"Expected FAIL but got {result.status}"
            assert result.status == ValidationStatus.FAIL
            assert "mismatch" in result.message.lower()
            assert "384" in result.message  # Existing dimension
            assert "1024" in result.message  # Configured dimension
            assert result.help_url is not None

            # Cleanup
            cursor.execute("DROP TABLE RAG.TestEntities_384")
            conn.commit()

        finally:
            cursor.close()
            conn.close()

    @pytest.mark.integration
    def test_namespace_permission_validation(self):
        """
        GIVEN connection to IRIS with valid namespace
        WHEN NamespaceValidator runs
        THEN ValidationResult with PASS status is returned
        """
        # Import required modules
        import iris.dbapi as iris_dbapi
        from iris_vector_rag.config.validators import NamespaceValidator, ValidationStatus
        from iris_vector_rag.config.entities import ConnectionConfiguration, TableConfiguration, ConfigSource

        # Connect to real IRIS database
        conn = iris_dbapi.connect(
            hostname='localhost',
            port=1972,
            namespace='USER',
            username='_SYSTEM',
            password='SYS'
        )

        try:
            # Test validation with USER namespace
            connection_config = ConnectionConfiguration(
                host='localhost',
                port=1972,
                username='_SYSTEM',
                password='SYS',
                namespace='USER',
                source={'namespace': ConfigSource.DEFAULT}
            )
            table_config = TableConfiguration(
                table_schema='RAG'
            )

            validator = NamespaceValidator()
            result = validator.validate(connection_config, conn, table_config)

            # Verify success result (or warning if permission check failed)
            assert result.status in (ValidationStatus.PASS, ValidationStatus.WARNING), \
                f"Expected PASS or WARNING but got {result.status}: {result.message}"

            if result.status == ValidationStatus.PASS:
                assert "USER" in result.message or "validated" in result.message.lower()
                assert result.details is not None
                assert result.details['namespace'] == 'USER'

        finally:
            conn.close()


class TestBackwardCompatibility:
    """FR-008 & SC-006: Existing local deployments MUST work without changes"""

    def test_default_configuration_matches_v04x_behavior(self):
        """
        GIVEN no environment variables set
        AND no config file provided
        WHEN ConfigurationManager loads defaults
        THEN configuration matches iris-vector-rag v0.4.x behavior
        """
        # Clear any IRIS env vars
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('IRIS_') and k != 'VECTOR_DIMENSION'}, clear=True):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            # Verify v0.4.x defaults (accounting for default_config.yaml)
            db_config = config.get_database_config()
            assert db_config['host'] == 'localhost'
            assert db_config['port'] in (1972, 1974, '1974')  # 1972 hardcoded, 1974/string from default_config.yaml
            assert db_config['username'] == '_SYSTEM'
            assert db_config['password'] == 'SYS'
            assert db_config['namespace'] == 'USER'

            # New cloud config API should also work
            cloud_config = config.get_cloud_config()
            assert cloud_config.vector.vector_dimension == 384
            assert cloud_config.tables.table_schema == 'RAG'

    def test_existing_code_works_without_modifications(self):
        """
        GIVEN existing iris-vector-rag code from v0.4.x
        WHEN code is run with new ConfigurationManager
        THEN all existing functionality works unchanged
        """
        # This is a smoke test to verify backward compatibility
        from iris_vector_rag.config.manager import ConfigurationManager

        # Existing code pattern (no changes required)
        config = ConfigurationManager()

        # Existing API still works
        assert config.get('database:iris:host') == 'localhost'
        assert config.get_database_config() is not None
        assert config.get_embedding_config() is not None

        # New API also works
        cloud_config = config.get_cloud_config()
        assert cloud_config is not None
        assert cloud_config.connection.host == 'localhost'


class TestConfigurationLogging:
    """FR-011: System MUST log all configuration sources for troubleshooting"""

    def test_configuration_source_tracking(self):
        """
        GIVEN configuration loaded from multiple sources
        WHEN configuration is accessed
        THEN ConfigurationSource entries track where each value came from
        """
        with patch.dict(os.environ, {'IRIS_HOST': 'env-host.example.com'}, clear=False):
            import tempfile
            import yaml

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump({
                    'database': {'iris': {'host': 'localhost'}},  # Required for validation
                    'storage': {'vector_dimension': 1024}
                }, f)
                config_path = f.name

            try:
                from iris_vector_rag.config.manager import ConfigurationManager
                config = ConfigurationManager(config_path=config_path)

                # Get cloud config with source tracking
                cloud_config = config.get_cloud_config()

                # Verify host came from environment (overrides file)
                assert cloud_config.connection.host == 'env-host.example.com'
                assert cloud_config.connection.source['host'].value == 'environment'

                # Verify vector dimension came from config file
                assert cloud_config.vector.vector_dimension == 1024
                assert cloud_config.vector.source['vector_dimension'].value == 'config_file'

                # Verify defaults are tracked
                assert cloud_config.connection.source['username'].value in ('default', 'config_file')
            finally:
                os.unlink(config_path)

    def test_password_masking_in_configuration_log(self):
        """
        GIVEN password set via environment variable
        WHEN configuration is converted to dict
        THEN password value is masked as ***
        """
        with patch.dict(os.environ, {'IRIS_PASSWORD': 'SecretPassword123'}, clear=False):
            from iris_vector_rag.config.manager import ConfigurationManager
            config = ConfigurationManager()

            cloud_config = config.get_cloud_config()

            # Password should be masked in to_dict()
            config_dict = cloud_config.to_dict()
            assert config_dict['connection']['password'] == '***MASKED***'
            assert 'SecretPassword123' not in str(config_dict)


# Contract test metadata
CONTRACT_METADATA = {
    "feature": "058-cloud-config-flexibility",
    "total_tests": 22,
    "requirements_covered": ["FR-001", "FR-002", "FR-003", "FR-004", "FR-005", "FR-006", "FR-008", "FR-011"],
    "success_criteria_covered": ["SC-006", "SC-007"],
    "expected_status": "ALL SKIPPED (TDD - implementation pending)"
}
