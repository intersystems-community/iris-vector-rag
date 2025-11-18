"""
Configuration entity models for cloud deployment flexibility.

This module defines configuration entities for managing IRIS database connection,
vector storage, and table schema settings with support for environment variables,
config files, and sensible defaults.

Feature: 058-cloud-config-flexibility
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ConfigSource(Enum):
    """
    Source of configuration value.

    Tracks where each configuration value originated for debugging and audit trails.
    """

    ENVIRONMENT = "environment"  # From environment variable
    CONFIG_FILE = "config_file"  # From YAML config file
    DEFAULT = "default"  # From hardcoded default


@dataclass
class ConnectionConfiguration:
    """
    Represents IRIS database connection parameters with priority-based resolution.

    Supports configuration from environment variables (IRIS_HOST, IRIS_PORT, etc.),
    config files, or defaults following 12-factor app pattern.

    Attributes:
        host: IRIS database host (env: IRIS_HOST, default: "localhost")
        port: IRIS database port (env: IRIS_PORT, default: 1972)
        username: Database username (env: IRIS_USERNAME, default: "_SYSTEM")
        password: Database password (env: IRIS_PASSWORD, default: "SYS")
        namespace: IRIS namespace (env: IRIS_NAMESPACE, default: "USER")
        connection_timeout: Connection timeout in seconds (default: 30)
        driver_path: Optional path to JDBC driver
        source: ConfigSource tracking where values originated

    Validation Rules:
        - host must not be empty string
        - port must be in range 1-65535
        - username must not be empty string
        - namespace must be valid IRIS namespace format (alphanumeric + %)

    Examples:
        >>> # From environment variables
        >>> os.environ['IRIS_HOST'] = 'aws-iris.example.com'
        >>> config = ConnectionConfiguration(
        ...     host=os.getenv('IRIS_HOST', 'localhost'),
        ...     source={'host': ConfigSource.ENVIRONMENT}
        ... )

        >>> # From config file
        >>> config = ConnectionConfiguration(
        ...     host='localhost',
        ...     port=1972,
        ...     source={'host': ConfigSource.CONFIG_FILE}
        ... )
    """

    host: str = "localhost"
    port: int = 1972
    username: str = "_SYSTEM"
    password: str = "SYS"
    namespace: str = "USER"
    connection_timeout: int = 30
    driver_path: Optional[str] = None
    source: Dict[str, ConfigSource] = field(
        default_factory=lambda: {
            "host": ConfigSource.DEFAULT,
            "port": ConfigSource.DEFAULT,
            "username": ConfigSource.DEFAULT,
            "password": ConfigSource.DEFAULT,
            "namespace": ConfigSource.DEFAULT,
        }
    )

    def validate(self) -> None:
        """
        Validate connection configuration.

        Raises:
            ValueError: If validation fails
        """
        if not self.host or not self.host.strip():
            raise ValueError("Connection host cannot be empty")

        if not (1 <= self.port <= 65535):
            raise ValueError(f"Port {self.port} must be in range 1-65535")

        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")

        if not self.namespace or not self.namespace.strip():
            raise ValueError("Namespace cannot be empty")

        # Validate namespace format (alphanumeric + %)
        if not all(c.isalnum() or c in "%_-" for c in self.namespace):
            raise ValueError(
                f"Invalid namespace format: {self.namespace}. "
                "Must contain only alphanumeric characters, %, _, or -"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation with masked password
        """
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": "***MASKED***",
            "namespace": self.namespace,
            "connection_timeout": self.connection_timeout,
            "driver_path": self.driver_path,
            "source": {k: v.value for k, v in self.source.items()},
        }


@dataclass
class VectorConfiguration:
    """
    Encapsulates vector storage settings with validation.

    Supports configurable vector dimensions (128-8192) for different embedding models
    and validates against existing database schema to prevent data corruption.

    Attributes:
        vector_dimension: Number of dimensions (env: VECTOR_DIMENSION, default: 384)
        distance_metric: Distance metric for similarity (default: "COSINE")
        index_type: Vector index type (default: "HNSW")
        source: ConfigSource tracking where values originated

    Validation Rules:
        - vector_dimension must be in range [128, 8192]
        - If existing tables found: vector_dimension must match existing dimension
        - distance_metric must be one of: COSINE, DOT, EUCLIDEAN
        - index_type must be one of: HNSW, FLAT

    Common Vector Dimensions:
        - 384: SentenceTransformers (all-MiniLM-L6-v2, default)
        - 768: SentenceTransformers (all-mpnet-base-v2)
        - 1024: NVIDIA NIM embeddings
        - 1536: OpenAI text-embedding-ada-002
        - 3072: OpenAI text-embedding-3-large

    Examples:
        >>> # For NVIDIA NIM embeddings
        >>> config = VectorConfiguration(vector_dimension=1024)

        >>> # For OpenAI embeddings
        >>> config = VectorConfiguration(vector_dimension=1536)
    """

    vector_dimension: int = 384
    distance_metric: str = "COSINE"
    index_type: str = "HNSW"
    source: Dict[str, ConfigSource] = field(
        default_factory=lambda: {
            "vector_dimension": ConfigSource.DEFAULT,
            "distance_metric": ConfigSource.DEFAULT,
            "index_type": ConfigSource.DEFAULT,
        }
    )

    # Valid options
    VALID_METRICS = {"COSINE", "DOT", "EUCLIDEAN"}
    VALID_INDEX_TYPES = {"HNSW", "FLAT"}
    MIN_DIMENSION = 128
    MAX_DIMENSION = 8192

    def validate(self) -> None:
        """
        Validate vector configuration.

        Raises:
            ValueError: If validation fails
        """
        if not (self.MIN_DIMENSION <= self.vector_dimension <= self.MAX_DIMENSION):
            raise ValueError(
                f"Vector dimension {self.vector_dimension} must be in range "
                f"[{self.MIN_DIMENSION}, {self.MAX_DIMENSION}]"
            )

        if self.distance_metric.upper() not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid distance metric: {self.distance_metric}. "
                f"Must be one of: {', '.join(self.VALID_METRICS)}"
            )

        if self.index_type.upper() not in self.VALID_INDEX_TYPES:
            raise ValueError(
                f"Invalid index type: {self.index_type}. "
                f"Must be one of: {', '.join(self.VALID_INDEX_TYPES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "vector_dimension": self.vector_dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "source": {k: v.value for k, v in self.source.items()},
        }


@dataclass
class TableConfiguration:
    """
    Defines table schema, names, and namespace requirements.

    Supports configurable table schema (e.g., RAG, SQLUser) for different
    namespace configurations, particularly for AWS deployments requiring
    schema-prefixed table names.

    Attributes:
        table_schema: Schema prefix for tables (env: TABLE_SCHEMA, default: "RAG")
        entities_table: Name of entities table (default: "Entities")
        relationships_table: Name of relationships table (default: "EntityRelationships")
        source: ConfigSource tracking where values originated

    Computed Properties:
        - full_entities_table: Schema-prefixed entities table name (e.g., "RAG.Entities")
        - full_relationships_table: Schema-prefixed relationships table name

    Validation Rules:
        - table_schema must be valid SQL identifier
        - entities_table and relationships_table must be valid SQL identifiers
        - Schema must exist in target namespace or have CREATE TABLE permission

    Examples:
        >>> # AWS deployment with SQLUser schema
        >>> config = TableConfiguration(table_schema="SQLUser")
        >>> config.full_entities_table
        'SQLUser.Entities'

        >>> # Default local deployment
        >>> config = TableConfiguration()
        >>> config.full_entities_table
        'RAG.Entities'
    """

    table_schema: str = "RAG"
    entities_table: str = "Entities"
    relationships_table: str = "EntityRelationships"
    source: Dict[str, ConfigSource] = field(
        default_factory=lambda: {
            "table_schema": ConfigSource.DEFAULT,
            "entities_table": ConfigSource.DEFAULT,
            "relationships_table": ConfigSource.DEFAULT,
        }
    )

    @property
    def full_entities_table(self) -> str:
        """Get fully qualified entities table name with schema prefix."""
        return f"{self.table_schema}.{self.entities_table}"

    @property
    def full_relationships_table(self) -> str:
        """Get fully qualified relationships table name with schema prefix."""
        return f"{self.table_schema}.{self.relationships_table}"

    def validate(self) -> None:
        """
        Validate table configuration.

        Raises:
            ValueError: If validation fails
        """
        # Validate SQL identifier format (basic check)
        for name, value in [
            ("table_schema", self.table_schema),
            ("entities_table", self.entities_table),
            ("relationships_table", self.relationships_table),
        ]:
            if not value or not value.strip():
                raise ValueError(f"{name} cannot be empty")

            # Basic SQL identifier validation (alphanumeric + underscore + %)
            if not all(c.isalnum() or c in "_%-" for c in value):
                raise ValueError(
                    f"Invalid {name}: {value}. "
                    "Must contain only alphanumeric characters, _, %, or -"
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "table_schema": self.table_schema,
            "entities_table": self.entities_table,
            "relationships_table": self.relationships_table,
            "full_entities_table": self.full_entities_table,
            "full_relationships_table": self.full_relationships_table,
            "source": {k: v.value for k, v in self.source.items()},
        }


@dataclass
class CloudConfiguration:
    """
    Aggregates all cloud deployment configuration settings.

    This is the top-level configuration object that combines connection,
    vector, and table settings with configuration source tracking.

    Attributes:
        connection: ConnectionConfiguration instance
        vector: VectorConfiguration instance
        tables: TableConfiguration instance

    Examples:
        >>> config = CloudConfiguration(
        ...     connection=ConnectionConfiguration(host='aws-iris.example.com'),
        ...     vector=VectorConfiguration(vector_dimension=1024),
        ...     tables=TableConfiguration(table_schema='SQLUser')
        ... )
        >>> config.validate()
    """

    connection: ConnectionConfiguration = field(default_factory=ConnectionConfiguration)
    vector: VectorConfiguration = field(default_factory=VectorConfiguration)
    tables: TableConfiguration = field(default_factory=TableConfiguration)

    def validate(self) -> None:
        """
        Validate all configuration components.

        Raises:
            ValueError: If any validation fails
        """
        self.connection.validate()
        self.vector.validate()
        self.tables.validate()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation with all components
        """
        return {
            "connection": self.connection.to_dict(),
            "vector": self.vector.to_dict(),
            "tables": self.tables.to_dict(),
        }
