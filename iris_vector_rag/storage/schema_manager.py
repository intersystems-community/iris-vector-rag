#!/usr/bin/env python3
"""
Schema management system for IRIS RAG tables with automatic migration support.

This module provides robust schema versioning, configuration tracking, and
automatic migration capabilities for vector dimensions and other schema changes.
"""

import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, cast

from iris_vector_rag.exceptions import DestructiveMigrationError, SchemaIndexError
from iris_vector_rag.integrations.ivg import (
    IVG_REQUIRED_TABLES,
    IVG_SCHEMA,
    IVG_UPGRADE_HINT,
    InitializationResult,
    ValidationResult,
    assert_ivg_compatible,
    create_graph_engine,
    get_ivg_version,
    pre_4_layout_tables,
)
from iris_vector_rag.integrations.ivg_bm25 import (
    Bm25UpgradeReport,
    upgrade_bm25_indexes,
)
from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.collation import EXACT_CASE_COLUMNS, ensure_exact_collation
from iris_vector_rag.storage.table_shape import (
    CANONICAL_CHUNK_SHAPE,
    CANONICAL_DOCUMENT_SHAPE,
    CatalogUnavailable,
    ColumnRole,
    TableShape,
    TableShapeError,
    VectorSpec,
    create_table_statement,
)
from iris_vector_rag.storage.text_search import (
    TEXT_INDEX_NAME,
    create_text_index_statement,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IvgUpgrade:
    """`SchemaManager.upgrade_iris_vector_graph`: the layout step and the BM25 step."""

    #: ivg's `UpgradeReport` for the 4.0 storage layout.
    storage: Any
    #: BM25 indexes brought to schema 2 (ivg 4.2.0).
    bm25: Bm25UpgradeReport

    @property
    def ok(self) -> bool:
        return self.bm25.ok


def _canonical_column_names(shape: TableShape) -> list:
    """The columns a canonical table has, in the order it declares them.

    Recorded in the schema metadata so a later `needs_migration` check compares the
    table against the shape the library creates, not against a hand-kept list that
    named columns no creation path ever emitted (feature 082, FR-006).
    """
    return [
        column.name
        for column in sorted(shape.columns.values(), key=lambda col: col.position)
    ]


COLUMN_OVERRIDE_KEYS = (
    "id_column",
    "text_column",
    "title_column",
    "metadata_column",
    "embedding_column",
)


def _drop_column_overrides(config: Dict[str, Any], table_name: str) -> Dict[str, Any]:
    """Remove column-name overrides from a stored configuration, with a warning.

    These keys used to tell the store which column filled which role. The table now
    answers that question itself -- `TableShape` reads it from the catalog -- so a
    configuration that still carries them is a second account of a fact that has one
    source (feature 082, FR-001). Dropping them silently would leave an operator
    believing a rename took effect, so say which keys were ignored.
    """
    stale = [key for key in COLUMN_OVERRIDE_KEYS if key in config]
    if not stale:
        return config
    warnings.warn(
        f"Column overrides in the schema configuration for {table_name} are ignored: "
        f"{', '.join(stale)}. The table's own columns decide which column fills which "
        "role; remove these keys from the SchemaMetadata configuration.",
        DeprecationWarning,
        stacklevel=3,
    )
    return {key: value for key, value in config.items() if key not in stale}


def validate_schema_prefix(prefix: str) -> None:
    """
    Validate schema prefix against [A-Za-z][A-Za-z0-9_]* pattern.

    Raises:
        ValueError: If prefix is empty or contains invalid characters.
    """
    if not prefix:
        raise ValueError("Schema prefix cannot be empty")
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", prefix):
        raise ValueError(
            f"Invalid schema prefix '{prefix}': must match [A-Za-z][A-Za-z0-9_]*"
        )
    logger.debug(f"Schema prefix '{prefix}' validated successfully")


def resolve_schema_prefix(ctor_arg: Optional[str], config_manager) -> str:
    """
    Resolve schema prefix using priority: explicit argument > config_manager > env > 'RAG'.

    One rule for every class that names tables, so the schema manager and the stores
    beside it cannot disagree about where the tables are.

    Returns:
        Validated schema prefix string.
    """
    if ctor_arg is not None:
        validate_schema_prefix(ctor_arg)
        return ctor_arg

    # Config manager reads IRIS_SCHEMA_PREFIX env var and config file
    if config_manager is not None:
        try:
            prefix = config_manager.get_schema_prefix()
            if prefix and isinstance(prefix, str) and prefix != "RAG":
                validate_schema_prefix(prefix)
                return prefix
            elif prefix and isinstance(prefix, str) and prefix == "RAG":
                return "RAG"
        except AttributeError:
            pass  # Config manager doesn't have get_schema_prefix yet

    # Check env var directly as fallback
    env_prefix = os.environ.get("IRIS_SCHEMA_PREFIX")
    if env_prefix:
        validate_schema_prefix(env_prefix)
        return env_prefix

    return "RAG"


class SchemaManager:
    """
    Universal vector dimension authority and schema manager for IRIS RAG.

    Features:
    - Central authority for ALL vector dimensions across all tables
    - Tracks vector dimensions and other configuration parameters
    - Automatically detects configuration changes
    - Performs safe schema migrations
    - Maintains schema version history
    - Provides simple dimension API for all components
    """

    # CLASS-LEVEL CACHING: only _config_loaded is shared (optimization for config loading)
    # _schema_validation_cache and _tables_validated are now instance-level (see __init__)
    _config_loaded = False  # Flag to prevent redundant config loading

    def __init__(
        self,
        connection_manager=None,
        config_manager=None,
        connection_string: Optional[str] = None,
        base_embedding_dimension: Optional[int] = None,
        schema_prefix: Optional[str] = None,
    ):
        if config_manager is None:
            from iris_vector_rag.config.manager import ConfigurationManager

            config_manager = ConfigurationManager()

        if connection_manager is None:
            from iris_vector_rag.core.connection import ConnectionManager

            connection_manager = ConnectionManager(config_manager)

        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.schema_version = "1.0.0"
        self.pipeline_type: Optional[str] = None

        # Cache for dimension lookups
        self._dimension_cache = {}

        # Instance-level caches (not class-level, so different prefixes don't collide)
        self._schema_validation_cache: dict = {}
        self._tables_validated: set = set()
        # Qualified names whose text index this instance has already ensured.
        self._text_indexed: set = set()

        # Configurable schema prefix (FR-001 through FR-008)
        self.schema_prefix = self._resolve_schema_prefix(schema_prefix, config_manager)

        # Lazy initialization flag (Feature 078: Pure Constructors)
        self._initialized = False

        # The embedding dimension is the active backend's real output width (G6 point 5),
        # resolved on first use so that constructing a SchemaManager never loads a model.
        # An explicit argument is the caller stating the width and is applied whichever
        # branch below runs (ST-08).
        self._base_embedding_dimension: Optional[int] = None
        self._table_configs_data: Optional[Dict[str, Any]] = None
        self._model_dimensions_data: Optional[Dict[str, int]] = None

        # Load and validate configuration on initialization (only if not already loaded)
        if not SchemaManager._config_loaded:
            self._load_and_validate_config()
            SchemaManager._config_loaded = True
        else:
            # Config already loaded by another instance - set instance attributes from config
            self.base_embedding_model = self.config_manager.get(
                "embedding_model.name", "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.debug(
                "Schema Manager: Using cached configuration from previous instance"
            )

        if base_embedding_dimension is not None:
            self.base_embedding_dimension = base_embedding_dimension

        # REMOVED: ensure_schema_metadata_table() call
        # Feature 078: Pure Constructors - schema initialization moved to lazy init on first use

    def _validate_schema_prefix(self, prefix: str) -> None:
        """
        Validate schema prefix against [A-Za-z][A-Za-z0-9_]* pattern.

        Raises:
            ValueError: If prefix is empty or contains invalid characters.
        """
        validate_schema_prefix(prefix)

    def _resolve_schema_prefix(self, ctor_arg: Optional[str], config_manager) -> str:
        """
        Resolve schema prefix using priority: constructor arg > config_manager > 'RAG'.

        Returns:
            Validated schema prefix string.
        """
        return resolve_schema_prefix(ctor_arg, config_manager)

    def _qn(self, table_name: str) -> str:
        """
        Qualify a table name with the schema prefix.

        Args:
            table_name: Unqualified table name (e.g., "SourceDocuments")

        Returns:
            Fully qualified name (e.g., "MYAPP.SourceDocuments")
        """
        return f"{self.schema_prefix}.{table_name}"

    def _load_and_validate_config(self):
        """Load configuration from config manager and validate it makes sense."""
        logger.info("Schema Manager: Loading and validating configuration...")

        # Load base embedding model configuration
        self.base_embedding_model = self.config_manager.get(
            "embedding_model.name", "sentence-transformers/all-MiniLM-L6-v2"
        )

    @property
    def base_embedding_dimension(self) -> int:
        """The one embedding dimension every table, migration and ivg engine uses.

        An explicit constructor argument, else the embedding manager's dimension: the
        active backend's real output width, with `embeddings.dimension` an override
        checked against it. There is no default; it raises `EmbeddingError` when nothing
        can say.
        """
        if getattr(self, "_base_embedding_dimension", None) is None:
            from ..embeddings.manager import EmbeddingManager

            dimension = EmbeddingManager(self.config_manager).get_embedding_dimension()
            self._set_base_embedding_dimension(dimension)
        return cast(int, self._base_embedding_dimension)

    @base_embedding_dimension.setter
    def base_embedding_dimension(self, dimension: int) -> None:
        self._set_base_embedding_dimension(dimension)

    def _set_base_embedding_dimension(self, dimension: int) -> None:
        self._base_embedding_dimension = dimension
        self._validate_configuration()
        self._table_configs_data = None
        self._model_dimensions_data = None
        getattr(self, "_dimension_cache", {}).clear()

    @property
    def _table_configs(self) -> Dict[str, Any]:
        if getattr(self, "_table_configs_data", None) is None:
            self._build_table_configurations()
        return cast(Dict[str, Any], self._table_configs_data)

    @property
    def _model_dimensions(self) -> Dict[str, int]:
        if getattr(self, "_model_dimensions_data", None) is None:
            self._build_model_dimension_mapping()
        return cast(Dict[str, int], self._model_dimensions_data)

    def _detect_iris_vector_graph(self) -> bool:
        """Detect iris-vector-graph package availability without importing it."""
        try:
            import importlib.util

            return importlib.util.find_spec("iris_vector_graph") is not None
        except Exception:
            return False

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists, defaulting to this manager's schema.

        The name is matched without regard to case: callers spell it the way their
        configuration does, IRIS stores it the way the DDL declared it, and feature 074
        established that the difference must not read as "no such table".
        """
        if "." in table_name:
            schema, table = table_name.split(".", 1)
        else:
            schema, table = self.schema_prefix, table_name

        return self._snapshot(table, schema=schema) is not None

    def _snapshot(self, table_name: str, *, schema: Optional[str] = None):
        """The catalog's own account of a table, or None if it does not exist.

        Read fresh every time rather than through a resolver's cache: this class is
        the one that creates and drops tables, so a cached answer here would describe
        the schema as it stood before its own last statement. All catalog SQL lives in
        `IRISCatalogReader`, which is what makes it testable against a real instance
        and replayable without one (feature 082, FR-013).
        """
        connection = self.connection_manager.get_connection()
        reader = IRISCatalogReader(connection)
        return reader.snapshot(schema or self.schema_prefix, table_name)

    def _catalog_columns(self, table_name: str) -> set:
        """Lower-cased column names of an existing table; empty if it has none."""
        snapshot = self._snapshot(table_name)
        if snapshot is None:
            return set()
        return {str(row["COLUMN_NAME"]).lower() for row in snapshot.columns}

    def validate_graph_prerequisites(self, pipeline_type: str = "graphrag"):
        """Validate iris-vector-graph package and required tables."""
        is_graphrag = (pipeline_type or "").lower() in {
            "graphrag",
            "hybrid_graphrag",
            "hybridgraphrag",
        }

        if not self._detect_iris_vector_graph():
            message = (
                "iris-vector-graph package is required for GraphRAG pipelines. "
                "Install with `pip install iris-vector-graph`."
            )
            if is_graphrag:
                raise ImportError(message)
            return ValidationResult(
                is_valid=False,
                package_installed=False,
                missing_tables=[],
                error_message=message,
            )

        try:
            import iris_vector_graph as ivg_module

            installed_version = assert_ivg_compatible(ivg_module)
        except ImportError as exc:
            if is_graphrag:
                raise
            return ValidationResult(
                is_valid=False,
                package_installed=True,
                missing_tables=[],
                error_message=str(exc),
                installed_version=(
                    get_ivg_version(ivg_module) if "ivg_module" in locals() else None
                ),
            )

        required_tables = list(IVG_REQUIRED_TABLES)
        missing_tables = [
            table for table in required_tables if not self._ivg_table_exists(table)
        ]

        if missing_tables:
            error_parts = []
            error_parts.append(
                f"Missing IVG tables in {IVG_SCHEMA}: {', '.join(missing_tables)}"
            )
            error_parts.append("Run ensure_iris_vector_graph_tables() to remediate.")
            error_message = " ".join(error_parts)
            logger.warning(
                "PPR unavailable: graph prerequisites not satisfied. %s",
                error_message,
            )
            return ValidationResult(
                is_valid=False,
                package_installed=True,
                missing_tables=missing_tables,
                error_message=error_message,
                installed_version=installed_version,
            )

        legacy_tables = self._ivg_legacy_layout_tables()
        if legacy_tables:
            error_message = (
                f"Pre-4.0 layout in {IVG_SCHEMA}: {', '.join(legacy_tables)}. "
                f"{IVG_UPGRADE_HINT}"
            )
            logger.warning(
                "PPR unavailable: graph prerequisites not satisfied. %s",
                error_message,
            )
            return ValidationResult(
                is_valid=False,
                package_installed=True,
                missing_tables=[],
                error_message=error_message,
                installed_version=installed_version,
                legacy_layout_tables=legacy_tables,
            )

        logger.info("PPR available: graph prerequisites satisfied")

        return ValidationResult(
            is_valid=True,
            package_installed=True,
            missing_tables=[],
            error_message="",
            installed_version=installed_version,
        )

    def ensure_iris_vector_graph_tables(self, pipeline_type: str = "graphrag"):
        """Ensure required iris-vector-graph tables exist and return initialization result."""
        start_time = time.time()

        is_graphrag = (pipeline_type or "").lower() in {
            "graphrag",
            "hybrid_graphrag",
            "hybridgraphrag",
        }

        if not self._detect_iris_vector_graph():
            message = (
                "iris-vector-graph package is required for GraphRAG pipelines. "
                "Install with `pip install iris-vector-graph`."
            )
            if is_graphrag:
                raise ImportError(message)
            return InitializationResult(
                package_detected=False,
                tables_attempted=[],
                tables_created={},
                total_time_seconds=time.time() - start_time,
                error_messages={},
            )

        tables_attempted = list(IVG_REQUIRED_TABLES)
        tables_created: Dict[str, bool] = {}
        error_messages: Dict[str, str] = {}
        ivg_status: Dict[str, Any] = {}
        installed_version: Optional[str] = None
        legacy_tables: List[str] = []

        try:
            import iris_vector_graph as ivg_module

            installed_version = assert_ivg_compatible(ivg_module)
            ivg_status = self._initialize_ivg_schema()
        except ImportError:
            if is_graphrag:
                raise
            message = "iris-vector-graph is not compatible with this IVR version"
            error_messages[IVG_SCHEMA] = message
            tables_created = {table: False for table in tables_attempted}
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.error("IVG schema initialization failed: %s", message)
            error_messages[IVG_SCHEMA] = message
            tables_created = {table: False for table in tables_attempted}
        else:
            for table in tables_attempted:
                try:
                    exists = self._ivg_table_exists(table)
                    tables_created[table] = exists
                    if not exists:
                        error_messages[table] = (
                            f"{IVG_SCHEMA}.{table} was not present after IVG initialization"
                        )
                except Exception as exc:
                    tables_created[table] = False
                    error_messages[table] = f"{type(exc).__name__}: {exc}"

            # initialize_schema adds 4.x DDL but does not move rows; an install that
            # predates 4.0 stays in the old layout until the operator migrates it.
            try:
                legacy_tables = self._ivg_legacy_layout_tables()
            except Exception as exc:
                error_messages[IVG_SCHEMA] = (
                    f"Could not read the {IVG_SCHEMA} layout: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if legacy_tables:
                    error_messages[IVG_SCHEMA] = (
                        f"Pre-4.0 layout in {IVG_SCHEMA}: "
                        f"{', '.join(legacy_tables)}. {IVG_UPGRADE_HINT}"
                    )

        if error_messages:
            logger.error(
                "IVG table initialization failed for %s",
                ", ".join(sorted(error_messages.keys())),
            )
        else:
            logger.info("IVG table initialization completed successfully")

        total_time_seconds = time.time() - start_time
        return InitializationResult(
            package_detected=True,
            tables_attempted=tables_attempted,
            tables_created=tables_created,
            total_time_seconds=total_time_seconds,
            error_messages=error_messages,
            installed_version=installed_version,
            ivg_status=ivg_status or {},
            legacy_layout_tables=legacy_tables,
        )

    def _ivg_table_exists(self, table_name: str) -> bool:
        """Check for an IVG-owned table in Graph_KG."""
        return self.table_exists(f"{IVG_SCHEMA}.{table_name}")

    def _ivg_columns(self, table_name: str) -> set:
        """Lower-cased column names of a Graph_KG table; empty if it does not exist."""
        snapshot = self._snapshot(table_name, schema=IVG_SCHEMA)
        if snapshot is None:
            return set()
        return {str(row["COLUMN_NAME"]).lower() for row in snapshot.columns}

    def _ivg_legacy_layout_tables(self) -> List[str]:
        """Required Graph_KG tables still in the layout ivg used before 4.0."""
        return pre_4_layout_tables(
            {table: self._ivg_columns(table) for table in IVG_REQUIRED_TABLES}
        )

    def upgrade_iris_vector_graph_storage(self, dry_run: bool = False, **options):
        """Migrate a pre-4.0 Graph_KG install to the iris-vector-graph 4.x layout.

        Never called implicitly: the migration moves rows and quarantines the ones it
        cannot place, which is an operator's decision. Runs ivg's own sequence --
        `initialize_schema()` for the 4.x DDL, then
        `iris_vector_graph.migrations.upgrade_to_4_0_0`. Re-runnable; a step whose work
        is done reports itself skipped.

        A dry run writes nothing, DDL included, so it skips `initialize_schema()`.
        A step ivg cannot predict on a genuine pre-4.0 install (its input column is
        one a real run adds) comes back with `blocked_because` saying which call
        adds it; `UpgradeReport.blocked` names those steps.

        `options` go to `upgrade_to_4_0_0` unchanged (`steps`, `vector_resolver`,
        `document_resolver`, `edge_resolver`, `edge_dimension`, `dtype`). Returns its
        `UpgradeReport`.
        """
        from iris_vector_graph.migrations import upgrade_to_4_0_0

        connection = self.connection_manager.get_connection()
        if not dry_run:
            self._graph_engine(connection).initialize_schema()
        return upgrade_to_4_0_0(connection, dry_run=dry_run, **options)

    def upgrade_bm25_indexes(
        self,
        names: Optional[Iterable[str]] = None,
        dry_run: bool = False,
        continue_on_error: bool = False,
    ) -> Bm25UpgradeReport:
        """Bring BM25 indexes to ivg 4.2.0's schema 2.

        ivg 4.2.0's `Insert` and `Delete` refuse an index built earlier until
        `engine.bm25_upgrade(name)` has run once (`Search` is unaffected). This runs it
        for every full-text index in the namespace, or for `names`, skipping the ones
        already converted, so it can be repeated. A dry run reads each index and writes
        nothing. One pass over each index's posting lists and writers are refused while
        it runs: run it as a job. Scores change afterwards (docs/IVG_4X_UPGRADE.md).

        An index that cannot be upgraded raises `Bm25UpgradeError` (its `report` holds
        what was done before it); `continue_on_error=True` records it and goes on.

        Needs the 4.2.0 `Graph.KG.BM25Index` class on the server, which
        `initialize_schema()` (and so `upgrade_iris_vector_graph`) deploys.
        """
        engine = self._graph_engine(self.connection_manager.get_connection())
        return upgrade_bm25_indexes(
            engine, names=names, dry_run=dry_run, continue_on_error=continue_on_error
        )

    def upgrade_iris_vector_graph(
        self,
        dry_run: bool = False,
        bm25_indexes: Optional[Iterable[str]] = None,
        continue_on_error: bool = False,
        **options,
    ) -> IvgUpgrade:
        """Everything an iris-vector-graph upgrade needs, in ivg's order.

        1. `upgrade_iris_vector_graph_storage` -- `initialize_schema()` (which deploys
           the `Graph.KG` classes of the installed ivg) and the 4.0 layout migration;
        2. `upgrade_bm25_indexes` -- BM25 schema 2, which needs the classes step 1 put
           on the server.

        `dry_run` previews both and writes neither. `bm25_indexes` limits step 2 to
        those names; `continue_on_error` is step 2's (see `upgrade_bm25_indexes`);
        `options` go to step 1. Re-runnable.
        """
        storage = self.upgrade_iris_vector_graph_storage(dry_run=dry_run, **options)
        bm25 = self.upgrade_bm25_indexes(
            names=bm25_indexes, dry_run=dry_run, continue_on_error=continue_on_error
        )
        return IvgUpgrade(storage=storage, bm25=bm25)

    def _graph_engine(self, connection):
        """An ivg engine on `connection`, logging in as the session's account.

        ivg 4.2.0 opens the engine's class-method connection (the class deploy
        included) with the credentials it is given; it no longer uses `_SYSTEM`.
        A connection manager that has no session identity gets an engine without
        credentials, which then shares `connection`.
        """
        from ..session import SessionIdentity

        options: Dict[str, Any] = {"embedding_dimension": self.base_embedding_dimension}
        identity = getattr(
            getattr(self.connection_manager, "session", None), "identity", None
        )
        if isinstance(identity, SessionIdentity):
            options["credentials"] = {
                "host": identity.host,
                "port": identity.port,
                "namespace": identity.namespace,
                "username": identity.username,
                "password": identity.password,
            }
        return create_graph_engine(connection, **options)

    def _initialize_ivg_schema(self) -> Dict[str, Any]:
        """Delegate Graph_KG schema creation to iris-vector-graph."""
        connection = self.connection_manager.get_connection()
        status = self._graph_engine(connection).initialize_schema()
        return status if isinstance(status, dict) else {"status": status}

    def _validate_configuration(self):
        """Validate that configuration values make sense."""
        errors = []

        # Validate base embedding dimension
        if (
            not isinstance(self.base_embedding_dimension, int)
            or self.base_embedding_dimension <= 0
        ):
            errors.append(
                f"Invalid base embedding dimension: {self.base_embedding_dimension}"
            )

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ Configuration validation passed")

    def _build_model_dimension_mapping(self):
        """Build unified model-to-dimension mapping from configuration."""
        # Start with known model dimensions
        from ..embeddings.manager import KNOWN_MODEL_DIMENSIONS

        dimensions = dict(KNOWN_MODEL_DIMENSIONS)

        # Add configured models
        dimensions[self.base_embedding_model] = self.base_embedding_dimension
        self._model_dimensions_data = dimensions

        logger.debug(f"Model-dimension mapping: {len(dimensions)} models configured")

    def _build_table_configurations(self):
        """Build table-specific configurations from config."""
        # Table-specific configurations based on config
        self._table_configs_data = {
            "SourceDocuments": {
                "embedding_column": "embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
            },
            "DocumentChunks": {
                "embedding_column": "chunk_embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
            },
            "Entities": {
                "embedding_column": "embedding",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
                "supports_vector_search": True,
                "supports_graph_traversal": True,
            },
            "EntityRelationships": {
                "embedding_column": "relation_embedding",
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
                "supports_vector_search": True,
                "supports_graph_traversal": True,
            },
            # IRIS Graph Core Tables for Hybrid Search
            "rdf_labels": {
                "embedding_column": None,
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,
                "dimension": 0,  # No embeddings
                "supports_vector_search": False,
                "supports_graph_traversal": True,
                "table_type": "graph_metadata",
                "created_by": "iris_vector_graph",
            },
            "rdf_props": {
                "embedding_column": None,
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,
                "dimension": 0,  # No embeddings
                "supports_vector_search": False,
                "supports_graph_traversal": True,
                "table_type": "graph_properties",
                "created_by": "iris_vector_graph",
            },
            "rdf_edges": {
                "embedding_column": None,
                "uses_document_embeddings": False,
                "default_model": self.base_embedding_model,
                "dimension": 0,  # No embeddings
                "supports_vector_search": False,
                "supports_graph_traversal": True,
                "table_type": "graph_relationships",
                "created_by": "iris_vector_graph",
            },
            "kg_NodeEmbeddings_optimized": {
                "embedding_column": "emb",
                "uses_document_embeddings": True,
                "default_model": self.base_embedding_model,
                "dimension": self.base_embedding_dimension,
                "supports_vector_search": True,
                "supports_graph_traversal": False,
                "table_type": "optimized_vectors",
                "created_by": "iris_vector_graph",
                "requires_hnsw_index": True,
            },
        }

        logger.debug(
            f"Table configurations: {len(self._table_configs_data)} tables configured"
        )

    def ensure_schema_metadata_table(self):
        """Create schema metadata table if it doesn't exist."""
        # Lazy init guard (Feature 078: Pure Constructors)
        if self._initialized:
            logger.debug("Schema metadata already initialized, skipping")
            return

        connection = self.connection_manager.get_connection()
        if connection is None:
            raise ConnectionError(
                "Missing configuration: IRIS connection unavailable (check IRIS_* settings)"
            )
        cursor = connection.cursor()

        try:
            # Try different schema approaches in order of preference
            # One place only. Every reader and writer names the table through `_qn`, so a
            # copy in the connecting user's schema would never be read.
            schema_attempts = [(self.schema_prefix, self._qn("SchemaMetadata"))]

            for schema_name, table_name in schema_attempts:
                try:
                    create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        table_name VARCHAR(255) NOT NULL,
                        schema_version VARCHAR(50) NOT NULL,
                        vector_dimension INTEGER,
                        embedding_model VARCHAR(255),
                        configuration VARCHAR(MAX),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (table_name)
                    )
                    """
                    cursor.execute(create_sql)
                    connection.commit()
                    logger.info(
                        f"✅ Schema metadata table ensured in {schema_name} schema"
                    )
                    # Feature 078: Mark initialization as complete after successful DDL
                    self._initialized = True
                    break
                except Exception as schema_error:
                    logger.warning(
                        f"Failed to create schema metadata table in {schema_name} schema: {schema_error}"
                    )
                    if (schema_name, table_name) == schema_attempts[
                        -1
                    ]:  # Last schema attempt
                        # Instead of raising, log warning and continue without metadata table
                        logger.error(
                            f"Schema metadata table creation failed in {schema_name}. Continuing without metadata table."
                        )
                        logger.warning(
                            "This may affect schema versioning but basic functionality will work."
                        )
                        # Feature 078: Mark initialization as complete even on graceful failure
                        self._initialized = True
                        return  # Exit gracefully
                    continue

        except Exception as e:
            logger.error(f"Failed to create schema metadata table: {e}")
            logger.warning(
                "Continuing without schema metadata table. Basic functionality will work."
            )
            # Feature 078: Mark initialization as complete even on failure
            self._initialized = True
            # Don't raise - allow the system to continue without metadata table
        finally:
            cursor.close()

    def get_current_schema_config(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get current schema configuration for a table."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                f"""
                SELECT schema_version, vector_dimension, embedding_model, configuration
                FROM {self._qn("SchemaMetadata")}
                WHERE table_name = ?
            """,
                [table_name],
            )

            result = cursor.fetchone()
            if result:
                # Handle different result formats gracefully
                if len(result) == 4:
                    # Expected format: (schema_version, vector_dim, embedding_model, config_json)
                    schema_version, vector_dim, embedding_model, config_json = result
                    config = json.loads(config_json) if config_json else {}
                    config = _drop_column_overrides(config, table_name)

                    # This won't actually exist in config yet
                    vector_data_type = config.get(
                        "vector_data_type", "FLOAT"
                    )  # Default to FLOAT

                    return {
                        "schema_version": schema_version,
                        "vector_dimension": vector_dim,
                        "embedding_model": embedding_model,
                        "vector_data_type": vector_data_type,
                        "configuration": config,
                    }
                elif len(result) == 1:
                    # Legacy or corrupted format: only one value returned
                    logger.warning(
                        f"Schema metadata for {table_name} has unexpected format (1 value instead of 4). This may indicate corrupted metadata."
                    )
                    return None
                else:
                    # Other unexpected formats
                    logger.warning(
                        f"Schema metadata for {table_name} has unexpected format ({len(result)} values instead of 4). This may indicate corrupted metadata."
                    )
                    return None
            return None

        except Exception as e:
            logger.error(f"Failed to get schema config for {table_name}: {e}")
            return None
        finally:
            cursor.close()

    def _get_expected_schema_config(
        self, table_name: str, pipeline_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get expected schema configuration based on current system config and pipeline requirements."""
        # Get model and dimension from centralized methods
        model_name = self.get_embedding_model(table_name)
        expected_dim = self.get_vector_dimension(table_name, model_name)

        # Get vector data type from environment or configuration, default to FLOAT
        vector_data_type = os.environ.get(
            "IRIS_VECTOR_DATA_TYPE"
        ) or self.config_manager.get("storage:iris:vector_data_type", "FLOAT")

        # Base configuration
        config = {
            "schema_version": self.schema_version,
            "vector_dimension": expected_dim,
            "embedding_model": model_name,
            "vector_data_type": vector_data_type,
            "configuration": {
                "managed_by_schema_manager": True,
                "supports_vector_search": True,
                "auto_migration": True,
            },
        }

        # Enhanced: Get table requirements from pipeline if specified
        if pipeline_type:
            config.update(
                self._get_table_requirements_config(table_name, pipeline_type)
            )

        # Table-specific configurations
        if table_name == "SourceDocuments":
            config["configuration"].update(
                {
                    "table_type": "document_storage",
                    "created_by": "BasicRAG",
                    "expected_columns": _canonical_column_names(
                        CANONICAL_DOCUMENT_SHAPE
                    ),
                }
            )
        elif table_name == "DocumentChunks":
            config["configuration"].update(
                {
                    "table_type": "chunk_storage",
                    "created_by": "BasicRAG",
                    "expected_columns": _canonical_column_names(CANONICAL_CHUNK_SHAPE),
                }
            )
        elif table_name == "Entities":
            config["configuration"].update(
                {
                    "table_type": "entity_storage",
                    "created_by": "GraphRAG",
                    "expected_columns": [
                        "entity_id",
                        "entity_name",
                        "entity_type",
                        "source_doc_id",
                        "description",
                        "embedding",
                        "created_at",
                    ],
                    "foreign_keys": [
                        {
                            "column": "source_doc_id",
                            "references": f"{self._qn('SourceDocuments')}(id)",
                            "on_delete": "CASCADE",
                        }
                    ],
                    "indexes": [
                        "idx_entities_name_lower",
                        "idx_entities_type",
                        "idx_entities_source_doc",
                    ],
                }
            )
        elif table_name == "EntityRelationships":
            config["configuration"].update(
                {
                    "table_type": "relationship_storage",
                    "created_by": "GraphRAG",
                    "expected_columns": [
                        "relationship_id",
                        "source_entity_id",
                        "target_entity_id",
                        "relationship_type",
                        "metadata",
                        "created_at",
                    ],
                    "foreign_keys": [
                        {
                            "column": "source_entity_id",
                            "references": f"{self._qn('Entities')}(entity_id)",
                            "on_delete": "CASCADE",
                        },
                        {
                            "column": "target_entity_id",
                            "references": f"{self._qn('Entities')}(entity_id)",
                            "on_delete": "CASCADE",
                        },
                    ],
                    "indexes": ["idx_rel_source", "idx_rel_target"],
                }
            )
        # IRIS Graph Core Tables
        elif table_name == "rdf_labels":
            config["configuration"].update(
                {
                    "table_type": "graph_metadata",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "s",  # subject
                        "label",  # entity type/label
                    ],
                    "indexes": [
                        "idx_labels_label_s",
                        "idx_labels_s_label",
                    ],
                }
            )
        elif table_name == "rdf_props":
            config["configuration"].update(
                {
                    "table_type": "graph_properties",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "s",  # subject
                        "key",  # property key
                        "val",  # property value
                    ],
                    "indexes": [
                        "idx_props_s_key",
                        "idx_props_key_val",
                    ],
                }
            )
        elif table_name == "rdf_edges":
            config["configuration"].update(
                {
                    "table_type": "graph_relationships",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "edge_id",  # primary key
                        "s",  # subject
                        "p",  # predicate
                        "o_id",  # object
                        "qualifiers",  # JSON metadata with confidence
                    ],
                    "indexes": [
                        "idx_edges_s_p",
                        "idx_edges_p_oid",
                        "idx_edges_s",
                    ],
                }
            )
        elif table_name == "kg_NodeEmbeddings_optimized":
            config["configuration"].update(
                {
                    "table_type": "optimized_vectors",
                    "created_by": "iris_vector_graph",
                    "expected_columns": [
                        "id",  # entity identifier
                        "emb",  # vector embedding (optimized VECTOR type)
                    ],
                    "indexes": [
                        "HNSW_NodeEmb_Optimized",  # HNSW index for fast similarity
                    ],
                    "requires_hnsw_optimization": True,
                }
            )

        return config

    def _get_table_requirements_config(
        self, table_name: str, pipeline_type: Optional[str]
    ) -> Dict[str, Any]:
        """Extract table configuration from pipeline requirements."""
        try:
            from ..validation.requirements import get_pipeline_requirements

            requirements = get_pipeline_requirements(pipeline_type)

            # Find the table requirement for this table
            for table_req in requirements.required_tables:
                if table_req.name == table_name:
                    return {
                        "text_content_type": table_req.text_content_type,
                        "supports_vector_search": table_req.supports_vector_search,
                    }

            # Check optional tables too
            for table_req in requirements.optional_tables:
                if table_req.name == table_name:
                    return {
                        "text_content_type": table_req.text_content_type,
                        "supports_vector_search": table_req.supports_vector_search,
                    }

        except Exception as e:
            logger.warning(f"Could not get table requirements for {pipeline_type}: {e}")

        # Default configuration
        return {"text_content_type": "LONGVARCHAR", "supports_vector_search": True}

    def needs_migration(
        self, table_name: str, pipeline_type: Optional[str] = None
    ) -> bool:
        """Check if table needs migration based on configuration and physical structure.

        Uses class-level caching to avoid repeated expensive validation checks.
        """
        # Check class-level cache first (shared across all instances)
        cache_key = f"{table_name}:{pipeline_type or 'default'}"
        if cache_key in self._schema_validation_cache:
            cached_result = self._schema_validation_cache[cache_key]
            logger.debug(
                f"Schema validation cache HIT for {table_name} (cached: {cached_result})"
            )
            return cached_result

        # Cache MISS - perform full validation
        logger.debug(
            f"Schema validation cache MISS for {table_name} - performing full validation"
        )

        current_config = self.get_current_schema_config(table_name)
        expected_config = self._get_expected_schema_config(table_name, pipeline_type)

        if not current_config:
            logger.info(f"Table {table_name} has no schema metadata - migration needed")
            self._schema_validation_cache[cache_key] = True
            return True

        # Compare top-level schema config fields
        keys_to_check = [
            "schema_version",
            "vector_dimension",
            "embedding_model",
            "vector_data_type",
        ]
        for key in keys_to_check:
            if current_config.get(key) != expected_config.get(key):
                logger.info(
                    f"Migration required for {table_name} due to {key} mismatch: "
                    f"expected={expected_config.get(key)}, actual={current_config.get(key)}"
                )
                self._schema_validation_cache[cache_key] = True
                return True

        # Compare nested configuration keys
        nested_keys = [
            "managed_by_schema_manager",
            "supports_vector_search",
            "auto_migration",
        ]
        for key in nested_keys:
            cur_val = current_config.get("configuration", {}).get(key)
            exp_val = expected_config.get("configuration", {}).get(key)
            if cur_val != exp_val:
                logger.info(
                    f"Migration required for {table_name} due to config.{key} mismatch: "
                    f"expected={exp_val}, actual={cur_val}"
                )
                self._schema_validation_cache[cache_key] = True
                return True

        # ✅ NEW: Check that all expected columns exist in physical table
        expected_columns = expected_config.get("configuration", {}).get(
            "expected_columns", []
        )
        if expected_columns:
            try:
                physical_columns = self.verify_table_structure(table_name)
                physical_col_names = {col.lower() for col in physical_columns.keys()}

                for col in expected_columns:
                    if col.lower() not in physical_col_names:
                        logger.info(
                            f"Migration required for {table_name}: missing column '{col}' in physical schema."
                        )
                        self._schema_validation_cache[cache_key] = True
                        return True
            except Exception as e:
                logger.warning(
                    f"Could not verify physical structure of {table_name}: {e}"
                )
                self._schema_validation_cache[cache_key] = True
                return True  # Be conservative and migrate if in doubt

        # Schema is valid - cache the result
        logger.info(f"✅ Schema validation PASSED for {table_name} - caching result")
        self._schema_validation_cache[cache_key] = False
        return False

    def migrate_table(
        self,
        table_name: str,
        preserve_data: bool = False,
        pipeline_type: Optional[str] = None,
        allow_destructive: bool = False,
    ) -> bool:
        """
        Migrate table to match expected configuration.

        Every migration rebuilds its table with DROP and CREATE, and DDL autocommits, so
        a failed or refused step cannot be rolled back. A table that holds rows is
        therefore not touched unless the caller says the rows may go.

        Args:
            table_name: Name of table to migrate
            preserve_data: Not implemented, and does not unlock a populated table.
            pipeline_type: Pipeline whose requirements shape the expected columns.
            allow_destructive: Permit dropping a table that holds rows.

        Returns:
            True if migration successful, False otherwise

        Raises:
            DestructiveMigrationError: a table the migration would drop holds rows and
                `allow_destructive` is not True.
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # The chunk migrator leaves an existing table alone when asked to preserve data.
            leaves_alone = preserve_data and table_name.lower() == "documentchunks"
            if not allow_destructive and not leaves_alone:
                self._refuse_populated(cursor, table_name)
            return self._migrate_table_unguarded(
                connection, cursor, table_name, pipeline_type, preserve_data
            )
        finally:
            cursor.close()

    #: Tables each migration drops, by lower-cased request name.
    _MIGRATION_DROPS = {
        "sourcedocuments": ("SourceDocuments", "DocumentChunks"),
        "documentchunks": ("DocumentChunks",),
        "entities": ("Entities",),
        "entityrelationships": ("EntityRelationships",),
        "rdf_labels": ("rdf_labels",),
        "rdf_props": ("rdf_props",),
        "rdf_edges": ("rdf_edges",),
        "kg_nodeembeddings_optimized": ("kg_NodeEmbeddings_optimized",),
    }

    def _refuse_populated(self, cursor, table_name: str) -> None:
        """Raise `DestructiveMigrationError` if migrating would drop a table with rows.

        `SourceDocuments` also takes `DocumentChunks` with it (its foreign key). A table
        that cannot be read is not assumed empty: the failure is raised as it is.
        """
        for target in self._MIGRATION_DROPS.get(table_name.lower(), ()):
            if self._snapshot(target) is None:
                continue
            qualified = self._qn(target)
            cursor.execute(f"SELECT TOP 1 1 FROM {qualified}")
            if cursor.fetchone() is not None:
                raise DestructiveMigrationError(
                    f"migrate_table({table_name!r}) would drop {qualified}, which holds "
                    f"rows, and DDL cannot be rolled back. Pass allow_destructive=True "
                    f"to say the rows may go.",
                    table=qualified,
                )

    def _migrate_table_unguarded(
        self, connection, cursor, table_name, pipeline_type, preserve_data
    ) -> bool:
        """Dispatch to the table's migration; the caller has already cleared it."""
        try:
            expected_config = self._get_expected_schema_config(
                table_name, pipeline_type
            )

            if table_name == "SourceDocuments":
                success = self._migrate_source_documents_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    # The rebuilt table has no indexes; give it its text index too.
                    self._text_indexed.discard(
                        ".".join(self._split_table("SourceDocuments")).lower()
                    )
                    self._ensure_text_index_once("SourceDocuments")
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "DocumentChunks":
                success = self._migrate_document_chunks_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "Entities":
                success = self._migrate_entities_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name == "EntityRelationships":
                success = self._migrate_entity_relationships_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            # IRIS Graph Core table migrations (case-insensitive)
            elif table_name.lower() == "rdf_labels":
                success = self._migrate_rdf_labels_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name.lower() == "rdf_props":
                success = self._migrate_rdf_props_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name.lower() == "rdf_edges":
                success = self._migrate_rdf_edges_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False
            elif table_name.lower() == "kg_nodeembeddings_optimized":
                success = self._migrate_kg_node_embeddings_optimized_table(
                    cursor, expected_config, preserve_data
                )
                if success:
                    connection.commit()
                    return True
                else:
                    connection.rollback()
                    return False

            # Add other table migrations as needed
            logger.warning(f"No migration handler for table {table_name}")
            return False

        except SchemaIndexError:
            # The table was rebuilt but its text index could not be: say so.
            connection.rollback()
            raise
        except Exception as e:
            logger.error(f"Migration failed for {table_name}: {e}")
            connection.rollback()
            return False

    def _migrate_source_documents_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Recreate SourceDocuments in the canonical shape.

        Destructive by construction: it drops the table. Only an explicit
        `migrate_table` call reaches here -- nothing decides on its own that a table
        looks wrong enough to be rebuilt (feature 082, FR-007).
        """
        table_name = self._qn("SourceDocuments")
        try:
            create_sql = self._canonical_create_sql(
                CANONICAL_DOCUMENT_SHAPE,
                "SourceDocuments",
                expected_config["vector_dimension"],
            )
            logger.info(f"🔧 Recreating {table_name} in the canonical shape")

            # Handle foreign key constraints from referencing tables
            logger.info("Checking for foreign key constraints on SourceDocuments...")
            known_constraints = [
                ("DOCUMENTCHUNKSFKEY2", "DocumentChunks"),
            ]

            for constraint_name, referencing_table in known_constraints:
                # Check if referencing table exists first. The predicate this replaces
                # compared an upper-cased name against the catalog's stored spelling,
                # so it never matched and the constraint was never dropped.
                if self._snapshot(referencing_table) is None:
                    logger.info(
                        f"Referencing table {self._qn(referencing_table)} does not exist — skipping constraint drop"
                    )
                    continue

                try:
                    logger.info(
                        f"Dropping foreign key constraint {constraint_name} from {self._qn(referencing_table)}"
                    )
                    cursor.execute(
                        f"ALTER TABLE {self._qn(referencing_table)} DROP CONSTRAINT {constraint_name}"
                    )
                    logger.info(f"✓ Successfully dropped constraint {constraint_name}")
                except Exception as fk_error:
                    logger.warning(
                        f"Could not drop foreign key {constraint_name}: {fk_error}"
                    )
                    try:
                        logger.info(
                            f"Attempting to drop referencing table {self._qn(referencing_table)}"
                        )
                        cursor.execute(
                            f"DROP TABLE IF EXISTS {self._qn(referencing_table)}"
                        )
                        logger.info(
                            f"✓ Dropped referencing table {self._qn(referencing_table)}"
                        )
                    except Exception as table_error:
                        logger.warning(
                            f"Could not drop referencing table {referencing_table}: {table_error}"
                        )

            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"Dropped existing {table_name} table")

            cursor.execute(create_sql)
            logger.info(f"✅ Successfully created {table_name} table")

            # doc_id is UNIQUE via the table constraint, so created_at is the one index
            try:
                cursor.execute(
                    f"CREATE INDEX idx_source_docs_created ON {table_name} (created_at)"
                )
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

            try:
                self._update_schema_metadata(cursor, "SourceDocuments", expected_config)
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            return True

        except Exception as e:
            logger.error(f"Failed to migrate SourceDocuments table: {e}")
            return False

    def _canonical_create_sql(self, shape, table_name: str, dimension: Any) -> str:
        """The `CREATE TABLE` for a canonical table, rendered from its shape.

        Every creation path in this file goes through here, so there is one answer to
        what a document or chunk table looks like. Two hand-written literals for one
        table name are how a fresh instance ended up with a `SourceDocuments` the
        library could not read back (feature 082, FR-006).

        The element type is the shape's own: `is_canonical` is true only of a
        `VECTOR(FLOAT, n)` embedding, so reading it from the configuration or the
        environment would let a creation path emit a table the library then refuses to
        call canonical (FR-010). The dimension follows the embedding model and is a
        real choice, so it still comes from the caller.
        """
        declared = next(
            column.vector
            for column in shape.columns.values()
            if column.vector is not None
        )
        return create_table_statement(
            shape,
            schema=self.schema_prefix,
            table=table_name,
            vector=VectorSpec(declared.element_type, int(dimension)),
        )

    def _update_schema_metadata(self, cursor, table_name: str, config: Dict[str, Any]):
        """Update schema metadata for a table."""
        try:
            # Use MERGE or INSERT/UPDATE pattern
            cursor.execute(
                f"DELETE FROM {self._qn('SchemaMetadata')} WHERE table_name = ?",
                [table_name],
            )

            # Handle configuration serialization safely
            configuration_json = None
            if "configuration" in config:
                try:
                    configuration_json = json.dumps(config["configuration"])
                except (TypeError, ValueError) as json_error:
                    logger.warning(
                        f"Could not serialize configuration for {table_name}: {json_error}"
                    )
                    configuration_json = json.dumps({"error": "serialization_failed"})

            cursor.execute(
                f"""
                INSERT INTO {self._qn("SchemaMetadata")}
                (table_name, schema_version, vector_dimension, embedding_model, configuration, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                [
                    table_name,
                    config.get("schema_version"),
                    config.get("vector_dimension"),
                    config.get("embedding_model"),
                    configuration_json,
                ],
            )

            logger.info(f"✅ Updated schema metadata for {table_name}")

        except Exception as e:
            logger.error(f"Failed to update schema metadata for {table_name}: {e}")
            raise

    def _migrate_document_chunks_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Recreate DocumentChunks in the canonical shape.

        As with the document table, this drops what is there: it runs only when a
        caller asks for a migration, and it leaves an existing table alone when asked
        to preserve data rather than altering it into a third shape (FR-007).
        """
        try:
            create_sql = self._canonical_create_sql(
                CANONICAL_CHUNK_SHAPE,
                "DocumentChunks",
                expected_config["vector_dimension"],
            )

            table_exists = self._snapshot("DocumentChunks") is not None

            if table_exists and preserve_data:
                logger.info(
                    f"{self._qn('DocumentChunks')} exists and data is preserved - "
                    "leaving its shape as it is"
                )
                self._update_schema_metadata(cursor, "DocumentChunks", expected_config)
                return True

            # If no preservation or table doesn't exist, create it fresh
            if table_exists:
                cursor.execute(f"DROP TABLE IF EXISTS {self._qn('DocumentChunks')}")

            cursor.execute(create_sql)

            for index_sql in [
                f"CREATE INDEX idx_chunks_document_id ON {self._qn('DocumentChunks')} (document_id)",
            ]:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index on DocumentChunks: {e}")

            self._update_schema_metadata(cursor, "DocumentChunks", expected_config)
            logger.info("✅ DocumentChunks table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate DocumentChunks table: {e}")
            return False

    def get_vector_dimension(
        self, table_name: str = "SourceDocuments", model_name: str = None
    ) -> int:
        """
        Universal method to get vector dimension for any table.
        This is the SINGLE SOURCE OF TRUTH for all vector dimensions.

        Enforces the schema manager's view of the world based on validated configuration.

        Args:
            table_name: Name of the table (SourceDocuments, DocumentTokenEmbeddings, etc.)
            model_name: Optional specific model name override

        Returns:
            Vector dimension for the table
        """
        # Check cache first
        cache_key = f"{table_name}:{model_name or 'default'}"
        if cache_key in self._dimension_cache:
            return self._dimension_cache[cache_key]

        # Primary method: Get dimension directly from table config (config-driven)
        if table_name in self._table_configs:
            dimension = self._table_configs[table_name]["dimension"]

            # If model override specified, use model mapping
            if (
                model_name
                and model_name != self._table_configs[table_name]["default_model"]
            ):
                if model_name in self._model_dimensions:
                    dimension = self._model_dimensions[model_name]
                else:
                    logger.warning(
                        f"Unknown model '{model_name}' for {table_name}, using table default: {dimension}"
                    )

        else:
            # Fallback: Try model-based lookup for unknown tables
            if not model_name:
                model_name = self.base_embedding_model

            if model_name in self._model_dimensions:
                dimension = self._model_dimensions[model_name]
            else:
                # HARD FAIL - no dangerous fallbacks that hide configuration issues
                error_msg = f"CRITICAL: Unknown table '{table_name}' and unknown model '{model_name}' - schema manager cannot determine dimension"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Cache the result
        self._dimension_cache[cache_key] = dimension

        logger.debug(
            f"Schema Manager: {table_name} with model {model_name or 'default'} -> {dimension}D"
        )
        return dimension

    def validate_vector_dimension(
        self, table_name: str, provided_dimension: int, context: str = ""
    ) -> None:
        """
        Validate that a provided dimension matches schema manager's expectation.

        Args:
            table_name: Name of the table
            provided_dimension: Dimension provided by caller
            context: Context string for error messages

        Raises:
            ValueError: If dimension doesn't match
        """
        expected_dimension = self.get_vector_dimension(table_name)
        if provided_dimension != expected_dimension:
            error_msg = f"Dimension mismatch for {table_name}"
            if context:
                error_msg += f" in {context}"
            error_msg += f": provided {provided_dimension}D, schema manager expects {expected_dimension}D"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_embedding_model(self, table_name: str = "SourceDocuments") -> str:
        """
        Get the embedding model name for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Model name string
        """
        if table_name in self._table_configs:
            return self._table_configs[table_name]["default_model"]

        # Fallback to config manager
        embedding_config = self.config_manager.get_embedding_config()
        return embedding_config.get("model", "all-MiniLM-L6-v2")

    def register_model(self, model_name: str, dimension: int) -> None:
        """
        Register a new model and its dimension.

        Args:
            model_name: Name of the embedding model
            dimension: Vector dimension for this model
        """
        self._model_dimensions[model_name] = dimension
        # Clear cache to force recalculation
        self._dimension_cache.clear()
        logger.info(f"Registered model {model_name} with dimension {dimension}")

    def validate_dimension_consistency(self) -> Dict[str, Any]:
        """
        Validate that all tables have consistent dimensions with their models.

        Returns:
            Dictionary with validation results
        """
        results = {"consistent": True, "issues": [], "table_dimensions": {}}

        for table_name in self._table_configs.keys():
            try:
                current_config = self.get_current_schema_config(table_name)
                expected_dimension = self.get_vector_dimension(table_name)

                results["table_dimensions"][table_name] = {
                    "expected": expected_dimension,
                    "current": (
                        current_config.get("vector_dimension")
                        if current_config
                        else None
                    ),
                }

                if (
                    current_config
                    and current_config.get("vector_dimension") != expected_dimension
                ):
                    results["consistent"] = False
                    results["issues"].append(
                        {
                            "table": table_name,
                            "issue": "dimension_mismatch",
                            "expected": expected_dimension,
                            "current": current_config.get("vector_dimension"),
                        }
                    )

            except Exception as e:
                results["consistent"] = False
                results["issues"].append(
                    {"table": table_name, "issue": "validation_error", "error": str(e)}
                )

        return results

    def get_schema_status(self) -> Dict[str, Any]:
        """Get status of all managed schemas."""
        tables = list(self._table_configs.keys())
        status = {}

        for table in tables:
            current_config = self.get_current_schema_config(table)
            expected_config = self._get_expected_schema_config(table)
            needs_migration = self.needs_migration(table)

            status[table] = {
                "current_config": current_config,
                "expected_config": expected_config,
                "needs_migration": needs_migration,
                "status": "migration_needed" if needs_migration else "up_to_date",
                "vector_dimension": self.get_vector_dimension(table),
            }

        return status

    def _migrate_entities_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate Entities table with proper schema and foreign keys."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")

            # Try multiple table name approaches
            # One name, the qualified one. A retry on the bare name would create the table
            # in the connecting user's schema, where nothing reads it.
            table_attempts = [self._qn("Entities")]

            # Ensure SourceDocuments.doc_id is unique so Entities FK can reference it
            try:
                cursor.execute(
                    f"CREATE UNIQUE INDEX idx_source_docs_doc_id_unique ON {self._qn('SourceDocuments')} (doc_id)"
                )
                logger.info(
                    f"✅ Ensured unique index on {self._qn('SourceDocuments')}(doc_id)"
                )
            except Exception as e:
                # Index may already exist or another benign condition; proceed
                logger.debug(f"Unique index ensure for SourceDocuments.doc_id: {e}")

            for table_name in table_attempts:
                try:
                    logger.info(
                        f"🔧 Attempting to create Entities table as {table_name}"
                    )

                    # Drop existing table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Dropped existing {table_name} table")

                    # Create with correct schema
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        entity_id VARCHAR(255) PRIMARY KEY,
                        entity_name VARCHAR(1000) NOT NULL,
                        entity_type VARCHAR(255) COLLATE EXACT NOT NULL,
                        source_doc_id VARCHAR(255) NOT NULL,
                        description VARCHAR(4000) NULL,
                        confidence DOUBLE DEFAULT 1.0,
                        embedding VECTOR({vector_data_type}, {vector_dim}) NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_doc_id) REFERENCES {self._qn("SourceDocuments")}(doc_id) ON DELETE CASCADE
                    )
                    """
                    cursor.execute(create_sql)
                    logger.info(f"✅ Successfully created {table_name} table")

                    # Create indexes for GraphRAG query performance
                    indexes = [
                        f"CREATE INDEX idx_entities_name ON {table_name} (entity_name)",
                        f"CREATE INDEX idx_entities_type ON {table_name} (entity_type)",
                        f"CREATE INDEX idx_entities_source_doc ON {table_name} (source_doc_id)",
                    ]
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                            logger.debug(f"Created index: {index_sql}")
                        except Exception as e:
                            logger.warning(f"Failed to create index: {e}")

                    # Update schema metadata
                    try:
                        self._update_schema_metadata(
                            cursor, "Entities", expected_config
                        )
                    except Exception as meta_error:
                        logger.debug(f"Schema metadata update failed: {meta_error}")

                    logger.info(
                        f"✅ Entities table created successfully as {table_name}"
                    )
                    return True

                except Exception as table_error:
                    logger.warning(
                        f"Failed to create table as {table_name}: {table_error}"
                    )
                    if table_name == table_attempts[-1]:
                        logger.error("All table creation attempts failed")
                        return False
                    continue

            return False

        except Exception as e:
            logger.error(f"Failed to migrate Entities table: {e}")
            return False

    def _migrate_entity_relationships_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate EntityRelationships table with proper schema and foreign keys."""
        try:
            # Try multiple table name approaches
            table_attempts = [self._qn("EntityRelationships")]

            for table_name in table_attempts:
                try:
                    logger.info(
                        f"🔧 Attempting to create EntityRelationships table as {table_name}"
                    )

                    # Drop existing table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Dropped existing {table_name} table")

                    # Create with correct schema
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        relationship_id VARCHAR(255) PRIMARY KEY,
                        source_entity_id VARCHAR(255) NOT NULL,
                        target_entity_id VARCHAR(255) NOT NULL,
                        relationship_type VARCHAR(255) COLLATE EXACT NOT NULL,
                        metadata VARCHAR(4000) NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_entity_id) REFERENCES {self._qn("Entities")}(entity_id) ON DELETE CASCADE,
                        FOREIGN KEY (target_entity_id) REFERENCES {self._qn("Entities")}(entity_id) ON DELETE CASCADE
                    )
                    """
                    cursor.execute(create_sql)
                    logger.info(f"✅ Successfully created {table_name} table")

                    # Create indexes for GraphRAG traversal performance
                    indexes = [
                        f"CREATE INDEX idx_rel_source ON {table_name} (source_entity_id)",
                        f"CREATE INDEX idx_rel_target ON {table_name} (target_entity_id)",
                    ]
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                            logger.debug(f"Created index: {index_sql}")
                        except Exception as e:
                            logger.warning(f"Failed to create index: {e}")

                    # Update schema metadata
                    try:
                        self._update_schema_metadata(
                            cursor, "EntityRelationships", expected_config
                        )
                    except Exception as meta_error:
                        logger.debug(f"Schema metadata update failed: {meta_error}")

                    logger.info(
                        f"✅ EntityRelationships table created successfully as {table_name}"
                    )
                    return True

                except Exception as table_error:
                    logger.warning(
                        f"Failed to create table as {table_name}: {table_error}"
                    )
                    if table_name == table_attempts[-1]:
                        logger.error("All table creation attempts failed")
                        return False
                    continue

            return False

        except Exception as e:
            logger.error(f"Failed to migrate EntityRelationships table: {e}")
            return False

    def ensure_vector_hnsw_index(
        self, cursor, table: str, column: str, index_name: str, try_acorn: bool = True
    ) -> None:
        """
        Ensure HNSW vector index exists on the specified table and column.

        Attempts to create index with ACORN=1 optimization first, falls back to standard HNSW.
        This function is idempotent - no-op if index already exists.

        Args:
            cursor: Database cursor
            table: Qualified table name (e.g., via self._qn("SourceDocuments"))
            column: Vector column name (e.g., "embedding")
            index_name: Index name (e.g., "idx_SourceDocuments_embedding")
            try_acorn: Whether to attempt ACORN=1 optimization (default: True)
        """
        try:
            reader = IRISCatalogReader(self.connection_manager.get_connection())
            schema, _, bare = table.rpartition(".")
            existing = reader.index_names(
                schema or self.schema_prefix, bare, cursor=cursor
            )

            if index_name.upper() in existing:
                logger.debug(
                    f"HNSW index {index_name} already exists on {table}.{column}"
                )
                return

            # Try to create index with ACORN=1 first
            if try_acorn:
                try:
                    acorn_sql = f"CREATE INDEX {index_name} ON {table}({column}) AS HNSW WITH (ACORN=1)"
                    logger.info(
                        f"Creating HNSW index with ACORN=1: {index_name} on {table}.{column}"
                    )
                    cursor.execute(acorn_sql)
                    logger.info(
                        f"✅ Successfully created HNSW index with ACORN=1: {index_name}"
                    )
                    return
                except Exception as acorn_error:
                    logger.warning(
                        f"ACORN=1 syntax not supported, falling back to standard HNSW: {acorn_error}"
                    )

            # Fallback to standard HNSW
            standard_sql = f"CREATE INDEX {index_name} ON {table}({column}) AS HNSW"
            logger.info(
                f"Creating standard HNSW index: {index_name} on {table}.{column}"
            )
            cursor.execute(standard_sql)
            logger.info(f"✅ Successfully created standard HNSW index: {index_name}")

        except Exception as e:
            logger.error(
                f"Failed to create HNSW index {index_name} on {table}.{column}: {e}"
            )
            logger.warning(
                "Continuing without vector index - this may impact performance"
            )
            # Re-raise to comply with Constitutional Principle VI: Explicit Error Handling
            # Caller can decide whether to continue or abort based on business logic
            raise

    def ensure_all_vector_indexes(self) -> List[str]:
        """
        Ensure all required vector indexes exist at startup.

        Creates HNSW indexes (with ACORN=1 when available) for all vector columns
        across the RAG system.

        Returns:
            The names of the indexes that now exist (created or already there). A table
            without the vector column is left out, not reported as indexed.

        Raises:
            SchemaIndexError: one or more indexes could not be ensured. Nothing is
                committed, and the message names every index that failed.
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            vector_indexes = [
                {
                    "table": self._qn("SourceDocuments"),
                    "column": "embedding",
                    "index_name": "idx_SourceDocuments_embedding",
                },
                {
                    "table": self._qn("Entities"),
                    "column": "embedding",
                    "index_name": "idx_Entities_embedding",
                },
            ]

            logger.info("Ensuring HNSW vector indexes...")
            errors = []
            ensured: List[str] = []

            for index_def in vector_indexes:
                try:
                    bare = index_def["table"].rpartition(".")[2]
                    snapshot = self._snapshot(bare)
                    if snapshot is None:
                        raise SchemaIndexError(
                            f"{index_def['table']} does not exist", table=bare
                        )
                    if index_def["column"].upper() not in {
                        str(c["COLUMN_NAME"]).upper() for c in snapshot.columns
                    }:
                        # The standard Entities table carries no vector column (entity
                        # embeddings live in iris-vector-graph), so there is nothing to
                        # index; leaving it out of the result says so.
                        logger.info(
                            f"{index_def['table']} has no {index_def['column']} "
                            f"column; no vector index to ensure"
                        )
                        continue
                    self.ensure_vector_hnsw_index(
                        cursor,
                        index_def["table"],
                        index_def["column"],
                        index_def["index_name"],
                        try_acorn=True,
                    )
                    ensured.append(index_def["index_name"])
                except Exception as index_error:
                    # Keep going so the message names every failure, then raise.
                    errors.append(f"{index_def['index_name']}: {index_error}")
                    logger.error(
                        f"Failed to ensure index {index_def['index_name']}: {index_error}"
                    )

            if errors:
                raise SchemaIndexError(
                    "Vector index creation failed for " + "; ".join(errors),
                    failed=[e.split(":", 1)[0] for e in errors],
                )

            connection.commit()
            logger.info("✅ Vector index ensure process completed")
            return ensured

        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()

    # ========== FULL-TEXT INDEX (text leg of text / hybrid / rrf retrieval) ==========

    def _split_table(self, table_name: str) -> tuple:
        """``(schema, table)`` for a bare or qualified name; bare means this manager's."""
        if "." in table_name:
            schema, table = table_name.split(".", 1)
            return schema, table
        return self.schema_prefix, table_name

    def has_text_index(self, table_name: str = "SourceDocuments") -> bool:
        """Whether the table carries the full-text index ``ensure_text_index`` creates.

        Raises:
            SchemaIndexError: the catalog could not be read. "Could not tell" is not
                "absent": a caller that treated it as absent would try to create it.
        """
        schema, table = self._split_table(table_name)
        try:
            names = IRISCatalogReader(
                self.connection_manager.get_connection()
            ).index_names(schema, table)
        except CatalogUnavailable as exc:
            raise SchemaIndexError(
                f"Cannot tell whether {schema}.{table} has its full-text index: {exc}",
                table=table,
            ) from exc
        return TEXT_INDEX_NAME.upper() in names

    def _ensure_text_index_once(self, table_name: str) -> None:
        """`ensure_text_index`, skipped once this instance has seen it succeed."""
        key = ".".join(self._split_table(table_name)).lower()
        if key not in self._text_indexed:
            self.ensure_text_index(table_name)
            self._text_indexed.add(key)

    def ensure_text_index(self, table_name: str = "SourceDocuments") -> str:
        """Ensure the table has a full-text (iFind) index on its text column.

        The text leg of ``text``, ``hybrid`` and ``rrf`` retrieval searches through this
        index. The column is the one the catalog says fills the text role, whatever the
        table calls it (``text_content``, ``content``, ...). The index is
        ``%iFind.Index.Basic`` with English tokenising and lower-casing.

        Idempotent and non-destructive: an existing index is left alone, and a missing
        one is built over the rows already in the table (no data is read out, dropped
        or rewritten), so an existing deployment calls this once to gain text search.
        The build is synchronous and its cost grows with the table.

        Args:
            table_name: Bare (this manager's schema) or ``Schema.Table``.

        Returns:
            The name of the index, which now exists.

        Raises:
            SchemaIndexError: the table is missing, has no column to index, or the
                database refused the index. Nothing is swallowed.
        """
        schema, table = self._split_table(table_name)
        snapshot = self._snapshot(table, schema=schema)
        if snapshot is None:
            raise SchemaIndexError(
                f"Cannot create the text index: {schema}.{table} does not exist",
                table=table,
            )
        try:
            column = TableShape.from_snapshot(snapshot).column_for(ColumnRole.TEXT).name
        except TableShapeError as exc:
            raise SchemaIndexError(
                f"Cannot create the text index on {schema}.{table}: {exc}", table=table
            ) from exc

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            existing = IRISCatalogReader(connection).index_names(
                schema, snapshot.table, cursor=cursor
            )
            if TEXT_INDEX_NAME.upper() in existing:
                logger.debug(
                    f"Text index {TEXT_INDEX_NAME} already on {schema}.{table}"
                )
                return TEXT_INDEX_NAME
            logger.info(
                f"Building text index {TEXT_INDEX_NAME} on {schema}.{table}.{column}"
            )
            language = self.config_manager.get("storage:iris:text_language")
            cursor.execute(
                create_text_index_statement(
                    schema, snapshot.table, column, language=language
                )
            )
            connection.commit()
            return TEXT_INDEX_NAME
        except Exception as exc:
            connection.rollback()
            raise SchemaIndexError(
                f"Could not create the text index {TEXT_INDEX_NAME} on "
                f"{schema}.{table}({column}): {exc}",
                table=table,
                column=column,
            ) from exc
        finally:
            cursor.close()

    # ========== AUDIT TESTING METHODS ==========
    # These methods replace direct SQL anti-patterns in integration tests

    def verify_table_structure(
        self, table_name: str, schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify table structure using proper abstractions (replaces direct SQL in tests).

        Args:
            table_name: Table name without schema (e.g., 'SourceDocuments')
            schema_name: SQL schema containing the table; this manager's own by default.
                The old default was the literal `RAG`, which reported an empty structure
                for every deployment that configured a different schema prefix.

        Returns:
            Dictionary mapping column names to data types
        """
        try:
            snapshot = self._snapshot(table_name, schema=schema_name)
        except Exception as e:  # noqa: BLE001 - callers treat {} as "cannot tell"
            logger.error(f"Failed to verify table structure for {table_name}: {e}")
            return {}
        if snapshot is None:
            return {}
        return {str(row["COLUMN_NAME"]): row["DATA_TYPE"] for row in snapshot.columns}

    # ========== IRIS GRAPH CORE TABLE MIGRATION METHODS ==========

    def _migrate_rdf_labels_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate rdf_labels table for entity type/label mapping."""
        try:
            logger.info("🔧 Migrating rdf_labels table for IRIS Graph Core")

            # Drop existing table if it exists (for now - data preservation not implemented)
            cursor.execute(f"DROP TABLE IF EXISTS {self._qn('rdf_labels')}")
            logger.info("Dropped existing rdf_labels table")

            # Create table with proper schema
            create_sql = f"""
            CREATE TABLE {self._qn('rdf_labels')} (
                s      VARCHAR(256) NOT NULL,
                label  VARCHAR(128) NOT NULL
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created rdf_labels table")

            # Create indexes for GraphRAG performance
            indexes = [
                f"CREATE INDEX idx_labels_label_s ON {self._qn('rdf_labels')}(label, s)",
                f"CREATE INDEX idx_labels_s_label ON {self._qn('rdf_labels')}(s, label)",
            ]
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    logger.debug(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            # Update schema metadata
            try:
                self._update_schema_metadata(cursor, "rdf_labels", expected_config)
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ rdf_labels table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate rdf_labels table: {e}")
            return False

    def _migrate_rdf_props_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate rdf_props table for entity properties."""
        try:
            logger.info("🔧 Migrating rdf_props table for IRIS Graph Core")

            # Drop existing table if it exists
            cursor.execute(f"DROP TABLE IF EXISTS {self._qn('rdf_props')}")
            logger.info("Dropped existing rdf_props table")

            # Create table with proper schema
            create_sql = f"""
            CREATE TABLE {self._qn('rdf_props')} (
                s      VARCHAR(256) NOT NULL,
                key    VARCHAR(128) NOT NULL,
                val    VARCHAR(4000)
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created rdf_props table")

            # Create indexes for property lookup performance
            indexes = [
                f"CREATE INDEX idx_props_s_key ON {self._qn('rdf_props')}(s, key)",
                f"CREATE INDEX idx_props_key_val ON {self._qn('rdf_props')}(key, val)",
            ]
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    logger.debug(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            # Update schema metadata
            try:
                self._update_schema_metadata(cursor, "rdf_props", expected_config)
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ rdf_props table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate rdf_props table: {e}")
            return False

    def _migrate_rdf_edges_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate rdf_edges table for graph relationships with JSON confidence."""
        try:
            logger.info("🔧 Migrating rdf_edges table for IRIS Graph Core")

            # Drop existing table if it exists
            cursor.execute(f"DROP TABLE IF EXISTS {self._qn('rdf_edges')}")
            logger.info("Dropped existing rdf_edges table")

            # Create table with proper schema including JSON qualifiers
            # IRIS uses IDENTITY instead of GENERATED BY DEFAULT AS IDENTITY
            create_sql = f"""
            CREATE TABLE {self._qn('rdf_edges')} (
                edge_id    BIGINT IDENTITY PRIMARY KEY,
                s          VARCHAR(256) NOT NULL,
                p          VARCHAR(128) NOT NULL,
                o_id       VARCHAR(256) NOT NULL,
                qualifiers VARCHAR(4000)
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created rdf_edges table")

            # Create indexes for graph traversal performance
            indexes = [
                f"CREATE INDEX idx_edges_s_p ON {self._qn('rdf_edges')}(s, p)",
                f"CREATE INDEX idx_edges_p_oid ON {self._qn('rdf_edges')}(p, o_id)",
                f"CREATE INDEX idx_edges_s ON {self._qn('rdf_edges')}(s)",
            ]
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    logger.debug(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            # Update schema metadata
            try:
                self._update_schema_metadata(cursor, "rdf_edges", expected_config)
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ rdf_edges table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate rdf_edges table: {e}")
            return False

    def _migrate_kg_node_embeddings_optimized_table(
        self, cursor, expected_config: Dict[str, Any], preserve_data: bool
    ) -> bool:
        """Migrate kg_NodeEmbeddings_optimized table for HNSW vector search."""
        try:
            vector_dim = expected_config["vector_dimension"]
            vector_data_type = expected_config.get("vector_data_type", "FLOAT")

            logger.info(
                f"🔧 Migrating kg_NodeEmbeddings_optimized table to {vector_dim}D vectors with {vector_data_type} type"
            )

            # Drop existing table if it exists
            cursor.execute(
                f"DROP TABLE IF EXISTS {self._qn('kg_NodeEmbeddings_optimized')}"
            )
            logger.info("Dropped existing kg_NodeEmbeddings_optimized table")

            # Create table with optimized VECTOR column
            create_sql = f"""
            CREATE TABLE {self._qn('kg_NodeEmbeddings_optimized')} (
                id   VARCHAR(256) PRIMARY KEY,
                emb  VECTOR({vector_data_type}, {vector_dim}) NOT NULL
            )
            """
            cursor.execute(create_sql)
            logger.info("✅ Successfully created kg_NodeEmbeddings_optimized table")

            # Create HNSW index for high-performance vector search
            try:
                self.ensure_vector_hnsw_index(
                    cursor,
                    self._qn("kg_NodeEmbeddings_optimized"),
                    "emb",
                    "HNSW_NodeEmb_Optimized",
                    try_acorn=True,
                )
                logger.info("✅ HNSW index created for optimized vector table")
            except Exception as index_error:
                logger.error(f"HNSW index creation failed: {index_error}")
                return False

            # Update schema metadata
            try:
                self._update_schema_metadata(
                    cursor, "kg_NodeEmbeddings_optimized", expected_config
                )
            except Exception as meta_error:
                logger.debug(f"Schema metadata update failed: {meta_error}")

            logger.info("✅ kg_NodeEmbeddings_optimized table migrated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate kg_NodeEmbeddings_optimized table: {e}")
            return False

    @classmethod
    def create_schema_manager(
        cls, pipeline_type: str, connection_manager, config_manager
    ):
        """
        Factory method to create appropriate schema manager for pipeline type.

        Args:
            pipeline_type: Type of pipeline (e.g., 'hybrid_graphrag', 'graphrag')
            connection_manager: Database connection manager
            config_manager: Configuration manager

        Returns:
            Appropriate schema manager instance
        """
        # Get pipeline requirements
        requirements = config_manager.get_pipeline_requirements(pipeline_type)
        schema_manager_type = requirements.get("schema_manager", "SchemaManager")

        if schema_manager_type == "HybridGraphRAGSchemaManager":
            logger.warning(
                "HybridGraphRAGSchemaManager is deprecated, using SchemaManager"
            )
        if pipeline_type in ["graphrag", "hybrid_graphrag"]:
            # Enhanced schema manager for graph-based pipelines
            logger.info(f"Creating enhanced SchemaManager for {pipeline_type}")
            manager = cls(connection_manager, config_manager)
            manager.pipeline_type = pipeline_type
            return manager
        else:
            logger.info(f"Creating standard SchemaManager for {pipeline_type}")
            return cls(connection_manager, config_manager)

    def ensure_pipeline_schema(self, pipeline_type: str) -> bool:
        """
        Ensure schema for specific pipeline type based on requirements.

        Args:
            pipeline_type: The pipeline type to ensure schema for

        Returns:
            True if all schema requirements were met successfully
        """
        try:
            logger.info(f"Ensuring schema for pipeline type: {pipeline_type}")
            requirements = self.config_manager.get_pipeline_requirements(pipeline_type)
            schema_requirements = requirements.get("schema_requirements", {})

            # Ensure required tables
            required_tables = schema_requirements.get("tables", [])
            for table_name in required_tables:
                if not self.ensure_table_schema(table_name, pipeline_type):
                    logger.error(f"Failed to ensure required table: {table_name}")
                    return False
                else:
                    logger.info(f"✅ Table {table_name} ensured for {pipeline_type}")

            # Create required indexes
            required_indexes = schema_requirements.get("indexes", [])
            for index_config in required_indexes:
                try:
                    ensured = self._ensure_index(index_config)
                except SchemaIndexError as index_error:
                    logger.error(f"Index not created: {index_error}")
                    ensured = False
                if not ensured:
                    if index_config.get("required", True):
                        logger.error(
                            f"Failed to create required index: {index_config.get('name')}"
                        )
                        return False
                    else:
                        logger.warning(
                            f"Optional index not created: {index_config.get('name')}"
                        )

            logger.info(f"✅ Schema ensured successfully for {pipeline_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure schema for {pipeline_type}: {e}")
            return False

    def ensure_table_schema(
        self, table_name: str, pipeline_type: Optional[str] = None
    ) -> bool:
        """
        Ensure a specific table exists with proper schema.

        Args:
            table_name: Name of the table to ensure
            pipeline_type: Optional pipeline type for context

        Returns:
            True if table exists or was created successfully
        """
        try:
            # Check if this is a standard RAG table
            standard_tables = [
                "SourceDocuments",
                "DocumentChunks",
                "Entities",
                "EntityRelationships",
                "Communities",
            ]
            if table_name in standard_tables:
                ready = self._ensure_standard_table(table_name)
                if ready and table_name == "SourceDocuments":
                    # Raises SchemaIndexError: a document table without its text
                    # index would make hybrid/rrf retrieval silently vector-only.
                    self._ensure_text_index_once(table_name)
                return ready

            # IVG owns its physical graph schema. Keep this compatibility entry point
            # for older IVR callers, but delegate creation to IVG instead of emitting
            # parallel RAG/current-schema DDL here.
            iris_graph_tables = list(IVG_REQUIRED_TABLES)
            iris_graph_tables_lower = [t.lower() for t in iris_graph_tables]
            if table_name.lower() in iris_graph_tables_lower:
                logger.info(
                    "Ensuring IVG schema via iris-vector-graph for %s", table_name
                )
                self._initialize_ivg_schema()
                return self._ivg_table_exists(table_name)
            else:
                # For other tables, delegate to specialized managers
                logger.warning(
                    f"Table {table_name} not recognized as standard RAG or iris-vector-graph table"
                )
                return False

        except SchemaIndexError:
            raise
        except Exception as e:
            logger.error(f"Failed to ensure table {table_name}: {e}")
            return False

    def _ensure_standard_table(self, table_name: str) -> bool:
        """Ensure standard RAG tables exist."""
        try:
            # Check class-level cache first to avoid repeated validation spam
            # `Entities` is re-checked every time because its column-adding path below
            # still runs; the document and chunk tables are not altered at all now, so
            # once seen they are done.
            if table_name in self._tables_validated and table_name != "Entities":
                logger.debug(
                    f"Table {self._qn(table_name)} already validated (cached) - skipping check"
                )
                return True

            snapshot = self._snapshot(table_name)

            conn = self.connection_manager.get_connection()
            cursor = conn.cursor()

            if snapshot is not None:
                # An existing document table is used in whatever shape it has: the
                # resolved shape tells the store which column holds the text, so
                # adding a canonical column to a legacy table buys nothing and costs
                # the caller a silent schema change (feature 082, FR-007). An explicit
                # `migrate_table` call is the only way to change a table's shape.
                if table_name == "Entities":
                    try:
                        columns = {
                            str(row["COLUMN_NAME"]).lower() for row in snapshot.columns
                        }

                        if "entity_id" not in columns:
                            cursor.execute(
                                f"ALTER TABLE {self._qn('Entities')} ADD entity_id VARCHAR(255)"
                            )
                            if "id" in columns:
                                cursor.execute(
                                    f"UPDATE {self._qn('Entities')} SET entity_id = id WHERE entity_id IS NULL"
                                )
                            conn.commit()
                            columns.add("entity_id")

                        if "entity_name" not in columns:
                            cursor.execute(
                                f"ALTER TABLE {self._qn('Entities')} ADD entity_name VARCHAR(255)"
                            )
                            if "name" in columns:
                                cursor.execute(
                                    f"UPDATE {self._qn('Entities')} SET entity_name = name WHERE entity_name IS NULL"
                                )
                            conn.commit()
                            columns.add("entity_name")

                        if "source_doc_id" not in columns:
                            cursor.execute(
                                f"ALTER TABLE {self._qn('Entities')} ADD source_doc_id VARCHAR(255)"
                            )
                            if "doc_id" in columns:
                                cursor.execute(
                                    f"UPDATE {self._qn('Entities')} SET source_doc_id = doc_id WHERE source_doc_id IS NULL"
                                )
                            conn.commit()
                    except Exception as exc:
                        logger.warning(f"Failed to ensure Entities columns: {exc}")

                if table_name in EXACT_CASE_COLUMNS:
                    # A table from an earlier release has the case-folding default.
                    ensure_exact_collation(
                        self.connection_manager,
                        self.schema_prefix,
                        table_name,
                        raise_on_failure=False,
                    )

                logger.debug(
                    f"Table {self._qn(table_name)} exists - caching validation result"
                )
                self._tables_validated.add(table_name)  # Cache the result
                cursor.close()
                return True

            # Create table based on type
            if table_name == "SourceDocuments":
                success = self._create_source_documents_table(cursor)
            elif table_name == "Entities":
                success = self._create_entities_table(cursor)
            elif table_name == "EntityRelationships":
                success = self._create_entity_relationships_table(cursor)
            elif table_name == "Communities":
                success = self._create_communities_table(cursor)
            elif table_name == "DocumentChunks":
                success = self._create_document_chunks_table(cursor)
            else:
                logger.error(f"Unknown standard table: {table_name}")
                cursor.close()
                return False

            if success:
                conn.commit()
                logger.info(f"✅ Successfully created table {self._qn(table_name)}")
            else:
                conn.rollback()
                logger.error(f"❌ Failed to create table {self._qn(table_name)}")

            cursor.close()
            return success

        except Exception as e:
            logger.error(f"Error ensuring standard table {table_name}: {e}")
            try:
                conn.rollback()
                cursor.close()
            except Exception:
                pass
            return False

    def _create_source_documents_table(self, cursor) -> bool:
        """Create SourceDocuments in the canonical shape."""
        try:
            cursor.execute(
                self._canonical_create_sql(
                    CANONICAL_DOCUMENT_SHAPE,
                    "SourceDocuments",
                    self.get_vector_dimension("SourceDocuments"),
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create SourceDocuments table: {e}")
            return False

    def _create_entities_table(self, cursor) -> bool:
        """Create Entities table."""
        try:
            create_sql = f"""
            CREATE TABLE {self._qn("Entities")} (
                entity_id VARCHAR(255) NOT NULL,
                entity_name VARCHAR(500) NOT NULL,
                entity_type VARCHAR(100) COLLATE EXACT,
                description VARCHAR(2000),
                confidence DOUBLE DEFAULT 1.0,
                source_doc_id VARCHAR(255),
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (entity_id)
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            indexes = [
                f"CREATE INDEX idx_entities_name ON {self._qn('Entities')} (entity_name)",
                f"CREATE INDEX idx_entities_type ON {self._qn('Entities')} (entity_type)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True
        except Exception as e:
            logger.error(f"Failed to create Entities table: {e}")
            return False

    def _create_entity_relationships_table(self, cursor) -> bool:
        """Create EntityRelationships table."""
        try:
            create_sql = f"""
            CREATE TABLE {self._qn("EntityRelationships")} (
                relationship_id VARCHAR(255) NOT NULL,
                source_entity_id VARCHAR(255) NOT NULL,
                target_entity_id VARCHAR(255) NOT NULL,
                relationship_type VARCHAR(255) COLLATE EXACT,
                weight DOUBLE DEFAULT 1.0,
                confidence DOUBLE DEFAULT 1.0,
                source_document VARCHAR(255),
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (relationship_id)
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            indexes = [
                f"CREATE INDEX idx_relationships_source ON {self._qn('EntityRelationships')} (source_entity_id)",
                f"CREATE INDEX idx_relationships_target ON {self._qn('EntityRelationships')} (target_entity_id)",
                f"CREATE INDEX idx_relationships_type ON {self._qn('EntityRelationships')} (relationship_type)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            return True
        except Exception as e:
            logger.error(f"Failed to create EntityRelationships table: {e}")
            return False

    def _create_communities_table(self, cursor) -> bool:
        """Create Communities table for GraphRAG community detection."""
        try:
            create_sql = f"""
            CREATE TABLE {self._qn("Communities")} (
                community_id VARCHAR(255) NOT NULL,
                name VARCHAR(500),
                description VARCHAR(5000),
                entity_ids VARCHAR(10000),
                entity_count INT DEFAULT 0,
                hierarchy_level INT DEFAULT 0,
                parent_community_id VARCHAR(255),
                summary VARCHAR(5000),
                confidence DOUBLE DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (community_id)
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            indexes = [
                f"CREATE INDEX idx_communities_level ON {self._qn('Communities')} (hierarchy_level)",
                f"CREATE INDEX idx_communities_parent ON {self._qn('Communities')} (parent_community_id)",
            ]

            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as idx_error:
                    logger.warning(f"Index creation warning: {idx_error}")

            logger.info("✅ Communities table created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Communities table: {e}")
            return False

    def _create_document_chunks_table(self, cursor) -> bool:
        """Create DocumentChunks in the canonical shape."""
        try:
            cursor.execute(
                self._canonical_create_sql(
                    CANONICAL_CHUNK_SHAPE,
                    "DocumentChunks",
                    self.get_vector_dimension("DocumentChunks"),
                )
            )
            cursor.execute(
                f"CREATE INDEX idx_chunks_document_id ON {self._qn('DocumentChunks')} (document_id)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create DocumentChunks table: {e}")
            return False

    _IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    def _ensure_index(self, index_config: Dict[str, Any]) -> bool:
        """Ensure the index `index_config` describes exists, in this manager's schema.

        `index_config` carries `name`, `table` (unqualified) and `columns` (or `column`); `type`
        defaults to a plain index, and `HNSW` goes through `ensure_vector_hnsw_index`.

        Returns:
            True if the index exists afterwards; False if the database refused it.

        Raises:
            SchemaIndexError: the configuration is incomplete or names an unsafe
                identifier. A configuration that cannot be acted on is not "ensured".
        """
        name = index_config.get("name")
        table = index_config.get("table")
        columns = index_config.get("columns") or index_config.get("column")
        if isinstance(columns, str):
            columns = [columns]
        if not table:
            raise SchemaIndexError(
                f"Index {name!r} names no table", config=index_config
            )
        if not columns:
            raise SchemaIndexError(
                f"Index {name!r} on {table!r} names no columns", config=index_config
            )
        for identifier in (name, table, *columns):
            if not isinstance(identifier, str) or not self._IDENTIFIER.match(
                identifier
            ):
                raise SchemaIndexError(
                    f"Unsafe or missing identifier {identifier!r} in index "
                    f"configuration",
                    config=index_config,
                )

        if index_config.get("parameters"):
            raise SchemaIndexError(
                f"Index {name!r} asks for parameters {index_config['parameters']} that "
                f"this manager cannot pass on; creating it without them would build a "
                f"different index than the one configured",
                config=index_config,
            )
        index_type = str(index_config.get("type", "BTREE")).upper()
        qualified = self._qn(table)
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            if index_type == "HNSW":
                if len(columns) != 1:
                    raise SchemaIndexError(
                        f"HNSW index {name!r} takes one vector column, got {columns}",
                        config=index_config,
                    )
                self.ensure_vector_hnsw_index(
                    cursor, qualified, columns[0], name, try_acorn=True
                )
            else:
                existing = IRISCatalogReader(connection).index_names(
                    self.schema_prefix, table, cursor=cursor
                )
                if name.upper() not in existing:
                    cursor.execute(
                        f"CREATE INDEX {name} ON {qualified} ({', '.join(columns)})"
                    )
            connection.commit()
            return True
        except SchemaIndexError:
            connection.rollback()
            raise
        except Exception as e:
            connection.rollback()
            logger.error(f"Failed to create index {name} on {qualified}: {e}")
            return False
        finally:
            cursor.close()
