"""Compatibility helpers for iris-vector-graph integration."""

from __future__ import annotations

import importlib
import re
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 4.0.0 is the storage-layout break: embeddings and node stores became graph-scoped.
# 5.0 removes what 4.x deprecates, so pyproject caps below it.
MIN_IVG_VERSION = "4.0.0"
IVG_SCHEMA = "Graph_KG"
IVG_REQUIRED_TABLES: Tuple[str, ...] = (
    "nodes",
    "rdf_labels",
    "rdf_props",
    "rdf_edges",
    "kg_NodeEmbeddings",
    "kg_NodeEmbeddings_optimized",
)

# Columns a table has only once it is in the 4.x layout. The embedding tables lost
# `id` for `(graph_id, node_id)`; ivg 4.x's `WHERE node_id = ?` fails on the old ones.
IVG_4X_REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "nodes": ("graph_id",),
    "rdf_labels": ("graph_id",),
    "rdf_props": ("graph_id",),
    "rdf_edges": ("graph_id",),
    "kg_NodeEmbeddings": ("graph_id", "node_id"),
    "kg_NodeEmbeddings_optimized": ("graph_id", "node_id"),
}

IVG_UPGRADE_HINT = (
    "Graph_KG predates the iris-vector-graph 4.x layout. Run "
    "SchemaManager.upgrade_iris_vector_graph_storage(dry_run=True) to preview, then "
    "without dry_run to migrate (it wraps iris_vector_graph.migrations.upgrade_to_4_0_0)."
)


class ValidationResult(BaseModel):
    """Result of checking whether IVG is usable for GraphRAG."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    package_installed: bool
    missing_tables: List[str] = Field(default_factory=list)
    error_message: str = ""
    installed_version: Optional[str] = None
    minimum_version: str = MIN_IVG_VERSION
    schema_name: str = IVG_SCHEMA
    legacy_layout_tables: List[str] = Field(default_factory=list)


class InitializationResult(BaseModel):
    """Result of delegating IVG schema initialization to iris-vector-graph."""

    model_config = ConfigDict(frozen=True)

    package_detected: bool
    tables_attempted: List[str] = Field(default_factory=list)
    tables_created: Dict[str, bool] = Field(default_factory=dict)
    total_time_seconds: float = 0.0
    error_messages: Dict[str, str] = Field(default_factory=dict)
    installed_version: Optional[str] = None
    minimum_version: str = MIN_IVG_VERSION
    schema_name: str = IVG_SCHEMA
    ivg_status: Dict[str, Any] = Field(default_factory=dict)
    legacy_layout_tables: List[str] = Field(default_factory=list)

    @field_validator("total_time_seconds")
    @classmethod
    def _non_negative_duration(cls, value: float) -> float:
        return max(float(value), 0.0)


def import_ivg_module():
    """Import iris_vector_graph or raise ImportError with a package-specific message."""
    return importlib.import_module("iris_vector_graph")


def get_ivg_version(module: Any | None = None) -> Optional[str]:
    """Return the active IVG version, including source-tree checkouts."""
    module = module or import_ivg_module()
    source_version = _source_tree_version(module)
    if source_version:
        return source_version

    version = getattr(module, "__version__", None)
    if version and version != "unknown":
        return str(version)

    try:
        return metadata.version("iris-vector-graph")
    except metadata.PackageNotFoundError:
        return None


def assert_ivg_compatible(
    module: Any | None = None,
    minimum_version: str = MIN_IVG_VERSION,
) -> Optional[str]:
    """Validate the imported IVG module is new enough for IVR's 4.x integration."""
    module = module or import_ivg_module()
    installed_version = get_ivg_version(module)
    if installed_version and _version_key(installed_version) >= _version_key(
        minimum_version
    ):
        return installed_version

    if _has_4x_surface(module) and not installed_version:
        return None

    version_text = installed_version or "unknown"
    raise ImportError(
        "HybridGraphRAG requires iris-vector-graph "
        f">={minimum_version}; found {version_text}. "
        "Install or expose the current IVG checkout before using graph features."
    )


def create_graph_engine(
    connection,
    embedding_dimension: Optional[int] = None,
    **kwargs,
):
    """Create an IVG engine; `embedding_dimension` and other kwargs pass through.

    ivg 4.x auto-detects the stored vector dtype when `vector_dtype` is not given.
    """
    module = import_ivg_module()
    assert_ivg_compatible(module)
    engine_cls = getattr(module, "IRISGraphEngine")
    if embedding_dimension is not None:
        kwargs.setdefault("embedding_dimension", int(embedding_dimension))
    return engine_cls(connection, **kwargs)


def pre_4_layout_tables(
    columns_by_table: Mapping[str, Iterable[str]],
) -> List[str]:
    """Required tables that exist but still have the pre-4.0 column layout.

    A table that is absent, or reported with no columns, is not a layout finding --
    that is what `missing_tables` reports. Returned in `IVG_REQUIRED_TABLES` order.
    """
    legacy = []
    for table in IVG_REQUIRED_TABLES:
        columns = {str(c).lower() for c in columns_by_table.get(table, ())}
        if not columns:
            continue
        if not set(IVG_4X_REQUIRED_COLUMNS[table]) <= columns:
            legacy.append(table)
    return legacy


def _source_tree_version(module: Any) -> Optional[str]:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None

    for parent in Path(module_file).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            data = tomllib.loads(pyproject.read_text())
        except Exception:
            continue
        project = data.get("project", {})
        if project.get("name") == "iris-vector-graph":
            version = project.get("version")
            return str(version) if version else None
    return None


def _version_key(version: str) -> Sequence[int]:
    parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _has_4x_surface(module: Any) -> bool:
    """The engine's public names plus the 4.0 layout migration, which 2.x lacks."""
    required_attrs = (
        "IRISGraphEngine",
        "GraphStore",
        "IVGResult",
        "EmbedSelector",
        "IndexConfig",
        "EngineStatus",
    )
    if not all(hasattr(module, attr) for attr in required_attrs):
        return False
    try:
        migrations = importlib.import_module(f"{module.__name__}.migrations")
    except ImportError:
        return False
    return hasattr(migrations, "upgrade_to_4_0_0")
