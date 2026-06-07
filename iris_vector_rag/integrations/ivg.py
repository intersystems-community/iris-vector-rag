"""Compatibility helpers for iris-vector-graph integration."""

from __future__ import annotations

import importlib
import re
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_IVG_VERSION = "2.0.0"
IVG_SCHEMA = "Graph_KG"
IVG_REQUIRED_TABLES: Tuple[str, ...] = (
    "nodes",
    "rdf_labels",
    "rdf_props",
    "rdf_edges",
    "kg_NodeEmbeddings",
    "kg_NodeEmbeddings_optimized",
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
    """Validate the imported IVG module is new enough for IVR's 2.x integration."""
    module = module or import_ivg_module()
    installed_version = get_ivg_version(module)
    if installed_version and _version_key(installed_version) >= _version_key(
        minimum_version
    ):
        return installed_version

    if _has_v2_surface(module) and not installed_version:
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
    """Create an IVG engine using the 2.x-compatible constructor contract."""
    module = import_ivg_module()
    assert_ivg_compatible(module)
    engine_cls = getattr(module, "IRISGraphEngine")
    if embedding_dimension is not None:
        kwargs.setdefault("embedding_dimension", int(embedding_dimension))
    return engine_cls(connection, **kwargs)


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


def _has_v2_surface(module: Any) -> bool:
    required_attrs = (
        "IRISGraphEngine",
        "GraphStore",
        "IVGResult",
        "EmbedSelector",
        "IndexConfig",
        "EngineStatus",
    )
    return all(hasattr(module, attr) for attr in required_attrs)
