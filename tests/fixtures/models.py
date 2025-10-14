"""
Data models for fixture infrastructure.

Defines core data structures for fixture metadata, manifests, and load results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class FixtureSourceType(Enum):
    """Types of fixture sources."""

    DAT = "dat"  # iris-devtools .DAT binary format
    JSON = "json"  # JSON document files
    PROGRAMMATIC = "prog"  # Python code generation


@dataclass
class FixtureMetadata:
    """Metadata describing a test fixture."""

    # Identity
    name: str
    version: str
    description: str

    # Provenance
    created_at: str  # ISO 8601 format
    created_by: str  # "iris-devtools", "manual", "script"
    source_type: str  # Will be converted to FixtureSourceType

    # Content
    tables: List[str]
    row_counts: Dict[str, int]
    checksum: str  # SHA256 hash

    # Dependencies
    requires_embeddings: bool = False
    embedding_model: Optional[str] = None
    embedding_dimension: int = 384

    # IRIS-specific
    namespace: str = "USER"
    iris_version_min: Optional[str] = None
    backend_mode: Optional[str] = None

    # Versioning and migration (T089)
    schema_version: str = "1.0"  # Manifest format version
    migration_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "source_type": self.source_type,
            "tables": self.tables,
            "row_counts": self.row_counts,
            "checksum": self.checksum,
            "requires_embeddings": self.requires_embeddings,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "namespace": self.namespace,
            "iris_version_min": self.iris_version_min,
            "backend_mode": self.backend_mode,
            "schema_version": self.schema_version,
            "migration_history": self.migration_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FixtureMetadata":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class FixtureManifest:
    """Registry of all available fixtures."""

    version: str = "1.0.0"  # Manifest format version
    fixtures: Dict[str, FixtureMetadata] = field(default_factory=dict)

    def register(self, metadata: FixtureMetadata) -> None:
        """Register a new fixture."""
        self.fixtures[metadata.name] = metadata

    def get(
        self, name: str, version: Optional[str] = None
    ) -> Optional[FixtureMetadata]:
        """
        Get fixture by name and optional version.

        Args:
            name: Fixture name
            version: Semantic version or None for latest

        Returns:
            FixtureMetadata or None if not found
        """
        if name not in self.fixtures:
            return None

        fixture = self.fixtures[name]

        # If version specified, check match
        if version and fixture.version != version:
            return None

        return fixture

    def list_fixtures(
        self, filter_by: Optional[Dict[str, Any]] = None
    ) -> List[FixtureMetadata]:
        """
        List all fixtures with optional filtering.

        Args:
            filter_by: Optional filters (e.g., {"source_type": "dat"})

        Returns:
            List of FixtureMetadata matching filters
        """
        fixtures = list(self.fixtures.values())

        if not filter_by:
            return fixtures

        # Apply filters
        filtered = []
        for fixture in fixtures:
            match = True
            for key, value in filter_by.items():
                if not hasattr(fixture, key):
                    match = False
                    break
                if getattr(fixture, key) != value:
                    match = False
                    break
            if match:
                filtered.append(fixture)

        return filtered

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "fixtures": {
                name: metadata.to_dict()
                for name, metadata in self.fixtures.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FixtureManifest":
        """Deserialize from dictionary."""
        manifest = cls(version=data.get("version", "1.0.0"))
        for name, metadata_dict in data.get("fixtures", {}).items():
            metadata = FixtureMetadata.from_dict(metadata_dict)
            manifest.register(metadata)
        return manifest

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FixtureManifest":
        """Load manifest from JSON file."""
        if not path.exists():
            return cls()  # Return empty manifest

        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class FixtureLoadResult:
    """Result of a fixture load operation."""

    # Identity
    fixture_name: str
    fixture_version: str

    # Performance
    load_time_seconds: float
    rows_loaded: int

    # Validation
    checksum_valid: bool
    tables_loaded: List[str]
    tables_failed: List[str] = field(default_factory=list)

    # Status
    success: bool = True
    error_message: Optional[str] = None

    def summary(self) -> str:
        """Human-readable summary."""
        if self.success:
            return (
                f"Fixture '{self.fixture_name}' v{self.fixture_version} loaded successfully:\n"
                f"  Time: {self.load_time_seconds:.2f}s\n"
                f"  Rows: {self.rows_loaded}\n"
                f"  Tables: {', '.join(self.tables_loaded)}"
            )
        else:
            return (
                f"Fixture '{self.fixture_name}' v{self.fixture_version} FAILED:\n"
                f"  Error: {self.error_message}\n"
                f"  Tables loaded: {', '.join(self.tables_loaded)}\n"
                f"  Tables failed: {', '.join(self.tables_failed)}"
            )


@dataclass
class MigrationResult:
    """
    Result of a fixture migration operation.

    Reference: T089 - Implement FixtureManager.migrate() for schema migration
    """

    # Migration details
    success: bool
    old_version: str
    new_version: str
    changes_applied: List[str]

    # Performance
    migration_time: float

    # Error handling
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging."""
        return {
            "success": self.success,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "changes_applied": self.changes_applied,
            "migration_time": self.migration_time,
            "error_message": self.error_message,
        }
