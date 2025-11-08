"""
IRIS EMBEDDING Configuration Models.

Feature: 051-add-native-iris
Purpose: Define configuration data models for IRIS EMBEDDING integration
         with validation, type safety, and serialization support.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import sys


@dataclass
class EmbeddingConfig:
    """
    Represents IRIS %Embedding.Config table entries with model settings
    and optional entity extraction configuration.

    This configuration controls how EMBEDDING columns auto-vectorize text data
    and optionally extract entities for GraphRAG knowledge graphs.
    """

    # Required fields
    name: str
    model_name: str
    hf_cache_path: str
    python_path: str
    embedding_class: str

    # Optional fields with defaults
    description: Optional[str] = None
    enable_entity_extraction: bool = False
    entity_types: List[str] = field(default_factory=list)
    batch_size: int = 32
    device_preference: str = "auto"  # cuda, mps, cpu, auto

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got {self.batch_size}")

        # Validate device_preference
        valid_devices = ["cuda", "mps", "cpu", "auto"]
        if self.device_preference not in valid_devices:
            raise ValueError(
                f"device_preference must be one of {valid_devices}, got '{self.device_preference}'"
            )

        # Validate entity extraction configuration
        if self.enable_entity_extraction and not self.entity_types:
            raise ValueError(
                "If enable_entity_extraction=True, entity_types must not be empty"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingConfig":
        """Create EmbeddingConfig from dictionary."""
        return cls(**data)

    def to_iris_json(self) -> Dict[str, Any]:
        """
        Convert to IRIS %Embedding.Config Configuration JSON format.

        Returns dictionary suitable for storing in IRIS Configuration field.
        """
        return {
            "modelName": self.model_name,
            "hfCachePath": self.hf_cache_path,
            "pythonPath": self.python_path,
            "batchSize": self.batch_size,
            "devicePreference": self.device_preference,
            "enableEntityExtraction": self.enable_entity_extraction,
            "entityTypes": self.entity_types
        }


@dataclass
class ValidationResult:
    """
    Result of embedding configuration validation.

    Used by validate_embedding_config() to report validation status,
    errors, and warnings before creating tables with EMBEDDING columns.
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Additional validation metadata
    model_name: Optional[str] = None
    cache_path_valid: bool = False
    python_path_valid: bool = False
    device: Optional[str] = None

    def add_error(self, error: str):
        """Add an error and mark validation as invalid."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add a warning (doesn't invalidate configuration)."""
        self.warnings.append(warning)


def validate_embedding_config(config: EmbeddingConfig) -> ValidationResult:
    """
    Validate EMBEDDING configuration before table creation (FR-010).

    Performs pre-flight validation to catch configuration errors early
    and provide actionable error messages.

    Args:
        config: EmbeddingConfig to validate

    Returns:
        ValidationResult with validation status, errors, and warnings

    Validation checks:
        1. Model file exists at hf_cache_path or can be downloaded
        2. Python executable exists at python_path
        3. Required Python packages installed (sentence-transformers, torch)
        4. If entity extraction enabled, entity_types not empty
        5. Device preference is valid
    """
    result = ValidationResult(valid=True)
    result.model_name = config.model_name

    # Check 1: Validate hf_cache_path exists and is writable
    cache_path = Path(config.hf_cache_path)
    if cache_path.exists():
        if not cache_path.is_dir():
            result.add_error(
                f"INVALID_CACHE_PATH: hf_cache_path '{config.hf_cache_path}' "
                "exists but is not a directory"
            )
        elif not os.access(cache_path, os.W_OK):
            result.add_error(
                f"INVALID_CACHE_PATH: hf_cache_path '{config.hf_cache_path}' "
                "is not writable"
            )
        else:
            result.cache_path_valid = True
    else:
        # Cache path doesn't exist - that's OK, it will be created
        result.add_warning(
            f"Cache path '{config.hf_cache_path}' does not exist. "
            "It will be created when first downloading models."
        )
        result.cache_path_valid = True

    # Check 2: Validate python_path is executable
    python_path = Path(config.python_path)
    if not python_path.exists():
        result.add_error(
            f"INVALID_PYTHON_PATH: Python executable not found at '{config.python_path}'. "
            "Verify python_path in configuration."
        )
    elif not os.access(python_path, os.X_OK):
        result.add_error(
            f"INVALID_PYTHON_PATH: '{config.python_path}' exists but is not executable"
        )
    else:
        result.python_path_valid = True

    # Check 3: Verify required packages (basic check)
    # Note: Full package check would require running pip list in target Python
    try:
        import sentence_transformers
        import torch
        result.add_warning(
            "Package validation: sentence-transformers and torch found in current environment. "
            "Ensure they are also installed in python_path environment."
        )
    except ImportError as e:
        result.add_error(
            f"MISSING_DEPENDENCIES: Required Python packages missing in current environment: {e}. "
            "Run: pip install sentence-transformers torch"
        )

    # Check 4: Validate device preference
    if config.device_preference not in ["cuda", "mps", "cpu", "auto"]:
        result.add_error(
            f"INVALID_DEVICE: device_preference must be one of [cuda, mps, cpu, auto], "
            f"got '{config.device_preference}'"
        )
    else:
        # Detect actual device that will be used
        try:
            import torch
            if config.device_preference == "auto":
                if torch.cuda.is_available():
                    result.device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    result.device = "mps"
                else:
                    result.device = "cpu"
            else:
                result.device = config.device_preference
        except ImportError:
            result.device = "unknown (torch not available)"

    # Check 5: Validate entity extraction configuration
    if config.enable_entity_extraction:
        if not config.entity_types:
            result.add_error(
                "INVALID_ENTITY_CONFIG: enable_entity_extraction=True but entity_types is empty. "
                "Provide at least one entity type (e.g., ['Disease', 'Medication'])"
            )
        else:
            # Validate entity type naming
            for entity_type in config.entity_types:
                if not entity_type or not isinstance(entity_type, str):
                    result.add_error(
                        f"INVALID_ENTITY_TYPE: Entity type must be non-empty string, got '{entity_type}'"
                    )
                elif not entity_type[0].isupper():
                    result.add_warning(
                        f"Entity type '{entity_type}' should start with uppercase letter by convention"
                    )

    # Check 6: Validate model exists or can be downloaded
    # This is a lightweight check - full validation would download the model
    model_cache_dir = cache_path / "models--" / config.model_name.replace("/", "--")
    if model_cache_dir.exists():
        result.add_warning(
            f"Model '{config.model_name}' found in cache at {model_cache_dir}"
        )
    else:
        result.add_warning(
            f"MODEL_NOT_CACHED: Model '{config.model_name}' not found in cache. "
            f"It will be downloaded from HuggingFace on first use. "
            f"To pre-download: huggingface-cli download {config.model_name}"
        )

    return result


@dataclass
class ClearCacheResult:
    """Result of cache clearing operation."""

    models_cleared: int
    memory_freed_mb: float

    def __str__(self) -> str:
        return f"Cleared {self.models_cleared} models, freed {self.memory_freed_mb:.1f}MB"


# Helper function for creating configurations
def create_embedding_config(
    name: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hf_cache_path: str = "/var/lib/huggingface",
    python_path: str = "/usr/bin/python3",
    enable_entity_extraction: bool = False,
    entity_types: Optional[List[str]] = None,
    **kwargs
) -> EmbeddingConfig:
    """
    Helper function to create EmbeddingConfig with sensible defaults.

    Args:
        name: Unique configuration identifier
        model_name: HuggingFace model ID
        hf_cache_path: Path to HuggingFace cache directory
        python_path: Path to Python executable
        enable_entity_extraction: Whether to extract entities
        entity_types: List of entity types to extract
        **kwargs: Additional fields (description, batch_size, device_preference)

    Returns:
        EmbeddingConfig instance

    Example:
        >>> config = create_embedding_config(
        ...     name="medical_embeddings",
        ...     enable_entity_extraction=True,
        ...     entity_types=["Disease", "Medication"]
        ... )
    """
    return EmbeddingConfig(
        name=name,
        model_name=model_name,
        hf_cache_path=hf_cache_path,
        python_path=python_path,
        embedding_class="%Embedding.SentenceTransformers",
        enable_entity_extraction=enable_entity_extraction,
        entity_types=entity_types or [],
        **kwargs
    )
