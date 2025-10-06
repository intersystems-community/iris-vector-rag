"""
General-Purpose Ontology Plugin System for IRIS RAG

This module provides a dynamic, domain-agnostic ontology plugin system that can
work with ANY ontology format and domain without hardcoded assumptions.

Key Features:
- Single general-purpose plugin that works with any domain
- Dynamic ontology loading from OWL, RDF, SKOS formats
- Auto-detection of domain from ontology metadata
- Support for custom domain definitions via configuration
- No hardcoded domain-specific plugins or assumptions

Available Plugin:
- GeneralOntologyPlugin: Universal ontology plugin for any domain
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .general_ontology import DomainConfiguration, GeneralOntologyPlugin

logger = logging.getLogger(__name__)

__all__ = [
    "GeneralOntologyPlugin",
    "DomainConfiguration",
    "get_ontology_plugin",
    "create_ontology_plugin",
    "load_custom_domain_definition",
    "list_supported_formats",
]


def get_ontology_plugin(
    ontology_source: Optional[str] = None,
    domain_config: Optional[Dict[str, Any]] = None,
) -> GeneralOntologyPlugin:
    """
    Get a general-purpose ontology plugin instance.

    Args:
        ontology_source: Path to ontology file or None for empty plugin
        domain_config: Optional custom domain configuration

    Returns:
        GeneralOntologyPlugin instance
    """
    # Convert domain config dict to DomainConfiguration if provided
    domain_configuration = None
    if domain_config:
        domain_configuration = DomainConfiguration(**domain_config)

    plugin = GeneralOntologyPlugin(
        ontology_path=ontology_source or "", domain_config=domain_configuration
    )

    # Load ontology if source provided
    if ontology_source and Path(ontology_source).exists():
        try:
            plugin.load_ontology_from_file(ontology_source)
            logger.info(f"Loaded ontology from {ontology_source}")
        except Exception as e:
            logger.error(f"Failed to load ontology from {ontology_source}: {e}")

    return plugin


def create_ontology_plugin(
    ontology_sources: List[Dict[str, Any]], auto_detect_domain: bool = True
) -> GeneralOntologyPlugin:
    """
    Create ontology plugin from multiple sources.

    Args:
        ontology_sources: List of ontology source configurations
        auto_detect_domain: Whether to auto-detect domain from ontologies

    Returns:
        GeneralOntologyPlugin with loaded ontologies

    Example:
        sources = [
            {"type": "owl", "path": "medical.owl"},
            {"type": "skos", "path": "terms.skos"}
        ]
        plugin = create_ontology_plugin(sources)
    """
    plugin = GeneralOntologyPlugin()
    plugin.auto_detect_domain = auto_detect_domain

    # Load from multiple sources
    for source_config in ontology_sources:
        source_path = source_config.get("path")
        source_type = source_config.get("type", "auto")

        if source_path and Path(source_path).exists():
            try:
                hierarchy = plugin.load_ontology_from_file(source_path, source_type)
                logger.info(f"Loaded {source_type} ontology from {source_path}")
            except Exception as e:
                logger.error(
                    f"Failed to load {source_type} ontology from {source_path}: {e}"
                )

    return plugin


def load_custom_domain_definition(definition_path: str) -> DomainConfiguration:
    """
    Load custom domain definition from JSON file.

    Args:
        definition_path: Path to JSON file containing domain definition

    Returns:
        DomainConfiguration object

    Example JSON format:
    {
        "domain_name": "biomedical",
        "description": "Biomedical research domain",
        "entity_types": {
            "PROTEIN": ["protein", "enzyme", "antibody"],
            "GENE": ["gene", "allele", "locus"]
        },
        "extraction_patterns": {
            "PROTEIN": ["\\b[A-Z][a-z]+\\d+\\b"]
        },
        "synonyms": {
            "protein": ["enzyme", "polypeptide"]
        }
    }
    """
    try:
        with open(definition_path, "r", encoding="utf-8") as f:
            definition_data = json.load(f)

        return DomainConfiguration(**definition_data)

    except Exception as e:
        logger.error(f"Failed to load domain definition from {definition_path}: {e}")
        raise


def create_domain_specific_plugin(
    domain_name: str,
    ontology_sources: List[Dict[str, Any]],
    custom_mappings: Optional[Dict[str, List[str]]] = None,
    custom_patterns: Optional[Dict[str, List[str]]] = None,
) -> GeneralOntologyPlugin:
    """
    Create a domain-specific plugin with custom configuration.

    Args:
        domain_name: Name of the domain
        ontology_sources: List of ontology source configurations
        custom_mappings: Custom entity type mappings
        custom_patterns: Custom extraction patterns

    Returns:
        Configured GeneralOntologyPlugin
    """
    # Create domain configuration
    domain_config = DomainConfiguration(
        domain_name=domain_name,
        entity_types=custom_mappings or {},
        extraction_patterns=custom_patterns or {},
    )

    # Create plugin with domain config
    plugin = GeneralOntologyPlugin(domain_config=domain_config)

    # Load ontology sources
    for source_config in ontology_sources:
        source_path = source_config.get("path")
        source_type = source_config.get("type", "auto")

        if source_path and Path(source_path).exists():
            try:
                plugin.load_ontology_from_file(source_path, source_type)
            except Exception as e:
                logger.error(f"Failed to load ontology from {source_path}: {e}")

    return plugin


def list_supported_formats() -> List[str]:
    """
    List all supported ontology formats.

    Returns:
        List of supported format names
    """
    return ["owl", "rdf", "skos", "xml", "ttl", "n3"]


def validate_ontology_source(source_config: Dict[str, Any]) -> bool:
    """
    Validate an ontology source configuration.

    Args:
        source_config: Source configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["path"]

    # Check required fields
    for field in required_fields:
        if field not in source_config:
            logger.error(f"Missing required field '{field}' in ontology source config")
            return False

    # Check file exists
    source_path = source_config["path"]
    if not Path(source_path).exists():
        logger.error(f"Ontology file not found: {source_path}")
        return False

    # Check format if specified
    source_type = source_config.get("type", "auto")
    if source_type != "auto" and source_type not in list_supported_formats():
        logger.error(f"Unsupported ontology format: {source_type}")
        return False

    return True


def create_plugin_from_config(config: Dict[str, Any]) -> GeneralOntologyPlugin:
    """
    Create ontology plugin from configuration dictionary.

    Args:
        config: Configuration dictionary with ontology settings

    Returns:
        Configured GeneralOntologyPlugin

    Example config:
    {
        "type": "general",
        "auto_detect_domain": true,
        "sources": [
            {"type": "owl", "path": "ontology.owl"}
        ],
        "custom_domains": {
            "enabled": true,
            "definition_path": "custom_domain.json"
        }
    }
    """
    # Extract configuration values
    auto_detect = config.get("auto_detect_domain", True)
    sources = config.get("sources", [])
    custom_domains_config = config.get("custom_domains", {})

    # Load custom domain definition if specified
    domain_config = None
    if custom_domains_config.get("enabled", False):
        definition_path = custom_domains_config.get("definition_path")
        if definition_path and Path(definition_path).exists():
            try:
                domain_config = load_custom_domain_definition(definition_path)
            except Exception as e:
                logger.warning(f"Failed to load custom domain definition: {e}")

    # Create plugin
    plugin = GeneralOntologyPlugin(domain_config=domain_config)
    plugin.auto_detect_domain = auto_detect

    # Load ontology sources
    for source_config in sources:
        if validate_ontology_source(source_config):
            try:
                source_path = source_config["path"]
                source_type = source_config.get("type", "auto")
                plugin.load_ontology_from_file(source_path, source_type)
            except Exception as e:
                logger.error(f"Failed to load ontology source: {e}")

    return plugin


# Legacy compatibility functions (for backward compatibility during transition)
def list_available_domains() -> List[str]:
    """
    List available domains (legacy compatibility).

    Note: In the general-purpose system, domains are auto-detected
    from loaded ontologies rather than hardcoded.

    Returns:
        List containing "general" as the universal domain
    """
    logger.warning(
        "list_available_domains() is deprecated. "
        "Use auto-detection or custom domain definitions instead."
    )
    return ["general"]


# Maintain backward compatibility with old function signature
def get_ontology_plugin_legacy(domain: str) -> Optional[GeneralOntologyPlugin]:
    """
    Legacy function for backward compatibility.

    Args:
        domain: Domain name (ignored in general-purpose system)

    Returns:
        GeneralOntologyPlugin instance
    """
    logger.warning(
        f"get_ontology_plugin('{domain}') is deprecated. "
        "Use get_ontology_plugin() with ontology sources instead."
    )
    return GeneralOntologyPlugin()
