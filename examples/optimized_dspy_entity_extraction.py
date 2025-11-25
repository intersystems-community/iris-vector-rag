"""
Example: Using Pre-Optimized DSPy Programs for Entity Extraction

This example demonstrates how to use the new optimized_program_path parameter
to load pre-trained DSPy programs for improved entity extraction accuracy.

Feature: 063-dspy-optimization
"""

from pathlib import Path
from iris_vector_rag.services.entity_extraction import OntologyAwareEntityExtractor
from iris_vector_rag.config.manager import ConfigurationManager
from langchain_core.documents import Document


def example_with_optimized_program():
    """Example: Load and use a pre-optimized DSPy program."""

    # 1. Configure the entity extraction service
    config = {
        "entity_extraction": {
            "method": "ontology_hybrid",
            "confidence_threshold": 0.7,
            "entity_types": ["PERSON", "ORG", "LOCATION"],
            "max_entities": 100
        },
        "ontology": {
            "enabled": False
        },
        "llm": {
            "model": "gpt-4o-mini",  # Or your preferred model
            "provider": "openai"
        }
    }

    config_manager = ConfigurationManager(config)

    # 2. Initialize with optimized program path (NEW FEATURE!)
    optimized_path = "entity_extractor_optimized.json"

    extractor = OntologyAwareEntityExtractor(
        config_manager=config_manager,
        optimized_program_path=optimized_path  # <-- NEW optional parameter
    )

    print(f"✓ Initialized with optimized program: {optimized_path}")
    print(f"  Stored path: {extractor.optimized_program_path}")

    # 3. When you call extract_batch_with_dspy, it will automatically load
    #    the optimized program if the file exists

    documents = [
        Document(
            page_content="John Doe works at Acme Corporation in New York.",
            metadata={"id": "doc1"}
        ),
        Document(
            page_content="Jane Smith is the CEO of Tech Industries based in San Francisco.",
            metadata={"id": "doc2"}
        )
    ]

    print("\nExtracting entities from documents...")

    # This will use the optimized DSPy program (if file exists)
    try:
        results = extractor.extract_batch_with_dspy(
            documents,
            entity_types=["PERSON", "ORG", "LOCATION"]
        )

        print(f"✓ Extracted entities from {len(results)} documents")
        for doc_id, entities in results.items():
            print(f"\n  Document {doc_id}:")
            for entity in entities:
                print(f"    - {entity.text} ({entity.entity_type})")

    except Exception as e:
        print(f"Note: Extraction failed (likely missing file): {e}")
        print("This is expected if entity_extractor_optimized.json doesn't exist")


def example_without_optimized_program():
    """Example: Backward compatible - works without optimized_program_path."""

    config = {
        "entity_extraction": {
            "method": "ontology_hybrid",
            "confidence_threshold": 0.7,
            "entity_types": ["PERSON", "ORG"],
            "max_entities": 100
        },
        "ontology": {
            "enabled": False
        }
    }

    config_manager = ConfigurationManager(config)

    # Initialize WITHOUT optimized program (backward compatible)
    extractor = OntologyAwareEntityExtractor(
        config_manager=config_manager
        # optimized_program_path NOT specified - uses default extraction
    )

    print("\n✓ Initialized without optimized program (backward compatible)")
    print(f"  Optimized path: {extractor.optimized_program_path}")  # Will be None


def example_graceful_fallback():
    """Example: Graceful fallback when file doesn't exist."""

    config = {
        "entity_extraction": {
            "method": "ontology_hybrid",
            "confidence_threshold": 0.7,
            "entity_types": ["PERSON"],
            "max_entities": 100
        },
        "ontology": {
            "enabled": False
        }
    }

    config_manager = ConfigurationManager(config)

    # Specify a file that doesn't exist
    extractor = OntologyAwareEntityExtractor(
        config_manager=config_manager,
        optimized_program_path="nonexistent_file.json"  # File doesn't exist
    )

    print("\n✓ Initialized with nonexistent file (graceful fallback)")
    print(f"  When extract_batch_with_dspy runs, it will:")
    print(f"  1. Check if file exists")
    print(f"  2. Log a warning if it doesn't")
    print(f"  3. Continue with standard extraction (no errors!)")


if __name__ == "__main__":
    print("=" * 70)
    print("DSPy Optimized Program Loading - Examples")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Example 1: Using Pre-Optimized DSPy Program")
    print("=" * 70)
    example_with_optimized_program()

    print("\n" + "=" * 70)
    print("Example 2: Backward Compatible (No Optimization)")
    print("=" * 70)
    example_without_optimized_program()

    print("\n" + "=" * 70)
    print("Example 3: Graceful Fallback")
    print("=" * 70)
    example_graceful_fallback()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key Benefits:
  ✓ Library-First Design: Clean API parameter (not environment variable)
  ✓ Backward Compatible: Optional parameter with None default
  ✓ Graceful Fallback: Clear logging when optimization unavailable
  ✓ Easy to Use: Just pass the path to your optimized program

Usage:
  extractor = OntologyAwareEntityExtractor(
      config_manager=config_manager,
      optimized_program_path="entity_extractor_optimized.json"
  )

For more information, see:
  - DSPY_OPTIMIZATION_INTEGRATION.md in hipporag2-pipeline
  - Contract tests: tests/contract/test_optimized_program_loading.py
""")
