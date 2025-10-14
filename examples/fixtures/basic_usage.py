#!/usr/bin/env python3
"""
Basic usage examples for FixtureManager.

This script demonstrates common fixture management operations:
1. Scanning for available fixtures
2. Loading fixtures into IRIS
3. Generating embeddings
4. Cleanup

Prerequisites:
- IRIS database running (docker-compose up -d)
- iris-devtools installed
- IRIS_PORT environment variable set

Run:
    IRIS_PORT=21972 python examples/fixtures/basic_usage.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.fixtures.manager import FixtureManager
from common.iris_dbapi_connector import get_iris_dbapi_connection


def print_header(message: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {message}")
    print(f"{'='*70}\n")


def example_1_scan_fixtures():
    """Example 1: Scan for available fixtures."""
    print_header("Example 1: Scan for Available Fixtures")

    manager = FixtureManager()

    # Scan fixtures directory
    manifest = manager.scan_fixtures(rescan=True)

    print(f"Found {len(manifest.fixtures)} fixture(s):\n")

    for name, metadata in manifest.fixtures.items():
        print(f"Name:         {metadata.name}")
        print(f"Version:      {metadata.version}")
        print(f"Description:  {metadata.description}")
        print(f"Source Type:  {metadata.source_type}")
        print(f"Tables:       {', '.join(metadata.tables)}")
        print(f"Total Rows:   {sum(metadata.row_counts.values())}")
        print()


def example_2_list_and_filter():
    """Example 2: List and filter fixtures."""
    print_header("Example 2: List and Filter Fixtures")

    manager = FixtureManager()
    manager.scan_fixtures()

    # List all fixtures
    all_fixtures = manager.list_fixtures()
    print(f"Total fixtures: {len(all_fixtures)}\n")

    # Filter by source type
    dat_fixtures = manager.list_fixtures(filter_by={"source_type": "dat"})
    print(f".DAT fixtures: {len(dat_fixtures)}")
    for fixture in dat_fixtures:
        print(f"  - {fixture.name} v{fixture.version}")

    # Filter by embeddings requirement
    embedding_fixtures = manager.list_fixtures(filter_by={"requires_embeddings": True})
    print(f"\nFixtures with embeddings: {len(embedding_fixtures)}")
    for fixture in embedding_fixtures:
        print(f"  - {fixture.name} (model: {fixture.embedding_model}, dim: {fixture.embedding_dimension})")


def example_3_get_fixture_info():
    """Example 3: Get detailed fixture information."""
    print_header("Example 3: Get Fixture Information")

    manager = FixtureManager()
    manager.scan_fixtures()

    fixtures = manager.list_fixtures()
    if not fixtures:
        print("No fixtures available. Create fixtures first.")
        return

    # Get first available fixture
    fixture_name = fixtures[0].name
    metadata = manager.get_fixture(fixture_name)

    if metadata:
        print(f"Fixture: {metadata.name}")
        print(f"Version: {metadata.version}")
        print(f"Description: {metadata.description}")
        print(f"Created: {metadata.created_at}")
        print(f"Created By: {metadata.created_by}")
        print(f"Checksum: {metadata.checksum}")
        print(f"Namespace: {metadata.namespace}")
        print(f"\nTable Details:")
        for table, count in metadata.row_counts.items():
            print(f"  {table:<40} {count:>6} rows")


def example_4_load_fixture():
    """Example 4: Load fixture into IRIS."""
    print_header("Example 4: Load Fixture into IRIS")

    try:
        # Verify IRIS connection
        conn = get_iris_dbapi_connection()
        print(" IRIS connection successful\n")
    except Exception as e:
        print(f" IRIS connection failed: {e}")
        print("\nMake sure IRIS is running:")
        print("  docker-compose up -d")
        print("  export IRIS_PORT=21972")
        return

    manager = FixtureManager(connection=conn)
    manager.scan_fixtures()

    fixtures = manager.list_fixtures()
    if not fixtures:
        print("No fixtures available. Create fixtures first.")
        return

    fixture_name = fixtures[0].name
    print(f"Loading fixture: {fixture_name}\n")

    # Load fixture
    result = manager.load_fixture(
        fixture_name=fixture_name,
        cleanup_first=True,
        validate_checksum=True,
        generate_embeddings=False,
    )

    if result.success:
        print(result.summary())
        print(f"\n Fixture loaded successfully!")
    else:
        print(f" Fixture loading failed: {result.error_message}")


def example_5_load_with_embeddings():
    """Example 5: Load fixture and generate embeddings."""
    print_header("Example 5: Load Fixture with Embeddings")

    try:
        conn = get_iris_dbapi_connection()
    except Exception as e:
        print(f" IRIS connection failed: {e}")
        return

    manager = FixtureManager(connection=conn)
    manager.scan_fixtures()

    # Find fixture that requires embeddings
    fixtures = manager.list_fixtures(filter_by={"requires_embeddings": True})
    if not fixtures:
        print("No fixtures with embeddings available.")
        return

    fixture_name = fixtures[0].name
    print(f"Loading fixture: {fixture_name}")
    print(f"Embedding model: {fixtures[0].embedding_model}")
    print(f"Embedding dimension: {fixtures[0].embedding_dimension}\n")

    # Load with embedding generation
    result = manager.load_fixture(
        fixture_name=fixture_name,
        cleanup_first=True,
        generate_embeddings=True,
    )

    if result.success:
        print(f" Fixture loaded in {result.load_time_seconds:.2f}s")
        print(f" {result.rows_loaded} rows loaded")
        print(f" Embeddings generated for:")
        for table in result.tables_loaded:
            print(f"  - {table}")
    else:
        print(f" Failed: {result.error_message}")


def example_6_cleanup():
    """Example 6: Cleanup fixture data."""
    print_header("Example 6: Cleanup Fixture Data")

    try:
        conn = get_iris_dbapi_connection()
    except Exception as e:
        print(f" IRIS connection failed: {e}")
        return

    manager = FixtureManager(connection=conn)

    # Define tables to cleanup
    tables = [
        "RAG.SourceDocuments",
        "RAG.DocumentChunks",
        "RAG.Entities",
        "RAG.EntityRelationships",
    ]

    print("Cleaning up tables:")
    for table in tables:
        print(f"  - {table}")

    deleted = manager.cleanup_fixture_data(tables)

    print(f"\n Deleted {deleted} total rows")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  FixtureManager - Usage Examples")
    print("="*70)

    try:
        example_1_scan_fixtures()
        example_2_list_and_filter()
        example_3_get_fixture_info()
        example_4_load_fixture()
        # example_5_load_with_embeddings()  # Uncomment if you want embedding generation
        # example_6_cleanup()  # Uncomment to cleanup after examples

        print("\n" + "="*70)
        print("  All examples completed successfully!")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
