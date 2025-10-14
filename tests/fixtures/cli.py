#!/usr/bin/env python
"""
Command-line interface for test fixture management.

Provides a user-friendly CLI for loading, creating, and managing test fixtures.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import time

from .manager import FixtureManager, FixtureError
from .models import FixtureSourceType


def print_header(message: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {message}")
    print(f"{'='*70}\n")


def print_success(message: str):
    """Print success message."""
    print(f"[OK] {message}")


def print_error(message: str):
    """Print error message."""
    print(f"[ERROR] {message}", file=sys.stderr)


def print_warning(message: str):
    """Print warning message."""
    print(f"[WARN] {message}")


def cmd_list(args, manager: FixtureManager):
    """List all available fixtures."""
    print_header("Available Test Fixtures")

    # Scan for fixtures
    manifest = manager.scan_fixtures(rescan=True)

    if not manifest.fixtures:
        print_warning("No fixtures found")
        return

    # Apply filters
    filter_by = {}
    if args.source_type:
        filter_by["source_type"] = args.source_type
    if args.requires_embeddings is not None:
        filter_by["requires_embeddings"] = args.requires_embeddings

    fixtures = manifest.list_fixtures(filter_by=filter_by if filter_by else None)

    if not fixtures:
        print_warning("No fixtures match the specified filters")
        return

    # Print table header
    print(f"{'Name':<30} {'Version':<10} {'Type':<8} {'Tables':<15} {'Rows':<8} {'Embeddings'}")
    print("-" * 95)

    # Print each fixture
    for fixture in fixtures:
        total_rows = sum(fixture.row_counts.values())
        tables_str = f"{len(fixture.tables)} tables"
        embeddings_str = "Required" if fixture.requires_embeddings else "-"

        print(f"{fixture.name:<30} {fixture.version:<10} {fixture.source_type:<8} "
              f"{tables_str:<15} {total_rows:<8} {embeddings_str}")

    print(f"\nTotal: {len(fixtures)} fixture(s)")


def cmd_info(args, manager: FixtureManager):
    """Show detailed fixture information."""
    print_header(f"Fixture Information: {args.fixture_name}")

    # Get fixture metadata
    metadata = manager.get_fixture(args.fixture_name, version=args.version)

    if metadata is None:
        print_error(f"Fixture '{args.fixture_name}' not found")
        sys.exit(1)

    # Print detailed information
    print(f"Name:         {metadata.name}")
    print(f"Version:      {metadata.version}")
    print(f"Description:  {metadata.description}")
    print(f"Source Type:  {metadata.source_type}")
    print(f"Namespace:    {metadata.namespace}")
    print(f"Created:      {metadata.created_at}")
    print(f"Created By:   {metadata.created_by}")
    print(f"Checksum:     {metadata.checksum}")
    print()

    print("Tables:")
    for table, count in metadata.row_counts.items():
        print(f"  - {table:<40} {count:>6} rows")

    total_rows = sum(metadata.row_counts.values())
    print(f"\nTotal rows: {total_rows}")

    if metadata.requires_embeddings:
        print(f"\nEmbeddings:   Required")
        print(f"  Model:      {metadata.embedding_model or 'default'}")
        print(f"  Dimension:  {metadata.embedding_dimension}")


def cmd_validate(args, manager: FixtureManager):
    """Validate fixture integrity."""
    print_header(f"Validating Fixture: {args.fixture_name}")

    try:
        # Get fixture metadata
        metadata = manager.get_fixture(args.fixture_name, version=args.version)

        if metadata is None:
            print_error(f"Fixture '{args.fixture_name}' not found")
            sys.exit(1)

        # Get fixture path
        fixture_dir = manager._get_fixture_path(metadata)
        if not fixture_dir.exists():
            print_error(f"Fixture directory not found: {fixture_dir}")
            sys.exit(1)

        print_success("Fixture directory exists")

        # Validate manifest
        manifest_file = fixture_dir / "manifest.json"
        if manifest_file.exists():
            print_success("Manifest file exists")
        else:
            print_error("Manifest file missing")
            sys.exit(1)

        # Validate checksum
        try:
            manager._validate_checksum(fixture_dir, metadata)
            print_success(f"Checksum valid: {metadata.checksum}")
        except Exception as e:
            print_error(f"Checksum validation failed: {e}")
            sys.exit(1)

        # Validate row counts
        print_success(f"Row counts: {sum(metadata.row_counts.values())} total rows")

        print()
        print_success("Fixture validation passed")

    except FixtureError as e:
        print_error(str(e))
        sys.exit(1)


def cmd_load(args, manager: FixtureManager):
    """Load fixture into database."""
    print_header(f"Loading Fixture: {args.fixture_name}")

    try:
        result = manager.load_fixture(
            fixture_name=args.fixture_name,
            version=args.version,
            validate_checksum=not args.no_validate_checksum,
            cleanup_first=args.cleanup_first,
            generate_embeddings=args.generate_embeddings,
        )

        if result.success:
            print()
            print(result.summary())
            print()
            print_success(f"Fixture loaded in {result.load_time_seconds:.2f} seconds")
        else:
            print_error(f"Fixture loading failed: {result.error_message}")
            sys.exit(1)

    except FixtureError as e:
        print_error(str(e))
        sys.exit(1)


def cmd_workflow(args, manager: FixtureManager):
    """Interactive fixture creation workflow."""
    print_header("Interactive Fixture Creation")

    # Get fixture name
    print("Fixture name:")
    fixture_name = input("> ").strip()
    if not fixture_name:
        print_error("Fixture name is required")
        sys.exit(1)

    # Get description
    print("\nDescription:")
    description = input("> ").strip()

    # Get version
    print("\nVersion [1.0.0]:")
    version = input("> ").strip() or "1.0.0"

    # Get tables
    print("\nTables (comma-separated):")
    print("Example: RAG.SourceDocuments,RAG.Entities,RAG.EntityRelationships")
    tables_str = input("> ").strip()
    if not tables_str:
        print_error("At least one table is required")
        sys.exit(1)
    tables = [t.strip() for t in tables_str.split(",")]

    # Get embeddings preference
    print("\nGenerate embeddings? [y/N]:")
    gen_embeddings = input("> ").strip().lower() == "y"

    embedding_model = None
    embedding_dimension = 384

    if gen_embeddings:
        print("\nEmbedding model [all-MiniLM-L6-v2]:")
        embedding_model = input("> ").strip() or "all-MiniLM-L6-v2"

        print("\nEmbedding dimension [384]:")
        dim_str = input("> ").strip()
        if dim_str:
            try:
                embedding_dimension = int(dim_str)
            except ValueError:
                print_error("Invalid dimension value")
                sys.exit(1)

    # Preview configuration
    print_header("Preview Configuration")
    print(f"Name:         {fixture_name}")
    print(f"Version:      {version}")
    print(f"Description:  {description}")
    print(f"Tables:       {', '.join(tables)}")
    print(f"Embeddings:   {'Yes' if gen_embeddings else 'No'}")
    if gen_embeddings:
        print(f"  Model:      {embedding_model}")
        print(f"  Dimension:  {embedding_dimension}")
    print()

    # Confirm
    print("Proceed? [Y/n]:")
    proceed = input("> ").strip().lower()
    if proceed == "n":
        print("Cancelled")
        sys.exit(0)

    # Create fixture
    print_header("Creating Fixture")

    try:
        metadata = manager.create_fixture(
            name=fixture_name,
            tables=tables,
            description=description,
            version=version,
            generate_embeddings=gen_embeddings,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

        print()
        print_success(f"Fixture '{metadata.name}' v{metadata.version} created successfully!")
        print(f"\nLocation: tests/fixtures/dat/{metadata.name}/")
        print(f"To use: make fixture-load FIXTURE={metadata.name}")

    except FixtureError as e:
        print_error(str(e))
        sys.exit(1)


def cmd_create(args, manager: FixtureManager):
    """Create fixture from command-line arguments."""
    print_header(f"Creating Fixture: {args.fixture_name}")

    if not args.tables:
        print_error("At least one table is required (--tables)")
        sys.exit(1)

    tables = [t.strip() for t in args.tables.split(",")]

    try:
        metadata = manager.create_fixture(
            name=args.fixture_name,
            tables=tables,
            description=args.description or f"Test fixture: {args.fixture_name}",
            version=args.version or "1.0.0",
            generate_embeddings=args.generate_embeddings,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
        )

        print()
        print_success(f"Fixture '{metadata.name}' v{metadata.version} created successfully!")
        print(f"\nLocation: tests/fixtures/dat/{metadata.name}/")
        print(f"To use: make fixture-load FIXTURE={metadata.name}")

    except FixtureError as e:
        print_error(str(e))
        sys.exit(1)


def cmd_snapshot(args, manager: FixtureManager):
    """Create quick database snapshot."""
    print_header(f"Creating Snapshot: {args.fixture_name}")

    # Auto-detect tables from RAG schema
    print("Auto-detecting RAG tables...")

    default_tables = [
        "RAG.SourceDocuments",
        "RAG.DocumentChunks",
        "RAG.Entities",
        "RAG.EntityRelationships",
    ]

    try:
        metadata = manager.create_fixture(
            name=args.fixture_name,
            tables=default_tables,
            description=f"Database snapshot created on {time.strftime('%Y-%m-%d %H:%M:%S')}",
            version=args.version or time.strftime("%Y%m%d.%H%M%S"),
            generate_embeddings=False,  # Snapshots don't generate embeddings
        )

        print()
        print_success(f"Snapshot '{metadata.name}' created successfully!")
        print(f"\nLocation: tests/fixtures/dat/{metadata.name}/")

    except FixtureError as e:
        print_error(str(e))
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test Fixture Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all fixtures
  python -m tests.fixtures.cli list

  # Show fixture info
  python -m tests.fixtures.cli info medical-graphrag-20

  # Load fixture
  python -m tests.fixtures.cli load medical-graphrag-20

  # Create fixture interactively
  python -m tests.fixtures.cli workflow

  # Create fixture from command line
  python -m tests.fixtures.cli create my-fixture --tables RAG.SourceDocuments,RAG.Entities

  # Quick snapshot
  python -m tests.fixtures.cli snapshot snapshot-20250114
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all available fixtures")
    list_parser.add_argument("--source-type", choices=["dat", "json", "prog"], help="Filter by source type")
    list_parser.add_argument("--requires-embeddings", action="store_true", help="Filter fixtures requiring embeddings")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed fixture information")
    info_parser.add_argument("fixture_name", help="Fixture name")
    info_parser.add_argument("--version", help="Specific version")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate fixture integrity")
    validate_parser.add_argument("fixture_name", help="Fixture name")
    validate_parser.add_argument("--version", help="Specific version")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load fixture into database")
    load_parser.add_argument("fixture_name", help="Fixture name")
    load_parser.add_argument("--version", help="Specific version")
    load_parser.add_argument("--no-validate-checksum", action="store_true", help="Skip checksum validation")
    load_parser.add_argument("--cleanup-first", action="store_true", help="Clean database before loading")
    load_parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings after loading")

    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Interactive fixture creation")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create fixture from current database")
    create_parser.add_argument("fixture_name", help="Fixture name")
    create_parser.add_argument("--tables", required=True, help="Comma-separated list of tables")
    create_parser.add_argument("--description", help="Fixture description")
    create_parser.add_argument("--version", help="Semantic version (default: 1.0.0)")
    create_parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings")
    create_parser.add_argument("--embedding-model", help="Embedding model name")
    create_parser.add_argument("--embedding-dimension", type=int, default=384, help="Embedding dimension")

    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Quick database snapshot")
    snapshot_parser.add_argument("fixture_name", help="Snapshot name")
    snapshot_parser.add_argument("--version", help="Version (default: timestamp)")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize fixture manager
    manager = FixtureManager()

    # Execute command
    commands = {
        "list": cmd_list,
        "info": cmd_info,
        "validate": cmd_validate,
        "load": cmd_load,
        "workflow": cmd_workflow,
        "create": cmd_create,
        "snapshot": cmd_snapshot,
    }

    try:
        commands[args.command](args, manager)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
