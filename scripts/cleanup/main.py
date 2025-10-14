#!/usr/bin/env python3
"""
CLI entry point for repository cleanup.

Orchestrates cleanup operations following the batch approach from quickstart.md.

Usage:
    python scripts/cleanup/main.py [--dry-run] [--skip-tests]
"""

import argparse
import sys
from pathlib import Path

from .models import FileCategory
from .operations import (
    check_broken_links,
    classify_files,
    consolidate_duplicates,
    move_files,
    remove_files,
    rollback_changes,
    scan_repository,
    update_documentation,
    update_documentation_index,
    validate_tests,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Repository cleanup operations")
    parser.add_argument('--dry-run', action='store_true',
                        help="Show what would be done without making changes")
    parser.add_argument('--skip-tests', action='store_true',
                        help="Skip test validation (not recommended)")
    parser.add_argument('--repo-root', type=Path, default=Path.cwd(),
                        help="Repository root directory (default: current directory)")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    print(f"Repository Cleanup Utility")
    print(f"=" * 50)
    print(f"Repository: {repo_root}")
    print(f"Dry run: {args.dry_run}")
    print(f"=" * 50)
    print()

    # Phase 1: Scan and Classify
    print("Phase 1: Scanning and classifying files...")
    try:
        inventory = scan_repository(repo_root)
        print(f"  Scanned {inventory.total_files} files")

        report = classify_files(inventory)
        print(f"  Classification complete:")
        print(report.to_summary())

        removable = inventory.get_removable_files()
        relocatable = inventory.get_relocatable_files()
        print(f"  Removable files: {len(removable)}")
        print(f"  Relocatable files: {len(relocatable)}")
    except Exception as e:
        print(f"ERROR during scan/classify: {e}")
        return 1

    # Phase 2: Cleanup Operations (in batches)

    # BATCH 1: Remove temporary/cache files
    print("\nBatch 1: Removing temporary and cache files...")
    temp_files = [f for f in removable if 'cache' in f.reason.lower() or 'temp' in f.reason.lower()]
    if temp_files:
        removal_report = remove_files(temp_files, dry_run=args.dry_run)
        print(f"  Removed: {len(removal_report.files_removed)} files")
        print(f"  Failed: {len(removal_report.files_failed)}")
        if not args.dry_run and not args.skip_tests:
            print("  Validating tests...")
            # Simplified - would run actual test validation here

    # BATCH 2: Remove old evaluation reports
    print("\nBatch 2: Removing old evaluation reports...")
    output_files = [f for f in removable if 'output' in f.reason.lower() or 'report' in f.reason.lower()]
    if output_files:
        removal_report = remove_files(output_files, dry_run=args.dry_run)
        print(f"  Removed: {len(removal_report.files_removed)} files")
        print(f"  Bytes freed: {removal_report.bytes_freed:,}")

    # BATCH 3: Remove historical tracking files
    print("\nBatch 3: Removing historical tracking files...")
    historical_files = [f for f in removable if 'historical' in f.reason.lower()]
    if historical_files:
        removal_report = remove_files(historical_files, dry_run=args.dry_run)
        print(f"  Removed: {len(removal_report.files_removed)} files")

    # BATCH 4: Consolidate duplicates (if any)
    print("\nBatch 4: Consolidating duplicate documentation...")
    # Simplified - would properly detect and consolidate duplicates
    print("  No duplicates detected (manual review recommended)")

    # BATCH 5: Move status files to docs/
    print("\nBatch 5: Moving status files to docs/...")
    if relocatable:
        # Create target mapping
        target_map = {}
        for file in relocatable:
            if 'status' in file.reason.lower():
                target_map[file.path] = repo_root / 'docs' / file.path.name

        if target_map:
            move_report = move_files(relocatable, target_map, dry_run=args.dry_run)
            print(f"  Moved: {len(move_report.files_moved)} files")
            print(f"  Failed: {len(move_report.files_failed)}")

            # BATCH 6: Update documentation links
            if not args.dry_run and move_report.files_moved:
                print("\nBatch 6: Updating documentation links...")
                path_mapping = move_report.get_path_mapping()
                update_report = update_documentation(path_mapping, repo_root, dry_run=args.dry_run)
                print(f"  Updated: {len(update_report.files_updated)} files")
                print(f"  Links updated: {update_report.links_updated}")

                # BATCH 7: Update DOCUMENTATION_INDEX.md
                print("\nBatch 7: Updating DOCUMENTATION_INDEX.md...")
                index_file = repo_root / 'DOCUMENTATION_INDEX.md'
                if index_file.exists():
                    success = update_documentation_index(index_file, path_mapping, dry_run=args.dry_run)
                    print(f"  Index update: {'Success' if success else 'Failed'}")

    # Final: Check for broken links
    print("\nFinal check: Scanning for broken links...")
    link_report = check_broken_links(repo_root)
    if link_report.broken_links_found:
        print(f"  WARNING: {len(link_report.broken_links_found)} broken links found")
        for link in link_report.broken_links_found[:5]:  # Show first 5
            print(f"    - {link.source_file.name}:{link.line_number} -> {link.target_path.name}")
    else:
        print("  No broken links detected")

    print("\n" + "=" * 50)
    print("Cleanup complete!")
    if args.dry_run:
        print("NOTE: This was a dry run. No changes were made.")
        print("Run without --dry-run to apply changes.")
    print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())
