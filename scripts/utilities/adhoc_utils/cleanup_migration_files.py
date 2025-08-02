#!/usr/bin/env python3
"""
Clean up temporary migration files that are no longer needed.
"""

import os
import shutil
from datetime import datetime

# Files to remove (temporary migration scripts)
files_to_remove = [
    "check_schema_status.py",
    "check_v2_migration_status.py",
    "complete_sourcedocuments_migration_final.py",
    "complete_sourcedocuments_migration_simple.py",
    "complete_sourcedocuments_migration.py",
    "complete_sourcedocuments_rename_final.py",
    "complete_sourcedocuments_workaround.py",
    "complete_v2_table_rename_auto.py",
    "complete_v2_table_rename.py",
    "debug_basic_rag_embeddings.py",
    "find_all_tables.py",
    "force_sourcedocuments_migration.py",
    "migrate_all_pipelines.py",
    "migrate_document_chunks_v2_jdbc.py",
    "migrate_document_chunks_v2_only.py",
    "remove_compiled_class_dependency.py",
    "test_basic_rag_final_performance.py",
    "test_basic_rag_performance.py",
    "test_basic_rag_with_retrieval.py",
    "test_basic_rag_working.py",
    "test_hnsw_performance_comparison.py",
    "test_hnsw_performance_final.py",
    "test_refactored_debug.py",
    "test_v2_rag_jdbc.py",
    "test_v2_rag_simple.py",
    "test_v2_rag_techniques.py",
    "update_pipelines_for_current_tables.py",
    "verify_basic_rag_retrieval.py",
    "verify_final_hnsw_state.py",
    "verify_v2_index_types.py",
    "validate_complete_hnsw_migration.py",
    "validate_hnsw_migration_simple.py",
    "validate_hnsw_final.py",
    "check_actual_tables.py",
    "check_tables_simple.py",
    "drop_sourcedocuments_dependencies.py"
]

# Files to keep (for reference)
files_to_keep = [
    "validate_hnsw_correct_schema.py",  # Working validation script
    "HNSW_MIGRATION_STATUS_FINAL.md",    # Final status document
    "test_jdbc_connection.py",           # JDBC test utility
]

# Backup .pre_v2_update files
backup_files = []
for f in os.listdir("."):
    if f.endswith(".pre_v2_update"):
        backup_files.append(f)

def main():
    """Clean up migration files."""
    print("Migration File Cleanup")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create archive directory
    archive_dir = f"archive/migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(archive_dir, exist_ok=True)
    print(f"\nCreated archive directory: {archive_dir}")
    
    # Archive files before deletion
    archived_count = 0
    deleted_count = 0
    
    print("\nArchiving and removing temporary files...")
    for filename in files_to_remove:
        if os.path.exists(filename):
            try:
                # Archive first
                shutil.copy2(filename, os.path.join(archive_dir, filename))
                # Then remove
                os.remove(filename)
                print(f"  ✓ Archived and removed: {filename}")
                archived_count += 1
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Error with {filename}: {str(e)}")
    
    # Archive backup files
    print("\nArchiving .pre_v2_update backup files...")
    for filename in backup_files:
        if os.path.exists(filename):
            try:
                shutil.copy2(filename, os.path.join(archive_dir, filename))
                os.remove(filename)
                print(f"  ✓ Archived and removed: {filename}")
                archived_count += 1
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Error with {filename}: {str(e)}")
    
    # Report on files to keep
    print("\nFiles kept for reference:")
    for filename in files_to_keep:
        if os.path.exists(filename):
            print(f"  ✓ Kept: {filename}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Files archived: {archived_count}")
    print(f"Files deleted: {deleted_count}")
    print(f"Archive location: {archive_dir}")
    
    print("\n✅ Cleanup completed successfully!")
    print("\nNext steps:")
    print("1. Review the archive directory to ensure nothing important was removed")
    print("2. Run 'python validate_hnsw_correct_schema.py' to verify system still works")
    print("3. Commit all changes with a comprehensive message")


if __name__ == "__main__":
    main()