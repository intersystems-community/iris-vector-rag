#!/usr/bin/env python3
"""
Fix embedding column datatype from varchar to VECTOR.

This script migrates the SourceDocuments table to use proper VECTOR type
for embeddings instead of varchar.
"""

from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

def main():
    print("=" * 60)
    print("Fixing embedding column datatype")
    print("=" * 60)

    # Create managers
    connection_manager = ConnectionManager()
    config_manager = ConfigurationManager()
    schema_manager = SchemaManager(connection_manager, config_manager)

    # Check current schema
    print("\n1. Checking current schema...")
    current_config = schema_manager.get_current_schema_config("SourceDocuments")
    if current_config:
        print(f"   Current config: {current_config}")
    else:
        print("   No schema metadata found")

    # Check physical structure
    print("\n2. Checking physical table structure...")
    structure = schema_manager.verify_table_structure("SourceDocuments")
    if "EMBEDDING" in structure:
        print(f"   embedding column type: {structure['EMBEDDING']}")
    else:
        print("   embedding column not found")

    # Migrate table
    print("\n3. Migrating SourceDocuments table...")
    print("   This will:")
    print("   - Drop the existing table")
    print("   - Recreate it with proper VECTOR type")
    print("   - Data will be lost (you'll need to reload)")

    success = schema_manager.migrate_table("SourceDocuments", preserve_data=False)

    if success:
        print("\n✅ Migration successful!")

        # Verify new structure
        print("\n4. Verifying new structure...")
        new_structure = schema_manager.verify_table_structure("SourceDocuments")
        if "EMBEDDING" in new_structure:
            print(f"   embedding column type: {new_structure['EMBEDDING']}")
        else:
            print("   embedding column not found")

        print("\n5. Next steps:")
        print("   - Run 'make load-data' to reload documents with proper embeddings")
        print("   - Re-run tests")
    else:
        print("\n❌ Migration failed!")
        print("   Check logs for details")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
