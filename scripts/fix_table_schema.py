#!/usr/bin/env python3
"""
Script to force recreation of SourceDocuments table with correct schema.
This fixes the ID column issue where the table has an auto-generated ID column
but the code expects to use doc_id as the primary key.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_rag.storage.schema_manager import SchemaManager
from common.iris_connection_manager import IRISConnectionManager
from iris_rag.config.manager import ConfigurationManager

def main():
    """Force recreation of SourceDocuments table with correct schema."""
    print("🔧 Fixing SourceDocuments table schema...")
    
    # Initialize components
    config_manager = ConfigurationManager()
    connection_manager = IRISConnectionManager(config_manager)
    schema_manager = SchemaManager(connection_manager, config_manager)
    
    # Force table recreation by calling ensure_table_schema
    # This should detect the schema mismatch and recreate the table
    print("📋 Ensuring SourceDocuments table schema...")
    success = schema_manager.ensure_table_schema('SourceDocuments')
    
    if success:
        print("✅ SourceDocuments table schema fixed successfully!")
        print("   - ID column: doc_id (VARCHAR, primary key)")
        print("   - Content column: text_content")
        print("   - No auto-generated ID column")
    else:
        print("❌ Failed to fix SourceDocuments table schema")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())