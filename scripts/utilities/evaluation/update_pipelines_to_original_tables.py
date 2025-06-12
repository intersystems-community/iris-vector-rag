#!/usr/bin/env python3
"""
Update all pipelines to use original table names instead of V2
"""

import os
import sys
import re
from pathlib import Path

def update_table_references(file_path):
    """Update table references in a single file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace V2 table references with original table names
    replacements = [
        # Case-insensitive replacements for SQL queries
        (r'(?i)SOURCEDOCUMENTS_V2', 'SourceDocuments'),
        (r'(?i)DOCUMENTCHUNKS_V2', 'DocumentChunks'),
        (r'(?i)DOCUMENTTOKENEMBEDDINGS_V2', 'DocumentTokenEmbeddings'),
        # Also handle without underscore
        (r'(?i)SOURCEDOCUMENTSV2', 'SourceDocuments'),
        (r'(?i)DOCUMENTCHUNKSV2', 'DocumentChunks'),
        (r'(?i)DOCUMENTTOKENEMBEDDINGSV2', 'DocumentTokenEmbeddings'),
        # Handle quoted versions
        (r'"SOURCEDOCUMENTS_V2"', '"SourceDocuments"'),
        (r'"DOCUMENTCHUNKS_V2"', '"DocumentChunks"'),
        (r'"DOCUMENTTOKENEMBEDDINGS_V2"', '"DocumentTokenEmbeddings"'),
        (r"'SOURCEDOCUMENTS_V2'", "'SourceDocuments'"),
        (r"'DOCUMENTCHUNKS_V2'", "'DocumentChunks'"),
        (r"'DOCUMENTTOKENEMBEDDINGS_V2'", "'DocumentTokenEmbeddings'"),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE if pattern.startswith('(?i)') else 0)
    
    if content != original_content:
        # Create backup
        backup_path = str(file_path) + '.pre_table_fix'
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(original_content)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Update table references in all pipeline files"""
    
    # Files and directories to update
    targets = [
        # Pipeline files
        'basic_rag/pipeline_jdbc.py',
        'hyde/pipeline.py',
        'crag/pipeline_jdbc_fixed.py',
        'colbert/pipeline.py',
        'noderag/pipeline.py',
        'graphrag/pipeline_jdbc_fixed.py',
        'hybrid_ifind_rag/pipeline.py',
        # Common files that might have table references
        'common/db_vector_search.py',
        'common/chunk_retrieval.py',
        'common/jdbc_safe_retrieval.py',
    ]
    
    # Also search for any .py files in these directories
    search_dirs = [
        'basic_rag',
        'hyde',
        'crag',
        'colbert',
        'noderag',
        'graphrag',
        'hybrid_ifind_rag',
        'common'
    ]
    
    all_files = set()
    
    # Add specific targets
    for target in targets:
        if os.path.exists(target):
            all_files.add(Path(target))
    
    # Add all .py files from directories
    for dir_name in search_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            all_files.update(dir_path.glob('*.py'))
    
    fixed_files = []
    
    for file_path in all_files:
        try:
            if update_table_references(file_path):
                fixed_files.append(file_path)
                print(f"✅ Updated: {file_path}")
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  - Checked {len(all_files)} files")
    print(f"  - Updated {len(fixed_files)} files")
    
    if fixed_files:
        print("\n📝 Updated files:")
        for f in sorted(fixed_files):
            print(f"  - {f}")

if __name__ == "__main__":
    main()