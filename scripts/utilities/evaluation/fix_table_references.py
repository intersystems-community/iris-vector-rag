#!/usr/bin/env python3
"""
Fix table references in pipelines to use existing table names
"""

import os
import sys
import re
from pathlib import Path

def fix_table_references(file_path):
    """Fix table references in a single file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace V2 table references with actual table names
    replacements = [
        # SourceDocuments_V2 -> SourceDocuments (since V2 exists but pipelines should use it)
        (r'SOURCEDOCUMENTS_V2', 'SOURCEDOCUMENTS_V2'),  # Keep V2 reference since it exists
        (r'SourceDocuments_V2', 'SourceDocuments_V2'),
        # DocumentChunks_V2 -> DocumentChunks (already migrated)
        (r'DOCUMENTCHUNKS_V2', 'DOCUMENTCHUNKS'),
        (r'DocumentChunks_V2', 'DocumentChunks'),
        # DocumentTokenEmbeddings_V2 -> DocumentTokenEmbeddings (already migrated)
        (r'DOCUMENTTOKENEMBEDDINGS_V2', 'DOCUMENTTOKENEMBEDDINGS'),
        (r'DocumentTokenEmbeddings_V2', 'DocumentTokenEmbeddings'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix table references in all pipeline files"""
    
    # Directories to search
    dirs_to_fix = [
        'basic_rag',
        'hyde', 
        'crag',
        'colbert',
        'noderag',
        'graphrag',
        'hybrid_ifind_rag',
        'common'
    ]
    
    fixed_files = []
    
    for dir_name in dirs_to_fix:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            continue
            
        for py_file in dir_path.glob('*.py'):
            if fix_table_references(py_file):
                fixed_files.append(py_file)
    
    if fixed_files:
        print(f"Fixed {len(fixed_files)} files:")
        for f in fixed_files:
            print(f"  - {f}")
    else:
        print("No files needed fixing")

if __name__ == "__main__":
    main()