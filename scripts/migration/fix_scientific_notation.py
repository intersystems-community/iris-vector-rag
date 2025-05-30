#!/usr/bin/env python3
"""
Fix scientific notation in embedding strings to avoid IRIS SQL parameter confusion
"""

import os
import re

def fix_embedding_format(content):
    """Fix embedding string formatting to avoid scientific notation"""
    
    # Find all lines that create query_embedding_str
    pattern = r"query_embedding_str = ','.join\(map\(str, query_embedding\)\)"
    
    # Replace with a version that formats floats to avoid scientific notation
    replacement = "query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])"
    
    modified = re.sub(pattern, replacement, content)
    
    # Also fix any existing query_embedding_str assignments that might use numpy arrays
    pattern2 = r"query_embedding_str = ','.join\(map\(str, query_embedding\[0\]\)\)"
    replacement2 = "query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding[0]])"
    modified = re.sub(pattern2, replacement2, modified)
    
    return modified

def process_file(filepath):
    """Process a single file"""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Apply fixes
    modified = fix_embedding_format(content)
    
    if modified != content:
        with open(filepath, 'w') as f:
            f.write(modified)
        print(f"  ✅ Fixed embedding format in {filepath}")
        return True
    else:
        print(f"  ℹ️  No changes needed in {filepath}")
        return False

def main():
    """Fix all V2 pipeline files"""
    print("🔧 Fixing scientific notation in embedding strings")
    print("=" * 60)
    
    # List of V2 pipeline files to fix
    v2_files = [
        'basic_rag/pipeline_v2.py',
        'crag/pipeline_v2.py',
        'hyde/pipeline_v2.py',
        'noderag/pipeline_v2.py',
        'graphrag/pipeline_v2.py',
        'hybrid_ifind_rag/pipeline_v2.py'
    ]
    
    fixed_count = 0
    for filepath in v2_files:
        if os.path.exists(filepath):
            if process_file(filepath):
                fixed_count += 1
        else:
            print(f"  ⚠️  File not found: {filepath}")
    
    print(f"\n✅ Fixed {fixed_count} files")

if __name__ == "__main__":
    main()