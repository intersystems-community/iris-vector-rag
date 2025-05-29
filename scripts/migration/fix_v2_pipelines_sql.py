#!/usr/bin/env python3
"""
Fix V2 pipelines to use string formatting for embeddings in TO_VECTOR calls
IRIS doesn't support parameter placeholders inside TO_VECTOR function
"""

import os
import re

def fix_pipeline_file(filepath):
    """Fix SQL queries in a pipeline file"""
    print(f"Fixing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find TO_VECTOR with parameter placeholder
    # This will match TO_VECTOR(?, 'DOUBLE', 384) or similar
    pattern = r"TO_VECTOR\(\s*\?\s*,\s*'DOUBLE'\s*,\s*(\d+)\s*\)"
    
    # Replace with string formatting
    def replacer(match):
        dimensions = match.group(1)
        return f"TO_VECTOR('{{{query_embedding_str}}}', 'DOUBLE', {dimensions})"
    
    # First pass: replace the TO_VECTOR calls
    content_fixed = re.sub(pattern, replacer, content)
    
    # Second pass: fix the cursor.execute calls to remove parameters
    # Find patterns like cursor.execute(sql_query, (query_embedding_str, query_embedding_str))
    execute_pattern = r"cursor\.execute\(([^,]+),\s*\([^)]*query_embedding_str[^)]*\)\)"
    content_fixed = re.sub(execute_pattern, r"cursor.execute(\1)", content_fixed)
    
    # Third pass: ensure we're using f-strings for SQL queries
    # Replace sql_query = """ with sql_query = f"""
    content_fixed = re.sub(r'sql_query\s*=\s*"""', 'sql_query = f"""', content_fixed)
    content_fixed = re.sub(r'sql\s*=\s*"""', 'sql = f"""', content_fixed)
    
    # Save the fixed content
    with open(filepath, 'w') as f:
        f.write(content_fixed)
    
    print(f"✅ Fixed {filepath}")

def main():
    """Fix all V2 pipeline files"""
    v2_pipelines = [
        'crag/pipeline_v2.py',
        'hyde/pipeline_v2.py',
        'noderag/pipeline_v2.py',
        'graphrag/pipeline_v2.py',
        'hybrid_ifind_rag/pipeline_v2.py'
    ]
    
    for pipeline in v2_pipelines:
        if os.path.exists(pipeline):
            try:
                fix_pipeline_file(pipeline)
            except Exception as e:
                print(f"❌ Error fixing {pipeline}: {e}")
        else:
            print(f"⚠️  File not found: {pipeline}")
    
    # Also fix the Decimal issue in HybridiFindRAG
    print("\nFixing Decimal issue in HybridiFindRAG V2...")
    try:
        with open('hybrid_ifind_rag/pipeline_v2.py', 'r') as f:
            content = f.read()
        
        # Fix the Decimal multiplication issue
        content = content.replace(
            "scores.get('basic', 0) * 0.3",
            "float(scores.get('basic', 0)) * 0.3"
        )
        content = content.replace(
            "scores.get('graph', 0) * 0.4",
            "float(scores.get('graph', 0)) * 0.4"
        )
        content = content.replace(
            "scores.get('hyde', 0) * 0.3",
            "float(scores.get('hyde', 0)) * 0.3"
        )
        
        with open('hybrid_ifind_rag/pipeline_v2.py', 'w') as f:
            f.write(content)
        
        print("✅ Fixed Decimal issue in HybridiFindRAG V2")
    except Exception as e:
        print(f"❌ Error fixing Decimal issue: {e}")

if __name__ == "__main__":
    main()