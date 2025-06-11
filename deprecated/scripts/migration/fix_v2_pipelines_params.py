#!/usr/bin/env python3
"""
Fix V2 pipelines to use string formatting instead of parameterized queries
to avoid IRIS parameter parsing issues with colons.
"""

import os
import glob

def fix_pipeline_file(filepath):
    """Fix a single pipeline file to use string formatting instead of parameters."""
    print(f"\n🔧 Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix patterns where we use cursor.execute with parameters
    # Pattern 1: cursor.execute(sql, [param1, param2, ...])
    if 'cursor.execute(sql, [' in content:
        print("  📍 Found parameterized query pattern")
        
        # Replace TO_VECTOR(?, DOUBLE, 384) with TO_VECTOR('{query_embedding_str}', 'FLOAT', 384)
        content = content.replace("TO_VECTOR(?, DOUBLE, 384)", "TO_VECTOR('{query_embedding_str}', 'FLOAT', 384)")
        content = content.replace("TO_VECTOR(?, DOUBLE, 128)", "TO_VECTOR('{query_embedding_str}', 'FLOAT', 128)")
        content = content.replace("TO_VECTOR(?, DOUBLE)", "TO_VECTOR('{query_embedding_str}', 'FLOAT')")
        
        # Replace similarity threshold placeholder
        content = content.replace(") > ?", ") > {similarity_threshold}")
        
        # Fix the execute call to use f-string formatting
        content = content.replace('cursor.execute(sql, [query_embedding_str, query_embedding_str, similarity_threshold])', 
                                'cursor.execute(sql.format(query_embedding_str=query_embedding_str, similarity_threshold=similarity_threshold))')
        
        # Also handle other parameter patterns
        content = content.replace('cursor.execute(sql, [query_embedding_str])', 
                                'cursor.execute(sql.format(query_embedding_str=query_embedding_str))')
    
    # Fix entity queries in GraphRAG
    if 'TO_VECTOR(embedding, ?)' in content:
        print("  📍 Found GraphRAG entity query pattern")
        content = content.replace("TO_VECTOR(embedding, ?)", "TO_VECTOR(embedding, 'FLOAT', 384)")
        content = content.replace("TO_VECTOR(?, ?)", "TO_VECTOR('{query_embedding_str}', 'FLOAT', 384)")
    
    # Fix any remaining ? placeholders in VECTOR operations
    if 'VECTOR_COSINE' in content and '?' in content:
        print("  📍 Found remaining VECTOR_COSINE placeholders")
        # This is a more complex case - need to handle it carefully
        lines = content.split('\n')
        in_sql = False
        sql_var = None
        
        for i, line in enumerate(lines):
            if 'sql = """' in line or 'sql = f"""' in line:
                in_sql = True
                # Change to f-string if not already
                lines[i] = line.replace('sql = """', 'sql = f"""')
            elif '"""' in line and in_sql:
                in_sql = False
            elif in_sql and 'VECTOR_COSINE' in line and '?' in line:
                # Replace ? with {param_name}
                lines[i] = line.replace('?', '{query_embedding_str}')
        
        content = '\n'.join(lines)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print("  ✅ Fixed parameterized queries")
        return True
    else:
        print("  ℹ️  No changes needed")
        return False

def main():
    """Fix all V2 pipeline files."""
    print("🔍 Fixing V2 pipelines to avoid IRIS parameter parsing issues...")
    
    # Find all pipeline_v2.py files
    v2_files = glob.glob("*/pipeline_v2.py")
    
    if not v2_files:
        print("❌ No V2 pipeline files found!")
        return
    
    print(f"📊 Found {len(v2_files)} V2 pipeline files")
    
    fixed_count = 0
    for filepath in v2_files:
        if fix_pipeline_file(filepath):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count}/{len(v2_files)} files")
    
    # Also check for any _v2.py files
    other_v2_files = glob.glob("*/*_v2.py")
    if other_v2_files:
        print(f"\n📊 Found {len(other_v2_files)} other V2 files")
        for filepath in other_v2_files:
            if 'pipeline' in filepath:
                if fix_pipeline_file(filepath):
                    fixed_count += 1

if __name__ == "__main__":
    main()