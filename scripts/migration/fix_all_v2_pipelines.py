#!/usr/bin/env python3
"""
Fix all V2 pipelines to use string formatting instead of parameterized queries
to avoid IRIS parameter parsing issues with colons.
"""

import os
import glob
import re

def fix_pipeline_file(filepath):
    """Fix a single pipeline file to use string formatting instead of parameters."""
    print(f"\n🔧 Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Fix 1: Replace parameterized TO_VECTOR calls
    if 'TO_VECTOR(?' in content:
        content = re.sub(r'TO_VECTOR\(\s*\?\s*,\s*DOUBLE\s*,\s*(\d+)\s*\)', 
                        r"TO_VECTOR('{query_embedding_str}', 'DOUBLE', \1)", content)
        content = re.sub(r'TO_VECTOR\(\s*\?\s*,\s*\'?DOUBLE\'?\s*\)', 
                        r"TO_VECTOR('{query_embedding_str}', 'FLOAT')", content)
        content = re.sub(r'TO_VECTOR\(\s*\?\s*\)', 
                        r"TO_VECTOR('{query_embedding_str}')", content)
        changes_made.append("Fixed TO_VECTOR parameterized queries")
    
    # Fix 2: Replace similarity threshold placeholders
    if ') > ?' in content:
        content = re.sub(r'\)\s*>\s*\?', r') > {similarity_threshold}', content)
        changes_made.append("Fixed similarity threshold placeholders")
    
    # Fix 3: Replace TOP ? with TOP {top_k}
    if 'TOP ?' in content or 'TOP %s' in content:
        content = re.sub(r'TOP\s+\?', 'TOP {top_k}', content)
        content = re.sub(r'TOP\s+%s', 'TOP {top_k}', content)
        changes_made.append("Fixed TOP placeholders")
    
    # Fix 4: Update cursor.execute calls
    if 'cursor.execute(sql, [' in content:
        # Find all cursor.execute calls with parameters
        pattern = r'cursor\.execute\(sql,\s*\[[^\]]+\]\)'
        matches = re.findall(pattern, content)
        for match in matches:
            # Extract the parameters
            params_match = re.search(r'\[([^\]]+)\]', match)
            if params_match:
                params = params_match.group(1)
                # Build the format call
                param_names = []
                for param in params.split(','):
                    param = param.strip()
                    if 'query_embedding_str' in param:
                        param_names.append('query_embedding_str=query_embedding_str')
                    elif 'similarity_threshold' in param:
                        param_names.append('similarity_threshold=similarity_threshold')
                    elif 'top_k' in param:
                        param_names.append('top_k=top_k')
                
                if param_names:
                    new_call = f'cursor.execute(sql.format({", ".join(param_names)}))'
                    content = content.replace(match, new_call)
        changes_made.append("Fixed cursor.execute parameterized calls")
    
    # Fix 5: Ensure SQL strings are f-strings where needed
    if 'sql = """' in content and '{' in content:
        content = re.sub(r'sql\s*=\s*"""', 'sql = f"""', content)
        changes_made.append("Converted SQL strings to f-strings")
    
    # Fix 6: Handle GraphRAG entity queries
    if 'TO_VECTOR(embedding, ?)' in content:
        content = content.replace("TO_VECTOR(embedding, ?)", "TO_VECTOR(embedding, 'FLOAT', 384)")
        changes_made.append("Fixed GraphRAG entity queries")
    
    # Fix 7: Handle any remaining ? in VECTOR operations
    if 'VECTOR_COSINE' in content and '?' in content:
        # Find lines with VECTOR_COSINE and ?
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'VECTOR_COSINE' in line and '?' in line:
                # Replace remaining ? with appropriate placeholders
                if 'query_embedding_str' not in line:
                    line = line.replace('?', '{query_embedding_str}')
                    lines[i] = line
        content = '\n'.join(lines)
        changes_made.append("Fixed remaining VECTOR_COSINE placeholders")
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Applied fixes: {', '.join(changes_made)}")
        return True
    else:
        print("  ℹ️  No changes needed")
        return False

def main():
    """Fix all V2 pipeline files."""
    print("🔍 Fixing all V2 pipelines to avoid IRIS parameter parsing issues...")
    
    # Find all pipeline_v2.py files
    v2_files = []
    
    # Look for pipeline_v2.py files
    v2_files.extend(glob.glob("*/pipeline_v2.py"))
    
    # Also look for any other V2 pipeline files
    v2_files.extend(glob.glob("*/*_pipeline_v2.py"))
    
    # Remove duplicates
    v2_files = list(set(v2_files))
    
    if not v2_files:
        print("❌ No V2 pipeline files found!")
        return
    
    print(f"📊 Found {len(v2_files)} V2 pipeline files")
    
    fixed_count = 0
    for filepath in sorted(v2_files):
        if fix_pipeline_file(filepath):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count}/{len(v2_files)} files")
    
    # Also check the original pipelines that might use V2 tables
    print("\n🔍 Checking original pipelines for V2 table usage...")
    original_files = glob.glob("*/pipeline.py")
    for filepath in sorted(original_files):
        if '_v2' not in filepath.lower():
            with open(filepath, 'r') as f:
                content = f.read()
            if '_V2' in content or 'SourceDocuments_V2' in content:
                print(f"\n⚠️  {filepath} uses V2 tables, checking for issues...")
                if fix_pipeline_file(filepath):
                    fixed_count += 1

if __name__ == "__main__":
    main()