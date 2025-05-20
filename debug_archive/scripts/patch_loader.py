#!/usr/bin/env python3
"""
Patch script to remove TO_VECTOR function calls from loader.py
This fixes compatibility issues with test environments.
"""

import os
import re

def patch_to_vector_references(file_path):
    print(f"Patching {file_path} to remove TO_VECTOR references...")
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace TO_VECTOR in SQL statements
    content = re.sub(r'TO_VECTOR\(\?, DOUBLE\)', r'?', content)
    content = re.sub(r'embedding VARCHAR\(4000\), VECTOR\(DOUBLE, \d+\)', r'embedding VARCHAR(4000)', content)
    content = re.sub(r'INSERT INTO SourceDocuments \(doc_id, text_content, embedding\) \n\s+VALUES \(\?, \?, TO_VECTOR\(\?, DOUBLE\)\)', 
                    r'INSERT INTO SourceDocuments (doc_id, text_content, embedding) \n                    VALUES (?, ?, ?)', content)
    content = re.sub(r'INSERT INTO DocumentTokenEmbeddings \n\s+\(doc_id, token_sequence_index, token_text, token_embedding, metadata_json\) \n\s+VALUES \(\?, \?, \?, TO_VECTOR\(\?, DOUBLE\), \?\)',
                    r'INSERT INTO DocumentTokenEmbeddings \n                                (doc_id, token_sequence_index, token_text, token_embedding, metadata_json) \n                                VALUES (?, ?, ?, ?, ?)', content)
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Successfully patched {file_path}")

if __name__ == "__main__":
    # Path to the loader.py file
    loader_path = os.path.join("eval", "loader.py")
    
    # Check if the file exists
    if os.path.exists(loader_path):
        patch_to_vector_references(loader_path)
    else:
        print(f"Error: File {loader_path} not found")
