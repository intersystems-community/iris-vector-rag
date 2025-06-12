#!/usr/bin/env python3
"""
Simple XML to 10K Documents Script
Processes XML files from data/pmc_100k_downloaded to scale to 10,000 documents
"""

import sys
import os
import json
import glob
import time
import xml.etree.ElementTree as ET
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from common.iris_connector import get_iris_connection
from sentence_transformers import SentenceTransformer

def parse_xml_file(xml_path):
    """Parse XML file and extract content"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract PMC ID from filename
        pmc_id = os.path.basename(xml_path).replace('.xml', '')
        
        # Extract title
        title = ""
        title_elements = root.findall(".//article-title")
        if title_elements:
            title = ''.join(title_elements[0].itertext()).strip()
        
        # Extract abstract
        abstract = ""
        abstract_elements = root.findall(".//abstract")
        if abstract_elements:
            abstract = ''.join(abstract_elements[0].itertext()).strip()
        
        # Extract body text
        full_text = ""
        body_elements = root.findall(".//body")
        if body_elements:
            full_text = ''.join(body_elements[0].itertext()).strip()
        
        # If no body, try sections
        if not full_text:
            sec_elements = root.findall(".//sec")
            full_text = ' '.join(''.join(sec.itertext()).strip() for sec in sec_elements)
        
        return {
            'pmcid': pmc_id,
            'title': title,
            'abstract': abstract,
            'full_text': full_text
        }
    except Exception as e:
        print(f"❌ Error parsing {xml_path}: {e}")
        return None

def main():
    print("🚀 SIMPLE XML TO 10K SCALING")
    print("=" * 50)
    
    # Initialize
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Check current state
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    current_docs = cursor.fetchone()[0]
    print(f"📊 Current documents: {current_docs:,}")
    
    target_docs = 10000
    needed_docs = target_docs - current_docs
    print(f"🎯 Need to add: {needed_docs:,} documents")
    
    if needed_docs <= 0:
        print("✅ Already at target!")
        return
    
    # Get existing document IDs
    cursor.execute("SELECT doc_id FROM RAG.SourceDocuments")
    existing_ids = {row[0] for row in cursor.fetchall()}
    print(f"📋 Found {len(existing_ids):,} existing IDs")
    
    # Find XML files
    xml_files = glob.glob('data/pmc_100k_downloaded/**/*.xml', recursive=True)
    print(f"📁 Found {len(xml_files):,} XML files")
    
    # Initialize embedding model
    print("🤖 Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Process XML files
    added_count = 0
    processed_count = 0
    
    for xml_file in xml_files:
        if added_count >= needed_docs:
            break
            
        processed_count += 1
        doc = parse_xml_file(xml_file)
        
        if doc and doc['pmcid'] not in existing_ids:
            # Only process documents with some content
            if doc['title'] or doc['abstract'] or doc['full_text']:
                try:
                    # Generate embedding
                    text_for_embedding = f"{doc['title']} {doc['abstract']}".strip()
                    if not text_for_embedding:
                        text_for_embedding = doc['full_text'][:500] if doc['full_text'] else "No content"
                    
                    embedding = embedding_model.encode([text_for_embedding])[0]
                    vector_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                    
                    # Insert document
                    cursor.execute("""
                        INSERT INTO RAG.SourceDocuments
                        (doc_id, title, text_content, metadata, embedding)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        doc['pmcid'],
                        doc['title'],
                        f"{doc['abstract']}\n\n{doc['full_text']}".strip(),
                        json.dumps({'source': 'xml_scaling', 'file': xml_file}),
                        vector_str
                    ))
                    
                    added_count += 1
                    existing_ids.add(doc['pmcid'])
                    
                    if added_count % 100 == 0:
                        conn.commit()
                        print(f"✅ Added {added_count:,}/{needed_docs:,} documents")
                        
                except Exception as e:
                    print(f"❌ Error inserting {doc['pmcid']}: {e}")
        
        if processed_count % 500 == 0:
            print(f"📊 Processed {processed_count:,} files, added {added_count:,} documents")
    
    conn.commit()
    
    # Final check
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    final_docs = cursor.fetchone()[0]
    
    print("\n" + "=" * 50)
    print("📊 SCALING COMPLETE")
    print(f"📈 Documents: {current_docs:,} → {final_docs:,}")
    print(f"✅ Added: {final_docs - current_docs:,} documents")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()