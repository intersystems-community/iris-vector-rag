#!/usr/bin/env python3
"""
Token Embedding Backfill Plan - Focused Analysis and Strategy
"""

import sys
import os
# Ensure project root is in path for generated script and this script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.iris_connector import get_iris_connection # Updated import
import json
from datetime import datetime

def get_current_state():
    """Get the current state of token embeddings"""
    print("🔍 CURRENT TOKEN EMBEDDING STATE")
    print("=" * 50)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Total documents
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    total_docs = cursor.fetchone()[0]
    
    # Documents with token embeddings
    cursor.execute("""
        SELECT COUNT(DISTINCT doc_id) 
        FROM RAG.DocumentTokenEmbeddings 
        WHERE token_embedding IS NOT NULL
    """)
    docs_with_tokens = cursor.fetchone()[0]
    
    # Documents missing token embeddings
    docs_missing = total_docs - docs_with_tokens
    coverage_percent = (docs_with_tokens / total_docs) * 100
    
    print(f"📊 Total documents: {total_docs:,}")
    print(f"✅ Documents with token embeddings: {docs_with_tokens:,}")
    print(f"❌ Documents missing token embeddings: {docs_missing:,}")
    print(f"📈 Current coverage: {coverage_percent:.1f}%")
    
    # Check recent documents (last 100)
    cursor.execute("""
        SELECT sd.doc_id
        FROM RAG.SourceDocuments sd
        WHERE sd.doc_id NOT IN (
            SELECT DISTINCT doc_id 
            FROM RAG.DocumentTokenEmbeddings 
            WHERE token_embedding IS NOT NULL
        )
        ORDER BY sd.doc_id DESC
        LIMIT 10
    """)
    
    recent_missing = cursor.fetchall()
    print(f"\n🔍 Recent documents missing token embeddings:")
    for doc_id, in recent_missing:
        print(f"  • {doc_id}")
    
    cursor.close()
    conn.close()
    
    return {
        'total_docs': total_docs,
        'docs_with_tokens': docs_with_tokens,
        'docs_missing': docs_missing,
        'coverage_percent': coverage_percent,
        'recent_missing': [doc_id for doc_id, in recent_missing]
    }

def analyze_trajectory():
    """Analyze if current process is generating token embeddings"""
    print("\n🎯 TRAJECTORY ANALYSIS")
    print("=" * 30)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Check the most recent 20 documents
    cursor.execute("""
        SELECT TOP 20 sd.doc_id,
               CASE WHEN dte.doc_id IS NOT NULL THEN 1 ELSE 0 END as has_tokens
        FROM RAG.SourceDocuments sd
        LEFT JOIN (
            SELECT DISTINCT doc_id 
            FROM RAG.DocumentTokenEmbeddings 
            WHERE token_embedding IS NOT NULL
        ) dte ON sd.doc_id = dte.doc_id
        ORDER BY sd.doc_id DESC
    """)
    
    recent_docs = cursor.fetchall()
    recent_with_tokens = sum(has_tokens for _, has_tokens in recent_docs)
    recent_rate = recent_with_tokens / len(recent_docs) if recent_docs else 0
    
    print(f"📊 Recent 20 documents with token embeddings: {recent_with_tokens}/20 ({recent_rate*100:.1f}%)")
    
    if recent_rate > 0.8:
        print("✅ Current process IS generating token embeddings for new documents")
        trajectory_status = "GOOD"
    elif recent_rate > 0.5:
        print("⚠️  Current process is PARTIALLY generating token embeddings")
        trajectory_status = "PARTIAL"
    else:
        print("❌ Current process is NOT generating token embeddings consistently")
        trajectory_status = "BROKEN"
    
    cursor.close()
    conn.close()
    
    return {
        'recent_rate': recent_rate,
        'recent_with_tokens': recent_with_tokens,
        'recent_total': len(recent_docs),
        'status': trajectory_status
    }

def estimate_backfill_effort():
    """Estimate the effort required for backfill"""
    print("\n⏱️  BACKFILL EFFORT ESTIMATION")
    print("=" * 40)
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Get average tokens per document from existing data
    cursor.execute("""
        SELECT AVG(CAST(token_count AS FLOAT)) as avg_tokens
        FROM (
            SELECT doc_id, COUNT(*) as token_count
            FROM RAG.DocumentTokenEmbeddings
            WHERE token_embedding IS NOT NULL
            GROUP BY doc_id
        ) doc_stats
    """)
    
    result = cursor.fetchone()
    avg_tokens = result[0] if result[0] else 200
    
    # Get documents missing token embeddings
    cursor.execute("""
        SELECT COUNT(*) 
        FROM RAG.SourceDocuments sd
        WHERE sd.doc_id NOT IN (
            SELECT DISTINCT doc_id 
            FROM RAG.DocumentTokenEmbeddings 
            WHERE token_embedding IS NOT NULL
        )
    """)
    
    docs_missing = cursor.fetchone()[0]
    
    # Estimate processing requirements
    total_tokens_needed = docs_missing * avg_tokens
    
    # Estimate time based on ColBERT processing speed (~100 tokens/second)
    tokens_per_second = 100
    estimated_seconds = total_tokens_needed / tokens_per_second
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24
    
    print(f"📊 Backfill requirements:")
    print(f"  Documents missing embeddings: {docs_missing:,}")
    print(f"  Average tokens per document: {avg_tokens:.0f}")
    print(f"  Total tokens to process: {total_tokens_needed:,.0f}")
    print(f"  Estimated processing time: {estimated_hours:.1f} hours ({estimated_days:.1f} days)")
    
    cursor.close()
    conn.close()
    
    return {
        'docs_missing': docs_missing,
        'avg_tokens_per_doc': avg_tokens,
        'total_tokens_needed': total_tokens_needed,
        'estimated_hours': estimated_hours,
        'estimated_days': estimated_days
    }

def create_backfill_strategy(current_state, trajectory, effort):
    """Create a comprehensive backfill strategy"""
    print("\n📋 BACKFILL STRATEGY")
    print("=" * 30)
    
    strategy = {
        'current_situation': {
            'total_docs': current_state['total_docs'],
            'coverage_percent': current_state['coverage_percent'],
            'docs_missing': current_state['docs_missing']
        },
        'trajectory_assessment': trajectory['status'],
        'effort_estimate': effort,
        'recommendations': [],
        'timeline_to_100k': {}
    }
    
    # Assess current trajectory
    if trajectory['status'] == 'BROKEN':
        strategy['recommendations'].append({
            'priority': 'CRITICAL',
            'action': 'Fix current ingestion process',
            'details': 'Only {:.1f}% of recent documents have token embeddings'.format(trajectory['recent_rate'] * 100),
            'impact': 'Without fixing this, we will have gaps in token embedding coverage'
        })
    
    # Backfill recommendations
    if effort['estimated_days'] < 1:
        strategy['recommendations'].append({
            'priority': 'HIGH',
            'action': 'Run immediate full backfill',
            'details': f'Process {effort["docs_missing"]:,} documents in {effort["estimated_hours"]:.1f} hours',
            'impact': 'Complete token embedding coverage achieved quickly'
        })
    elif effort['estimated_days'] < 7:
        strategy['recommendations'].append({
            'priority': 'HIGH',
            'action': 'Run backfill over weekend',
            'details': f'Process {effort["docs_missing"]:,} documents in {effort["estimated_days"]:.1f} days',
            'impact': 'Complete token embedding coverage with minimal disruption'
        })
    else:
        strategy['recommendations'].append({
            'priority': 'MEDIUM',
            'action': 'Run incremental backfill in batches',
            'details': f'Process in daily batches over {effort["estimated_days"]:.0f} days',
            'impact': 'Gradual improvement in token embedding coverage'
        })
    
    # Timeline to 100k documents
    current_docs = current_state['total_docs']
    docs_to_100k = 100000 - current_docs
    
    strategy['timeline_to_100k'] = {
        'current_docs': current_docs,
        'docs_needed': docs_to_100k,
        'backfill_needed': current_state['docs_missing'],
        'process_status': trajectory['status']
    }
    
    print("🎯 Key Recommendations:")
    for i, rec in enumerate(strategy['recommendations'], 1):
        print(f"  {i}. [{rec['priority']}] {rec['action']}")
        print(f"     Details: {rec['details']}")
        print(f"     Impact: {rec['impact']}")
        print()
    
    print(f"🚀 Path to 100k documents:")
    print(f"  Current: {current_docs:,} documents")
    print(f"  Need: {docs_to_100k:,} more documents")
    print(f"  Backfill needed: {current_state['docs_missing']:,} documents")
    
    if trajectory['status'] == 'GOOD':
        print(f"  ✅ New documents will have token embeddings")
    else:
        print(f"  ❌ Need to fix token embedding generation first")
    
    return strategy

def create_backfill_script():
    """Create a script to perform the backfill"""
    print("\n🛠️  CREATING BACKFILL SCRIPT")
    print("=" * 35)
    
    script_content = '''#!/usr/bin/env python3
"""
Token Embedding Backfill Script
Generates ColBERT token embeddings for documents that don't have them
"""

import sys
import os
# Ensure project root is in path for generated script
project_root_generated = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_generated not in sys.path:
    sys.path.insert(0, project_root_generated)

from src.common.iris_connector import get_iris_connection # Updated import
from src.working.colbert.doc_encoder import ColBERTDocumentEncoder # Updated import
import time
from datetime import datetime

def get_documents_without_tokens(batch_size=100):
    """Get documents that don't have token embeddings"""
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT TOP {batch_size} sd.doc_id, sd.content
        FROM RAG.SourceDocuments sd
        WHERE sd.doc_id NOT IN (
            SELECT DISTINCT doc_id 
            FROM RAG.DocumentTokenEmbeddings 
            WHERE token_embedding IS NOT NULL
        )
        ORDER BY sd.doc_id
    """)
    
    docs = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return docs

def process_document_tokens(doc_id, content, encoder):
    """Process a single document and store its token embeddings"""
    try:
        # Generate token embeddings
        token_embeddings = encoder.encode_document(content, doc_id)
        
        # Store in database
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        for token_data in token_embeddings:
            cursor.execute("""
                INSERT INTO RAG.DocumentTokenEmbeddings 
                (doc_id, token_sequence_index, token_text, token_embedding, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                token_data['doc_id'],
                token_data['token_sequence_index'],
                token_data['token_text'],
                token_data['token_embedding'],
                token_data.get('metadata_json', '{}')
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return len(token_embeddings)
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        return 0

def main():
    """Main backfill function"""
    print(f"🚀 Starting token embedding backfill - {datetime.now()}")
    
    # Initialize encoder
    encoder = ColBERTDocumentEncoder()
    
    batch_size = 100
    total_processed = 0
    total_tokens = 0
    
    while True:
        # Get next batch of documents
        docs = get_documents_without_tokens(batch_size)
        
        if not docs:
            print("✅ No more documents to process")
            break
        
        print(f"📊 Processing batch of {len(docs)} documents...")
        batch_start = time.time()
        
        for doc_id, content in docs:
            tokens_generated = process_document_tokens(doc_id, content, encoder)
            total_tokens += tokens_generated
            total_processed += 1
            
            if total_processed % 10 == 0:
                elapsed = time.time() - batch_start
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"  Processed {total_processed} docs, {total_tokens} tokens ({rate:.1f} docs/sec)")
        
        batch_elapsed = time.time() - batch_start
        print(f"  Batch completed in {batch_elapsed:.1f} seconds")
    
    print(f"🎉 Backfill completed!")
    print(f"  Total documents processed: {total_processed}")
    print(f"  Total tokens generated: {total_tokens}")

if __name__ == "__main__":
    main()
'''
    
    with open('backfill_token_embeddings.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created backfill_token_embeddings.py")
    print("   Usage: python3 backfill_token_embeddings.py")
    
    return 'backfill_token_embeddings.py'

def main():
    """Main analysis function"""
    print("🔍 TOKEN EMBEDDING BACKFILL ANALYSIS & PLANNING")
    print("=" * 60)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Get current state
    current_state = get_current_state()
    
    # Step 2: Analyze trajectory
    trajectory = analyze_trajectory()
    
    # Step 3: Estimate effort
    effort = estimate_backfill_effort()
    
    # Step 4: Create strategy
    strategy = create_backfill_strategy(current_state, trajectory, effort)
    
    # Step 5: Create backfill script
    script_path = create_backfill_script()
    
    # Save complete analysis
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'current_state': current_state,
        'trajectory': trajectory,
        'effort_estimate': effort,
        'strategy': strategy,
        'backfill_script': script_path
    }
    
    with open('token_embedding_backfill_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n📄 Complete analysis saved to: token_embedding_backfill_analysis.json")
    print(f"🛠️  Backfill script created: {script_path}")
    
    return analysis_results

if __name__ == "__main__":
    main()