#!/usr/bin/env python3
"""
Simple LIST ERROR Check
Basic investigation of data causing LIST ERROR issues
"""

import sys
import json
import logging
from datetime import datetime

# Add the project root to the path
sys.path.append('.')

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simple_data_check():
    """Simple check for basic data integrity issues"""
    logger.info("🔍 Running simple data integrity check...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'basic_stats': {},
        'sample_data': {},
        'issues_found': []
    }
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # 1. Basic table counts
        logger.info("Getting basic table counts...")
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        source_count = cursor.fetchone()[0]
        results['basic_stats']['source_documents'] = source_count
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        results['basic_stats']['token_embeddings'] = token_count
        
        # 2. Check for NULL/empty embeddings
        logger.info("Checking for NULL/empty embeddings...")
        
        cursor.execute("""
        SELECT COUNT(*) FROM RAG.SourceDocuments 
        WHERE embedding IS NULL OR embedding = ''
        """)
        null_embeddings = cursor.fetchone()[0]
        results['basic_stats']['null_source_embeddings'] = null_embeddings
        
        cursor.execute("""
        SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings 
        WHERE token_embedding IS NULL OR token_embedding = ''
        """)
        null_token_embeddings = cursor.fetchone()[0]
        results['basic_stats']['null_token_embeddings'] = null_token_embeddings
        
        # 3. Sample a few embeddings to check format
        logger.info("Sampling embedding formats...")
        
        cursor.execute("""
        SELECT TOP 5 doc_id, LENGTH(embedding), SUBSTRING(embedding, 1, 50)
        FROM RAG.SourceDocuments 
        WHERE embedding IS NOT NULL AND embedding <> ''
        """)
        sample_embeddings = cursor.fetchall()
        
        results['sample_data']['source_embeddings'] = [
            {
                'doc_id': row[0],
                'length': row[1],
                'sample': row[2]
            } for row in sample_embeddings
        ]
        
        # 4. Check for obvious format issues
        logger.info("Checking for format issues...")
        
        # Check for brackets in embeddings (common LIST ERROR cause)
        cursor.execute("""
        SELECT COUNT(*) FROM RAG.SourceDocuments 
        WHERE embedding LIKE '%[%' OR embedding LIKE '%]%'
        """)
        bracket_count = cursor.fetchone()[0]
        
        if bracket_count > 0:
            results['issues_found'].append({
                'type': 'BRACKET_FORMAT',
                'count': bracket_count,
                'severity': 'HIGH',
                'description': 'Embeddings contain brackets which cause LIST ERROR'
            })
        
        # Check for quotes in embeddings
        cursor.execute("""
        SELECT COUNT(*) FROM RAG.SourceDocuments 
        WHERE embedding LIKE '%"%'
        """)
        quote_count = cursor.fetchone()[0]
        
        if quote_count > 0:
            results['issues_found'].append({
                'type': 'QUOTE_FORMAT',
                'count': quote_count,
                'severity': 'HIGH',
                'description': 'Embeddings contain quotes which may cause LIST ERROR'
            })
        
        # 5. Check token embeddings for similar issues
        logger.info("Checking token embedding formats...")
        
        cursor.execute("""
        SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings 
        WHERE token_embedding LIKE '%[%' OR token_embedding LIKE '%]%'
        """)
        token_bracket_count = cursor.fetchone()[0]
        
        if token_bracket_count > 0:
            results['issues_found'].append({
                'type': 'TOKEN_BRACKET_FORMAT',
                'count': token_bracket_count,
                'severity': 'HIGH',
                'description': 'Token embeddings contain brackets which cause LIST ERROR'
            })
        
        cursor.close()
        conn.close()
        
        logger.info("Simple data check completed successfully")
        
    except Exception as e:
        logger.error(f"Error during simple data check: {e}")
        results['error'] = str(e)
    
    return results

def main():
    """Run simple data check and save report"""
    results = simple_data_check()
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"simple_list_error_check_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("SIMPLE LIST ERROR CHECK REPORT")
    print("="*80)
    
    print(f"\nReport saved to: {report_file}")
    print(f"Timestamp: {results['timestamp']}")
    
    # Print basic stats
    if 'basic_stats' in results:
        print(f"\nBASIC STATISTICS:")
        for key, value in results['basic_stats'].items():
            print(f"  {key}: {value:,}")
    
    # Print sample data
    if 'sample_data' in results and 'source_embeddings' in results['sample_data']:
        print(f"\nSAMPLE EMBEDDINGS:")
        for sample in results['sample_data']['source_embeddings']:
            print(f"  {sample['doc_id']}: length={sample['length']}, sample='{sample['sample']}...'")
    
    # Print issues found
    if results.get('issues_found'):
        print(f"\nISSUES FOUND ({len(results['issues_found'])}):")
        for i, issue in enumerate(results['issues_found'], 1):
            print(f"  {i}. [{issue['severity']}] {issue['type']}: {issue['description']}")
            print(f"     Count: {issue['count']:,}")
    else:
        print("\nNo obvious format issues found in sample data.")
    
    # Print error if any
    if 'error' in results:
        print(f"\nERROR: {results['error']}")
    
    print("\n" + "="*80)
    
    return results

if __name__ == "__main__":
    main()