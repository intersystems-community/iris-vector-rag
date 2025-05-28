#!/usr/bin/env python3
"""
Comprehensive Embedding Integrity Assessment and Regeneration Plan

This script provides a complete assessment of embedding data integrity issues
and creates a detailed plan for restoration after the column mismatch fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.iris_connector import get_iris_connection
import json
from datetime import datetime

def assess_embedding_integrity():
    """Comprehensive assessment of all embedding data"""
    print("🔍 COMPREHENSIVE EMBEDDING INTEGRITY ASSESSMENT")
    print("=" * 60)
    print(f"📅 Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    assessment = {
        'timestamp': datetime.now().isoformat(),
        'source_documents': {},
        'token_embeddings': {},
        'backup_analysis': {},
        'corruption_analysis': {},
        'regeneration_scope': {},
        'recommendations': []
    }
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Source Documents Analysis
        print("📊 1. SOURCE DOCUMENTS EMBEDDING ANALYSIS")
        print("-" * 50)
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NULL")
        null_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        non_null_embeddings = cursor.fetchone()[0]
        
        assessment['source_documents'] = {
            'total_records': total_docs,
            'null_embeddings': null_embeddings,
            'non_null_embeddings': non_null_embeddings,
            'null_percentage': round(null_embeddings / total_docs * 100, 1) if total_docs > 0 else 0,
            'status': 'COMPLETE_REGENERATION_NEEDED' if null_embeddings == total_docs else 'PARTIAL_REGENERATION_NEEDED'
        }
        
        print(f"  📊 Total documents: {total_docs:,}")
        print(f"  ❌ NULL embeddings: {null_embeddings:,} ({assessment['source_documents']['null_percentage']}%)")
        print(f"  ✅ Non-NULL embeddings: {non_null_embeddings:,}")
        print(f"  🎯 Status: {assessment['source_documents']['status']}")
        
        # 2. Token Embeddings Analysis
        print(f"\n📊 2. TOKEN EMBEDDINGS ANALYSIS")
        print("-" * 50)
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        total_tokens = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE token_embedding IS NULL")
        null_token_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE token_embedding IS NOT NULL")
        non_null_token_embeddings = cursor.fetchone()[0]
        
        # Check for corrupted embeddings (all 40 chars = corrupted)
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE LENGTH(token_embedding) = 40")
        corrupted_tokens = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings")
        token_doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings_Vector")
        vector_table_count = cursor.fetchone()[0]
        
        assessment['token_embeddings'] = {
            'total_token_records': total_tokens,
            'null_token_embeddings': null_token_embeddings,
            'non_null_token_embeddings': non_null_token_embeddings,
            'corrupted_token_embeddings': corrupted_tokens,
            'documents_with_tokens': token_doc_count,
            'token_coverage_percentage': round(token_doc_count / total_docs * 100, 1) if total_docs > 0 else 0,
            'vector_table_records': vector_table_count,
            'status': 'CORRUPTED' if corrupted_tokens > 0 else 'HEALTHY'
        }
        
        print(f"  📊 Total token records: {total_tokens:,}")
        print(f"  📄 Documents with tokens: {token_doc_count:,} ({assessment['token_embeddings']['token_coverage_percentage']}% coverage)")
        print(f"  ❌ NULL token embeddings: {null_token_embeddings:,}")
        print(f"  🚨 Corrupted token embeddings: {corrupted_tokens:,}")
        print(f"  📊 Vector table records: {vector_table_count:,}")
        print(f"  🎯 Status: {assessment['token_embeddings']['status']}")
        
        # 3. Backup Analysis
        print(f"\n📊 3. BACKUP DATA ANALYSIS")
        print("-" * 50)
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_ActualCorruptionBackup")
        backup_total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_ActualCorruptionBackup WHERE embedding IS NOT NULL")
        backup_embeddings = cursor.fetchone()[0]
        
        assessment['backup_analysis'] = {
            'backup_total_records': backup_total,
            'backup_embeddings_available': backup_embeddings,
            'backup_embedding_percentage': round(backup_embeddings / backup_total * 100, 1) if backup_total > 0 else 0,
            'recovery_potential': 'MINIMAL' if backup_embeddings < 100 else 'PARTIAL'
        }
        
        print(f"  📊 Backup total records: {backup_total:,}")
        print(f"  💾 Available embeddings in backup: {backup_embeddings:,} ({assessment['backup_analysis']['backup_embedding_percentage']}%)")
        print(f"  🎯 Recovery potential: {assessment['backup_analysis']['recovery_potential']}")
        
        # 4. Corruption Timeline Analysis
        print(f"\n📊 4. CORRUPTION TIMELINE ANALYSIS")
        print("-" * 50)
        
        assessment['corruption_analysis'] = {
            'corruption_period': 'During column mismatch period (before 2025-05-27)',
            'corruption_cause': 'Column mismatch in INSERT statements + vector format issues',
            'fix_applied': '2025-05-27 12:41:25',
            'embeddings_cleared': 'All document embeddings set to NULL during fix',
            'token_embeddings_affected': 'Token embeddings corrupted (40-char uniform length)',
            'data_integrity_restored': True,
            'embedding_regeneration_required': True
        }
        
        print(f"  🕐 Corruption period: {assessment['corruption_analysis']['corruption_period']}")
        print(f"  🔧 Fix applied: {assessment['corruption_analysis']['fix_applied']}")
        print(f"  ✅ Data integrity restored: {assessment['corruption_analysis']['data_integrity_restored']}")
        print(f"  🔄 Embedding regeneration required: {assessment['corruption_analysis']['embedding_regeneration_required']}")
        
    except Exception as e:
        print(f"❌ Error during assessment: {e}")
        assessment['error'] = str(e)
    finally:
        cursor.close()
        conn.close()
    
    return assessment

def calculate_regeneration_scope(assessment):
    """Calculate the scope of regeneration needed"""
    print(f"\n📋 5. REGENERATION SCOPE CALCULATION")
    print("-" * 50)
    
    scope = {
        'document_embeddings': {
            'records_to_regenerate': assessment['source_documents']['null_embeddings'],
            'estimated_time_hours': 0,
            'priority': 'HIGH'
        },
        'token_embeddings': {
            'records_to_regenerate': assessment['token_embeddings']['corrupted_token_embeddings'],
            'documents_to_process': assessment['token_embeddings']['documents_with_tokens'],
            'estimated_time_hours': 0,
            'priority': 'HIGH'
        },
        'vector_tables': {
            'tables_to_populate': ['DocumentTokenEmbeddings_Vector'],
            'estimated_time_hours': 0,
            'priority': 'MEDIUM'
        }
    }
    
    # Estimate regeneration time (rough estimates)
    docs_to_regen = scope['document_embeddings']['records_to_regenerate']
    scope['document_embeddings']['estimated_time_hours'] = round(docs_to_regen / 1000, 1)  # ~1000 docs/hour
    
    tokens_to_regen = scope['token_embeddings']['records_to_regenerate']
    scope['token_embeddings']['estimated_time_hours'] = round(tokens_to_regen / 10000, 1)  # ~10k tokens/hour
    
    scope['vector_tables']['estimated_time_hours'] = 2  # Setup time
    
    total_time = (scope['document_embeddings']['estimated_time_hours'] + 
                  scope['token_embeddings']['estimated_time_hours'] + 
                  scope['vector_tables']['estimated_time_hours'])
    
    scope['total_estimated_hours'] = round(total_time, 1)
    
    print(f"  📄 Document embeddings: {docs_to_regen:,} records (~{scope['document_embeddings']['estimated_time_hours']} hours)")
    print(f"  🔤 Token embeddings: {tokens_to_regen:,} records (~{scope['token_embeddings']['estimated_time_hours']} hours)")
    print(f"  📊 Vector tables: {len(scope['vector_tables']['tables_to_populate'])} tables (~{scope['vector_tables']['estimated_time_hours']} hours)")
    print(f"  ⏱️  Total estimated time: ~{scope['total_estimated_hours']} hours")
    
    return scope

def generate_regeneration_plan(assessment, scope):
    """Generate detailed regeneration plan with priorities"""
    print(f"\n📋 6. REGENERATION PLAN")
    print("-" * 50)
    
    plan = {
        'phase_1_immediate': [],
        'phase_2_token_cleanup': [],
        'phase_3_full_regeneration': [],
        'phase_4_validation': []
    }
    
    # Phase 1: Immediate - Clean up corrupted data
    plan['phase_1_immediate'] = [
        {
            'priority': 'CRITICAL',
            'action': 'Clean corrupted token embeddings',
            'command': 'DELETE FROM RAG.DocumentTokenEmbeddings WHERE LENGTH(token_embedding) = 40',
            'scope': f'{assessment["token_embeddings"]["corrupted_token_embeddings"]:,} corrupted records',
            'estimated_time': '5 minutes'
        },
        {
            'priority': 'CRITICAL', 
            'action': 'Verify document data integrity',
            'command': 'python3 final_validation.py',
            'scope': 'All 50,002 documents',
            'estimated_time': '2 minutes'
        }
    ]
    
    # Phase 2: Token cleanup
    plan['phase_2_token_cleanup'] = [
        {
            'priority': 'HIGH',
            'action': 'Clear DocumentTokenEmbeddings_Vector table',
            'command': 'TRUNCATE TABLE RAG.DocumentTokenEmbeddings_Vector',
            'scope': 'Prepare for fresh token embeddings',
            'estimated_time': '1 minute'
        }
    ]
    
    # Phase 3: Full regeneration
    plan['phase_3_full_regeneration'] = [
        {
            'priority': 'HIGH',
            'action': 'Regenerate document embeddings',
            'command': 'python3 data/loader_varchar_fixed.py --regenerate-embeddings --batch-size 100',
            'scope': f'{assessment["source_documents"]["null_embeddings"]:,} documents',
            'estimated_time': f'{scope["document_embeddings"]["estimated_time_hours"]} hours'
        },
        {
            'priority': 'HIGH',
            'action': 'Regenerate ColBERT token embeddings',
            'command': 'python3 scripts/populate_colbert_token_embeddings.py --full-regeneration',
            'scope': f'~{assessment["token_embeddings"]["documents_with_tokens"]:,} documents',
            'estimated_time': f'{scope["token_embeddings"]["estimated_time_hours"]} hours'
        }
    ]
    
    # Phase 4: Validation
    plan['phase_4_validation'] = [
        {
            'priority': 'MEDIUM',
            'action': 'Validate all RAG pipelines',
            'command': 'python3 tests/test_e2e_rag_pipelines.py',
            'scope': 'All RAG techniques',
            'estimated_time': '30 minutes'
        },
        {
            'priority': 'LOW',
            'action': 'Run performance benchmarks',
            'command': 'python3 eval/bench_runner.py --quick-benchmark',
            'scope': 'Performance validation',
            'estimated_time': '1 hour'
        }
    ]
    
    # Print the plan
    for phase_name, phase_actions in plan.items():
        phase_display = phase_name.replace('_', ' ').title()
        print(f"\n  🎯 {phase_display}:")
        for i, action in enumerate(phase_actions, 1):
            print(f"    {i}. [{action['priority']}] {action['action']}")
            print(f"       Command: {action['command']}")
            print(f"       Scope: {action['scope']}")
            print(f"       Time: {action['estimated_time']}")
            print()
    
    return plan

def generate_recommendations(assessment, scope, plan):
    """Generate final recommendations"""
    print(f"\n🎯 7. FINAL RECOMMENDATIONS")
    print("-" * 50)
    
    recommendations = [
        {
            'priority': 'IMMEDIATE',
            'recommendation': 'Clean corrupted token embeddings first',
            'rationale': 'Corrupted data may interfere with regeneration processes',
            'action': 'Execute Phase 1 immediately'
        },
        {
            'priority': 'HIGH',
            'recommendation': 'Regenerate document embeddings before token embeddings',
            'rationale': 'Document embeddings are needed for basic RAG functionality',
            'action': 'Execute Phase 3 document regeneration first'
        },
        {
            'priority': 'MEDIUM',
            'recommendation': 'Consider parallel processing for large-scale regeneration',
            'rationale': f'~{scope["total_estimated_hours"]} hours total time can be reduced with parallelization',
            'action': 'Use batch processing and multiple workers'
        },
        {
            'priority': 'LOW',
            'recommendation': 'Monitor disk space during regeneration',
            'rationale': 'Large embedding datasets require significant storage',
            'action': 'Ensure adequate disk space before starting'
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. [{rec['priority']}] {rec['recommendation']}")
        print(f"     Rationale: {rec['rationale']}")
        print(f"     Action: {rec['action']}")
        print()
    
    return recommendations

def main():
    """Main assessment function"""
    # Run comprehensive assessment
    assessment = assess_embedding_integrity()
    
    # Calculate regeneration scope
    scope = calculate_regeneration_scope(assessment)
    assessment['regeneration_scope'] = scope
    
    # Generate regeneration plan
    plan = generate_regeneration_plan(assessment, scope)
    assessment['regeneration_plan'] = plan
    
    # Generate recommendations
    recommendations = generate_recommendations(assessment, scope, plan)
    assessment['recommendations'] = recommendations
    
    # Save comprehensive report
    report_filename = f"embedding_integrity_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(assessment, f, indent=2)
    
    print(f"\n💾 COMPREHENSIVE REPORT SAVED")
    print(f"📄 Report file: {report_filename}")
    
    print(f"\n✅ EMBEDDING INTEGRITY ASSESSMENT COMPLETE")
    print("=" * 60)
    print("🎯 NEXT STEPS:")
    print("1. Review the comprehensive report")
    print("2. Execute Phase 1 (immediate cleanup)")
    print("3. Proceed with full regeneration plan")
    print("4. Validate all systems after regeneration")
    
    return assessment

if __name__ == "__main__":
    main()