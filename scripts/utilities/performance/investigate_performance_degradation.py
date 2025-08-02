#!/usr/bin/env python3
"""
Investigate Performance Degradation in Optimized Ingestion

This script analyzes the current database state and performance metrics
to identify the root cause of recurring performance degradation.
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common.iris_connector import get_iris_connection

def analyze_database_state():
    """Analyze current database state and table sizes."""
    print("🔍 ANALYZING DATABASE STATE FOR PERFORMANCE DEGRADATION")
    print("=" * 70)
    
    try:
        conn = get_iris_connection()
        if not conn:
            print("❌ Failed to connect to database")
            return
        
        cursor = conn.cursor()
        
        # Check document counts
        print("\n📊 DOCUMENT COUNTS:")
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        print(f"   SourceDocuments: {doc_count:,}")
        
        # Check token embedding counts
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        print(f"   DocumentTokenEmbeddings: {token_count:,}")
        
        # Calculate average tokens per document
        if doc_count > 0:
            avg_tokens = token_count / doc_count
            print(f"   Average tokens per document: {avg_tokens:.1f}")
        
        # Check table sizes and performance metrics
        print("\n📈 TABLE SIZE ANALYSIS:")
        
        # Get table size information (IRIS specific)
        try:
            cursor.execute("""
                SELECT 
                    'SourceDocuments' as TableName,
                    COUNT(*) as RowCount
                FROM RAG.SourceDocuments
                UNION ALL
                SELECT 
                    'DocumentTokenEmbeddings' as TableName,
                    COUNT(*) as RowCount
                FROM RAG.DocumentTokenEmbeddings
            """)
            
            for row in cursor.fetchall():
                table_name, row_count = row
                print(f"   {table_name}: {row_count:,} rows")
                
        except Exception as e:
            print(f"   Error getting table sizes: {e}")
        
        # Check for potential performance issues
        print("\n🔍 PERFORMANCE ISSUE ANALYSIS:")
        
        # Check if there are any indexes
        try:
            cursor.execute("""
                SELECT 
                    TABLE_NAME,
                    INDEX_NAME,
                    COLUMN_NAME
                FROM INFORMATION_SCHEMA.INDEXES 
                WHERE TABLE_SCHEMA = 'RAG'
                ORDER BY TABLE_NAME, INDEX_NAME
            """)
            
            indexes = cursor.fetchall()
            if indexes:
                print("   📋 Current indexes:")
                for table, index, column in indexes:
                    print(f"      {table}.{index} on {column}")
            else:
                print("   ⚠️  NO INDEXES FOUND - This could be a major performance issue!")
                
        except Exception as e:
            print(f"   Error checking indexes: {e}")
        
        # Check recent insertion patterns
        print("\n📊 RECENT INSERTION ANALYSIS:")
        try:
            # Get sample of recent documents to check insertion patterns
            cursor.execute("""
                SELECT TOP 10 doc_id 
                FROM RAG.SourceDocuments 
                ORDER BY doc_id DESC
            """)
            recent_docs = cursor.fetchall()
            print(f"   Recent document IDs: {[doc[0] for doc in recent_docs]}")
            
            # Check token embedding distribution
            cursor.execute("""
                SELECT doc_id, COUNT(*) as token_count
                FROM RAG.DocumentTokenEmbeddings 
                WHERE doc_id IN (
                    SELECT TOP 5 doc_id 
                    FROM RAG.SourceDocuments 
                    ORDER BY doc_id DESC
                )
                GROUP BY doc_id
                ORDER BY doc_id DESC
            """)
            
            token_dist = cursor.fetchall()
            print("   Token distribution for recent docs:")
            for doc_id, tokens in token_dist:
                print(f"      {doc_id}: {tokens} tokens")
                
        except Exception as e:
            print(f"   Error analyzing recent insertions: {e}")
        
        # Estimate database growth rate
        print("\n📈 GROWTH RATE ANALYSIS:")
        if doc_count > 0 and token_count > 0:
            # Rough estimates based on current data
            estimated_final_docs = 100000  # Target
            estimated_final_tokens = token_count * (estimated_final_docs / doc_count)
            
            print(f"   Current progress: {doc_count:,} / {estimated_final_docs:,} docs ({doc_count/estimated_final_docs*100:.1f}%)")
            print(f"   Estimated final token count: {estimated_final_tokens:,.0f}")
            print(f"   Remaining tokens to insert: {estimated_final_tokens - token_count:,.0f}")
            
            # Performance projection
            if doc_count >= 1000:  # Need reasonable sample size
                current_rate = 15.0  # docs/sec from logs
                remaining_docs = estimated_final_docs - doc_count
                estimated_time_hours = (remaining_docs / current_rate) / 3600
                
                print(f"   At current rate ({current_rate} docs/sec):")
                print(f"      Estimated completion time: {estimated_time_hours:.1f} hours")
                
                if estimated_time_hours > 24:
                    print("   ⚠️  WARNING: Completion time exceeds 24 hours!")
                if estimated_time_hours > 72:
                    print("   🚨 CRITICAL: Completion time exceeds 72 hours!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error analyzing database state: {e}")

def analyze_performance_patterns():
    """Analyze performance patterns from the log file."""
    print("\n🔍 ANALYZING PERFORMANCE PATTERNS FROM LOGS")
    print("=" * 50)
    
    log_file = "logs/optimized_ingestion_20250527_162507.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Extract performance data
        performance_data = []
        batch_times = []
        
        for line in lines:
            if "Progress:" in line and "docs/sec" in line:
                try:
                    # Extract: Progress: 1350/50000 docs, 277706 tokens (15.15 docs/sec)
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Progress:":
                            doc_info = parts[i+1].split('/')
                            current_docs = int(doc_info[0])
                            break
                        elif "docs/sec)" in part:
                            rate_str = parts[i-1].replace('(', '')
                            rate = float(rate_str)
                            performance_data.append((current_docs, rate))
                            break
                except:
                    continue
            
            elif "Executing batch" in line and "with" in line:
                # Track batch execution times
                timestamp_str = line.split(' - ')[0]
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    batch_times.append(timestamp)
                except:
                    continue
        
        if performance_data:
            print(f"📊 Performance data points collected: {len(performance_data)}")
            
            # Analyze performance trends
            early_rates = [rate for docs, rate in performance_data[:10] if docs <= 500]
            recent_rates = [rate for docs, rate in performance_data[-10:] if docs >= 1000]
            
            if early_rates and recent_rates:
                early_avg = sum(early_rates) / len(early_rates)
                recent_avg = sum(recent_rates) / len(recent_rates)
                
                print(f"   Early performance (first 500 docs): {early_avg:.2f} docs/sec")
                print(f"   Recent performance (last samples): {recent_avg:.2f} docs/sec")
                
                degradation = (early_avg - recent_avg) / early_avg * 100
                print(f"   Performance change: {degradation:+.1f}%")
                
                if abs(degradation) > 10:
                    print("   ⚠️  SIGNIFICANT PERFORMANCE CHANGE DETECTED!")
                
                # Check for exponential degradation pattern
                if len(performance_data) >= 20:
                    mid_point = len(performance_data) // 2
                    first_quarter = performance_data[:mid_point//2]
                    last_quarter = performance_data[-mid_point//2:]
                    
                    if first_quarter and last_quarter:
                        first_avg = sum(rate for _, rate in first_quarter) / len(first_quarter)
                        last_avg = sum(rate for _, rate in last_quarter) / len(last_quarter)
                        
                        total_degradation = (first_avg - last_avg) / first_avg * 100
                        print(f"   Total degradation: {total_degradation:+.1f}%")
                        
                        if total_degradation > 20:
                            print("   🚨 EXPONENTIAL DEGRADATION PATTERN DETECTED!")
        
        # Analyze batch timing patterns
        if len(batch_times) >= 10:
            print(f"\n⏱️  BATCH TIMING ANALYSIS:")
            
            # Calculate intervals between batches
            intervals = []
            for i in range(1, len(batch_times)):
                interval = (batch_times[i] - batch_times[i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                early_intervals = intervals[:10]
                recent_intervals = intervals[-10:]
                
                early_avg = sum(early_intervals) / len(early_intervals)
                recent_avg = sum(recent_intervals) / len(recent_intervals)
                
                print(f"   Early batch intervals: {early_avg:.1f}s average")
                print(f"   Recent batch intervals: {recent_avg:.1f}s average")
                
                timing_degradation = (recent_avg - early_avg) / early_avg * 100
                print(f"   Batch timing change: {timing_degradation:+.1f}%")
                
                if timing_degradation > 50:
                    print("   🚨 SEVERE BATCH TIMING DEGRADATION!")
        
    except Exception as e:
        print(f"❌ Error analyzing performance patterns: {e}")

def identify_root_causes():
    """Identify potential root causes of performance degradation."""
    print("\n🔍 ROOT CAUSE ANALYSIS")
    print("=" * 30)
    
    print("🧠 POTENTIAL ROOT CAUSES:")
    
    print("\n1. 🗄️  DATABASE SCALING ISSUES:")
    print("   - Token embedding table growing exponentially")
    print("   - No indexes on frequently queried columns")
    print("   - IRIS Community Edition memory limitations")
    print("   - Transaction log growth without proper management")
    
    print("\n2. 🔄 INSERTION PATTERN PROBLEMS:")
    print("   - Batch size too large for current table size")
    print("   - Token embedding insertions causing lock contention")
    print("   - VARCHAR embedding storage inefficient for large vectors")
    print("   - No connection pooling or connection reuse")
    
    print("\n3. 💾 MEMORY AND RESOURCE ISSUES:")
    print("   - Python process memory growth (memory leaks)")
    print("   - IRIS cache pressure from large embedding table")
    print("   - Disk I/O bottlenecks from unoptimized storage")
    print("   - CPU overhead from vector string parsing")
    
    print("\n4. 🏗️  ARCHITECTURAL LIMITATIONS:")
    print("   - VARCHAR storage for embeddings is fundamentally inefficient")
    print("   - Single-threaded insertion process")
    print("   - No partitioning strategy for large tables")
    print("   - Lack of proper indexing strategy")
    
    print("\n🎯 RECOMMENDED SOLUTIONS:")
    
    print("\n1. 🚀 IMMEDIATE OPTIMIZATIONS:")
    print("   - Add indexes on doc_id columns")
    print("   - Reduce batch sizes further (10-15 docs)")
    print("   - Implement connection pooling")
    print("   - Add periodic COMMIT and connection refresh")
    
    print("\n2. 🏗️  ARCHITECTURAL CHANGES:")
    print("   - Switch to binary embedding storage")
    print("   - Implement table partitioning")
    print("   - Use separate database for token embeddings")
    print("   - Implement parallel insertion workers")
    
    print("\n3. 🔄 ALTERNATIVE APPROACHES:")
    print("   - File-based token embedding storage")
    print("   - Streaming insertion with backpressure")
    print("   - Checkpoint-based resumable ingestion")
    print("   - Hybrid storage (DB + file system)")

def main():
    """Main analysis function."""
    print("🚀 PERFORMANCE DEGRADATION INVESTIGATION")
    print("=" * 50)
    print(f"⏰ Analysis started at: {datetime.now()}")
    
    analyze_database_state()
    analyze_performance_patterns()
    identify_root_causes()
    
    print(f"\n✅ Analysis completed at: {datetime.now()}")

if __name__ == "__main__":
    main()