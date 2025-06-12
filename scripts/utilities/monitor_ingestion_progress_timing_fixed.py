#!/usr/bin/env python3
"""
Fixed IRIS ingestion progress monitor with improved timing display.
This script tracks database growth, document counts, and system health using proper IRIS Python connector.
"""

import subprocess
import time
import json
import datetime
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection, IRISConnectionError

def get_container_stats():
    """Get Docker container statistics."""
    try:
        result = subprocess.run([
            'docker', 'stats', 'iris_db_rag_standalone', '--no-stream', '--format',
            'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}'
        ], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "Container stats unavailable"

def get_database_size():
    """Get current database size."""
    try:
        result = subprocess.run([
            'docker', 'exec', 'iris_db_rag_standalone', 
            'du', '-sh', '/usr/irissys/mgr/user/'
        ], capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\t')[0]
    except subprocess.CalledProcessError:
        return "Size unavailable"

def get_document_counts():
    """Get current document counts from IRIS using proper Python connector."""
    try:
        # Get IRIS connection
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        counts = {}
        
        # Check main documents table
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
            result = cursor.fetchone()
            counts['documents'] = result[0] if result else 0
        except Exception as e:
            print(f"   Warning: Could not count SourceDocuments: {e}")
            counts['documents'] = 0
        
        # Check chunks table
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            result = cursor.fetchone()
            counts['chunks'] = result[0] if result else 0
        except Exception as e:
            print(f"   Warning: Could not count DocumentChunks: {e}")
            counts['chunks'] = 0
        
        # Check ColBERT token embeddings
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            result = cursor.fetchone()
            counts['tokens'] = result[0] if result else 0
        except Exception as e:
            print(f"   Warning: Could not count DocumentTokenEmbeddings: {e}")
            counts['tokens'] = 0
        
        # Check knowledge graph nodes
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            result = cursor.fetchone()
            counts['kg_nodes'] = result[0] if result else 0
        except Exception as e:
            print(f"   Warning: Could not count KnowledgeGraphNodes: {e}")
            counts['kg_nodes'] = 0
        
        cursor.close()
        conn.close()
        
        return counts
        
    except IRISConnectionError as e:
        print(f"   Error: Could not connect to IRIS: {e}")
        return None
    except Exception as e:
        print(f"   Error: Database query failed: {e}")
        return None

def get_table_info():
    """Get information about table existence and structure."""
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Check which tables exist
        tables_query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = 'RAG'
        ORDER BY TABLE_NAME
        """
        
        cursor.execute(tables_query)
        tables = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return tables
        
    except Exception as e:
        print(f"   Error getting table info: {e}")
        return []

def get_volume_info():
    """Get Docker volume information."""
    try:
        result = subprocess.run([
            'docker', 'volume', 'inspect', 'rag-templates_iris_db_data'
        ], capture_output=True, text=True, check=True)
        volume_info = json.loads(result.stdout)[0]
        mountpoint = volume_info['Mountpoint']
        
        # Get volume size
        size_result = subprocess.run([
            'sudo', 'du', '-sh', mountpoint
        ], capture_output=True, text=True)
        
        if size_result.returncode == 0:
            volume_size = size_result.stdout.strip().split('\t')[0]
        else:
            volume_size = "Size unavailable (need sudo)"
            
        return f"Volume: {mountpoint} ({volume_size})"
    except:
        return "Volume info unavailable"

def format_elapsed_time(elapsed_seconds):
    """Format elapsed time in a human-readable way."""
    if elapsed_seconds < 60:
        return f"{elapsed_seconds:.1f} seconds"
    elif elapsed_seconds < 3600:
        minutes = elapsed_seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = elapsed_seconds / 3600
        return f"{hours:.1f} hours"

def monitor_progress(interval=30, duration=None):
    """Monitor ingestion progress."""
    print("🔍 IRIS Ingestion Progress Monitor (Timing Fixed)")
    print("=" * 50)
    start_datetime = datetime.datetime.now()
    start_time = time.time()  # Capture start time immediately
    print(f"Started at: {start_datetime}")
    print(f"Monitoring interval: {interval} seconds")
    if duration:
        print(f"Duration: {duration} seconds")
    print()
    
    # Test database connection first
    print("🔌 Testing database connection...")
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        print("✅ Database connection successful")
        
        # Show available tables
        tables = get_table_info()
        if tables:
            print(f"📋 Available RAG tables: {', '.join(tables)}")
        else:
            print("⚠️  No RAG tables found")
        print()
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Monitoring will continue with limited functionality")
        print()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = datetime.datetime.now()
            elapsed_seconds = time.time() - start_time
            
            print(f"\n📊 Update #{iteration} - {current_time.strftime('%H:%M:%S')}")
            print(f"⏱️  Elapsed: {format_elapsed_time(elapsed_seconds)}")
            
            # Database size
            db_size = get_database_size()
            print(f"💾 Database size: {db_size}")
            
            # Document counts
            counts = get_document_counts()
            if counts:
                print(f"📄 Documents: {counts['documents']:,}")
                if counts['chunks'] > 0:
                    print(f"🧩 Chunks: {counts['chunks']:,}")
                if counts['tokens'] > 0:
                    print(f"🔤 ColBERT tokens: {counts['tokens']:,}")
                if counts['kg_nodes'] > 0:
                    print(f"🕸️  Knowledge graph nodes: {counts['kg_nodes']:,}")
            else:
                print("📄 Document counts: unavailable")
            
            # Container stats
            print(f"🐳 Container stats:")
            stats = get_container_stats()
            print(f"   {stats}")
            
            # Volume info
            volume_info = get_volume_info()
            print(f"📁 {volume_info}")
            
            # Check if duration limit reached
            if duration and elapsed_seconds >= duration:
                print(f"\n✅ Monitoring completed after {format_elapsed_time(elapsed_seconds)}")
                break
                
            print(f"\n⏳ Next update in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        elapsed_seconds = time.time() - start_time
        print(f"\n🛑 Monitoring stopped by user after {format_elapsed_time(elapsed_seconds)}")
    except Exception as e:
        elapsed_seconds = time.time() - start_time
        print(f"\n❌ Error during monitoring: {e}")
        print(f"   Ran for {format_elapsed_time(elapsed_seconds)} before error")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor IRIS ingestion progress (Timing Fixed)")
    parser.add_argument("--interval", "-i", type=int, default=30, 
                       help="Monitoring interval in seconds (default: 30)")
    parser.add_argument("--duration", "-d", type=int, 
                       help="Total monitoring duration in seconds (default: unlimited)")
    
    args = parser.parse_args()
    
    monitor_progress(interval=args.interval, duration=args.duration)