#!/usr/bin/env python3
"""
Monitor IRIS ingestion progress while it's running.
This script tracks database growth, document counts, and system health.
"""

import subprocess
import time
import json
import datetime
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection

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
        except Exception:
            counts['documents'] = 0
        
        # Check chunks table
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            result = cursor.fetchone()
            counts['chunks'] = result[0] if result else 0
        except Exception:
            counts['chunks'] = 0
        
        # Check ColBERT token embeddings
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            result = cursor.fetchone()
            counts['tokens'] = result[0] if result else 0
        except Exception:
            counts['tokens'] = 0
        
        cursor.close()
        conn.close()
        
        return counts
        
    except Exception:
        return None

def get_volume_info():
    """Get Docker volume information."""
    try:
        result = subprocess.run([
            'docker', 'volume', 'inspect', 'rag-templates_iris_db_data'
        ], capture_output=True, text=True, check=True)
        volume_info = json.loads(result.stdout)[0]
        mountpoint = volume_info['Mountpoint']
        
        # Try to get volume size without sudo first
        size_result = subprocess.run([
            'du', '-sh', mountpoint
        ], capture_output=True, text=True)
        
        if size_result.returncode == 0:
            volume_size = size_result.stdout.strip().split('\t')[0]
        else:
            volume_size = "Size unavailable"
            
        return f"Volume: {mountpoint} ({volume_size})"
    except:
        return "Volume info unavailable"

def monitor_progress(interval=30, duration=None):
    """Monitor ingestion progress."""
    print("🔍 IRIS Ingestion Progress Monitor (Fixed)")
    print("=" * 50)
    start_time = time.time()  # Capture start time immediately
    print(f"Started at: {datetime.datetime.now()}")
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
            elapsed = time.time() - start_time
            
            print(f"\n📊 Update #{iteration} - {current_time.strftime('%H:%M:%S')}")
            # Format elapsed time appropriately - ensure minimum display of 0.1 seconds
            elapsed_display = max(elapsed, 0.1)
            if elapsed_display < 60:
                print(f"⏱️  Elapsed: {elapsed_display:.1f} seconds")
            elif elapsed_display < 3600:
                print(f"⏱️  Elapsed: {elapsed_display/60:.1f} minutes")
            else:
                print(f"⏱️  Elapsed: {elapsed_display/3600:.1f} hours")
            
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
            if duration and elapsed >= duration:
                print(f"\n✅ Monitoring completed after {duration} seconds")
                break
                
            print(f"\n⏳ Next update in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user after {elapsed/60:.1f} minutes")
    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor IRIS ingestion progress")
    parser.add_argument("--interval", "-i", type=int, default=30, 
                       help="Monitoring interval in seconds (default: 30)")
    parser.add_argument("--duration", "-d", type=int, 
                       help="Total monitoring duration in seconds (default: unlimited)")
    
    args = parser.parse_args()
    
    monitor_progress(interval=args.interval, duration=args.duration)