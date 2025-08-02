#!/usr/bin/env python3
"""
Monitor 100k Ingestion Progress
Real-time monitoring of the conservative ingestion process
"""

import time
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common.iris_connector import get_iris_connection

def get_process_status():
    """Check if the ingestion process is running."""
    try:
        result = os.popen("ps aux | grep run_conservative_ingestion | grep -v grep").read().strip()
        if result:
            parts = result.split()
            pid = parts[1]
            cpu = parts[2]
            mem = parts[3]
            return True, pid, cpu, mem
        return False, None, None, None
    except:
        return False, None, None, None

def get_database_counts():
    """Get current database counts."""
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        
        # Get latest document
        cursor.execute("""
            SELECT TOP 1 doc_id, title, created_at 
            FROM RAG.SourceDocuments 
            WHERE doc_id NOT LIKE 'TEST_%'
            ORDER BY created_at DESC
        """)
        latest = cursor.fetchone()
        
        conn.close()
        return doc_count, token_count, latest
    except Exception as e:
        return None, None, f"Error: {e}"

def get_log_progress():
    """Get progress from log files."""
    try:
        log_files = list(Path("logs").glob("conservative_ingestion_*.log"))
        if not log_files:
            return "No log files found"
        
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        # Get last few lines
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            if lines:
                last_lines = lines[-3:]
                return ''.join(last_lines).strip()
        return "No log content"
    except Exception as e:
        return f"Error reading logs: {e}"

def get_checkpoint_status():
    """Check checkpoint file."""
    checkpoint_file = Path("data/conservative_checkpoint.json")
    if checkpoint_file.exists():
        try:
            import json
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint
        except:
            return "Error reading checkpoint"
    return "No checkpoint file"

def print_status():
    """Print comprehensive status."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"100K INGESTION STATUS - {timestamp}")
    print(f"{'='*60}")
    
    # Process status
    running, pid, cpu, mem = get_process_status()
    if running:
        print(f"🟢 PROCESS: Running (PID: {pid}, CPU: {cpu}%, MEM: {mem}%)")
    else:
        print(f"🔴 PROCESS: Not running")
    
    # Database status
    doc_count, token_count, latest = get_database_counts()
    if doc_count is not None:
        print(f"📊 DATABASE:")
        print(f"   Documents: {doc_count:,}")
        print(f"   Token embeddings: {token_count:,}")
        if latest and len(latest) >= 2:
            print(f"   Latest: {latest[0]} - {latest[1][:50]}...")
    else:
        print(f"🔴 DATABASE: Connection error")
    
    # Log progress
    log_progress = get_log_progress()
    print(f"📝 LOG PROGRESS:")
    for line in log_progress.split('\n')[-2:]:
        if line.strip():
            print(f"   {line.strip()}")
    
    # Checkpoint status
    checkpoint = get_checkpoint_status()
    if isinstance(checkpoint, dict):
        print(f"💾 CHECKPOINT:")
        print(f"   Processed: {checkpoint.get('processed_count', 0):,}")
        print(f"   Last doc: {checkpoint.get('last_doc_id', 'None')}")
        print(f"   Time: {checkpoint.get('datetime', 'Unknown')}")
    
    print(f"{'='*60}")

def monitor_continuous():
    """Monitor continuously."""
    print("Starting continuous monitoring (Ctrl+C to stop)...")
    try:
        while True:
            print_status()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuous()
    else:
        print_status()
        print("\nFor continuous monitoring, run: python monitor_100k_ingestion.py --continuous")