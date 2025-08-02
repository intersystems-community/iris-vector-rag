#!/usr/bin/env python3
"""
Ingestion Status Checker

This script helps monitor the background ingestion process.
"""

import os
import sys
import subprocess
from datetime import datetime

def check_process_status():
    """Check if the ingestion process is still running."""
    try:
        # Look for the background ingestion process
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if 'run_background_ingestion.py' in line and 'grep' not in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time = parts[9]
                print(f"✅ Ingestion process is RUNNING")
                print(f"   PID: {pid}")
                print(f"   CPU: {cpu}%")
                print(f"   Memory: {mem}%")
                print(f"   Runtime: {time}")
                return True
        
        print("❌ Ingestion process is NOT running")
        return False
        
    except Exception as e:
        print(f"Error checking process status: {e}")
        return False

def check_log_progress():
    """Check the latest progress from the log file."""
    log_file = "ingestion_background.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"\n📋 Latest log entries from {log_file}:")
    print("=" * 60)
    
    try:
        # Get the last 15 lines of the log
        result = subprocess.run(
            ["tail", "-15", log_file],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        else:
            print("No recent log entries found")
            
    except Exception as e:
        print(f"Error reading log file: {e}")

def check_database_count():
    """Check current document count in database."""
    try:
        # Add the project root to the path
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        
        from common.iris_connector import get_iris_connection
        
        conn = get_iris_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            current_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            print(f"\n📊 Current database status:")
            print(f"   Documents in database: {current_count:,}")
            return current_count
        else:
            print("❌ Could not connect to database")
            return None
            
    except Exception as e:
        print(f"Error checking database: {e}")
        return None

def estimate_completion():
    """Estimate completion time based on current progress."""
    log_file = "ingestion_background.log"
    
    if not os.path.exists(log_file):
        return
    
    try:
        # Look for processing rate information in the log
        result = subprocess.run(
            ["grep", "docs/sec", log_file],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if lines:
                last_line = lines[-1]
                print(f"\n⏱️  Latest processing rate info:")
                print(f"   {last_line.split(' - ')[-1] if ' - ' in last_line else last_line}")
        
        # Look for document counts
        result = subprocess.run(
            ["grep", "Loaded.*SourceDocuments", log_file],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if lines:
                last_line = lines[-1]
                print(f"   {last_line.split(' - ')[-1] if ' - ' in last_line else last_line}")
                
    except Exception as e:
        print(f"Error estimating completion: {e}")

def main():
    """Main status check function."""
    print(f"🔍 Ingestion Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check if process is running
    is_running = check_process_status()
    
    # Check database count
    doc_count = check_database_count()
    
    # Show recent log progress
    check_log_progress()
    
    # Show completion estimate
    if is_running:
        estimate_completion()
        
        print(f"\n💡 To monitor continuously, run:")
        print(f"   tail -f ingestion_background.log")
        print(f"\n💡 To check status again later, run:")
        print(f"   python3 check_ingestion_status.py")
    else:
        print(f"\n⚠️  Process appears to have stopped. Check the log for details.")
        print(f"   To restart: nohup python3 run_background_ingestion.py > ingestion_background.log 2>&1 &")

if __name__ == "__main__":
    main()