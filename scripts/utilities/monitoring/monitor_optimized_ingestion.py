#!/usr/bin/env python3
"""
Monitor Optimized Ingestion Progress

This script monitors the optimized ingestion process and provides
real-time performance metrics and progress updates.
"""

import time
import os
import sys
from datetime import datetime

def monitor_ingestion_progress():
    """Monitor the optimized ingestion process."""
    print("🔍 MONITORING OPTIMIZED INGESTION PROGRESS")
    print("=" * 60)
    
    log_file = "optimized_ingestion_output.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📋 Monitoring log file: {log_file}")
    print(f"⏰ Started monitoring at: {datetime.now()}")
    print()
    
    last_position = 0
    last_file_count = 0
    last_doc_count = 0
    last_token_count = 0
    start_time = time.time()
    
    while True:
        try:
            # Check if process is still running
            result = os.system("ps -p 94718 > /dev/null 2>&1")
            if result != 0:
                print("🛑 Process 94718 is no longer running")
                break
            
            # Read new log entries
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
            
            # Parse progress information
            current_file_count = last_file_count
            current_doc_count = last_doc_count
            current_token_count = last_token_count
            current_rate = 0
            
            for line in new_lines:
                # File processing progress
                if "Processed" in line and "files in" in line and "files/s" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "Processed":
                                current_file_count = int(parts[i+1])
                                break
                    except:
                        pass
                
                # Document loading progress
                elif "Progress:" in line and "docs," in line and "docs/sec" in line:
                    try:
                        # Extract: Progress: 100/50000 docs, 21872 tokens (13.75 docs/sec)
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "Progress:":
                                doc_info = parts[i+1].split('/')
                                current_doc_count = int(doc_info[0])
                                break
                            elif "tokens" in part:
                                current_token_count = int(parts[i-1])
                            elif "docs/sec)" in part:
                                rate_str = parts[i-1].replace('(', '')
                                current_rate = float(rate_str)
                    except:
                        pass
                
                # Performance warnings
                elif "PERFORMANCE WARNING" in line or "DEGRADING PERFORMANCE" in line:
                    print(f"⚠️  {line.strip()}")
                
                # Success/completion messages
                elif "SUCCESS" in line or "completed successfully" in line:
                    print(f"✅ {line.strip()}")
            
            # Update progress if there were changes
            if (current_file_count != last_file_count or 
                current_doc_count != last_doc_count or 
                current_token_count != last_token_count):
                
                elapsed = time.time() - start_time
                
                print(f"\r📊 Progress Update ({datetime.now().strftime('%H:%M:%S')}):")
                
                if current_file_count > last_file_count:
                    print(f"   📄 Files processed: {current_file_count:,}")
                
                if current_doc_count > last_doc_count:
                    print(f"   📝 Documents loaded: {current_doc_count:,}")
                    print(f"   🔢 Token embeddings: {current_token_count:,}")
                    if current_rate > 0:
                        print(f"   ⚡ Current rate: {current_rate:.2f} docs/sec")
                        
                        # Performance assessment
                        if current_rate >= 10.0:
                            status = "🎉 EXCELLENT"
                        elif current_rate >= 5.0:
                            status = "✅ GOOD"
                        elif current_rate >= 2.0:
                            status = "⚠️  ACCEPTABLE"
                        else:
                            status = "❌ POOR"
                        print(f"   📈 Performance: {status}")
                
                print(f"   ⏱️  Elapsed time: {elapsed/60:.1f} minutes")
                print()
                
                last_file_count = current_file_count
                last_doc_count = current_doc_count
                last_token_count = current_token_count
            
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            time.sleep(5)
    
    # Final status check
    print("\n📊 FINAL STATUS CHECK")
    print("=" * 30)
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for completion or error messages
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if "completed successfully" in line:
                print("✅ Ingestion completed successfully!")
                break
            elif "failed" in line or "error" in line.lower():
                print(f"❌ Error detected: {line.strip()}")
                break
        else:
            print("⏳ Process may still be running or ended unexpectedly")
            
    except Exception as e:
        print(f"❌ Error reading final status: {e}")

if __name__ == "__main__":
    monitor_ingestion_progress()