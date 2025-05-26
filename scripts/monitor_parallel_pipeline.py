#!/usr/bin/env python3
"""
Parallel Download-Ingestion Pipeline Monitor

Monitors both download and ingestion processes running simultaneously
to provide real-time status updates and coordination.
"""

import os
import sys
import time
import json
import psutil
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection

class ParallelPipelineMonitor:
    """Monitor for parallel download-ingestion pipeline"""
    
    def __init__(self):
        self.data_dir = Path("data/pmc_100k_downloaded")
        self.download_checkpoint = self.data_dir / "download_checkpoint.pkl"
        self.ingestion_checkpoint = Path("ingestion_checkpoint.pkl")
        
    def get_download_status(self):
        """Get current download status"""
        try:
            # Count available XML files
            xml_files = list(self.data_dir.glob("**/*.xml"))
            available_count = len(xml_files)
            
            # Try to read download checkpoint for progress
            download_progress = "Unknown"
            if self.download_checkpoint.exists():
                try:
                    import pickle
                    with open(self.download_checkpoint, 'rb') as f:
                        checkpoint = pickle.load(f)
                        if hasattr(checkpoint, 'processed_count'):
                            download_progress = f"{checkpoint.processed_count:,}"
                        elif isinstance(checkpoint, dict) and 'processed_count' in checkpoint:
                            download_progress = f"{checkpoint['processed_count']:,}"
                except:
                    pass
            
            return {
                'available_files': available_count,
                'progress': download_progress,
                'status': 'Active' if available_count > 0 else 'Starting'
            }
        except Exception as e:
            return {
                'available_files': 0,
                'progress': 'Error',
                'status': f'Error: {e}'
            }
    
    def get_ingestion_status(self):
        """Get current ingestion status"""
        try:
            # Database connection
            conn = get_iris_connection()
            cursor = conn.cursor()
            
            # Get current document count
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id LIKE 'PMC%'")
            pmc_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            docs_with_embeddings = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            # Try to read ingestion checkpoint
            eta = "Unknown"
            current_target = "Unknown"
            if self.ingestion_checkpoint.exists():
                try:
                    import pickle
                    with open(self.ingestion_checkpoint, 'rb') as f:
                        checkpoint = pickle.load(f)
                        current_target = f"{checkpoint.current_docs:,}/{checkpoint.target_docs:,}"
                        
                        # Calculate ETA
                        elapsed = time.time() - checkpoint.start_time + checkpoint.total_ingestion_time
                        if elapsed > 0 and checkpoint.current_docs > 0:
                            rate = checkpoint.current_docs / elapsed
                            remaining = checkpoint.target_docs - checkpoint.current_docs
                            if rate > 0:
                                eta_seconds = remaining / rate
                                eta = str(timedelta(seconds=int(eta_seconds)))
                except:
                    pass
            
            return {
                'total_docs': total_docs,
                'pmc_docs': pmc_docs,
                'docs_with_embeddings': docs_with_embeddings,
                'progress': current_target,
                'eta': eta,
                'status': 'Active' if total_docs > 0 else 'Starting'
            }
        except Exception as e:
            return {
                'total_docs': 0,
                'pmc_docs': 0,
                'docs_with_embeddings': 0,
                'progress': 'Error',
                'eta': 'Error',
                'status': f'Error: {e}'
            }
    
    def get_system_status(self):
        """Get system resource status"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('.')
            
            return {
                'memory_percent': memory.percent,
                'memory_gb': memory.used / (1024**3),
                'cpu_percent': cpu,
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100
            }
        except Exception as e:
            return {
                'memory_percent': 0,
                'memory_gb': 0,
                'cpu_percent': 0,
                'disk_free_gb': 0,
                'disk_percent': 0,
                'error': str(e)
            }
    
    def display_status(self):
        """Display comprehensive status"""
        download_status = self.get_download_status()
        ingestion_status = self.get_ingestion_status()
        system_status = self.get_system_status()
        
        print("\n" + "="*80)
        print(f"🔄 PARALLEL DOWNLOAD-INGESTION PIPELINE STATUS")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Download Status
        print(f"\n📥 DOWNLOAD PROCESS:")
        print(f"   Status: {download_status['status']}")
        print(f"   Available Files: {download_status['available_files']:,}")
        print(f"   Progress: {download_status['progress']}")
        
        # Ingestion Status
        print(f"\n💾 INGESTION PROCESS:")
        print(f"   Status: {ingestion_status['status']}")
        print(f"   Total Documents: {ingestion_status['total_docs']:,}")
        print(f"   PMC Documents: {ingestion_status['pmc_docs']:,}")
        print(f"   With Embeddings: {ingestion_status['docs_with_embeddings']:,}")
        print(f"   Progress: {ingestion_status['progress']}")
        print(f"   ETA: {ingestion_status['eta']}")
        
        # System Status
        print(f"\n🖥️  SYSTEM RESOURCES:")
        print(f"   Memory: {system_status['memory_percent']:.1f}% ({system_status['memory_gb']:.1f} GB)")
        print(f"   CPU: {system_status['cpu_percent']:.1f}%")
        print(f"   Disk Free: {system_status['disk_free_gb']:.1f} GB ({100-system_status['disk_percent']:.1f}% free)")
        
        # Coordination Status
        available_files = download_status['available_files']
        ingested_docs = ingestion_status['pmc_docs']
        remaining_to_ingest = max(0, available_files - ingested_docs)
        
        print(f"\n🔗 COORDINATION STATUS:")
        print(f"   Files Available for Ingestion: {available_files:,}")
        print(f"   Files Already Ingested: {ingested_docs:,}")
        print(f"   Files Remaining to Ingest: {remaining_to_ingest:,}")
        
        if remaining_to_ingest > 0:
            print(f"   ✅ Pipeline is processing available data")
        else:
            print(f"   ⏳ Waiting for more downloads")
        
        print("="*80)
        
        return {
            'download': download_status,
            'ingestion': ingestion_status,
            'system': system_status,
            'coordination': {
                'available_files': available_files,
                'ingested_docs': ingested_docs,
                'remaining_to_ingest': remaining_to_ingest
            }
        }
    
    def monitor_continuous(self, interval=30):
        """Continuously monitor both processes"""
        print("🚀 Starting continuous monitoring of parallel pipeline...")
        print(f"📊 Updates every {interval} seconds. Press Ctrl+C to stop.")
        
        try:
            while True:
                status = self.display_status()
                
                # Check for alerts
                if status['system']['memory_percent'] > 90:
                    print(f"\n⚠️  HIGH MEMORY ALERT: {status['system']['memory_percent']:.1f}%")
                
                if status['system']['disk_free_gb'] < 5:
                    print(f"\n⚠️  LOW DISK SPACE ALERT: {status['system']['disk_free_gb']:.1f} GB free")
                
                # Wait for next update
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor parallel download-ingestion pipeline")
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true', help='Show status once and exit')
    
    args = parser.parse_args()
    
    monitor = ParallelPipelineMonitor()
    
    if args.once:
        monitor.display_status()
    else:
        monitor.monitor_continuous(args.interval)

if __name__ == "__main__":
    main()