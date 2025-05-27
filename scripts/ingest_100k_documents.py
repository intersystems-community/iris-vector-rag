#!/usr/bin/env python3
"""
Massive Scale Document Ingestion Pipeline (100K Documents)

Enterprise-scale document processing pipeline with:
- Batch processing with memory management
- Checkpointing to resume interrupted ingestion
- Progress monitoring and ETA calculations
- Optimized database operations for massive scale
- Support for both RAG and RAG_HNSW schemas

Usage:
    python scripts/ingest_100k_documents.py --target-docs 100000
    python scripts/ingest_100k_documents.py --resume-from-checkpoint
    python scripts/ingest_100k_documents.py --target-docs 50000 --batch-size 1000
"""

import os
import sys
import logging
import time
import json
import argparse
import psutil
import numpy as np
import threading
import pickle
import gc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import signal

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
from data.loader_varchar_fixed import load_documents_to_iris
from data.pmc_processor import process_pmc_files
from colbert.doc_encoder import get_colbert_doc_encoder

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest_100k_documents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IngestionCheckpoint:
    """Checkpoint data for resuming ingestion"""
    target_docs: int
    current_docs: int
    processed_files: List[str]
    failed_files: List[Dict[str, Any]]
    start_time: float
    last_checkpoint_time: float
    total_ingestion_time: float
    error_count: int
    batch_count: int
    schema_type: str  # 'RAG' or 'RAG_HNSW'

class IngestionMonitor:
    """System resource and progress monitor for ingestion"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        self.start_time = time.time()
        
    def start(self):
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("📊 Ingestion monitoring started")
        
    def stop(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        return self.metrics
        
    def _monitor_loop(self):
        while self.monitoring:
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                disk = psutil.disk_usage('.')
                
                metric = {
                    'timestamp': time.time(),
                    'memory_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'cpu_percent': cpu,
                    'disk_free_gb': disk.free / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100
                }
                self.metrics.append(metric)
                
                # Alert on resource issues
                if memory.percent > 90:
                    logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
                    gc.collect()  # Force garbage collection
                if disk.free < 5 * 1024**3:  # Less than 5GB free
                    logger.warning(f"⚠️ Low disk space: {disk.free/(1024**3):.1f}GB free")
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            time.sleep(15)  # Monitor every 15 seconds

class MassiveScaleIngestionPipeline:
    """Enterprise-grade document ingestion pipeline for 100k+ documents"""
    
    def __init__(self, data_dir: str = "data/pmc_oas_downloaded", checkpoint_interval: int = 600):
        self.data_dir = Path(data_dir)
        self.checkpoint_interval = checkpoint_interval  # seconds
        self.checkpoint_file = Path("ingestion_checkpoint.pkl")
        
        # Database connections
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        
        # Monitoring
        self.monitor = IngestionMonitor()
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Checkpoint data
        self.checkpoint: Optional[IngestionCheckpoint] = None
        self.last_checkpoint_save = time.time()
        
        logger.info(f"🚀 MassiveScaleIngestionPipeline initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"⏰ Checkpoint interval: {checkpoint_interval} seconds")
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        if not self.checkpoint:
            return
            
        try:
            self.checkpoint.last_checkpoint_time = time.time()
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.checkpoint, f)
            logger.info(f"💾 Checkpoint saved: {self.checkpoint.current_docs}/{self.checkpoint.target_docs} documents")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint from file"""
        if not self.checkpoint_file.exists():
            logger.info("📋 No checkpoint file found, starting fresh")
            return False
            
        try:
            with open(self.checkpoint_file, 'rb') as f:
                self.checkpoint = pickle.load(f)
            logger.info(f"📋 Checkpoint loaded: {self.checkpoint.current_docs}/{self.checkpoint.target_docs} documents")
            logger.info(f"⏱️ Previous session time: {self.checkpoint.total_ingestion_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def create_checkpoint(self, target_docs: int, schema_type: str = "RAG"):
        """Create new checkpoint"""
        self.checkpoint = IngestionCheckpoint(
            target_docs=target_docs,
            current_docs=0,
            processed_files=[],
            failed_files=[],
            start_time=time.time(),
            last_checkpoint_time=time.time(),
            total_ingestion_time=0.0,
            error_count=0,
            batch_count=0,
            schema_type=schema_type
        )
        logger.info(f"📋 New checkpoint created for {target_docs} documents ({schema_type} schema)")
    
    def should_save_checkpoint(self) -> bool:
        """Check if it's time to save checkpoint"""
        return time.time() - self.last_checkpoint_save >= self.checkpoint_interval
    
    def calculate_eta(self) -> str:
        """Calculate estimated time to completion"""
        if not self.checkpoint:
            return "Unknown"
            
        elapsed = time.time() - self.checkpoint.start_time + self.checkpoint.total_ingestion_time
        if elapsed == 0 or self.checkpoint.current_docs == 0:
            return "Unknown"
            
        rate = self.checkpoint.current_docs / elapsed
        remaining = self.checkpoint.target_docs - self.checkpoint.current_docs
        
        if rate == 0:
            return "Unknown"
            
        eta_seconds = remaining / rate
        eta_delta = timedelta(seconds=int(eta_seconds))
        return str(eta_delta)
    
    def setup_database_and_models(self, schema_type: str = "RAG") -> bool:
        """Setup database connection and models"""
        logger.info(f"🔧 Setting up database connection and models ({schema_type} schema)...")
        
        try:
            # Database connection
            self.connection = get_iris_connection()
            if not self.connection:
                raise Exception("Failed to get database connection")
            
            # Check current document count
            table_name = f"{schema_type}.SourceDocuments"
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            current_docs = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
            docs_with_embeddings = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"📊 Database ({schema_type}): {current_docs:,} total docs, {docs_with_embeddings:,} with embeddings")
            
            # Setup models
            self.embedding_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
            self.llm_func = get_llm_func(provider="stub")
            self.colbert_encoder = get_colbert_doc_encoder()
            
            logger.info("✅ Database and models setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def get_available_files(self) -> List[str]:
        """Get list of available PMC XML files"""
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return []
        
        xml_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(xml_files):,} XML files in {self.data_dir}")
        return xml_files
    
    def get_current_document_count(self, schema_type: str = "RAG") -> int:
        """Get current document count from database"""
        try:
            table_name = f"{schema_type}.SourceDocuments"
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0
    
    def get_processed_doc_ids(self, schema_type: str = "RAG") -> set:
        """Get set of already processed document IDs"""
        try:
            table_name = f"{schema_type}.SourceDocuments"
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT doc_id FROM {table_name}")
            doc_ids = {row[0] for row in cursor.fetchall()}
            cursor.close()
            return doc_ids
        except Exception as e:
            logger.error(f"❌ Error getting processed doc IDs: {e}")
            return set()
    
    def extract_pmc_id_from_path(self, file_path: str) -> str:
        """Extract PMC ID from file path"""
        try:
            # Extract PMC ID from path like: data/pmc_100k_downloaded/PMC174xxxxxx/PMC1748350588.xml
            import os
            filename = os.path.basename(file_path)
            if filename.startswith('PMC') and filename.endswith('.xml'):
                return filename[:-4]  # Remove .xml extension
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting PMC ID from {file_path}: {e}")
            return None
    
    def process_file_batch(self, file_batch: List[str], batch_num: int, total_batches: int) -> Dict[str, Any]:
        """Process a batch of files"""
        batch_start = time.time()
        logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(file_batch)} files)")
        
        batch_results = {
            'processed_count': 0,
            'loaded_count': 0,
            'error_count': 0,
            'processing_time': 0,
            'files_processed': [],
            'files_failed': []
        }
        
        try:
            # Process files in the batch
            all_documents = []
            for file_path in file_batch:
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping batch processing")
                    break
                    
                try:
                    # Fix API interface: extract_pmc_metadata expects individual file path
                    from data.pmc_processor import extract_pmc_metadata
                    document = extract_pmc_metadata(file_path)
                    if document and document.get('title') != 'Error':
                        all_documents.append(document)
                        batch_results['files_processed'].append(file_path)
                        batch_results['processed_count'] += 1
                    else:
                        logger.warning(f"⚠️ No valid document extracted from {file_path}")
                        batch_results['files_failed'].append({
                            'file': file_path,
                            'error': 'No valid document extracted or processing error',
                            'timestamp': time.time()
                        })
                        batch_results['error_count'] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {e}")
                    batch_results['files_failed'].append({
                        'file': file_path,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    batch_results['error_count'] += 1
            
            # Load documents to database if any were processed
            if all_documents and not self.shutdown_requested:
                logger.info(f"💾 Loading {len(all_documents)} documents to database...")
                load_result = load_documents_to_iris(
                    self.connection,
                    all_documents,
                    embedding_func=self.embedding_func,
                    colbert_doc_encoder_func=self.colbert_encoder,
                    batch_size=250  # Increased sub-batch size for better performance
                )
                batch_results['loaded_count'] = load_result.get('loaded_doc_count', 0)
                
                # Update checkpoint
                self.checkpoint.current_docs += batch_results['loaded_count']
                self.checkpoint.processed_files.extend(batch_results['files_processed'])
                self.checkpoint.failed_files.extend(batch_results['files_failed'])
                self.checkpoint.error_count += batch_results['error_count']
                self.checkpoint.batch_count += 1
            
            batch_results['processing_time'] = time.time() - batch_start
            
            # Memory cleanup
            del all_documents
            gc.collect()
            
            # Log batch results
            rate = batch_results['loaded_count'] / batch_results['processing_time'] if batch_results['processing_time'] > 0 else 0
            eta = self.calculate_eta()
            logger.info(f"✅ Batch {batch_num} completed: {batch_results['loaded_count']} docs loaded in {batch_results['processing_time']:.1f}s ({rate:.1f} docs/sec)")
            logger.info(f"📊 Progress: {self.checkpoint.current_docs}/{self.checkpoint.target_docs} documents, ETA: {eta}")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_num} failed: {e}")
            batch_results['error_count'] += len(file_batch)
            batch_results['processing_time'] = time.time() - batch_start
            return batch_results
    
    def ingest_to_target(self, target_docs: int, batch_size: int = 1000, resume: bool = False, schema_type: str = "RAG") -> int:
        """Ingest documents to reach target count"""
        # Load or create checkpoint
        if resume and self.load_checkpoint():
            if self.checkpoint.target_docs != target_docs:
                logger.warning(f"⚠️ Target count mismatch: checkpoint={self.checkpoint.target_docs}, requested={target_docs}")
                logger.info("📋 Updating checkpoint target count")
                self.checkpoint.target_docs = target_docs
        else:
            self.create_checkpoint(target_docs, schema_type)
        
        # Setup database and models
        if not self.setup_database_and_models(schema_type):
            logger.error("❌ Failed to setup database and models")
            return 0
        
        # Start monitoring
        self.monitor.start()
        
        try:
            # Get current count from database
            current_count = self.get_current_document_count(schema_type)
            self.checkpoint.current_docs = current_count
            
            if current_count >= target_docs:
                logger.info(f"🎯 Target already reached: {current_count} >= {target_docs}")
                return current_count
            
            needed = target_docs - current_count
            logger.info(f"📈 Need {needed} more documents to reach target of {target_docs}")
            logger.info(f"⏱️ ETA: {self.calculate_eta()}")
            
            # Get available files
            available_files = self.get_available_files()
            if not available_files:
                logger.error("❌ No XML files found to process")
                return current_count
            
            # Get already processed doc_ids from database to avoid duplicates
            processed_doc_ids = self.get_processed_doc_ids(schema_type)
            logger.info(f"📊 Found {len(processed_doc_ids)} existing documents in database")
            
            # Filter out already processed files (both from checkpoint and database)
            remaining_files = []
            for file_path in available_files:
                if file_path in self.checkpoint.processed_files:
                    continue  # Skip files in checkpoint
                
                # Extract PMC ID from file path to check against database
                pmc_id = self.extract_pmc_id_from_path(file_path)
                if pmc_id and pmc_id in processed_doc_ids:
                    continue  # Skip files already in database
                
                remaining_files.append(file_path)
            
            logger.info(f"📁 {len(remaining_files)} files remaining to process (after duplicate filtering)")
            
            if not remaining_files:
                logger.warning("⚠️ All available files have been processed")
                return current_count
            
            # Process files in batches
            total_batches = (len(remaining_files) + batch_size - 1) // batch_size
            logger.info(f"🔄 Processing {len(remaining_files)} files in {total_batches} batches of {batch_size}")
            
            for i in range(0, len(remaining_files), batch_size):
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping ingestion")
                    break
                
                if self.checkpoint.current_docs >= target_docs:
                    logger.info(f"🎯 Target reached: {self.checkpoint.current_docs}")
                    break
                
                batch_files = remaining_files[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                # Process batch
                batch_result = self.process_file_batch(batch_files, batch_num, total_batches)
                
                # Save checkpoint periodically
                if self.should_save_checkpoint():
                    self.save_checkpoint()
                    self.last_checkpoint_save = time.time()
                
                # Check memory usage and force cleanup if needed
                memory = psutil.virtual_memory()
                if memory.percent > 85:
                    logger.warning(f"⚠️ High memory usage ({memory.percent:.1f}%), forcing cleanup")
                    gc.collect()
            
            # Final count
            final_count = self.get_current_document_count(schema_type)
            self.checkpoint.current_docs = final_count
            
            return final_count
            
        finally:
            # Final checkpoint save
            if self.checkpoint:
                self.checkpoint.total_ingestion_time += time.time() - self.checkpoint.start_time
                self.save_checkpoint()
            
            # Stop monitoring
            monitoring_data = self.monitor.stop()
            
            # Generate summary report
            self.generate_summary_report(monitoring_data)
    
    def generate_summary_report(self, monitoring_data: List[Dict[str, Any]]):
        """Generate comprehensive ingestion summary report"""
        if not self.checkpoint:
            return
            
        report = {
            "ingestion_summary": {
                "target_docs": self.checkpoint.target_docs,
                "final_docs": self.checkpoint.current_docs,
                "success_rate": (self.checkpoint.current_docs / self.checkpoint.target_docs) * 100,
                "total_time_seconds": self.checkpoint.total_ingestion_time,
                "error_count": self.checkpoint.error_count,
                "batch_count": self.checkpoint.batch_count,
                "files_processed": len(self.checkpoint.processed_files),
                "files_failed": len(self.checkpoint.failed_files),
                "schema_type": self.checkpoint.schema_type
            },
            "performance_metrics": {
                "ingestion_rate_docs_per_second": self.checkpoint.current_docs / self.checkpoint.total_ingestion_time if self.checkpoint.total_ingestion_time > 0 else 0,
                "peak_memory_gb": max([m['memory_gb'] for m in monitoring_data]) if monitoring_data else 0,
                "avg_cpu_percent": sum([m['cpu_percent'] for m in monitoring_data]) / len(monitoring_data) if monitoring_data else 0,
                "disk_usage_gb": sum([m.get('disk_percent', 0) for m in monitoring_data]) / len(monitoring_data) if monitoring_data else 0
            },
            "error_details": {
                "failed_files": self.checkpoint.failed_files,
                "error_rate": (self.checkpoint.error_count / max(self.checkpoint.current_docs, 1)) * 100
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_file = f"ingestion_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("📊 INGESTION SUMMARY REPORT")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {self.checkpoint.target_docs:,} documents")
        logger.info(f"✅ Ingested: {self.checkpoint.current_docs:,} documents")
        logger.info(f"📈 Success Rate: {report['ingestion_summary']['success_rate']:.1f}%")
        logger.info(f"⏱️ Total Time: {self.checkpoint.total_ingestion_time:.1f} seconds")
        logger.info(f"🚀 Ingestion Rate: {report['performance_metrics']['ingestion_rate_docs_per_second']:.2f} docs/sec")
        logger.info(f"📦 Batches Processed: {self.checkpoint.batch_count}")
        logger.info(f"❌ Errors: {self.checkpoint.error_count}")
        logger.info(f"📄 Report saved: {report_file}")
        logger.info("=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Massive Scale Document Ingestion Pipeline")
    parser.add_argument("--target-docs", type=int, default=100000,
                       help="Target number of documents to ingest")
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                       help="Resume from existing checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/pmc_100k_downloaded",
                       help="Directory containing PMC XML files")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Number of files to process per batch")
    parser.add_argument("--checkpoint-interval", type=int, default=600,
                       help="Checkpoint save interval in seconds")
    parser.add_argument("--schema-type", type=str, default="RAG", choices=["RAG", "RAG_HNSW"],
                       help="Database schema to use")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Massive Scale Ingestion Pipeline - Target: {args.target_docs:,} documents")
    logger.info(f"📁 Data directory: {args.data_dir}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    logger.info(f"🗄️ Schema: {args.schema_type}")
    
    pipeline = MassiveScaleIngestionPipeline(args.data_dir, args.checkpoint_interval)
    
    try:
        final_count = pipeline.ingest_to_target(
            args.target_docs, 
            args.batch_size, 
            args.resume_from_checkpoint,
            args.schema_type
        )
        
        logger.info("=" * 80)
        logger.info("🎉 INGESTION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {args.target_docs:,} documents")
        logger.info(f"✅ Ingested: {final_count:,} documents")
        
        if final_count >= args.target_docs:
            logger.info("🎯 Target reached successfully!")
            return True
        else:
            logger.info(f"⚠️ Target not fully reached (missing {args.target_docs - final_count:,} documents)")
            return False
            
    except KeyboardInterrupt:
        logger.info("🛑 Ingestion interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)