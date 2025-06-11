#!/usr/bin/env python3
"""
Bulletproof 100K PMC Article Downloader

Enhanced version of the PMC downloader with enterprise-grade features:
- Comprehensive error recovery and retry logic
- Progress checkpointing to resume interrupted downloads
- Rate limiting to avoid overwhelming PMC servers
- File validation and corruption detection
- Comprehensive logging for unattended operation
- Signal handling for graceful shutdown
- Automatic checkpoint saving every 10 minutes

Usage:
    python scripts/download_100k_pmc_articles.py --target-count 100000
    python scripts/download_100k_pmc_articles.py --resume-from-checkpoint
    python scripts/download_100k_pmc_articles.py --target-count 50000 --checkpoint-interval 600
"""

import os
import sys
import logging
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
import json
import gzip
from urllib.parse import urlencode
import tarfile
import signal
import threading
import hashlib
import pickle
from datetime import datetime, timedelta
import psutil
from dataclasses import dataclass, asdict
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_100k_pmc_articles.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DownloadCheckpoint:
    """Checkpoint data for resuming downloads"""
    target_count: int
    current_count: int
    downloaded_files: List[str]
    failed_downloads: List[Dict[str, Any]]
    bulk_files_completed: List[str]
    pmc_ids_processed: List[str]
    start_time: float
    last_checkpoint_time: float
    total_download_time: float
    error_count: int
    retry_count: int
    
class DownloadMonitor:
    """System resource and progress monitor"""
    
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
        logger.info("📊 System monitoring started")
        
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
                if memory.percent > 85:
                    logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
                if disk.free < 10 * 1024**3:  # Less than 10GB free
                    logger.warning(f"⚠️ Low disk space: {disk.free/(1024**3):.1f}GB free")
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            time.sleep(30)  # Monitor every 30 seconds

class BulletproofPMCDownloader:
    """Enterprise-grade PMC data downloader with bulletproof features"""
    
    def __init__(self, output_dir: str = "data/pmc_100k_downloaded", checkpoint_interval: int = 600):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval  # seconds
        self.checkpoint_file = self.output_dir / "download_checkpoint.pkl"
        
        # API configuration
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_ftp_base = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
        
        # Rate limiting for NCBI API (max 3 requests per second)
        self.last_request_time = 0
        self.min_request_interval = 0.34  # Slightly more than 1/3 second
        
        # Retry configuration
        self.max_retries = 5
        self.retry_delay_base = 2  # Exponential backoff base
        
        # Monitoring
        self.monitor = DownloadMonitor()
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Checkpoint data
        self.checkpoint: Optional[DownloadCheckpoint] = None
        self.last_checkpoint_save = time.time()
        
        logger.info(f"🚀 BulletproofPMCDownloader initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
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
            logger.info(f"💾 Checkpoint saved: {self.checkpoint.current_count}/{self.checkpoint.target_count} documents")
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
            logger.info(f"📋 Checkpoint loaded: {self.checkpoint.current_count}/{self.checkpoint.target_count} documents")
            logger.info(f"⏱️ Previous session time: {self.checkpoint.total_download_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def create_checkpoint(self, target_count: int):
        """Create new checkpoint"""
        self.checkpoint = DownloadCheckpoint(
            target_count=target_count,
            current_count=0,
            downloaded_files=[],
            failed_downloads=[],
            bulk_files_completed=[],
            pmc_ids_processed=[],
            start_time=time.time(),
            last_checkpoint_time=time.time(),
            total_download_time=0.0,
            error_count=0,
            retry_count=0
        )
        logger.info(f"📋 New checkpoint created for {target_count} documents")
    
    def should_save_checkpoint(self) -> bool:
        """Check if it's time to save checkpoint"""
        return time.time() - self.last_checkpoint_save >= self.checkpoint_interval
    
    def rate_limit(self):
        """Enforce rate limiting for NCBI API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def validate_xml_file(self, file_path: Path) -> bool:
        """Validate that XML file is not corrupted"""
        try:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
                
            # Try to parse XML
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1KB
                if not content.strip():
                    return False
                    
            # Quick XML validation
            ET.parse(file_path)
            return True
        except Exception:
            return False
    
    def calculate_eta(self) -> str:
        """Calculate estimated time to completion"""
        if not self.checkpoint:
            return "Unknown"
            
        elapsed = time.time() - self.checkpoint.start_time + self.checkpoint.total_download_time
        if elapsed == 0 or self.checkpoint.current_count == 0:
            return "Unknown"
            
        rate = self.checkpoint.current_count / elapsed
    def get_pmc_id_list(self, query: str = "open access[filter]", max_ids: int = 100000) -> List[str]:
        """Get list of PMC IDs using E-search with robust error handling"""
        logger.info(f"🔍 Searching for PMC articles with query: {query}")
        
        # Skip if we already have processed IDs from checkpoint
        if self.checkpoint and self.checkpoint.pmc_ids_processed:
            remaining_ids = [pid for pid in self.checkpoint.pmc_ids_processed 
                           if pid not in [f.split('/')[-1].replace('.xml', '') for f in self.checkpoint.downloaded_files]]
            if remaining_ids:
                logger.info(f"📋 Using {len(remaining_ids)} PMC IDs from checkpoint")
                return remaining_ids[:max_ids]
        
        pmc_ids = []
        retstart = 0
        retmax = 10000  # Maximum allowed by NCBI
        
        while len(pmc_ids) < max_ids and not self.shutdown_requested:
            try:
                self.rate_limit()
                
                # Build search URL
                params = {
                    'db': 'pmc',
                    'term': query,
                    'retmode': 'json',
                    'retmax': min(retmax, max_ids - len(pmc_ids)),
                    'retstart': retstart,
                    'tool': 'rag_templates_100k',
                    'email': 'research@example.com'  # Replace with actual email
                }
                
                url = self.base_url + "esearch.fcgi?" + urlencode(params)
                
                def search_request():
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    return response.json()
                
                data = self.retry_with_backoff(search_request)
                
                if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                    batch_ids = data['esearchresult']['idlist']
                    pmc_ids.extend(batch_ids)
                    
                    logger.info(f"✅ Retrieved {len(batch_ids)} PMC IDs (total: {len(pmc_ids)})")
                    
                    if len(batch_ids) < retmax:
                        # No more results
                        break
                        
                    retstart += retmax
                else:
                    logger.warning("⚠️ No results found in E-search response")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error in E-search: {e}")
                self.checkpoint.error_count += 1
                break
        
        # Save PMC IDs to checkpoint
        if self.checkpoint:
            self.checkpoint.pmc_ids_processed = pmc_ids
            self.save_checkpoint()
        
        logger.info(f"🎯 Found {len(pmc_ids)} total PMC IDs")
        return pmc_ids
    
    def download_pmc_articles(self, pmc_ids: List[str], batch_size: int = 200) -> int:
        """Download PMC articles using E-fetch with comprehensive error handling"""
        logger.info(f"📥 Downloading {len(pmc_ids)} PMC articles in batches of {batch_size}")
        
        downloaded_count = 0
        
        for i in range(0, len(pmc_ids), batch_size):
            if self.shutdown_requested:
                logger.info("🛑 Shutdown requested, stopping download")
                break
                
            batch_ids = pmc_ids[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pmc_ids) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Downloading batch {batch_num}/{total_batches} ({len(batch_ids)} articles)")
            
            try:
                self.rate_limit()
                
                # Build fetch URL
                params = {
                    'db': 'pmc',
                    'id': ','.join(batch_ids),
                    'retmode': 'xml',
                    'tool': 'rag_templates_100k',
                    'email': 'research@example.com'  # Replace with actual email
                }
                
                url = self.base_url + "efetch.fcgi?" + urlencode(params)
                
                def fetch_request():
                    response = requests.get(url, timeout=120)  # Longer timeout for large batches
                    response.raise_for_status()
                    return response.content
                
                content = self.retry_with_backoff(fetch_request)
                
                # Parse the XML response
                root = ET.fromstring(content)
                
                # Extract individual articles
                articles = root.findall('.//article')
                
                for article in articles:
                    try:
                        # Extract PMC ID
                        pmc_id_elem = article.find('.//article-id[@pub-id-type="pmc"]')
                        if pmc_id_elem is not None:
                            pmc_id = pmc_id_elem.text
                            if not pmc_id.startswith('PMC'):
                                pmc_id = 'PMC' + pmc_id
                        else:
                            # Fallback to using batch ID
                            pmc_id = f"PMC{batch_ids[len(articles)]}" if len(articles) < len(batch_ids) else f"PMC{int(time.time())}"
                        
                        # Create directory structure
                        pmc_dir = self.output_dir / f"{pmc_id[:6]}xxxxxx"
                        pmc_dir.mkdir(exist_ok=True)
                        
                        # Save article XML
                        article_file = pmc_dir / f"{pmc_id}.xml"
                        
                        # Convert article element back to XML string
                        article_xml = ET.tostring(article, encoding='unicode')
                        
                        with open(article_file, 'w', encoding='utf-8') as f:
                            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                            f.write(article_xml)
                        
                        # Validate the file
                        if self.validate_xml_file(article_file):
                            downloaded_count += 1
                            self.checkpoint.current_count += 1
                            self.checkpoint.downloaded_files.append(str(article_file))
                            
                            if downloaded_count % 100 == 0:
                                eta = self.calculate_eta()
                                logger.info(f"✅ Downloaded {downloaded_count} articles... ETA: {eta}")
                        else:
                            logger.warning(f"⚠️ Invalid XML file: {article_file}")
                            article_file.unlink(missing_ok=True)
                            self.checkpoint.error_count += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing article: {e}")
                        self.checkpoint.error_count += 1
                        continue
                
                # Save checkpoint periodically
                if self.should_save_checkpoint():
                    self.save_checkpoint()
                    self.last_checkpoint_save = time.time()
                
            except Exception as e:
                logger.error(f"❌ Error downloading batch {batch_num}: {e}")
                self.checkpoint.failed_downloads.append({
                    'batch_num': batch_num,
                    'batch_ids': batch_ids,
                    'error': str(e),
                    'timestamp': time.time()
                })
                self.checkpoint.error_count += len(batch_ids)
                continue
        
        logger.info(f"🎉 Downloaded {downloaded_count} PMC articles")
        return downloaded_count
    
    def download_oa_bulk_files(self) -> int:
        """Download Open Access bulk files from PMC FTP with enhanced error handling"""
        logger.info("📦 Downloading Open Access bulk files from PMC FTP...")
        
        # Extended list of bulk files for 100k+ documents
        bulk_files = [
            "oa_comm/xml/oa_comm_xml.PMC000xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC001xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC002xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC003xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC004xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC005xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC006xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC007xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC008xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC009xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC010xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC011xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC012xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC013xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC014xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC015xxxxxx.baseline.2023.tar.gz",
        ]
        
        total_downloaded = 0
        
        for bulk_file in bulk_files:
            if self.shutdown_requested:
                logger.info("🛑 Shutdown requested, stopping bulk download")
                break
                
            filename = Path(bulk_file).name
            
            # Skip if already completed
            if filename in self.checkpoint.bulk_files_completed:
                logger.info(f"⏭️ Skipping {filename} (already completed)")
                continue
                
            url = self.pmc_ftp_base + bulk_file
            local_path = self.output_dir / filename
            
            if local_path.exists():
                logger.info(f"⏭️ Skipping {filename} (already exists)")
                self.checkpoint.bulk_files_completed.append(filename)
                continue
            
            logger.info(f"📥 Downloading {filename}...")
            
            try:
                def download_bulk_file():
                    response = requests.get(url, stream=True, timeout=300)
                    response.raise_for_status()
                    
                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                
                                if total_size > 0 and downloaded_size % (1024*1024*50) == 0:  # Every 50MB
                                    progress = (downloaded_size / total_size) * 100
                                    logger.info(f"📊 Progress: {progress:.1f}% ({downloaded_size/(1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB)")
                    
                    return downloaded_size
                
                downloaded_size = self.retry_with_backoff(download_bulk_file)
                logger.info(f"✅ Downloaded {filename} ({downloaded_size/(1024*1024):.1f}MB)")
                
                # Extract the archive
                logger.info(f"📂 Extracting {filename}...")
                extracted_count = 0
                
                try:
                    with tarfile.open(local_path, 'r:gz') as tar:
                        members = tar.getmembers()
                        for member in members:
                            if member.name.endswith('.xml'):
                                tar.extract(member, path=self.output_dir)
                                
                                # Validate extracted file
                                extracted_file = self.output_dir / member.name
                                if self.validate_xml_file(extracted_file):
                                    extracted_count += 1
                                    self.checkpoint.current_count += 1
                                    self.checkpoint.downloaded_files.append(str(extracted_file))
                                else:
                                    logger.warning(f"⚠️ Invalid extracted file: {extracted_file}")
                                    extracted_file.unlink(missing_ok=True)
                                    self.checkpoint.error_count += 1
                                
                                # Check if target reached
                                if self.checkpoint.current_count >= self.checkpoint.target_count:
                                    logger.info(f"🎯 Target reached during extraction: {self.checkpoint.current_count}")
                                    break
                    
                    total_downloaded += extracted_count
                    logger.info(f"✅ Extracted {extracted_count} valid XML files from {filename}")
                    
                    # Mark as completed
                    self.checkpoint.bulk_files_completed.append(filename)
                    
                    # Remove the archive to save space
                    local_path.unlink()
                    logger.info(f"🗑️ Removed archive {filename}")
                    
                except Exception as e:
                    logger.error(f"❌ Error extracting {filename}: {e}")
                    self.checkpoint.error_count += 1
                
                # Save checkpoint after each bulk file
                self.save_checkpoint()
                
                # Check if target reached
                if self.checkpoint.current_count >= self.checkpoint.target_count:
                    logger.info(f"🎯 Target reached: {self.checkpoint.current_count}")
                    break
                
            except Exception as e:
                logger.error(f"❌ Error downloading {filename}: {e}")
                self.checkpoint.failed_downloads.append({
                    'bulk_file': filename,
                    'error': str(e),
                    'timestamp': time.time()
                })
                self.checkpoint.error_count += 1
                continue
        
        logger.info(f"🎉 Total documents downloaded from bulk files: {total_downloaded}")
        return total_downloaded
    
    def get_current_document_count(self) -> int:
        """Count current XML documents"""
        count = len(list(self.output_dir.rglob("*.xml")))
        logger.info(f"📊 Current document count: {count}")
        return count
    
    def download_to_target(self, target_count: int, resume: bool = False) -> int:
        """Download documents to reach target count with bulletproof features"""
        # Load or create checkpoint
        if resume and self.load_checkpoint():
            if self.checkpoint.target_count != target_count:
                logger.warning(f"⚠️ Target count mismatch: checkpoint={self.checkpoint.target_count}, requested={target_count}")
                logger.info("📋 Updating checkpoint target count")
                self.checkpoint.target_count = target_count
        else:
            self.create_checkpoint(target_count)
        
        # Start monitoring
        self.monitor.start()
        
        try:
            current_count = self.get_current_document_count()
            self.checkpoint.current_count = current_count
            
            if current_count >= target_count:
                logger.info(f"🎯 Target already reached: {current_count} >= {target_count}")
                return current_count
            
            needed = target_count - current_count
            logger.info(f"📈 Need {needed} more documents to reach target of {target_count}")
            logger.info(f"⏱️ ETA: {self.calculate_eta()}")
            
            # First try bulk downloads (more efficient for large targets)
            if needed > 10000:
                logger.info("📦 Attempting bulk download for large target...")
                bulk_downloaded = self.download_oa_bulk_files()
                current_count = self.get_current_document_count()
                self.checkpoint.current_count = current_count
                
                if current_count >= target_count:
                    logger.info(f"🎯 Target reached with bulk download: {current_count}")
                    return current_count
            
            # If still need more, use E-utilities API
            remaining_needed = target_count - current_count
            if remaining_needed > 0 and not self.shutdown_requested:
                logger.info(f"🔍 Using E-utilities API to download {remaining_needed} more documents...")
                
                # Get PMC IDs
                pmc_ids = self.get_pmc_id_list(max_ids=remaining_needed)
                
                if pmc_ids and not self.shutdown_requested:
                    # Download articles
                    downloaded = self.download_pmc_articles(pmc_ids)
                    current_count = self.get_current_document_count()
                    self.checkpoint.current_count = current_count
            
            return current_count
            
        finally:
            # Final checkpoint save
            if self.checkpoint:
                self.checkpoint.total_download_time += time.time() - self.checkpoint.start_time
                self.save_checkpoint()
            
            # Stop monitoring
            monitoring_data = self.monitor.stop()
            
            # Generate summary report
            self.generate_summary_report(monitoring_data)
    
    def generate_summary_report(self, monitoring_data: List[Dict[str, Any]]):
        """Generate comprehensive download summary report"""
        if not self.checkpoint:
            return
            
        report = {
            "download_summary": {
                "target_count": self.checkpoint.target_count,
                "final_count": self.checkpoint.current_count,
                "success_rate": (self.checkpoint.current_count / self.checkpoint.target_count) * 100,
                "total_time_seconds": self.checkpoint.total_download_time,
                "error_count": self.checkpoint.error_count,
                "retry_count": self.checkpoint.retry_count,
                "files_downloaded": len(self.checkpoint.downloaded_files),
                "bulk_files_completed": len(self.checkpoint.bulk_files_completed),
                "failed_downloads": len(self.checkpoint.failed_downloads)
            },
            "performance_metrics": {
                "download_rate_docs_per_second": self.checkpoint.current_count / self.checkpoint.total_download_time if self.checkpoint.total_download_time > 0 else 0,
                "peak_memory_gb": max([m['memory_gb'] for m in monitoring_data]) if monitoring_data else 0,
                "avg_cpu_percent": sum([m['cpu_percent'] for m in monitoring_data]) / len(monitoring_data) if monitoring_data else 0,
                "disk_usage_gb": sum([m.get('disk_percent', 0) for m in monitoring_data]) / len(monitoring_data) if monitoring_data else 0
            },
            "error_details": {
                "failed_downloads": self.checkpoint.failed_downloads,
                "error_rate": (self.checkpoint.error_count / max(self.checkpoint.current_count, 1)) * 100
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.output_dir / f"download_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("📊 DOWNLOAD SUMMARY REPORT")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {self.checkpoint.target_count:,} documents")
        logger.info(f"✅ Downloaded: {self.checkpoint.current_count:,} documents")
        logger.info(f"📈 Success Rate: {report['download_summary']['success_rate']:.1f}%")
        logger.info(f"⏱️ Total Time: {self.checkpoint.total_download_time:.1f} seconds")
        logger.info(f"🚀 Download Rate: {report['performance_metrics']['download_rate_docs_per_second']:.2f} docs/sec")
        logger.info(f"❌ Errors: {self.checkpoint.error_count}")
        logger.info(f"🔄 Retries: {self.checkpoint.retry_count}")
        logger.info(f"📄 Report saved: {report_file}")
        logger.info("=" * 80)
        remaining = self.checkpoint.target_count - self.checkpoint.current_count
        
        if rate == 0:
            return "Unknown"
            
        eta_seconds = remaining / rate
        eta_delta = timedelta(seconds=int(eta_seconds))
        return str(eta_delta)
    
    def retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                    
                delay = self.retry_delay_base ** attempt
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                self.checkpoint.retry_count += 1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bulletproof 100K PMC Article Downloader")
    parser.add_argument("--target-count", type=int, default=100000,
                       help="Target number of documents to download")
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                       help="Resume from existing checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/pmc_100k_downloaded",
                       help="Output directory for downloaded data")
    parser.add_argument("--checkpoint-interval", type=int, default=600,
                       help="Checkpoint save interval in seconds")
    parser.add_argument("--bulk-only", action="store_true",
                       help="Only download bulk files (faster)")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Bulletproof PMC Downloader - Target: {args.target_count:,} documents")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"⏰ Checkpoint interval: {args.checkpoint_interval}s")
    
    downloader = BulletproofPMCDownloader(args.output_dir, args.checkpoint_interval)
    
    try:
        if args.bulk_only:
            # Only download bulk files
            logger.info("📦 Bulk-only mode enabled")
            downloader.create_checkpoint(args.target_count)
            downloaded = downloader.download_oa_bulk_files()
            final_count = downloader.get_current_document_count()
        else:
            # Full download to target
            final_count = downloader.download_to_target(args.target_count, args.resume_from_checkpoint)
        
        logger.info("=" * 80)
        logger.info("🎉 DOWNLOAD COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {args.target_count:,} documents")
        logger.info(f"✅ Downloaded: {final_count:,} documents")
        
        if final_count >= args.target_count:
            logger.info("🎯 Target reached successfully!")
            return True
        else:
            logger.info(f"⚠️ Target not fully reached (missing {args.target_count - final_count:,} documents)")
            return False
            
    except KeyboardInterrupt:
        logger.info("🛑 Download interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)