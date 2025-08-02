#!/usr/bin/env python3
"""
Fixed 100K PMC Article Downloader

Fixed version with correct NCBI FTP URLs and structure:
- Updated to use current oa_bulk/oa_comm/xml/ path structure
- Updated to use 2024-12-18 baseline files (current as of 2025)
- Enhanced error recovery and retry logic
- Progress checkpointing to resume interrupted downloads
- Rate limiting to avoid overwhelming PMC servers
- File validation and corruption detection
- Comprehensive logging for unattended operation

Usage:
    python scripts/download_100k_pmc_articles_fixed.py --target-count 100000
    python scripts/download_100k_pmc_articles_fixed.py --resume-from-checkpoint
    python scripts/download_100k_pmc_articles_fixed.py --target-count 50000 --checkpoint-interval 600
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
import tarfile
import signal
import pickle
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_100k_pmc_articles_fixed.log'),
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

class FixedPMCDownloader:
    """Fixed PMC data downloader with correct NCBI URLs"""
    
    def __init__(self, output_dir: str = "data/pmc_100k_downloaded", checkpoint_interval: int = 600):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file = self.output_dir / "download_checkpoint.pkl"
        
        # API configuration
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        # FIXED: Updated to correct FTP path structure
        self.pmc_ftp_base = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
        
        # Rate limiting for NCBI API (max 3 requests per second)
        self.last_request_time = 0
        self.min_request_interval = 0.34
        
        # Retry configuration
        self.max_retries = 5
        self.retry_delay_base = 2
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Checkpoint data
        self.checkpoint: Optional[DownloadCheckpoint] = None
        self.last_checkpoint_save = time.time()
        
        logger.info(f"🚀 Fixed PMC Downloader initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔗 FTP Base URL: {self.pmc_ftp_base}")
    
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
        """Validate that XML file contains real PMC content"""
        try:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
                
            # Try to parse XML
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)
                if not content.strip():
                    return False
                    
                # Check for mock content indicators
                if "Mock Article" in content or "mock content" in content:
                    logger.warning(f"⚠️ Mock content detected in {file_path}")
                    return False
                    
            # Quick XML validation
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Validate it's a real PMC article structure
            if root.tag != 'article':
                return False
                
            return True
        except Exception as e:
            logger.warning(f"⚠️ XML validation failed for {file_path}: {e}")
            return False
    
    def download_oa_bulk_files(self) -> int:
        """Download Open Access bulk files from PMC FTP with FIXED URLs"""
        logger.info("📦 Downloading Open Access bulk files from PMC FTP...")
        
        # FIXED: Updated bulk files list with correct 2024-12-18 baseline files
        bulk_files = [
            "oa_comm_xml.PMC000xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC001xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC002xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC003xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC004xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC005xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC006xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC007xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC008xxxxxx.baseline.2024-12-18.tar.gz",
            "oa_comm_xml.PMC009xxxxxx.baseline.2024-12-18.tar.gz",
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
                
            # FIXED: Use correct FTP base URL
            url = self.pmc_ftp_base + bulk_file
            local_path = self.output_dir / filename
            
            if local_path.exists():
                logger.info(f"⏭️ Skipping {filename} (already exists)")
                self.checkpoint.bulk_files_completed.append(filename)
                continue
            
            logger.info(f"📥 Downloading {filename} from {url}...")
            
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
                                
                                if total_size > 0 and downloaded_size % (1024*1024*50) == 0:
                                    progress = (downloaded_size / total_size) * 100
                                    logger.info(f"📊 Progress: {progress:.1f}%")
                    
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
                                    
                                    if extracted_count % 1000 == 0:
                                        logger.info(f"✅ Extracted {extracted_count} real PMC articles")
                                else:
                                    extracted_file.unlink(missing_ok=True)
                                    self.checkpoint.error_count += 1
                                
                                # Check if target reached
                                if self.checkpoint.current_count >= self.checkpoint.target_count:
                                    logger.info(f"🎯 Target reached: {self.checkpoint.current_count}")
                                    break
                    
                    total_downloaded += extracted_count
                    logger.info(f"✅ Extracted {extracted_count} valid real PMC XML files from {filename}")
                    
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
        
        logger.info(f"🎉 Total real PMC documents downloaded: {total_downloaded}")
        return total_downloaded
    
    def get_current_document_count(self) -> int:
        """Count current XML documents"""
        count = len(list(self.output_dir.rglob("*.xml")))
        logger.info(f"📊 Current document count: {count}")
        return count
    
    def download_to_target(self, target_count: int, resume: bool = False) -> int:
        """Download documents to reach target count"""
        # Load or create checkpoint
        if resume and self.load_checkpoint():
            if self.checkpoint.target_count != target_count:
                logger.info("📋 Updating checkpoint target count")
                self.checkpoint.target_count = target_count
        else:
            self.create_checkpoint(target_count)
        
        try:
            current_count = self.get_current_document_count()
            self.checkpoint.current_count = current_count
            
            if current_count >= target_count:
                logger.info(f"🎯 Target already reached: {current_count} >= {target_count}")
                return current_count
            
            needed = target_count - current_count
            logger.info(f"📈 Need {needed} more documents to reach target of {target_count}")
            
            # Try bulk downloads
            logger.info("📦 Attempting bulk download...")
            bulk_downloaded = self.download_oa_bulk_files()
            current_count = self.get_current_document_count()
            self.checkpoint.current_count = current_count
            
            return current_count
            
        finally:
            # Final checkpoint save
            if self.checkpoint:
                self.checkpoint.total_download_time += time.time() - self.checkpoint.start_time
                self.save_checkpoint()
            
            # Generate summary report
            self.generate_summary_report()
    
    def generate_summary_report(self):
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
            "validation_info": {
                "real_pmc_content": True,
                "mock_content_filtered": True,
                "xml_validation": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.output_dir / f"download_report_fixed_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("📊 DOWNLOAD SUMMARY REPORT")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {self.checkpoint.target_count:,} documents")
        logger.info(f"✅ Downloaded: {self.checkpoint.current_count:,} documents")
        logger.info(f"📈 Success Rate: {report['download_summary']['success_rate']:.1f}%")
        logger.info(f"⏱️ Total Time: {self.checkpoint.total_download_time:.1f} seconds")
        logger.info(f"❌ Errors: {self.checkpoint.error_count}")
        logger.info(f"📄 Report saved: {report_file}")
        logger.info("=" * 80)
    
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
    parser = argparse.ArgumentParser(description="Fixed 100K PMC Article Downloader")
    parser.add_argument("--target-count", type=int, default=100000,
                       help="Target number of documents to download")
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                       help="Resume from existing checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/pmc_100k_downloaded",
                       help="Output directory for downloaded data")
    parser.add_argument("--checkpoint-interval", type=int, default=600,
                       help="Checkpoint save interval in seconds")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Fixed PMC Downloader - Target: {args.target_count:,} documents")
    logger.info(f"📁 Output directory: {args.output_dir}")
    
    downloader = FixedPMCDownloader(args.output_dir, args.checkpoint_interval)
    
    try:
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