"""
PMC Batch Downloader with Progress Tracking and Validation

This module provides enterprise-scale batch downloading of PMC documents
with comprehensive progress tracking, validation, and monitoring capabilities.
"""

import logging
import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import xml.etree.ElementTree as ET

from .api_client import PMCAPIClient
from data.pmc_processor import extract_pmc_metadata

logger = logging.getLogger(__name__)

@dataclass
class DownloadProgress:
    """Progress tracking for batch downloads."""
    total_documents: int = 0
    downloaded_documents: int = 0
    validated_documents: int = 0
    failed_documents: int = 0
    current_file: str = ""
    start_time: float = 0
    last_update_time: float = 0
    download_speed_docs_per_second: float = 0
    estimated_time_remaining_seconds: float = 0
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.downloaded_documents / self.total_documents) * 100
    
    @property
    def elapsed_time_seconds(self) -> float:
        """Calculate elapsed time."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time
    
    def update_speed_and_eta(self):
        """Update download speed and ETA calculations."""
        elapsed = self.elapsed_time_seconds
        if elapsed > 0 and self.downloaded_documents > 0:
            self.download_speed_docs_per_second = self.downloaded_documents / elapsed
            
            remaining_docs = self.total_documents - self.downloaded_documents
            if self.download_speed_docs_per_second > 0:
                self.estimated_time_remaining_seconds = remaining_docs / self.download_speed_docs_per_second
            else:
                self.estimated_time_remaining_seconds = 0
        
        self.last_update_time = time.time()

@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    file_path: str
    pmc_id: str = ""
    title: str = ""
    abstract_length: int = 0
    error_message: str = ""
    file_size_bytes: int = 0
    validation_time_seconds: float = 0

class PMCBatchDownloader:
    """Enterprise-scale PMC document batch downloader."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize batch downloader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_client = PMCAPIClient(config.get('api_client', {}))
        
        # Download configuration
        self.target_document_count = config.get('target_document_count', 10000)
        self.batch_size = config.get('batch_size', 100)
        self.max_concurrent_downloads = config.get('max_concurrent_downloads', 4)
        self.validation_enabled = config.get('enable_validation', True)
        self.checkpoint_interval = config.get('checkpoint_interval', 500)
        
        # Paths
        self.download_dir = Path(config.get('download_directory', 'data/pmc_downloads'))
        self.checkpoint_file = Path(config.get('checkpoint_file', 'data/pmc_download_checkpoint.json'))
        
        # Progress tracking
        self.progress = DownloadProgress()
        self.progress_callbacks: List[Callable] = []
        self.validation_results: List[ValidationResult] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
    def add_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """Notify all progress callbacks."""
        self.progress.update_speed_and_eta()
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def _validate_document(self, file_path: Path) -> ValidationResult:
        """
        Validate a downloaded PMC document.
        
        Args:
            file_path: Path to the XML file
            
        Returns:
            ValidationResult with validation details
        """
        start_time = time.time()
        
        try:
            # Check file exists and has content
            if not file_path.exists():
                return ValidationResult(
                    is_valid=False,
                    file_path=str(file_path),
                    error_message="File does not exist",
                    validation_time_seconds=time.time() - start_time
                )
            
            file_size = file_path.stat().st_size
            if file_size == 0:
                return ValidationResult(
                    is_valid=False,
                    file_path=str(file_path),
                    error_message="File is empty",
                    file_size_bytes=file_size,
                    validation_time_seconds=time.time() - start_time
                )
            
            # Validate XML structure and extract metadata
            try:
                metadata = extract_pmc_metadata(str(file_path))
                
                # Check for valid content
                if not metadata.get('title') or metadata['title'] == "Error":
                    return ValidationResult(
                        is_valid=False,
                        file_path=str(file_path),
                        error_message="Invalid or missing title",
                        file_size_bytes=file_size,
                        validation_time_seconds=time.time() - start_time
                    )
                
                # Check for reasonable abstract length (medical papers should have abstracts)
                abstract = metadata.get('abstract', '')
                if len(abstract.strip()) < 50:  # Minimum reasonable abstract length
                    logger.warning(f"Short abstract in {file_path}: {len(abstract)} chars")
                
                return ValidationResult(
                    is_valid=True,
                    file_path=str(file_path),
                    pmc_id=metadata.get('pmc_id', ''),
                    title=metadata.get('title', ''),
                    abstract_length=len(abstract),
                    file_size_bytes=file_size,
                    validation_time_seconds=time.time() - start_time
                )
                
            except ET.ParseError as e:
                return ValidationResult(
                    is_valid=False,
                    file_path=str(file_path),
                    error_message=f"XML parse error: {str(e)}",
                    file_size_bytes=file_size,
                    validation_time_seconds=time.time() - start_time
                )
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                file_path=str(file_path),
                error_message=f"Validation error: {str(e)}",
                validation_time_seconds=time.time() - start_time
            )
    
    def _save_checkpoint(self):
        """Save download progress checkpoint."""
        try:
            checkpoint_data = {
                'progress': asdict(self.progress),
                'validation_results': [asdict(r) for r in self.validation_results],
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }
            
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            logger.debug(f"Checkpoint saved: {self.progress.downloaded_documents}/{self.progress.total_documents}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> bool:
        """
        Load download progress from checkpoint.
        
        Returns:
            True if checkpoint was loaded successfully
        """
        try:
            if not self.checkpoint_file.exists():
                return False
            
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore progress
            progress_data = checkpoint_data.get('progress', {})
            for key, value in progress_data.items():
                if hasattr(self.progress, key):
                    setattr(self.progress, key, value)
            
            # Restore validation results
            validation_data = checkpoint_data.get('validation_results', [])
            self.validation_results = [
                ValidationResult(**result) for result in validation_data
            ]
            
            logger.info(f"Checkpoint loaded: {self.progress.downloaded_documents}/{self.progress.total_documents} documents")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    def download_enterprise_dataset(self, 
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Download enterprise-scale PMC dataset (10,000+ documents).
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with download results and statistics
        """
        if progress_callback:
            self.add_progress_callback(progress_callback)
        
        # Initialize progress
        self.progress = DownloadProgress(
            total_documents=self.target_document_count,
            start_time=time.time()
        )
        
        # Try to load existing checkpoint
        checkpoint_loaded = self._load_checkpoint()
        
        logger.info(f"🚀 Starting enterprise PMC download: {self.target_document_count} documents")
        logger.info(f"📁 Download directory: {self.download_dir}")
        logger.info(f"🔄 Checkpoint loaded: {checkpoint_loaded}")
        
        try:
            # Create download directory
            self.download_dir.mkdir(parents=True, exist_ok=True)
            
            # Get available bulk files
            logger.info("📋 Fetching available bulk download files...")
            bulk_files = self.api_client.get_available_bulk_files()
            
            if not bulk_files:
                raise RuntimeError("No bulk download files available")
            
            # Select best file for target document count
            selected_file = None
            for file_info in bulk_files:
                if file_info['estimated_documents'] >= self.target_document_count:
                    selected_file = file_info
                    break
            
            if not selected_file:
                # Use the largest available file
                selected_file = bulk_files[0]
                logger.warning(f"No file with {self.target_document_count}+ docs, using largest: {selected_file['filename']}")
            
            logger.info(f"📦 Selected file: {selected_file['filename']} (~{selected_file['estimated_documents']} docs)")
            
            # Download bulk file if not already downloaded
            archive_path = self.download_dir / selected_file['filename']
            if not archive_path.exists():
                logger.info("⬇️ Downloading bulk file...")
                
                def download_progress(progress, downloaded, total):
                    self.progress.current_file = f"Downloading {selected_file['filename']}"
                    self._notify_progress()
                
                download_result = self.api_client.download_bulk_file(
                    selected_file['filename'],
                    self.download_dir,
                    download_progress
                )
                
                if not download_result['success']:
                    raise RuntimeError(f"Download failed: {download_result.get('error')}")
                
                logger.info(f"✅ Download complete: {download_result['size_bytes']:,} bytes")
            else:
                logger.info(f"📁 Using existing archive: {archive_path}")
            
            # Extract documents
            extract_dir = self.download_dir / "extracted"
            if self.progress.downloaded_documents < self.target_document_count:
                logger.info("📂 Extracting documents...")
                
                def extract_progress(progress, extracted, total):
                    self.progress.current_file = f"Extracting documents ({extracted}/{total})"
                    self.progress.downloaded_documents = extracted
                    self._notify_progress()
                
                extract_result = self.api_client.extract_bulk_file(
                    archive_path,
                    extract_dir,
                    max_documents=self.target_document_count,
                    progress_callback=extract_progress
                )
                
                if not extract_result['success']:
                    raise RuntimeError(f"Extraction failed: {extract_result.get('error')}")
                
                self.progress.downloaded_documents = extract_result['extracted_count']
                logger.info(f"✅ Extraction complete: {extract_result['extracted_count']} documents")
            
            # Validate documents if enabled
            if self.validation_enabled:
                logger.info("🔍 Validating documents...")
                self._validate_extracted_documents(extract_dir)
            
            # Final checkpoint save
            self._save_checkpoint()
            
            # Generate final results
            results = {
                'success': True,
                'total_documents': self.progress.total_documents,
                'downloaded_documents': self.progress.downloaded_documents,
                'validated_documents': self.progress.validated_documents,
                'failed_documents': self.progress.failed_documents,
                'download_time_seconds': self.progress.elapsed_time_seconds,
                'download_speed_docs_per_second': self.progress.download_speed_docs_per_second,
                'download_directory': str(self.download_dir),
                'extract_directory': str(extract_dir),
                'validation_enabled': self.validation_enabled,
                'validation_results': [asdict(r) for r in self.validation_results]
            }
            
            logger.info(f"🎉 Enterprise download complete!")
            logger.info(f"   📊 Downloaded: {results['downloaded_documents']:,} documents")
            logger.info(f"   ✅ Validated: {results['validated_documents']:,} documents")
            logger.info(f"   ⏱️ Time: {results['download_time_seconds']:.2f} seconds")
            logger.info(f"   🚀 Speed: {results['download_speed_docs_per_second']:.2f} docs/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"Enterprise download failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'downloaded_documents': self.progress.downloaded_documents,
                'download_time_seconds': self.progress.elapsed_time_seconds
            }
    
    def _validate_extracted_documents(self, extract_dir: Path):
        """Validate all extracted documents."""
        xml_files = list(extract_dir.glob("*.xml"))
        
        logger.info(f"Validating {len(xml_files)} XML files...")
        
        for i, xml_file in enumerate(xml_files):
            self.progress.current_file = f"Validating {xml_file.name}"
            
            validation_result = self._validate_document(xml_file)
            self.validation_results.append(validation_result)
            
            if validation_result.is_valid:
                self.progress.validated_documents += 1
            else:
                self.progress.failed_documents += 1
                logger.warning(f"Validation failed for {xml_file}: {validation_result.error_message}")
            
            # Update progress and save checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                self._notify_progress()
                self._save_checkpoint()
        
        # Final progress update
        self._notify_progress()
        
        logger.info(f"Validation complete: {self.progress.validated_documents} valid, {self.progress.failed_documents} failed")