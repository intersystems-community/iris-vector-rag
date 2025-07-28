"""
PMC API Client for Open Access Subset

This module provides a client for accessing the PMC Open Access Subset
to download real medical/scientific documents for enterprise-scale testing.

Uses the PMC OAI-PMH service and FTP access for bulk downloads.
"""

import logging
import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Generator
from urllib.parse import urljoin
from pathlib import Path
import ftplib
import gzip
import tarfile
import tempfile
import os

logger = logging.getLogger(__name__)

class PMCAPIClient:
    """Client for accessing PMC Open Access documents via API and FTP."""
    
    # PMC OAI-PMH endpoint for metadata
    OAI_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
    
    # PMC FTP server for bulk downloads
    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    FTP_BASE_PATH = "/pub/pmc/oa_bulk"
    
    # PMC E-utilities for individual document access
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PMC API client.
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config or {}
        self.session = requests.Session()
        
        # Configure session with reasonable defaults
        self.session.headers.update({
            'User-Agent': 'RAG-Templates-PMC-Downloader/1.0 (enterprise-testing)'
        })
        
        # Rate limiting configuration
        self.request_delay = self.config.get('request_delay_seconds', 0.5)
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout_seconds', 30)
        
    def get_available_bulk_files(self) -> List[Dict[str, Any]]:
        """
        Get list of available individual PMC files from the oa_package structure.
        
        Note: PMC no longer provides bulk archives. Instead, individual documents
        are stored as .tar.gz files in /pub/pmc/oa_package/XX/YY/ directories.
        
        Returns:
            List of dictionaries with file information for enterprise-scale download
        """
        try:
            ftp = ftplib.FTP(self.FTP_HOST)
            ftp.login()
            
            # Navigate to the oa_package directory
            package_base = "/pub/pmc/oa_package"
            ftp.cwd(package_base)
            
            all_files = []
            subdirs = ftp.nlst()
            
            # Limit exploration to avoid overwhelming the system
            # Sample from multiple directories to get diverse content
            sample_dirs = subdirs[:16] if len(subdirs) > 16 else subdirs
            
            logger.info(f"Exploring {len(sample_dirs)} directories out of {len(subdirs)} available")
            
            for subdir in sample_dirs:
                try:
                    # Navigate to subdirectory (e.g., "00")
                    ftp.cwd(f"{package_base}/{subdir}")
                    nested_dirs = ftp.nlst()
                    
                    # Sample nested directories (e.g., "00", "01", etc.)
                    sample_nested = nested_dirs[:4] if len(nested_dirs) > 4 else nested_dirs
                    
                    for nested_dir in sample_nested:
                        try:
                            # Navigate to final directory with actual files
                            final_path = f"{package_base}/{subdir}/{nested_dir}"
                            ftp.cwd(final_path)
                            
                            files = ftp.nlst()
                            tar_files = [f for f in files if f.endswith('.tar.gz') and f.startswith('PMC')]
                            
                            for tar_file in tar_files:
                                all_files.append({
                                    'filename': tar_file,
                                    'size_bytes': None,  # Skip size check for performance
                                    'ftp_path': f"{final_path}/{tar_file}",
                                    'estimated_documents': 1,  # Each file = 1 document
                                    'pmc_id': tar_file.replace('.tar.gz', ''),
                                    'directory': final_path
                                })
                            
                            # Limit total files to prevent memory issues
                            if len(all_files) >= 50000:  # Enough for enterprise testing
                                break
                                
                        except Exception as e:
                            logger.warning(f"Could not access {package_base}/{subdir}/{nested_dir}: {e}")
                            continue
                    
                    if len(all_files) >= 50000:
                        break
                        
                except Exception as e:
                    logger.warning(f"Could not access {package_base}/{subdir}: {e}")
                    continue
            
            ftp.quit()
            
            # Sort by PMC ID for consistent ordering
            all_files.sort(key=lambda x: x.get('pmc_id', ''))
            
            logger.info(f"Found {len(all_files)} individual PMC documents available for download")
            return all_files
            
        except Exception as e:
            logger.error(f"Error accessing PMC FTP: {e}")
            return []
    
    def _estimate_document_count(self, filename: str, size_bytes: Optional[int]) -> int:
        """
        Estimate number of documents in a bulk file.
        
        Args:
            filename: Name of the bulk file
            size_bytes: Size of the file in bytes
            
        Returns:
            Estimated number of documents
        """
        if size_bytes is None:
            return 0
        
        # Rough estimation: average PMC XML document is ~100KB compressed
        avg_doc_size_compressed = 100 * 1024  # 100KB
        estimated_docs = size_bytes // avg_doc_size_compressed
        
        # Apply filename-based adjustments
        if 'comm' in filename.lower():
            # Commercial use subset tends to be smaller
            estimated_docs = int(estimated_docs * 0.7)
        
        return max(estimated_docs, 1)
    
    def download_individual_file(self, file_info: Dict[str, Any], target_dir: Path,
                                progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Download an individual PMC file from the oa_package structure.
        
        Args:
            file_info: Dictionary with file information (from get_available_bulk_files)
            target_dir: Directory to save the downloaded file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with download results
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = file_info['filename']
        ftp_path = file_info['ftp_path']
        local_path = target_dir / filename
        
        try:
            ftp = ftplib.FTP(self.FTP_HOST)
            ftp.login()
            
            # Navigate to the specific directory
            directory = file_info['directory']
            ftp.cwd(directory)
            
            # Set binary mode for proper file transfer
            ftp.voidcmd('TYPE I')
            
            downloaded_bytes = 0
            start_time = time.time()
            
            def progress_handler(data):
                nonlocal downloaded_bytes
                downloaded_bytes += len(data)
                
                if progress_callback:
                    # Estimate progress based on downloaded bytes (no total size available)
                    progress_callback(downloaded_bytes, filename)
            
            logger.info(f"Downloading {filename} from {directory}")
            
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f'RETR {filename}', lambda data: (
                    f.write(data), progress_handler(data)
                ))
            
            ftp.quit()
            
            download_time = time.time() - start_time
            
            result = {
                'success': True,
                'local_path': str(local_path),
                'size_bytes': downloaded_bytes,
                'download_time_seconds': download_time,
                'download_speed_mbps': (downloaded_bytes / (1024 * 1024)) / download_time if download_time > 0 else 0,
                'pmc_id': file_info.get('pmc_id', filename.replace('.tar.gz', ''))
            }
            
            logger.info(f"Downloaded {filename}: {downloaded_bytes:,} bytes in {download_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return {
                'success': False,
                'error': str(e),
                'local_path': str(local_path) if 'local_path' in locals() else None,
                'pmc_id': file_info.get('pmc_id', filename.replace('.tar.gz', ''))
            }

    def download_bulk_file(self, filename: str, target_dir: Path,
                          progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        
        Note: PMC no longer provides bulk files. This method will return an error
        directing users to use download_multiple_files instead.
        """
        logger.error("PMC no longer provides bulk download files. Use download_multiple_files() instead.")
        return {
            'success': False,
            'error': 'PMC bulk downloads are no longer available. Use individual file downloads.',
            'local_path': None
        }
    
    def download_multiple_files(self, file_list: List[Dict[str, Any]], target_dir: Path,
                               max_files: Optional[int] = None,
                               progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Download multiple individual PMC files for enterprise-scale data collection.
        
        Args:
            file_list: List of file info dictionaries from get_available_bulk_files()
            target_dir: Directory to save downloaded files
            max_files: Maximum number of files to download
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with download results and statistics
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if max_files:
            file_list = file_list[:max_files]
        
        downloaded_files = []
        failed_downloads = []
        total_bytes = 0
        start_time = time.time()
        
        logger.info(f"Starting download of {len(file_list)} PMC files")
        
        for i, file_info in enumerate(file_list):
            try:
                def file_progress(bytes_downloaded, filename):
                    if progress_callback:
                        overall_progress = {
                            'current_file': i + 1,
                            'total_files': len(file_list),
                            'filename': filename,
                            'bytes_downloaded': bytes_downloaded,
                            'completed_files': len(downloaded_files),
                            'failed_files': len(failed_downloads)
                        }
                        progress_callback(overall_progress)
                
                result = self.download_individual_file(file_info, target_dir, file_progress)
                
                if result['success']:
                    downloaded_files.append(result)
                    total_bytes += result.get('size_bytes', 0)
                    logger.info(f"✅ Downloaded {result['pmc_id']} ({i+1}/{len(file_list)})")
                else:
                    failed_downloads.append({
                        'file_info': file_info,
                        'error': result.get('error', 'Unknown error')
                    })
                    logger.warning(f"❌ Failed to download {file_info.get('pmc_id', 'unknown')} ({i+1}/{len(file_list)})")
                
                # Rate limiting to be respectful to PMC servers
                time.sleep(self.request_delay)
                
            except Exception as e:
                failed_downloads.append({
                    'file_info': file_info,
                    'error': str(e)
                })
                logger.error(f"❌ Exception downloading {file_info.get('pmc_id', 'unknown')}: {e}")
        
        download_time = time.time() - start_time
        
        result = {
            'success': len(downloaded_files) > 0,
            'downloaded_count': len(downloaded_files),
            'failed_count': len(failed_downloads),
            'total_bytes': total_bytes,
            'download_time_seconds': download_time,
            'average_speed_mbps': (total_bytes / (1024 * 1024)) / download_time if download_time > 0 else 0,
            'downloaded_files': downloaded_files,
            'failed_downloads': failed_downloads,
            'target_dir': str(target_dir)
        }
        
        logger.info(f"Download complete: {len(downloaded_files)} successful, {len(failed_downloads)} failed")
        return result

    def extract_individual_file(self, archive_path: Path, extract_dir: Path,
                               progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Extract a single PMC document from its .tar.gz archive.
        
        Args:
            archive_path: Path to the individual .tar.gz file
            extract_dir: Directory to extract the document to
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with extraction results
        """
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        start_time = time.time()
        
        try:
            logger.info(f"Extracting {archive_path.name}")
            
            with tarfile.open(archive_path, 'r:gz') as tar:
                members = tar.getmembers()
                
                for member in members:
                    if member.isfile():
                        # Extract to extract_dir with clean filename
                        clean_name = Path(member.name).name
                        target_path = extract_dir / clean_name
                        
                        with tar.extractfile(member) as source:
                            if source:
                                with open(target_path, 'wb') as target:
                                    target.write(source.read())
                                
                                extracted_files.append(str(target_path))
                                
                                if progress_callback:
                                    progress_callback(len(extracted_files), clean_name)
            
            extraction_time = time.time() - start_time
            
            result = {
                'success': True,
                'extracted_count': len(extracted_files),
                'extracted_files': extracted_files,
                'extraction_time_seconds': extraction_time,
                'extract_dir': str(extract_dir),
                'source_archive': str(archive_path)
            }
            
            logger.info(f"Extracted {len(extracted_files)} files from {archive_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_count': len(extracted_files),
                'source_archive': str(archive_path)
            }

    def extract_bulk_file(self, archive_path: Path, extract_dir: Path,
                         max_documents: Optional[int] = None,
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        
        Note: This now works with individual files instead of bulk archives.
        """
        return self.extract_individual_file(archive_path, extract_dir, progress_callback)
    
    def search_pmc_ids(self, query: str, max_results: int = 10000) -> List[str]:
        """
        Search for PMC IDs using E-utilities.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of PMC IDs
        """
        try:
            # Use esearch to find PMC IDs
            search_url = urljoin(self.EUTILS_BASE, "esearch.fcgi")
            params = {
                'db': 'pmc',
                'term': query,
                'retmax': max_results,
                'retmode': 'xml',
                'usehistory': 'y'
            }
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            id_list = root.find('.//IdList')
            
            if id_list is not None:
                pmc_ids = [id_elem.text for id_elem in id_list.findall('Id')]
                logger.info(f"Found {len(pmc_ids)} PMC IDs for query: {query}")
                return pmc_ids
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching PMC IDs: {e}")
            return []
    
    def get_document_metadata(self, pmc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get metadata for PMC documents.
        
        Args:
            pmc_ids: List of PMC IDs
            
        Returns:
            List of metadata dictionaries
        """
        if not pmc_ids:
            return []
        
        try:
            # Use esummary to get metadata
            summary_url = urljoin(self.EUTILS_BASE, "esummary.fcgi")
            
            # Process in batches to avoid URL length limits
            batch_size = 200
            all_metadata = []
            
            for i in range(0, len(pmc_ids), batch_size):
                batch_ids = pmc_ids[i:i + batch_size]
                
                params = {
                    'db': 'pmc',
                    'id': ','.join(batch_ids),
                    'retmode': 'xml'
                }
                
                response = self.session.get(summary_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                for doc_sum in root.findall('.//DocSum'):
                    metadata = {'pmc_id': None}
                    
                    # Extract ID
                    id_elem = doc_sum.find('Id')
                    if id_elem is not None:
                        metadata['pmc_id'] = id_elem.text
                    
                    # Extract other metadata
                    for item in doc_sum.findall('.//Item'):
                        name = item.get('Name')
                        if name and item.text:
                            metadata[name.lower()] = item.text
                    
                    all_metadata.append(metadata)
                
                # Rate limiting
                time.sleep(self.request_delay)
            
            logger.info(f"Retrieved metadata for {len(all_metadata)} documents")
            return all_metadata
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return []